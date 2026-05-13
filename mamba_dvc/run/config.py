"""Campaign config parsing + variant expansion for the batch driver.

A campaign config (YAML) names a set of zarr stores, which deformations
to run per store, a base set of correlation parameters, and a sweep /
list of parameter variants. :class:`BatchSpec` parses and validates it;
expansion turns ``defaults x sweep + variants`` into a flat, deduped
list of :class:`Variant` objects, each carrying a stable identity hash
and its parameters pre-split into the three loop tiers (load / grid /
compute) the execution loop in :mod:`mamba_dvc.run.batch` needs.

See ``docs/plans/run-interface.md`` §3-§4 for the tier model and the
config schema.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

__all__ = ["BatchSpec", "SelectSpec", "Variant", "knob_names", "slug_value"]


# ``"all"`` / ``"none"`` are literal sentinels; a tuple of strings names
# an explicit subset of deformation entries.
SelectSpec = Literal["all", "none"] | tuple[str, ...]


# --------------------------------------------------------------- knob model

# Every parameter the driver can vary, with its built-in default. A
# config's ``defaults:`` block layers on top of these; ``sweep`` /
# ``variants`` layer on top of that. Values mirror
# ``correlate_multi_gpu``'s signature where they overlap, plus the three
# load-tier knobs (``mask`` / ``dry_shape`` / ``flow_convention``) that
# are consumed by the dataset layer rather than by dispatch.
_BUILTIN_DEFAULTS: dict[str, Any] = {
    # grid tier — changing these re-runs correlate AND invalidates the
    # cached GT-truth array (the POI lattice moves).
    "window": 96,
    "overlap": 0.5,
    # compute tier — changing these re-runs correlate only; ``positions``
    # is unchanged so the GT-truth cache survives.
    "mask_threshold": 0.9,
    "batch_size": "auto",
    "ncc_mode": "linear",
    "ncc_normalization": "overlap",
    "search_radius": None,
    "tukey_alpha": None,
    # load tier — changing these re-materializes the pair (zarr read;
    # rebuilds the GT field).
    "mask": None,  # None → profile/manifest default; "none" → no mask
    "dry_shape": None,  # None → full volume; [z, y, x] → centered subblock
    "flow_convention": None,  # None → defer to manifest/profile (warned)
}

_LOAD_KEYS: tuple[str, ...] = ("mask", "dry_shape", "flow_convention")
_GRID_KEYS: tuple[str, ...] = ("window", "overlap")
_COMPUTE_KEYS: tuple[str, ...] = tuple(
    k for k in _BUILTIN_DEFAULTS if k not in _LOAD_KEYS and k not in _GRID_KEYS
)

# Short prefixes for the auto-generated variant slug (the human-readable
# part of ``variant_id``). Only knobs that differ from the resolved
# defaults appear in a slug.
_SLUG_ABBREV: dict[str, str] = {
    "window": "w",
    "overlap": "ov",
    "mask_threshold": "mt",
    "batch_size": "bs",
    "ncc_mode": "ncc",
    "ncc_normalization": "nn",
    "search_radius": "sr",
    "tukey_alpha": "ta",
    "mask": "m",
    "dry_shape": "dry",
    "flow_convention": "fc",
}

_FLOW_CONVENTIONS: frozenset[str] = frozenset({"pull_back", "push_forward"})


def knob_names() -> frozenset[str]:
    """Names of every parameter knob the driver understands."""
    return frozenset(_BUILTIN_DEFAULTS)


# Top-level config keys we recognise. An unknown key is a hard error so
# typos (``store:`` for ``stores:``) don't silently no-op.
_KNOWN_TOP_KEYS: frozenset[str] = frozenset(
    {
        "campaign",
        "stores",
        "manifest",
        "strict",
        "select",
        "defaults",
        "sweep",
        "variants",
        "devices",
        "distance_bins",
        "prefetch",
        "out_dir",
    }
)


# ------------------------------------------------------------------- Variant


@dataclass(frozen=True)
class Variant:
    """One fully-resolved parameter set plus its stable identity.

    Parameters
    ----------
    variant_id
        ``f"{name}-{params_hash}"``. Used as the result filename stem;
        stable across runs of the same config.
    name
        Human-readable label — an explicit ``variants[].name`` from the
        config, or an auto-slug of the knobs that differ from the
        resolved defaults (``mt0.5``, ``mt0.3-w64``), or ``"base"`` when
        nothing differs.
    params_hash
        First 8 hex chars of ``sha256`` over the canonical (sorted-key)
        JSON of :attr:`params`. The true variant identity; the resume
        check matches on this.
    params
        The full resolved parameter dict (union of the three tier
        slices below).
    load_params
        ``mask`` / ``dry_shape`` / ``flow_convention`` — consumed by the
        dataset layer (``DvcDataset.load_pair`` and the manifest /
        profile convention resolution).
    grid_params
        ``window`` / ``overlap`` — forwarded to ``correlate_multi_gpu``;
        also key the GT-truth cache.
    compute_params
        The remaining correlation knobs (``mask_threshold``,
        ``batch_size``, ``ncc_mode``, ``ncc_normalization``,
        ``search_radius``, ``tukey_alpha``) — forwarded to
        ``correlate_multi_gpu``.
    """

    variant_id: str
    name: str
    params_hash: str
    params: Mapping[str, Any]
    load_params: Mapping[str, Any]
    grid_params: Mapping[str, Any]
    compute_params: Mapping[str, Any]


# ----------------------------------------------------------------- BatchSpec


@dataclass(frozen=True)
class BatchSpec:
    """Parsed + validated campaign config.

    Build with :meth:`from_yaml` (production) or :meth:`from_mapping`
    (tests / programmatic). All path fields are taken as-is: a relative
    path is relative to the process working directory at run time, not
    to the config file.

    Parameters
    ----------
    campaign
        Campaign name → results land under ``out_dir / campaign``.
    stores
        Zarr store paths to iterate over, in order.
    manifest
        Optional shared :class:`~mamba_dvc.io.manifest.StoreManifest`
        YAML applied to every store. ``None`` falls back to per-store
        discovery (sidecar YAML / embedded ``.zattrs``).
    strict
        Passed through to :meth:`DvcDataset.open`. ``True`` (the default)
        aborts on a store that fails verification; ``False`` opens it
        anyway and processes only its healthy entries (the broken ones
        land in ``broken_entries`` and drop out of ``list_*``). Set it
        ``False`` when a store has a known-bad sub-entry you want to skip
        rather than lose the whole store over.
    select_real, select_synthetic
        Which deformation entries to run per store: ``"all"``,
        ``"none"``, or an explicit tuple of names. Real entries have no
        ground truth — they are correlated and persisted but not scored.
    variants
        The expanded, deduped list of :class:`Variant` objects.
    devices
        GPU ids forwarded to ``correlate_multi_gpu``. ``None`` selects
        every visible device.
    distance_bins
        Boundary-stratification edges forwarded to ``evaluate_pair``.
        Empty (the default) skips the (expensive) EDT entirely.
    prefetch
        How many ``(deformation, mask, dry_shape)`` load groups the
        execution loop fetches ahead in a background thread while the
        current group's variants run (see :mod:`mamba_dvc.run.batch`).
        ``0`` disables prefetch (purely synchronous loads). ``1`` (the
        default) overlaps the *next* group's zarr read with the current
        group's correlate/evaluate — roughly halving campaign wall time
        when a load costs about as much as a group's variants. Larger
        values keep more loads queued but only one runs at a time (one
        loader thread), so ``1`` is the practical sweet spot; the
        prefetch is also skipped step-by-step when free host RAM looks
        too tight to hold another materialized pair.
    out_dir
        Root output directory; the campaign directory is
        ``out_dir / campaign``.
    source_path
        Path the config was read from (via :meth:`from_yaml`), or
        ``None``. The driver copies this into the campaign directory as
        ``config.snapshot.yaml``; when ``None`` it serialises
        :attr:`raw` instead.
    raw
        The parsed YAML mapping, retained for the config snapshot.
    """

    campaign: str
    stores: tuple[Path, ...]
    manifest: Path | None
    strict: bool
    select_real: SelectSpec
    select_synthetic: SelectSpec
    variants: tuple[Variant, ...]
    devices: tuple[int, ...] | None
    distance_bins: tuple[float, ...]
    prefetch: int
    out_dir: Path
    source_path: Path | None
    raw: Mapping[str, Any]

    # --------------------------------------------------------- factories

    @classmethod
    def from_yaml(cls, path: Path) -> BatchSpec:
        """Parse and validate a campaign config YAML file.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file is not a mapping or fails schema validation.
        """
        if not path.exists():
            raise FileNotFoundError(f"campaign config not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, Mapping):
            raise ValueError(f"campaign config at {path} must be a mapping")
        return cls.from_mapping(cast("Mapping[str, Any]", raw), source_path=path)

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, source_path: Path | None = None
    ) -> BatchSpec:
        """Build a :class:`BatchSpec` from an already-parsed mapping.

        Raises
        ------
        ValueError
            On any schema violation (unknown key, missing ``campaign``
            or ``stores``, malformed ``select`` / ``sweep`` / ``variants``,
            unknown parameter knob, bad value type).
        """
        unknown = set(raw) - _KNOWN_TOP_KEYS
        if unknown:
            raise ValueError(
                f"unknown campaign config key(s): {sorted(unknown)}; "
                f"known keys are {sorted(_KNOWN_TOP_KEYS)}"
            )

        campaign = raw.get("campaign")
        if not isinstance(campaign, str) or not campaign.strip():
            raise ValueError("campaign config must set a non-empty 'campaign' name")

        stores_raw = raw.get("stores")
        if (
            not isinstance(stores_raw, Sequence)
            or isinstance(stores_raw, str)
            or not stores_raw
        ):
            raise ValueError("campaign config 'stores' must be a non-empty list of paths")
        stores = tuple(Path(str(s)) for s in cast("Sequence[Any]", stores_raw))

        manifest_raw = raw.get("manifest")
        manifest = Path(str(manifest_raw)) if manifest_raw is not None else None

        strict_raw = raw.get("strict", True)
        if not isinstance(strict_raw, bool):
            raise ValueError(f"campaign config 'strict' must be a bool, got {strict_raw!r}")
        strict = strict_raw

        select_raw = raw.get("select", {})
        if not isinstance(select_raw, Mapping):
            raise ValueError("campaign config 'select' must be a mapping")
        select_map = cast("Mapping[str, Any]", select_raw)
        select_real = _parse_select(select_map.get("real", "all"), "select.real")
        select_synthetic = _parse_select(
            select_map.get("synthetic", "all"), "select.synthetic"
        )

        devices_raw = raw.get("devices")
        devices = _parse_devices(devices_raw)

        distance_bins = _parse_distance_bins(raw.get("distance_bins", []))

        prefetch = _parse_prefetch(raw.get("prefetch", 1))

        out_dir = Path(str(raw.get("out_dir", "results")))

        variants = _expand_variants(
            defaults_raw=raw.get("defaults", {}),
            sweep_raw=raw.get("sweep", {}),
            variants_raw=raw.get("variants", []),
        )

        return cls(
            campaign=campaign,
            stores=stores,
            manifest=manifest,
            strict=strict,
            select_real=select_real,
            select_synthetic=select_synthetic,
            variants=variants,
            devices=devices,
            distance_bins=distance_bins,
            prefetch=prefetch,
            out_dir=out_dir,
            source_path=source_path,
            raw=dict(raw),
        )

    # --------------------------------------------------------- accessors

    @property
    def campaign_dir(self) -> Path:
        """Directory the campaign's results live under (``out_dir / campaign``)."""
        return self.out_dir / self.campaign

    def select_names(
        self, real: Sequence[str], synthetic: Sequence[str]
    ) -> list[tuple[str, str]]:
        """Resolve the ``select`` spec against a store's actual entries.

        Parameters
        ----------
        real, synthetic
            The store's available real / synthetic deformation names
            (e.g. ``DvcDataset.list_real()`` / ``.list_synthetic()``).

        Returns
        -------
        list of tuple
            ``(name, kind)`` pairs where ``kind`` is ``"real"`` or
            ``"synthetic"``, in stable order (synthetic after real, each
            sorted). Names requested explicitly but absent from the store
            are silently skipped — the caller's resume / manifest logs
            record what actually ran.
        """
        out: list[tuple[str, str]] = []
        out.extend((n, "real") for n in _resolve_select(self.select_real, real))
        out.extend((n, "synthetic") for n in _resolve_select(self.select_synthetic, synthetic))
        return out


# ------------------------------------------------------------- parse helpers


def _parse_select(value: Any, where: str) -> SelectSpec:
    if value == "all" or value == "none":
        return value
    if isinstance(value, Sequence) and not isinstance(value, str):
        seq = cast("Sequence[Any]", value)
        if not all(isinstance(v, str) for v in seq):
            raise ValueError(f"{where} list entries must be strings")
        return tuple(str(v) for v in seq)
    raise ValueError(f"{where} must be 'all', 'none', or a list of names")


def _resolve_select(spec: SelectSpec, available: Sequence[str]) -> list[str]:
    if spec == "all":
        return list(available)
    if spec == "none":
        return []
    available_set = set(available)
    return [n for n in spec if n in available_set]


def _parse_devices(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("'devices' must be a list of GPU ids or null")
    try:
        ids = tuple(int(v) for v in cast("Sequence[Any]", value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'devices' entries must be ints, got {value!r}") from exc
    if not ids:
        raise ValueError("'devices' must be non-empty when set; use null for all visible")
    if any(d < 0 for d in ids):
        raise ValueError(f"'devices' entries must be non-negative, got {value!r}")
    return ids


def _parse_distance_bins(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("'distance_bins' must be a list of numbers")
    try:
        edges = tuple(float(v) for v in cast("Sequence[Any]", value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'distance_bins' entries must be numbers, got {value!r}") from exc
    if list(edges) != sorted(edges):
        raise ValueError(f"'distance_bins' must be non-decreasing, got {value!r}")
    return edges


def _parse_prefetch(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'prefetch' must be a non-negative int, got {value!r}")
    if value < 0:
        raise ValueError(f"'prefetch' must be a non-negative int, got {value!r}")
    return int(value)


# ------------------------------------------------------- variant expansion


def _normalize_value(key: str, value: Any) -> Any:
    """Coerce one knob value into its canonical, JSON-serialisable form.

    A ``[z, y, x]`` list (``window`` / ``dry_shape``) becomes a 3-tuple
    of ints; a cubic ``window`` stays an ``int``; everything else passes
    through after light per-knob validation. The canonical form is what
    gets hashed for the variant id — write ``window`` consistently as an
    int or as a list within a campaign so the ids stay stable.
    """
    if key not in _BUILTIN_DEFAULTS:
        raise ValueError(
            f"unknown parameter knob {key!r}; known knobs are {sorted(_BUILTIN_DEFAULTS)}"
        )

    if key in ("window", "dry_shape"):
        if value is None:
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            if key == "dry_shape":
                raise ValueError("'dry_shape' must be a [z, y, x] list or null, not a scalar")
            return int(value)
        if isinstance(value, Sequence) and not isinstance(value, str):
            parts = list(cast("Sequence[Any]", value))
            if len(parts) != 3 or not all(
                isinstance(p, int) and not isinstance(p, bool) for p in parts
            ):
                raise ValueError(f"{key!r} list must be three ints, got {value!r}")
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        raise ValueError(f"{key!r} must be an int (cubic) or a [z, y, x] list, got {value!r}")

    if key == "batch_size":
        if value == "auto":
            return "auto"
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
        raise ValueError(f"'batch_size' must be a positive int or 'auto', got {value!r}")

    if key == "mask":
        if value is None or isinstance(value, str):
            return value
        raise ValueError(f"'mask' must be a mask name, 'none', or null, got {value!r}")

    if key == "flow_convention":
        if value is None or value in _FLOW_CONVENTIONS:
            return value
        raise ValueError(
            f"'flow_convention' must be one of {sorted(_FLOW_CONVENTIONS)} or null, "
            f"got {value!r}"
        )

    if key in ("ncc_mode", "ncc_normalization"):
        if not isinstance(value, str):
            raise ValueError(f"{key!r} must be a string, got {value!r}")
        return value

    if key == "search_radius":
        if value is None:
            return None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return int(value)
        raise ValueError(f"'search_radius' must be a positive int or null, got {value!r}")

    # mask_threshold, overlap, tukey_alpha — float (tukey_alpha may be null).
    if value is None:
        if key == "tukey_alpha":
            return None
        raise ValueError(f"{key!r} must be a number, got null")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key!r} must be a number, got {value!r}")
    return float(value)


def _resolved_defaults(defaults_raw: object) -> dict[str, Any]:
    if not isinstance(defaults_raw, Mapping):
        raise ValueError("'defaults' must be a mapping")
    overrides = cast("Mapping[str, Any]", defaults_raw)
    resolved = dict(_BUILTIN_DEFAULTS)
    for key, value in overrides.items():
        resolved[str(key)] = _normalize_value(str(key), value)
    # Re-normalise the built-ins too so the canonical form is uniform
    # (e.g. ``mask_threshold`` becomes a float even if untouched).
    return {k: _normalize_value(k, v) for k, v in resolved.items()}


def _sweep_points(sweep_raw: object) -> list[dict[str, Any]]:
    """Cartesian product over the named ``sweep`` axes → list of overlay dicts.

    Empty / absent ``sweep`` yields ``[{}]`` (a single empty overlay), so
    the caller always materialises at least the defaults variant.
    """
    if sweep_raw is None or sweep_raw == {}:
        return [{}]
    if not isinstance(sweep_raw, Mapping):
        raise ValueError("'sweep' must be a mapping of knob → list of values")
    sweep = cast("Mapping[str, Any]", sweep_raw)
    axes: list[tuple[str, list[Any]]] = []
    for key, values in sweep.items():
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ValueError(f"'sweep.{key}' must be a list of values, got {values!r}")
        seq = cast("Sequence[Any]", values)
        if not seq:
            raise ValueError(f"'sweep.{key}' must be a non-empty list")
        axes.append((str(key), [_normalize_value(str(key), v) for v in seq]))
    if not axes:
        return [{}]
    keys = [k for k, _ in axes]
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*[vals for _, vals in axes]):
        out.append(dict(zip(keys, combo, strict=True)))
    return out


def _explicit_variants(variants_raw: object) -> list[tuple[str | None, dict[str, Any]]]:
    if variants_raw is None or variants_raw == []:
        return []
    if not isinstance(variants_raw, Sequence) or isinstance(variants_raw, str):
        raise ValueError("'variants' must be a list of mappings")
    entries = cast("Sequence[Any]", variants_raw)
    out: list[tuple[str | None, dict[str, Any]]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"'variants[{i}]' must be a mapping")
        overlay_raw = cast("Mapping[str, Any]", entry)
        name = overlay_raw.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError(f"'variants[{i}].name' must be a non-empty string")
        overlay = {
            str(k): _normalize_value(str(k), v) for k, v in overlay_raw.items() if k != "name"
        }
        out.append((str(name) if name is not None else None, overlay))
    return out


def _split_tiers(
    params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    load = {k: params[k] for k in _LOAD_KEYS}
    grid = {k: params[k] for k in _GRID_KEYS}
    compute = {k: params[k] for k in _COMPUTE_KEYS}
    return load, grid, compute


def _canonical_json(params: Mapping[str, Any]) -> str:
    """JSON of ``params`` with sorted keys and tuples flattened to lists."""

    def _plain(value: Any) -> Any:
        if isinstance(value, tuple):
            return [_plain(v) for v in cast("tuple[Any, ...]", value)]
        return value

    return json.dumps({k: _plain(params[k]) for k in sorted(params)}, sort_keys=True)


def _params_hash(params: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(params).encode("utf-8")).hexdigest()[:8]


def slug_value(value: Any) -> str:
    """Render a knob value as a short, filename-safe token.

    ``None`` → ``"none"``; floats use ``%g`` (so ``5.0`` → ``"5"`` but
    ``0.5`` → ``"0.5"``); tuples join on ``"x"`` (``(64, 64, 64)`` →
    ``"64x64x64"``). Used both for the auto-slug in :attr:`Variant.name`
    and for ``--only KEY=VALUE`` matching, so the two stay consistent.
    """
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "x".join(slug_value(v) for v in cast("tuple[Any, ...]", value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _auto_slug(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in _BUILTIN_DEFAULTS:  # stable, declaration order
        if params[key] != defaults[key]:
            parts.append(f"{_SLUG_ABBREV[key]}{slug_value(params[key])}")
    return "-".join(parts) if parts else "base"


def _expand_variants(
    *, defaults_raw: Any, sweep_raw: Any, variants_raw: Any
) -> tuple[Variant, ...]:
    """``defaults x sweep + variants`` → deduped tuple of :class:`Variant`.

    Sweep-generated variants are enumerated first, then the explicit
    ``variants`` entries; duplicates (identical resolved params) collapse
    to the first occurrence, so an explicit entry that re-states a sweep
    point keeps the sweep point's (auto-slug) name.
    """
    defaults = _resolved_defaults(defaults_raw)

    overlays: list[tuple[str | None, dict[str, Any]]] = []
    overlays.extend((None, pt) for pt in _sweep_points(sweep_raw))
    overlays.extend(_explicit_variants(variants_raw))

    seen: dict[str, Variant] = {}
    ordered: list[Variant] = []
    for name_override, overlay in overlays:
        params = dict(defaults)
        for key, value in overlay.items():
            params[key] = value  # already normalised
        h = _params_hash(params)
        if h in seen:
            continue
        name = name_override if name_override is not None else _auto_slug(params, defaults)
        load, grid, compute = _split_tiers(params)
        variant = Variant(
            variant_id=f"{name}-{h}",
            name=name,
            params_hash=h,
            params=params,
            load_params=load,
            grid_params=grid,
            compute_params=compute,
        )
        seen[h] = variant
        ordered.append(variant)
    return tuple(ordered)
