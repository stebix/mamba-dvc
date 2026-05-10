"""``mamba-dvc budget`` subcommand: VRAM-aware batch-size oracle.

Pure inspection -- does *not* run correlation. The :func:`inspect`
analogue for the batch-size oracle: "what would dispatch do on this
machine right now?". Calls :func:`mamba_dvc.gpu.budget.probe_free_vram`
to read live free VRAM and prints the recommendation alongside the
derivation, so a user can see *why* a number was chosen.

Plan: ``docs/plans/batchsize-oracle-v1.md`` §10.3.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mamba_dvc.cli._common import resolve_manifest
from mamba_dvc.core.ncc import NCCMode, NCCNormalization
from mamba_dvc.gpu.budget import (
    BudgetInputs,
    KernelFootprint,
    estimate_max_batch,
    is_cupy_available,
    kernel_footprint,
    probe_free_vram,
    resident_bytes,
)
from mamba_dvc.io.dataset import DvcDataset

__all__ = ["budget"]


# --------------------------------------------------------------- arg parsers


def _parse_volume_shape(spec: str) -> tuple[int, int, int]:
    """Parse ``"Z,Y,X"`` into a positive 3-tuple of ints."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise typer.BadParameter(
            f"--volume-shape must be three comma-separated ints, got {spec!r}"
        )
    try:
        z, y, x = (int(p) for p in parts)
    except ValueError as exc:
        raise typer.BadParameter(f"--volume-shape entries must be ints, got {spec!r}") from exc
    if z <= 0 or y <= 0 or x <= 0:
        raise typer.BadParameter(f"--volume-shape entries must be positive, got {spec!r}")
    return (z, y, x)


def _parse_window(spec: str) -> tuple[int, int, int]:
    """Parse ``"96"`` (cube) or ``"WZ,WY,WX"`` into a positive 3-tuple."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) == 1:
        try:
            w = int(parts[0])
        except ValueError as exc:
            raise typer.BadParameter(
                f"--window must be a positive int or three comma-separated ints, got {spec!r}"
            ) from exc
        if w <= 0:
            raise typer.BadParameter(f"--window must be positive, got {w}")
        return (w, w, w)
    if len(parts) == 3:
        try:
            wz, wy, wx = (int(p) for p in parts)
        except ValueError as exc:
            raise typer.BadParameter(f"--window entries must be ints, got {spec!r}") from exc
        if wz <= 0 or wy <= 0 or wx <= 0:
            raise typer.BadParameter(f"--window entries must be positive, got {spec!r}")
        return (wz, wy, wx)
    raise typer.BadParameter(
        f"--window must be a positive int or three comma-separated ints, got {spec!r}"
    )


def _parse_devices(spec: str) -> list[int]:
    """Parse ``"0,1,2,3"`` into a non-empty list of non-negative ints."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise typer.BadParameter("--devices must list at least one device id")
    try:
        ids = [int(p) for p in parts]
    except ValueError as exc:
        raise typer.BadParameter(f"--devices entries must be ints, got {spec!r}") from exc
    if any(d < 0 for d in ids):
        raise typer.BadParameter(f"--devices entries must be non-negative, got {spec!r}")
    return ids


def _enumerate_devices() -> list[int]:
    """Return every CUDA device visible to CuPy.

    Mirrors :func:`mamba_dvc.gpu.dispatch._resolve_device_ids` for the
    "no --devices flag" default. The caller has already gated on
    :func:`is_cupy_available`, so the import here is safe.
    """
    import cupy as cp  # pyright: ignore[reportMissingImports]

    n = int(cp.cuda.runtime.getDeviceCount())
    if n <= 0:
        raise RuntimeError("no CUDA devices visible to CuPy")
    return list(range(n))


# ------------------------------------------------------------------- report


@dataclass(frozen=True)
class _BudgetReport:
    """Resolved inputs + probe + recommendation, for shared rendering.

    The rich and JSON paths read this same object so the two outputs
    can never disagree on what was decided.
    """

    inputs: BudgetInputs
    per_device: tuple[tuple[int, int, int], ...]  # (device_id, free, total)
    headroom: float
    ceiling: int | None
    batch_size: int
    footprint: KernelFootprint
    resident: int


# ------------------------------------------------------------------ command


def budget(
    store: Annotated[
        Path | None,
        typer.Option(
            "--store",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Zarr store to size against; mutually exclusive with --volume-shape.",
        ),
    ] = None,
    volume_shape: Annotated[
        str | None,
        typer.Option(
            "--volume-shape",
            help="Hypothetical volume shape Z,Y,X; mutually exclusive with --store.",
        ),
    ] = None,
    window: Annotated[
        str,
        typer.Option(
            "--window",
            "-w",
            help="Subvolume size, either a single int (cubic) or WZ,WY,WX.",
        ),
    ] = "96",
    mode: Annotated[
        NCCMode,
        typer.Option(
            "--mode",
            help="FFT-NCC kernel selection.",
            case_sensitive=False,
        ),
    ] = NCCMode.LINEAR,
    normalization: Annotated[
        NCCNormalization,
        typer.Option(
            "--normalization",
            help="Per-POI denominator convention.",
            case_sensitive=False,
        ),
    ] = NCCNormalization.OVERLAP,
    devices: Annotated[
        str | None,
        typer.Option(
            "--devices",
            help="Comma-separated CUDA device ids; default: every visible device.",
        ),
    ] = None,
    headroom: Annotated[
        float,
        typer.Option(
            "--headroom",
            min=0.0,
            max=0.999,
            help="Fraction of free VRAM to reserve for plan cache + fragmentation.",
        ),
    ] = 0.15,
    ceiling: Annotated[
        int | None,
        typer.Option(
            "--ceiling",
            min=1,
            help="Optional upper clamp on batch_size (e.g. to match a numeric run).",
        ),
    ] = None,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable JSON report."),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable styled terminal output."),
    ] = False,
) -> None:
    """Show what ``batch_size`` ``correlate_multi_gpu`` would auto-pick.

    Probes free VRAM on each requested device, takes the minimum, and
    reports the largest batch that fits the cost model with the given
    ``headroom``. Exit code is ``0`` when a recommendation is rendered,
    ``1`` if the budget is too tight (RuntimeError from the recommender
    or a missing CuPy install), and ``2`` for malformed input.
    """
    # 1. --store XOR --volume-shape.
    if (store is None) == (volume_shape is None):
        typer.echo("error: pass exactly one of --store or --volume-shape", err=True)
        raise typer.Exit(2)

    # 2. CuPy is required: budget probes live free VRAM.
    if not is_cupy_available():
        typer.echo(
            "error: `mamba-dvc budget` requires CuPy; install it on this host or "
            "pass batch_size explicitly to correlate_multi_gpu",
            err=True,
        )
        raise typer.Exit(1)

    # 3. Resolve volume shape + has_mask (from the store, or from the flag).
    if store is not None:
        try:
            manifest_obj, _source = resolve_manifest(
                store, manifest_path=None, profile_override=None
            )
            ds = DvcDataset.open(store, manifest=manifest_obj, strict=False)
        except (FileNotFoundError, KeyError, LookupError, ValueError) as exc:
            typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
            raise typer.Exit(2) from exc
        ref_shape = ds.reference.shape
        resolved_volume_shape = (int(ref_shape[0]), int(ref_shape[1]), int(ref_shape[2]))
        has_mask = bool(ds.masks)
    else:
        # volume_shape is non-None here because of the XOR check above.
        assert volume_shape is not None
        resolved_volume_shape = _parse_volume_shape(volume_shape)
        # Without a store we have nothing to inspect; default to the
        # production case (mask present), matching dispatch's residency.
        has_mask = True

    # 4. Parse window (mode and normalization are typer-validated enums).
    resolved_window = _parse_window(window)

    # 5. Resolve device list.
    try:
        device_ids = _enumerate_devices() if devices is None else _parse_devices(devices)
    except RuntimeError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    # 6. BudgetInputs (validates window vs volume_shape, etc.).
    try:
        inputs = BudgetInputs(
            volume_shape=resolved_volume_shape,
            window=resolved_window,
            mode=mode,
            normalization=normalization,
            has_mask=has_mask,
            # v1 dispatch publishes the same SHM under both names when
            # deformed_mask is None, so the budget tool models the
            # falling-back case (the conservative pre-flight).
            deformed_mask_distinct=False,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    # 7. Probe each device. RuntimeError here means CuPy lost the device.
    try:
        per_device = tuple((d, *probe_free_vram(d)) for d in device_ids)
    except (RuntimeError, ValueError) as exc:
        # ValueError: negative device id (already filtered by parser, but
        # cheap to keep the safety net). RuntimeError: CuPy gone or device
        # gone. Either way, exit 1 -- not a parse error.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    min_free = min(free for _, free, _ in per_device)

    # 8. Recommend.
    try:
        kwargs: dict[str, float | int] = {"headroom_fraction": headroom}
        if ceiling is not None:
            kwargs["max_batch"] = ceiling
        batch = estimate_max_batch(min_free, inputs, **kwargs)  # pyright: ignore[reportArgumentType]
    except ValueError as exc:
        # estimate_max_batch raises ValueError for bad parameters
        # (out-of-range headroom, etc.) -- treat as malformed input.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    except RuntimeError as exc:
        # Underflow: budget too tight. Plan §5: clean message, no traceback.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    fp = kernel_footprint(inputs.window, inputs.mode, inputs.normalization)
    res = resident_bytes(inputs)
    report = _BudgetReport(
        inputs=inputs,
        per_device=per_device,
        headroom=headroom,
        ceiling=ceiling,
        batch_size=batch,
        footprint=fp,
        resident=res,
    )

    if json_out:
        typer.echo(_render_json(report))
    else:
        console = Console(no_color=no_color, width=120)
        _render_rich(console, report)

    raise typer.Exit(0)


# ----------------------------------------------------------------- rendering


def _render_rich(console: Console, report: _BudgetReport) -> None:
    """Print the three-panel rich report."""
    console.print(_inputs_panel(report))
    console.print(_vram_table(report))
    console.print(_recommendation_panel(report))


def _inputs_panel(report: _BudgetReport) -> Panel:
    """Top panel: the resolved :class:`BudgetInputs`."""
    inputs = report.inputs
    body = Text()
    body.append("volume shape  ", style="bold")
    body.append(f"{inputs.volume_shape}\n")
    body.append("window        ", style="bold")
    body.append(f"{inputs.window}\n")
    body.append("mode          ", style="bold")
    body.append(f"{inputs.mode.value}\n")
    body.append("normalization ", style="bold")
    body.append(f"{inputs.normalization.value}\n")
    body.append("has_mask      ", style="bold")
    body.append(f"{inputs.has_mask}\n")
    body.append("deformed_mask ", style="bold")
    body.append(f"distinct={inputs.deformed_mask_distinct}")
    return Panel(body, title="inputs", border_style="blue", expand=False)


def _vram_table(report: _BudgetReport) -> Table:
    """Per-device VRAM probe; the limiting device is highlighted."""
    min_free = min(free for _, free, _ in report.per_device)
    table = Table(title="VRAM probe", title_style="bold blue", expand=False)
    table.add_column("device", justify="right")
    table.add_column("free", justify="right")
    table.add_column("total", justify="right")
    table.add_column("%free", justify="right")
    table.add_column("", justify="left")  # limit marker
    for device_id, free, total in report.per_device:
        pct = 100.0 * free / total if total > 0 else 0.0
        marker = ""
        style = None
        if free == min_free:
            marker = "<- limit"
            style = "bold yellow"
        table.add_row(
            str(device_id),
            _human_bytes(free),
            _human_bytes(total),
            f"{pct:.1f}%",
            marker,
            style=style,
        )
    return table


def _recommendation_panel(report: _BudgetReport) -> Panel:
    """Recommendation + derivation + the satisfied inequality."""
    min_free = min(free for _, free, _ in report.per_device)
    fp = report.footprint
    available = int(min_free * (1.0 - report.headroom))
    spent = report.resident + fp.fixed_bytes + fp.per_poi_bytes * report.batch_size
    ceiling_label = "none" if report.ceiling is None else str(report.ceiling)

    body = Text()
    body.append(f"batch_size = {report.batch_size}\n", style="bold green")
    body.append("\nderivation\n", style="bold")
    body.append(f"  resident   = {_human_bytes(report.resident)}\n", style="dim")
    body.append(f"  per_poi    = {_human_bytes(fp.per_poi_bytes)}\n", style="dim")
    body.append(f"  fixed      = {_human_bytes(fp.fixed_bytes)}\n", style="dim")
    body.append(f"  headroom   = {report.headroom * 100:.1f}%\n", style="dim")
    body.append(f"  ceiling    = {ceiling_label}\n", style="dim")
    body.append("\ninequality (must hold)\n", style="bold")
    body.append(
        "  free * (1 - headroom) >= resident + fixed + per_poi * batch\n",
        style="dim italic",
    )
    body.append(
        f"  {_human_bytes(available)} >= {_human_bytes(spent)}",
        style="green" if available >= spent else "red",
    )
    return Panel(body, title="recommendation", border_style="green", expand=False)


def _human_bytes(n: int) -> str:
    """Format a byte count with binary units (KiB, MiB, ...)."""
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if f < 1024 or unit == "TiB":
            return f"{f:.2f} {unit}"
        f /= 1024
    return f"{f:.2f} TiB"


# --------------------------------------------------------------- json output


def _render_json(report: _BudgetReport) -> str:
    """Serialize the report as indented JSON parallel to ``--json`` on inspect."""
    inputs = report.inputs
    min_free = min(free for _, free, _ in report.per_device)
    limiting = next(d for d, free, _ in report.per_device if free == min_free)
    payload: dict[str, object] = {
        "inputs": {
            "volume_shape": list(inputs.volume_shape),
            "window": list(inputs.window),
            "mode": inputs.mode.value,
            "normalization": inputs.normalization.value,
            "has_mask": inputs.has_mask,
            "deformed_mask_distinct": inputs.deformed_mask_distinct,
        },
        "probe": {
            "devices": [
                {
                    "id": d,
                    "free_bytes": free,
                    "total_bytes": total,
                    "percent_free": (100.0 * free / total) if total > 0 else 0.0,
                }
                for d, free, total in report.per_device
            ],
            "min_free_bytes": min_free,
            "limiting_device": limiting,
        },
        "recommendation": {
            "batch_size": report.batch_size,
            "headroom_fraction": report.headroom,
            "ceiling": report.ceiling,
            "resident_bytes": report.resident,
            "per_poi_bytes": report.footprint.per_poi_bytes,
            "fixed_bytes": report.footprint.fixed_bytes,
        },
    }
    return _json.dumps(payload, indent=2)
