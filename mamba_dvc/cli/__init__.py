"""Command-line interface for ``mamba-dvc``.

Hosts the :class:`typer.Typer` app and the ``inspect`` subcommand.
Subcommands that share helpers (manifest discovery, profile loading)
import them from :mod:`mamba_dvc.cli._common`; subcommands that grow
into more than a screen of code get their own private module
(see :mod:`mamba_dvc.cli._budget`).

Entry point: ``[project.scripts] mamba-dvc = "mamba_dvc.cli:app"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
import zarr
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from mamba_dvc.cli._budget import budget
from mamba_dvc.cli._common import resolve_manifest
from mamba_dvc.io.dataset import BrokenEntry, DvcDataset
from mamba_dvc.io.verify import VerificationReport

__all__ = ["app", "budget", "inspect"]


app = typer.Typer(
    name="mamba-dvc",
    help="FFT-based digital volume correlation tooling.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:  # pyright: ignore[reportUnusedFunction]
    """Top-level callback that keeps subcommand dispatch enabled.

    Without this, Typer collapses a single-command app to a flat CLI
    (``mamba-dvc <args>``). The callback is a no-op but forces the
    subcommand layer (``mamba-dvc inspect <args>``) so additional
    commands slot in without changing the user-facing invocation.
    """


# Register subcommands defined in private modules. Splitting them out
# keeps this file focused on inspect; the registration happens here so
# every subcommand sits under the same Typer app and shares the root
# callback above.
app.command(name="budget")(budget)


# --------------------------------------------------------------------- inspect


@app.command()
def inspect(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Path to the .zarr store directory.",
        ),
    ],
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            "-m",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Explicit manifest YAML; overrides sidecar / .zattrs discovery.",
        ),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            "-p",
            help="Override the profile name (must be registered).",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show per-array chunks, codecs, and on-disk size.",
        ),
    ] = False,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable JSON inventory."),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable styled terminal output."),
    ] = False,
) -> None:
    """Open a zarr store and print its validity + parsed slot inventory.

    The store is opened with ``strict=False`` so a non-conforming store
    still renders whatever the verifier could parse. Exit code is
    ``0`` on a clean report, ``1`` if the verifier reported errors,
    and ``2`` for unrecoverable open failures (missing path, malformed
    manifest, unknown profile name).
    """
    try:
        manifest_obj, source = resolve_manifest(
            path, manifest_path=manifest, profile_override=profile
        )
    except (KeyError, ValueError) as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(2) from exc

    try:
        ds = DvcDataset.open(path, manifest=manifest_obj, strict=False)
    except (FileNotFoundError, KeyError, LookupError, ValueError) as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(2) from exc

    if json_out:
        typer.echo(_render_json(ds, path=path, manifest_source=source, verbose=verbose))
    else:
        # Force a generous render width so aligned leaf lines aren't
        # cropped at rich's default 80 cols (which kicks in whenever
        # stdout isn't a real TTY -- CliRunner, pipes, redirects).
        # Panels render with ``expand=False`` so they fit their content
        # rather than stretching to this width.
        console = Console(no_color=no_color, width=_RENDER_WIDTH)
        _render_rich(console, ds, path=path, manifest_source=source, verbose=verbose)

    raise typer.Exit(0 if ds.verification_report.ok else 1)


# ---------------------------------------------------------------- rendering


_RENDER_WIDTH = 240
"""Logical console width used for rendering.

Set generously so the aligned slot tree never hits rich's default 80-col
crop when stdout isn't a real TTY (``CliRunner``, pipes, redirects).
Panels are rendered with ``expand=False`` so they fit their actual
content width rather than stretching to this value.
"""


def _render_rich(
    console: Console,
    ds: DvcDataset,
    *,
    path: Path,
    manifest_source: str,
    verbose: bool,
) -> None:
    """Print header + validity panel + slot tree to ``console``."""
    console.print(_header_panel(ds, path=path, manifest_source=manifest_source))
    console.print(_validity_panel(ds.verification_report))
    console.print(_slot_tree(ds, path=path, verbose=verbose))


def _header_panel(ds: DvcDataset, *, path: Path, manifest_source: str) -> Panel:
    """Top panel: store path, resolved profile, manifest origin, spacing."""
    body = Text()
    body.append("store    ", style="bold")
    body.append(f"{path}\n")
    body.append("profile  ", style="bold")
    body.append(f"{ds.profile.name}\n")
    body.append("manifest ", style="bold")
    body.append(manifest_source)
    if ds.spacing is not None:
        body.append("\nspacing  ", style="bold")
        body.append(f"{tuple(ds.spacing.values)} {ds.spacing.unit}")
    return Panel(body, title="store", border_style="blue", expand=False)


def _validity_panel(report: VerificationReport) -> Panel:
    """Bordered panel summarising the verifier outcome."""
    if report.ok:
        body = Text("OK", style="bold green")
        if report.warnings:
            body.append(f"  ({len(report.warnings)} warning(s))", style="yellow")
            for msg in report.warnings:
                body.append(f"\n  ! {msg}", style="yellow")
        return Panel(body, title="validity", border_style="green", expand=False)

    n_err = len(report.errors)
    n_warn = len(report.warnings)
    body = Text(f"FAILED: {n_err} error(s), {n_warn} warning(s)", style="bold red")
    for msg in report.errors:
        body.append(f"\n  - {msg}", style="red")
    for msg in report.warnings:
        body.append(f"\n  ! {msg}", style="yellow")
    return Panel(body, title="validity", border_style="red", expand=False)


@dataclass(frozen=True)
class _LeafColumns:
    """Per-field column widths used to align array-leaf lines.

    Computed once per tree from a pre-walk of every array leaf. Carries
    the verbose flag so :func:`_array_line` can decide which fields to
    render without a separate parameter.
    """

    name_w: int
    shape_w: int
    dtype_w: int
    chunks_w: int
    codecs_w: int
    size_w: int
    verbose: bool


def _slot_tree(ds: DvcDataset, *, path: Path, verbose: bool) -> Tree:
    """Rich tree mirroring the on-disk group hierarchy."""
    tree = Tree(Text(path.name, style="bold"))

    profile = ds.profile
    default_mask = _resolve_default_mask(ds)
    columns = _compute_columns(ds, verbose=verbose)

    base_node = tree.add(Text(f"{profile.base_group}/", style="bold blue"))
    base_node.add(_array_line(profile.reference_name, ds.reference, ["reference"], columns))
    for name in sorted(ds.masks):
        tags = ["default"] if name == default_mask else []
        base_node.add(_array_line(name, ds.masks[name], tags, columns))

    real_names = ds.list_real()
    real_broken = [b for b in ds.broken_entries.values() if b.kind == "real"]
    real_node = tree.add(_group_header(profile.real_group, len(real_names), len(real_broken)))
    for name in real_names:
        entry = ds.deformations[name]
        sub = real_node.add(Text(name, style="cyan"))
        sub.add(_array_line(profile.deformed_name, entry.image, [], columns))
    for be in sorted(real_broken, key=lambda b: b.name):
        real_node.add(_broken_line(be))

    syn_names = ds.list_synthetic()
    syn_broken = [b for b in ds.broken_entries.values() if b.kind == "synthetic"]
    syn_node = tree.add(
        _group_header(profile.synthetic_group, len(syn_names), len(syn_broken))
    )
    for name in syn_names:
        entry = ds.deformations[name]
        sub = syn_node.add(Text(name, style="cyan"))
        sub.add(_array_line(profile.deformed_name, entry.image, [], columns))
        if entry.flow is not None:
            flow_tags = [
                profile.flow_defaults.axis_order,
                profile.flow_defaults.convention,
            ]
            sub.add(_array_line(profile.flow_name, entry.flow, flow_tags, columns))
    for be in sorted(syn_broken, key=lambda b: b.name):
        syn_node.add(_broken_line(be))

    return tree


def _group_header(group_name: str, n_healthy: int, n_broken: int) -> Text:
    """Build the ``iterations/  (4 healthy, 1 broken)`` header text.

    Without broken entries the count is rendered as ``(N entries)`` for
    backward compatibility with the existing happy-path layout. The
    "broken" count surfaces in red so a partial open is impossible to
    miss when scanning the tree.
    """
    parts: list[tuple[str, str]] = [(f"{group_name}/", "bold blue")]
    if n_broken == 0:
        parts.append((f"  ({n_healthy} entries)", "dim"))
    else:
        parts.append((f"  ({n_healthy} healthy, ", "dim"))
        parts.append((f"{n_broken} broken", "bold red"))
        parts.append((")", "dim"))
    return Text.assemble(*parts)


def _broken_line(be: BrokenEntry) -> Text:
    """Render one broken-entry leaf as ``<name>  [BROKEN: <reason>]``."""
    line = Text()
    line.append(be.name, style="dim red")
    line.append("  ")
    line.append(f"[BROKEN: {be.reason}]", style="bold red")
    return line


def _compute_columns(ds: DvcDataset, *, verbose: bool) -> _LeafColumns:
    """Pre-walk every array leaf and compute per-field max widths.

    Run once per tree so the alignment is global: every leaf, regardless
    of depth, uses the same column widths. The cost is one metadata
    pass over each :class:`zarr.Array` handle (no voxels touched).
    """
    leaves: list[tuple[str, zarr.Array[Any]]] = [
        (ds.profile.reference_name, ds.reference),
    ]
    leaves.extend((name, ds.masks[name]) for name in ds.masks)
    for entry in ds.deformations.values():
        leaves.append((ds.profile.deformed_name, entry.image))
        if entry.flow is not None:
            leaves.append((ds.profile.flow_name, entry.flow))

    name_w = max(len(name) for name, _ in leaves)
    shape_w = max(len(_shape_str(arr.shape)) for _, arr in leaves)
    dtype_w = max(len(str(arr.dtype)) for _, arr in leaves)
    chunks_w = 0
    codecs_w = 0
    size_w = 0
    if verbose:
        chunks_w = max(len(_shape_str(arr.chunks)) for _, arr in leaves)
        codecs_w = max(len(_codec_summary(arr)) for _, arr in leaves)
        size_w = max(len(_human_bytes(_array_nbytes(arr))) for _, arr in leaves)

    return _LeafColumns(
        name_w=name_w,
        shape_w=shape_w,
        dtype_w=dtype_w,
        chunks_w=chunks_w,
        codecs_w=codecs_w,
        size_w=size_w,
        verbose=verbose,
    )


def _array_line(
    name: str,
    array: zarr.Array[Any],
    tags: list[str],
    columns: _LeafColumns,
) -> Text:
    """Build the rich ``Text`` line for one zarr array leaf.

    Fields are padded to the widths in ``columns`` so leaves at any
    depth align in the same visual columns. Tags trail unpadded — they
    are decorative and absent on most lines.
    """
    line = Text()
    line.append(name.ljust(columns.name_w), style="cyan")
    line.append("  ")
    line.append(_shape_str(array.shape).ljust(columns.shape_w))
    line.append("  ")
    line.append(str(array.dtype).ljust(columns.dtype_w), style="dim")
    if columns.verbose:
        line.append("  chunks=", style="dim")
        line.append(_shape_str(array.chunks).ljust(columns.chunks_w), style="dim")
        line.append("  codecs=", style="dim")
        line.append(_codec_summary(array).ljust(columns.codecs_w), style="dim")
        line.append("  ")
        line.append(_human_bytes(_array_nbytes(array)).rjust(columns.size_w), style="dim")
    if tags:
        line.append("  ")
        for i, t in enumerate(tags):
            if i:
                line.append(" ")
            line.append(f"[{t}]", style="yellow")
    return line


def _resolve_default_mask(ds: DvcDataset) -> str | None:
    """Mirror :meth:`DvcDataset._resolve_mask_name` (manifest > profile)."""
    if ds.manifest is not None and ds.manifest.base.default_mask is not None:
        return ds.manifest.base.default_mask
    return ds.profile.default_mask


def _shape_str(shape: tuple[int, ...]) -> str:
    """Format a shape tuple as ``(a,b,c)`` without spaces."""
    return "(" + ",".join(str(int(s)) for s in shape) + ")"


def _human_bytes(n: int) -> str:
    """Format a byte count with binary units (KiB, MiB, ...)."""
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if f < 1024 or unit == "TiB":
            return f"{f:.2f} {unit}"
        f /= 1024
    return f"{f:.2f} TiB"


def _array_nbytes(array: zarr.Array[Any]) -> int:
    """Compute uncompressed in-memory size from shape * itemsize."""
    n = 1
    for s in array.shape:
        n *= int(s)
    return n * int(array.dtype.itemsize)


def _codec_summary(array: zarr.Array[Any]) -> str:
    """Short codec list, e.g. ``"bytes,zstd"``; ``"?"`` if unavailable."""
    codecs = getattr(array.metadata, "codecs", None)
    if not codecs:
        return "?"
    names = [type(c).__name__.removesuffix("Codec").lower() for c in codecs]
    return ",".join(names) or "?"


# --------------------------------------------------------------- json output


def _render_json(
    ds: DvcDataset,
    *,
    path: Path,
    manifest_source: str,
    verbose: bool,
) -> str:
    """Serialize the inventory + verifier report as indented JSON."""
    payload: dict[str, Any] = {
        "header": {
            "store": str(path),
            "profile": ds.profile.name,
            "manifest_source": manifest_source,
            "spacing": (
                {"values": list(ds.spacing.values), "unit": ds.spacing.unit}
                if ds.spacing is not None
                else None
            ),
        },
        "validity": {
            "ok": ds.verification_report.ok,
            "errors": list(ds.verification_report.errors),
            "warnings": list(ds.verification_report.warnings),
        },
        "slots": {
            "reference": _array_dict(ds.reference, verbose=verbose),
            "default_mask": _resolve_default_mask(ds),
            "masks": {
                name: _array_dict(arr, verbose=verbose) for name, arr in ds.masks.items()
            },
            "real": {
                name: {
                    "image": _array_dict(ds.deformations[name].image, verbose=verbose),
                }
                for name in ds.list_real()
            },
            "synthetic": {
                name: {
                    "image": _array_dict(ds.deformations[name].image, verbose=verbose),
                    "flow": (
                        _array_dict(flow, verbose=verbose)
                        if (flow := ds.deformations[name].flow) is not None
                        else None
                    ),
                }
                for name in ds.list_synthetic()
            },
            "broken": {
                name: {
                    "kind": be.kind,
                    "reason": be.reason,
                    "missing": list(be.missing),
                }
                for name, be in sorted(ds.broken_entries.items())
            },
        },
    }
    return json.dumps(payload, indent=2)


def _array_dict(array: zarr.Array[Any], *, verbose: bool) -> dict[str, Any]:
    """Serialize one zarr array's metadata into a JSON-friendly dict."""
    out: dict[str, Any] = {
        "shape": [int(s) for s in array.shape],
        "dtype": str(array.dtype),
    }
    if verbose:
        out["chunks"] = [int(c) for c in array.chunks]
        out["codecs"] = _codec_summary(array)
        out["nbytes"] = _array_nbytes(array)
    return out


if __name__ == "__main__":
    app()
