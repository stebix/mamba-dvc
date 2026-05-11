"""``mamba-dvc run`` subcommand: execute a batch displacement-field campaign.

Thin CLI over :mod:`mamba_dvc.run`: parse a campaign config, optionally
preview the plan (``--dry-run``), and otherwise execute it. The heavy
lifting — the materialize-once / iterate-many loop, resume, persistence
— lives in :mod:`mamba_dvc.run.batch`; this module only translates
flags and renders.

Plan: ``docs/plans/run-interface.md`` §1, §5.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mamba_dvc.run import BatchSpec, Job, JobResult, plan_jobs, run_batch

__all__ = ["run"]


# --------------------------------------------------------------- arg parsers


def _parse_only(items: list[str] | None) -> dict[str, str] | None:
    """Parse repeated ``KEY=VALUE`` ``--only`` flags into a dict."""
    if not items:
        return None
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"--only expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"--only key must be non-empty in {item!r}")
        out[key] = value.strip()
    return out


def _parse_devices(spec: str | None) -> tuple[int, ...] | None:
    if spec is None:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise typer.BadParameter("--devices must list at least one GPU id")
    try:
        ids = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise typer.BadParameter(f"--devices entries must be ints, got {spec!r}") from exc
    if any(d < 0 for d in ids):
        raise typer.BadParameter(f"--devices entries must be non-negative, got {spec!r}")
    return ids


# ------------------------------------------------------------------ command


def run(
    config: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Campaign config YAML.",
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Print the plan (stores x deformations x variants) and exit."
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force", help="Recompute variants whose result already exists on disk."
        ),
    ] = False,
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Filter variants by resolved param, e.g. mask_threshold=0.5 (repeatable).",
        ),
    ] = None,
    devices: Annotated[
        str | None,
        typer.Option(
            "--devices",
            help="Comma-separated GPU ids overriding the config's (e.g. 0,1 to dodge OOM).",
        ),
    ] = None,
    store: Annotated[
        list[Path] | None,
        typer.Option(
            "--store",
            help="Restrict to a subset of stores (by path or .zarr name; repeatable).",
        ),
    ] = None,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable styled terminal output."),
    ] = False,
) -> None:
    """Execute (or preview, with ``--dry-run``) a batch campaign config.

    Exit code is ``0`` on a clean run (or any dry-run), ``1`` if at
    least one job failed, and ``2`` for a malformed config or flags.
    """
    try:
        spec = BatchSpec.from_yaml(config)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(2) from exc

    only_filter = _parse_only(only)
    device_override = _parse_devices(devices)
    if device_override is not None:
        spec = replace(spec, devices=device_override)
    store_subset = store or None

    console = Console(no_color=no_color, width=140)

    if dry_run:
        try:
            jobs = plan_jobs(spec, force=force, only=only_filter, stores=store_subset)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from exc
        _render_plan(console, spec, jobs)
        raise typer.Exit(0)

    try:
        results = run_batch(spec, force=force, only=only_filter, stores=store_subset)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    _render_results(console, spec, results)
    n_failed = sum(1 for r in results if r.status == "failed")
    raise typer.Exit(1 if n_failed else 0)


# ----------------------------------------------------------------- rendering


def _render_plan(console: Console, spec: BatchSpec, jobs: list[Job]) -> None:
    """Print the campaign header, variant list, and per-store pending counts."""
    console.print(_header_panel(spec))

    variants = Text()
    for v in spec.variants:
        variants.append(f"  {v.variant_id}", style="cyan")
        variants.append("\n")
    console.print(
        Panel(
            variants,
            title=f"{len(spec.variants)} variant(s)",
            border_style="blue",
            expand=False,
        )
    )

    if not jobs:
        console.print(
            "[green]nothing to do[/green] — every selected variant is already on disk."
        )
        return

    by_store: Counter[tuple[str, str]] = Counter()
    for job in jobs:
        by_store[(job.store.name, job.deformation)] += 1

    table = Table(title=f"{len(jobs)} pending job(s)", title_style="bold blue", expand=False)
    table.add_column("store")
    table.add_column("deformation")
    table.add_column("pending variants", justify="right")
    for (store_name, deformation), count in sorted(by_store.items()):
        table.add_row(store_name, deformation, str(count))
    console.print(table)


def _render_results(console: Console, spec: BatchSpec, results: list[JobResult]) -> None:
    """Print a one-panel summary of an executed campaign."""
    counts = Counter(r.status for r in results)
    body = Text()
    body.append(f"campaign   {spec.campaign}\n", style="bold")
    body.append(f"results    {spec.campaign_dir}\n")
    body.append(f"ran        {len(results)} job(s)  ")
    body.append(f"ok={counts.get('ok', 0)} ", style="green")
    failed = counts.get("failed", 0)
    body.append(f"failed={failed}", style="red" if failed else "dim")
    if not results:
        body.append("\n\n(nothing pending — resume found everything on disk)")
    console.print(
        Panel(
            body, title="run complete", border_style="red" if failed else "green", expand=False
        )
    )

    for r in results:
        if r.status != "failed":
            continue
        last_line = ((r.error or "").strip().splitlines() or ["(no detail)"])[-1]
        phase = r.summary.get("phase", "?")
        loc = f"{r.job.store.name}/{r.job.deformation}/{r.job.variant.variant_id}"
        console.print(f"[red]FAILED[/red] {loc}  [dim]({phase}: {last_line})[/dim]")


def _header_panel(spec: BatchSpec) -> Panel:
    body = Text()
    body.append("campaign  ", style="bold")
    body.append(f"{spec.campaign}\n")
    body.append("output    ", style="bold")
    body.append(f"{spec.campaign_dir}\n")
    body.append("stores    ", style="bold")
    body.append(f"{len(spec.stores)}\n")
    body.append("select    ", style="bold")
    body.append(f"real={spec.select_real}  synthetic={spec.select_synthetic}\n")
    body.append("devices   ", style="bold")
    body.append("all visible" if spec.devices is None else str(list(spec.devices)))
    return Panel(body, title="campaign", border_style="blue", expand=False)
