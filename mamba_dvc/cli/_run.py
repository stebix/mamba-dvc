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
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from mamba_dvc.instrument import accumulating
from mamba_dvc.run import BatchSpec, Job, JobResult, NullObserver, plan_jobs, run_batch

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
    prefetch: Annotated[
        int | None,
        typer.Option(
            "--prefetch",
            min=0,
            help=(
                "How many load groups to fetch ahead in a background thread "
                "(0 disables). Overlapping the next pair's zarr read with the "
                "current pair's correlate roughly halves campaign wall time when "
                "a load costs about as much as a group's variants. Default: the "
                "config's value (1)."
            ),
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress the live progress line; still print the end-of-run summary.",
        ),
    ] = False,
    timing: Annotated[
        bool,
        typer.Option(
            "--timing",
            help=(
                "Accumulate per-phase wall times across the campaign and print a "
                "breakdown table at the end. Adds a small GPU-sync overhead "
                "(diagnostic mode); per-pair NCC sub-phases only appear on the "
                "single-GPU path."
            ),
        ),
    ] = False,
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
    if prefetch is not None:
        spec = replace(spec, prefetch=prefetch)
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

    timing_table: str | None = None
    try:
        if timing:
            with accumulating() as acc:
                results = _execute_campaign(
                    spec,
                    force=force,
                    only=only_filter,
                    stores=store_subset,
                    console=console,
                    quiet=quiet,
                )
            timing_table = acc.render(
                title="campaign phase breakdown — wall seconds (sub-phases nest under totals)"
            )
        else:
            results = _execute_campaign(
                spec,
                force=force,
                only=only_filter,
                stores=store_subset,
                console=console,
                quiet=quiet,
            )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc

    _render_results(console, spec, results)
    if timing_table is not None:
        console.print(
            Panel(Text(timing_table), title="timing", border_style="cyan", expand=False)
        )
    n_failed = sum(1 for r in results if r.status == "failed")
    raise typer.Exit(1 if n_failed else 0)


def _execute_campaign(
    spec: BatchSpec,
    *,
    force: bool,
    only: dict[str, str] | None,
    stores: list[Path] | None,
    console: Console,
    quiet: bool,
) -> list[JobResult]:
    """Run the campaign, picking a progress observer for the output target.

    ``--quiet`` → no live output; a non-terminal stdout (pipe, CI log) →
    one plain line per finished job; an interactive terminal → a Rich
    spinner/progress bar. The ``run_batch`` return value and the on-disk
    artifacts are identical in every case — only the live rendering
    differs (the library itself stays print-free).
    """
    if quiet:
        return run_batch(spec, force=force, only=only, stores=stores, observer=NullObserver())
    if not console.is_terminal:
        return run_batch(
            spec, force=force, only=only, stores=stores, observer=_PlainLogObserver(console)
        )
    with _campaign_progress(console) as progress:
        return run_batch(
            spec,
            force=force,
            only=only,
            stores=stores,
            observer=_RichProgressObserver(progress),
        )


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
        if r.status == "failed":
            console.print(_failure_line(r))


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


# ----------------------------------------------------------------- progress observers


def _loc(job: Job) -> str:
    """``<store>/<deformation>/<variant_id>`` — the canonical job locator string."""
    return f"{job.store.name}/{job.deformation}/{job.variant.variant_id}"


def _failure_line(result: JobResult) -> str:
    """Render a one-line ``FAILED <loc> (<phase>: <last traceback line>)`` summary."""
    phase = result.summary.get("phase", "?")
    last = ((result.error or "").strip().splitlines() or ["(no detail)"])[-1]
    return f"[red]FAILED[/red] {_loc(result.job)}  [dim]({phase}: {last})[/dim]"


def _campaign_progress(console: Console) -> Progress:
    """Build the campaign progress bar: spinner, current job, bar, m/n done, elapsed, ETA."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


class _RichProgressObserver(NullObserver):
    """Drives one Rich progress task from ``run_batch`` events.

    The owning :class:`~rich.progress.Progress` is started and stopped by
    its own ``with`` block (see :func:`_execute_campaign`); this observer
    only adds the task and updates its description/completion. When there
    is nothing pending it adds no task and stays inert — the end-of-run
    panel reports the no-op. Inherits the no-op ``on_batch_end`` from
    :class:`~mamba_dvc.run.NullObserver`.
    """

    def __init__(self, progress: Progress) -> None:
        self._progress = progress
        self._task: TaskID | None = None

    def on_batch_start(self, *, n_jobs: int, n_variants: int) -> None:
        """Add the progress task sized to the pending-job count (if any)."""
        if n_jobs:
            self._task = self._progress.add_task("starting", total=n_jobs)

    def on_pair_load_start(self, store: Path, deformation: str, *, n_variants: int) -> None:
        """Show the (slow) ``load_pair`` in the description line."""
        if self._task is not None:
            self._progress.update(
                self._task,
                description=f"[dim]{store.name}/{deformation}[/dim]  loading pair",
            )

    def on_job_start(self, job: Job) -> None:
        """Switch the description to the in-flight variant."""
        if self._task is not None:
            self._progress.update(
                self._task,
                description=(
                    f"[dim]{job.store.name}/{job.deformation}[/dim]  "
                    f"[cyan]{job.variant.name}[/cyan]  correlating"
                ),
            )

    def on_job_end(self, result: JobResult) -> None:
        """Advance the bar; print a line above it when the job failed."""
        if self._task is None:
            return
        self._progress.advance(self._task)
        if result.status == "failed":
            self._progress.console.print(_failure_line(result))


class _PlainLogObserver(NullObserver):
    """Emits one plain line per finished job — for non-terminal stdout (pipes, CI).

    No spinner or cursor control: just append-only ``done/total`` lines so
    a redirected log stays readable. Inherits the no-op ``on_job_start`` /
    ``on_batch_end`` from :class:`~mamba_dvc.run.NullObserver`.
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._done = 0
        self._total = 0

    def on_batch_start(self, *, n_jobs: int, n_variants: int) -> None:
        """Record the job total used in the per-job ``done/total`` prefix."""
        self._total = n_jobs

    def on_pair_load_start(self, store: Path, deformation: str, *, n_variants: int) -> None:
        """Note the start of a (slow) ``load_pair``."""
        self._console.print(
            f"[dim]loading[/dim]  {store.name}/{deformation}  ({n_variants} variant(s))"
        )

    def on_job_end(self, result: JobResult) -> None:
        """Print a one-line ``done/total  status  locator`` entry."""
        self._done += 1
        prefix = f"  {self._done}/{self._total}"
        if result.status == "failed":
            self._console.print(f"{prefix}  {_failure_line(result)}")
        else:
            secs = result.summary.get("wall_correlate_s", 0.0)
            self._console.print(
                f"{prefix}  [green]ok[/green]  {_loc(result.job)}  [dim]{secs}s[/dim]"
            )
