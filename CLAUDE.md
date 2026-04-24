# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FFT-based digital volume correlation (DVC) for synchrotron µCT of rat bone with screw implant. Target volume shape `(960, 1280, 1280)` float32, compute on 4× NVIDIA A6000 via CuPy. Python **3.14** is required (CuPy v14 / CUDA 13.x).

The repository is currently at scaffold stage: `pyproject.toml`, tooling, and the architecture plan exist; the `mamba_dvc/` package itself is not populated. **Read `docs/plans/overview.md` before writing any non-trivial code** — it is the source of truth for algorithm, module layout, API shapes, memory budget, and validation strategy.

## Commands

Dependencies are managed with **uv** (see `uv.lock`). The `.venv` is already in the repo.

```bash
uv sync                       # install deps incl. dev group
uv run ruff format .          # format
uv run ruff check --fix .     # lint (D/jaxtyping rules are enabled — see pyproject)
uv run pyright                # strict type check on mamba_dvc/
uv run pytest                 # tests
uv run pytest -m "not gpu"    # skip GPU-only tests on non-CUDA machines
uv run pytest -m slow         # only slow tests
uv run pytest tests/path/to/test_file.py::test_name   # single test
uv run pre-commit run --all-files   # full lint gate (mirror of CI)
pre-commit install            # one-time: enable git hooks
```

Before declaring a task done: `ruff format`, `ruff check`, `pyright`, `pytest` must all pass.

## Architecture intent

Planned package layout (`docs/plans/overview.md` §4 is authoritative):

```
mamba_dvc/
  io/           volume + mask + ground-truth field readers
  core/         grid, window, ncc, peakfit, outlier  ← the hot path
  pipeline/     correlate.py — pure single-pair driver
  gpu/          dispatch (multi-GPU sharding), memory (pinned buffers, streams)
  types.py      DisplacementField, GridSpec, MaskSpec
  validate/     synthetic.py (analytical) + known_fields.py (experimental GT)
  cli.py
```

Two non-negotiable invariants baked into the design:

- **Pure-core rule.** `pipeline.correlate.correlate()` is a pure function. All sequencing, prefetch, warm-start, and multi-pair driving live in a wrapping driver layer. This is what lets v2 features (iterative warp, pyramid, time series) slot in without refactor.
- **Single-pass algorithm only (v1).** One forward FFT NCC + 3D Gaussian subvoxel fit. No iteration, no pyramid, no IC-GN. See §2 of the plan for the SNR/accuracy justification — don't "helpfully" add an iteration loop.

Extension points for v2 (IC-GN, Padfield masked FFT, pyramid, densification, time series, per-frame deformed masks) are listed in §10 of the plan. Keep APIs aligned with those hooks — in particular `deformed_mask` already appears in the v1 signature even though v1 falls back to the shared mask.

## Multi-GPU model

One Python process per GPU via `multiprocessing.spawn` (Windows-safe). Volumes + mask are **replicated** on every GPU; POIs are **sharded block-wise along Z** to preserve spatial locality for the outlier test. No inter-GPU communication during NCC — gather happens once at the end, and the outlier test runs CPU-side on the assembled grid.

## Array typing and shape checking

- All public functions in `core/`, `pipeline/`, `io/`, `gpu/` annotate array inputs and outputs with **`jaxtyping`** (e.g. `Float32[np.ndarray, "z y x"]`). Shape symbols (`z`, `y`, `x`, `points`, `w`) are shared vocabulary — document them once in `types.py`, reuse everywhere. Use `batch` when a leading batch axis is introduced.
- Tests wrap target functions with `@jaxtyped(typechecker=beartype)` for runtime shape enforcement; production code relies on the static annotations only.
- `pyright` strict on `mamba_dvc/`. The `F722` ignore in `pyproject.toml` exists because jaxtyping shape strings look like forward references — don't remove it.
- Docstrings: NumPy convention, enforced via ruff's `D` rules (`pydocstyle.convention = "numpy"`). Tests and `validate/` are exempted in `per-file-ignores`.

## Library choices (locked for v1)

CuPy + cuFFT for GPU compute; NumPy for host; SciPy only for CPU reference implementations in tests. **Not using PyTorch or JAX in v1** — CuPy's explicit memory control fits the FFT-heavy workload better, and nothing in v1 needs autograd. Zarr v3 + tifffile for volume I/O; `tensorstore` is available for async / GDS reads when needed.
