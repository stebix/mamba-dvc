# Multi-GPU dispatch — plan & contract

The v1 multi-GPU layer that wraps `pipeline.correlate.correlate()`.
Implements plan `docs/plans/overview.md` §6: replicate volumes on every
GPU, shard POIs along lattice-Z, run the batched FFT NCC + peakfit
inner loop in a per-GPU worker process, gather and run the outlier
test on the parent. Lives in `mamba_dvc/gpu/`.

## 1. Position in the codebase

```
                ┌──────────────────────────┐
parent process  │ correlate_multi_gpu(...) │   gpu/dispatch.py
                │  build_grid              │
                │  filter_by_mask          │
                │  shard admitted_idx      │
                │  publish volumes (shm)   │
                └────┬───────┬────────┬────┘
                     │       │        │
                ┌────▼─┐ ┌───▼──┐ ┌───▼──┐
                │ W0   │ │ W1   │ │ ...  │   one process per GPU
                │ shm  │ │ shm  │ │      │   (multiprocessing.spawn)
                │ →H2D │ │ →H2D │ │      │
                │ helper│ │ helper│ │     │   _correlate_admitted_subset
                │ →D2H │ │ →D2H │ │      │   on cupy arrays
                │ pipe │ │ pipe │ │      │
                └────┬─┘ └─────┬┘ └─────┬┘
                     │         │        │
                ┌────▼─────────▼────────▼──┐
                │ scatter into full-length  │
                │ buffers; outlier test;    │
                │ DisplacementField         │
                └───────────────────────────┘
```

The single-device `pipeline.correlate.correlate()` and the multi-GPU
`gpu.dispatch.correlate_multi_gpu()` share the inner loop, mask
resolution, and parameter validation via `mamba_dvc.pipeline._internal`
(`correlate_admitted_subset`, `resolve_masks`, `normalize_window`,
`TUKEY_DEFAULTS`). One source of truth, no underscore-prefixed
imports across module boundaries — the `_internal` module name itself
signals "internal collaboration surface; not part of the public API".

## 2. Design invariants

| Invariant | Rationale |
|---|---|
| `correlate()` stays pure, single-device | Plan §4 pure-core rule: v2 iterative warp / time-series drivers wrap it inside their own loops, none of which can tolerate hidden state. |
| Dispatch is a sibling, not a layer above | The pure-core rule excludes spawning subprocesses from inside `correlate()`. Putting it in `gpu/dispatch.py` makes the layering explicit. |
| Per-shard helper output is **full-length sparse** | Each worker returns arrays of length `n_points` with the shard's slots populated and every other slot at zero (`MASKED`). The parent scatters per-shard slots into pre-allocated parent buffers. No merging logic; no max-reduction; no order dependency. |
| Outlier test runs on the parent after gather | The 3×3×3 lattice neighborhood would see invalid data if it ran inside a worker on a partial lattice and produce false positives at slab boundaries. Plan §6 calls this out. |
| One process per GPU (`multiprocessing.spawn`) | Windows-safe (no `fork`); avoids GIL contention; each worker pins itself to its assigned device before any CuPy import-side allocation. |
| Volumes are **replicated** to each GPU | Plan §6 §"Decomposition". Halo decomposition (each GPU owns a Z-slab + halo) was considered and rejected — 17 GB out of 49 GB on an A6000 is not tight, and replication keeps `extract_subvolumes` simple. |
| No inter-GPU communication during NCC | Sharing happens only at the boundaries: parent → workers (volumes via shared memory) and workers → parent (full-length sparse partial arrays via pipe). |

## 3. The dispatch seam: `_correlate_admitted_subset`

The refactor that made dispatch possible. `correlate()` was split:

- `mamba_dvc.pipeline._internal.correlate_admitted_subset(ref, def, mask, def_mask, grid, admitted_idx, ...) -> (disp, conf, status)`
  is the workhorse. It runs the batched FFT NCC + peakfit + per-POI
  search-radius gate on whichever subset of admitted POIs the caller
  hands it. It does **not** run mask admission (the caller did that)
  and does **not** run the outlier test (the caller does that on the
  full lattice).
- `correlate()` is now a thin wrapper: build grid, resolve masks, run
  `filter_by_mask`, call the helper with `admitted_idx = all admitted`,
  run the outlier test, build `DisplacementField`.
- `correlate_multi_gpu()` is the parallel wrapper: same setup steps,
  shard `admitted_idx` into N contiguous lattice-Z slabs, hand each
  shard to a worker process, scatter the partials, run the outlier
  test, build `DisplacementField`.

The helper is array-module-agnostic. When `reference` is a CuPy array
(workers replicate to device on attach), it runs entirely on device
and returns CuPy arrays. When `reference` is NumPy (the single-device
public API), it stays on host. Workers `cp.asnumpy()` once at the end
before the result hits the pipe.

### Sparse-output convention

Helper output for a shard with `admitted_idx = [3, 4, 5]` over an
`n_points = 8` lattice:

```
displacements: zeros(8, 3) with rows 3, 4, 5 populated (or zero if OUT_OF_RANGE)
confidence:    zeros(8)    with entries 3, 4, 5 populated
status:        full(MASKED) with entries 3, 4, 5 set OK or OUT_OF_RANGE
```

Parent scatter is `out[shard_admitted_idx] = shard_arr[shard_admitted_idx]`.
Across shards the indices are disjoint by construction (the partition
is a function of lattice-Z), so the scatter is order-independent.

POIs that no shard claims (truly mask-rejected; lattice-Z slabs that
ended up empty after filtering) keep their parent-init values:
`MASKED` / zero. The outlier overlay then runs on the assembled
arrays.

## 4. Sharding: contiguous lattice-Z slabs

`_shard_admitted_indices(admitted_idx, grid, n_shards)` decodes each
admitted flat index against `grid.grid_shape` to recover its
lattice-Z coordinate, then partitions Z into `n_shards` contiguous
chunks via `np.array_split` and returns one int64 array per shard.

Empty shards are valid work units. The helper handles a zero-length
`admitted_idx` cleanly (returns the all-MASKED defaults), and the
parent's per-shard scatter writes nothing — both consistent with the
sparse-output contract.

## 5. IPC

### Volumes: `multiprocessing.shared_memory` (`mamba_dvc/gpu/shm.py`)

A 6.3 GB float32 volume × 4 workers × 2 (ref + def) = 50 GB if
shipped via pickle. Shared memory is one allocation in the parent and
N zero-copy attaches in the workers. The parent context-manages
lifetime via `published()`; workers `attach()` then `close()` their
own handles before exit; the parent `close+unlink`s on context exit.

`unlink()` semantics differ between Windows and POSIX. We wrap it in
`contextlib.suppress(OSError)` so the same code runs on Windows
production hosts and Linux dev workstations.

When the I/O layer (zarr / tensorstore) lands, dispatch can pivot to
each worker independently opening the shared store. The
`gpu/shm.py` module remains useful as the in-memory IPC fallback.

### Per-shard partial results: unidirectional pipe

Each worker returns three small arrays (`disp`, `conf`, `status`),
total ~150 KB at v1 lattice scales. Pickle + pipe is sub-millisecond.
The pipe is also the channel for error messages — the worker wraps
its body in `try/except`, sends `("err", repr(exc), traceback,
gpu_id)` on failure, and the parent re-raises with the offending GPU
id and the shard size annotated.

## 6. Worker lifecycle: spawn-per-call (v1)

Each `correlate_multi_gpu()` call:

1. Parent allocates SHM for the four volumes (ref, def, mask,
   deformed_mask), spawns N processes via `ctx.Process(...)`.
2. Worker `import cupy; cp.cuda.Device(gpu_id).use()` (pin BEFORE any
   allocation), attaches SHM, uploads to GPU via `cp.asarray()`.
3. Worker calls `_correlate_admitted_subset` with its shard.
4. Worker `cp.asnumpy()`s the result, sends over the pipe, closes SHM
   handles, exits.
5. Parent `wait()`s on all pipes, scatters partials, joins workers,
   closes/unlinks SHM, runs outlier test, returns
   `DisplacementField`.

Cost: ~1.5 s spawn × N workers, plus ~3 s CUDA context init per
device. Significant for one pair (~6 s overhead on 4 A6000s) — a
non-issue for v2's time-series driver, where a persistent
`MultiGPUDispatcher` context manager will keep workers alive across
pairs.

## 7. `device_ids` semantics

| Value | Behaviour |
|---|---|
| `None` (default) | All CuPy-visible GPUs. Production dispatch on the 4× A6000 box. |
| `[0]` | In-process happy path. No spawn — pin to device 0, upload, run helper, D2H. The cheap baseline against which the multi-process path is regression-tested. |
| `[0, 0]` | Two workers pinned to GPU 0. Useful on single-GPU dev hosts to exercise the spawn / pipe / scatter plumbing. |
| `[0, 1, 2, 3]` | Production dispatch (explicit). |

`device_ids=[]` raises `ValueError`. CuPy unavailable raises
`RuntimeError` with a pointer to `correlate()` for host-only
execution.

## 8. Failure modes

| Symptom | Cause | Mitigation in v1 |
|---|---|---|
| Worker raises during NCC (e.g. CUDA OOM) | Helper or upload failed | `_worker_entry` catches, ships `("err", ..., gpu_id)`, parent re-raises with GPU id + shard size + worker traceback. |
| Worker dies hard (segfault, killed) | Driver bug, OOM-kill, etc. | Pipe closes; parent's `recv()` raises `EOFError`; parent re-raises with `exitcode`. |
| Parent crashes mid-dispatch | Test/user keyboard interrupt | `_run_multi_process` catches `BaseException`, `terminate()`s every alive worker, joins with timeout, then re-raises. SHM is released via the `published()` `finally` block. |
| Worker hangs (e.g. driver deadlock) | Rare | Currently no timeout. The parent's `wait()` has no deadline. v2 may add one. |

## 9. The Windows pipe-class gotcha (caught in v1 testing)

`multiprocessing.connection.Connection` and `PipeConnection` share no
common base class in Python's nominal types: on Linux `Pipe()` returns
`Connection`; on Windows `Pipe()` returns `PipeConnection`. An early
draft of `_gather_shards` filtered `wait()`'s output via
`isinstance(c, Connection)`, which silently dropped every ready pipe
on Windows — the parent looped on `wait()` forever while workers
exited normally.

Fix: iterate `wait()`'s output without an `isinstance` filter, with a
`recv in pipe_to_meta` runtime defense. Pyright complaints suppressed
inline. Documented in `dispatch.py:_gather_shards` so a future reader
does not "helpfully" reintroduce the filter.

## 10. Decisions deferred to v2

- **Persistent worker pool** (`MultiGPUDispatcher` context manager).
  Single-pair v1 spawns and tears down each call. Time-series wraps
  this cleanly.
- **Programmatic batch-size heuristic** in `gpu/dispatch.py::_resolve_batch_size`.
  Today the call site picks (default 64 in dispatch, 256 in
  `correlate()`). A future helper resolves from
  `(window, mode, normalization, free VRAM)`.
- **Pinned host buffers + async H2D streams** (`gpu/memory.py`).
  Matters for time-series double-buffering, not single-pair runs.
- **Halo decomposition.** Replication is fine at v1 memory budgets.
- **Switch to per-worker zarr opens.** Lands when `mamba_dvc/io/`
  lands; the `_run_multi_process` boundary stays the same.

## 11. Test surface

Pure logic (no GPU): `tests/gpu/test_dispatch.py::TestShardAdmittedIndices`.
Five tests pin the partition contract: union covers admitted exactly,
slabs are ascending and non-overlapping, sparse admission is
preserved, empty admitted gives empty shards, non-positive `n_shards`
raises.

Dispatch-seam helper (no GPU):
`tests/pipeline/test_correlate.py::TestCorrelateAdmittedSubset`. Six
tests pin: helper + outlier overlay equals `correlate()`, with and
without mask admission; disjoint-shard scatter equals one full call;
non-shard slots are MASKED with zero; empty admitted is a no-op;
out-of-range zeroes only at admitted slots.

End-to-end with GPU (`@pytest.mark.gpu @pytest.mark.slow`):
- `device_ids=[0]` matches `correlate()` (in-process happy path).
- `device_ids=[0, 0]` matches `correlate()` (multi-process spawn,
  one GPU). Exercises every part of the dispatch plumbing on a
  single-GPU dev host.
- Same with mask admission active. Pins that MASKED slots flow
  through the gather correctly.

Validation (no GPU):
`tests/gpu/test_dispatch.py::TestCorrelateMultiGpuValidation`. Empty
`device_ids`, shape mismatch, dtype mismatch raise `ValueError`.

The full multi-GPU production path (`device_ids=[0, 1, 2, 3]` on the
4× A6000 box) is not in CI but is the canonical performance smoke
test. Run manually before tagging a release.
