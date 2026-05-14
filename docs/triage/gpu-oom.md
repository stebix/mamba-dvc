# Multi-GPU OOM / 4→2 GPU fallback

`correlate_multi_gpu` periodically fails on the 4-GPU path during
`results_v3`; `_correlate_with_fallback` (`mamba_dvc/run/batch.py:524`)
silently retries on `device_ids[:2]` and the campaign carries on. The
retry was introduced in `79c477e` as a workaround for what was thought
to be host pinned-memory exhaustion (plan §9). This document captures
what the on-disk `results_v3/events.jsonl` actually shows and where the
current "fix" is still blind.

- **Affected entry point**: `mamba_dvc.run.batch.run_batch` →
  `_correlate_with_fallback` → `correlate_multi_gpu` (per-call path,
  not the persistent dispatcher).
- **Affected commits**: `79c477e` introduced the silent
  `except Exception: retry` swallow. `04bd525` narrowed it to
  `except MemoryError` and added a `correlate_retry` event. **The
  campaign log on disk predates `04bd525`**, so no retry events
  recorded, and the exception class that fires is still unknown.
- **Not affected**: any path with `len(device_ids) <= 2` (no fallback
  branch); `MultiGPUDispatcher` persistent path (different driver).

## Pattern in the log

`results_v3/events.jsonl`, 16 jobs completed at time of analysis (one
store fully done, second store one variant in). Each cell is the wall
time recorded against the `dispatch.compute` phase, with FAIL marking
jobs that ended up running a second `dispatch.compute` on
`n_devices=2`.

```
              mt0.9 (n=16 424)   mt0.7 (n=27 673)   mt0.5 (n=37 280)
fs004         FAIL 24 s          FAIL 26 s          OK   136 s
fs104         FAIL 31 s          FAIL 29 s          OK   142 s
fs204         FAIL 23 s          FAIL 34 s          FAIL 26 s
fs304         OK   99 s          OK  122 s          OK  121 s
fs402         OK   53 s          OK   70 s          OK   87 s
```

(`5L_PEEK_4w_000 / fs004 / base` — only further job present —
completed on 4 GPU at 34 s.)

Failure rate is **9 / 16, not 16 / 16**. Prior session-summary
characterisation as "8/8 always retries" was wrong.

## Three observations the data forces

1. **Inverse size correlation within the failing rows.** On `fs004` and
   `fs104` the *smallest* admitted-POI count fails, the *largest*
   succeeds. A pinned-host buffer OOM scales with shard size and the
   shape would be the opposite.
2. **Order dependence by deformation.** Failures cluster on the first
   three deformations processed (`fs004`, `fs104`, `fs204`) and stop
   for `fs304` / `fs402`. That is a state-cleanup signature, not a
   per-job budget. Whatever leaks across `correlate_multi_gpu` calls
   in the parent (or whatever does *not* leak on later jobs) is the
   load-bearing variable.
3. **`fs204` fails on every variant.** Only deformation that does, and
   it is the one `results_v3`'s `config.snapshot.yaml:17` already
   flagged as upstream-pathological ("MemoryError in zstd decode during
   load_pair" in `v2`). load_pair has its own phase record and is not
   what fails here — but the upstream-pathology label probably extends.

## Why the "fast failed compute" timings are misleading

The failing 4-GPU rows record 23-34 s, the succeeding ones 53-142 s.
The natural reading is "4-GPU completes fast, then dies in the gather
step." That reading is wrong:

`instrument.timed` is a `try / finally` (`mamba_dvc/instrument.py:171-176`):

```python
t0 = time.perf_counter()
try:
    yield
finally:
    if sync:
        _sync()
    log_phase(phase, time.perf_counter() - t0, **fields)
```

`log_phase` runs on the way out *whether or not* the block raised. So
every `dispatch.compute` row in `events.jsonl` is "wall time spent
inside the block before it exited," not "wall time of a successful
compute." The 24 s figures are exception-terminated wall, not
completed work.

This rules out the gather-OOM hypothesis as the most-likely site for
all failures: the exception happens *during* the compute block, almost
certainly inside a worker.

## Where the worker exception goes

Worker side (`mamba_dvc/gpu/dispatch.py:536-539`):

```python
except Exception as exc:
    # Broad catch is intentional: workers must always send a
    # message back so the parent's `recv` doesn't block forever.
    result_send.send(("err", repr(exc), traceback.format_exc(), gpu_id))
```

Parent side, in `_gather_shards` (`mamba_dvc/gpu/dispatch.py:596-601`):

```python
if msg[0] == "err":
    _, repr_exc, tb, gid = msg
    raise RuntimeError(
        f"worker on GPU {gid} (shard size {int(shard_idx.size)}) "
        f"raised: {repr_exc}\n{tb}"
    )
```

The worker's actual exception type is laundered through `repr(exc)` and
arrives at `_correlate_with_fallback` as a `RuntimeError`. Whatever
fires inside the worker — `cupy.cuda.memory.OutOfMemoryError`, pinned
`cudaMallocHost` failure, a `CUDARuntimeError`, an FFT plan failure —
is **not** a `MemoryError` by the time the fallback's `except` clause
sees it.

## What that means for the current fallback

`_correlate_with_fallback` (`mamba_dvc/run/batch.py:524-556`) catches
`MemoryError` only (post `04bd525`). With worker-origin failures
arriving as `RuntimeError`:

- The narrowed `except` will **not** catch the 9 / 16 failures we are
  trying to instrument. Those will bubble cleanly, which is what the
  narrowing was meant to do — they should have been caught by
  whichever real bug-tracking is appropriate, not silently retried.
- The narrowed `except` will only catch parent-side `MemoryError`:
  - `published(reference)` /  `published(deformed)` /
    `published(eff_mask)` (`dispatch.py:627-631`) — the SHM publish
    which copies the volume into a `multiprocessing.shared_memory`
    block.
  - `np.zeros((n_points, 3), dtype=np.float32)` and siblings in
    `_gather_shards` (`dispatch.py:561-563`).
  - The CPU outlier reduction (`dispatch.py:888-889`).
- Net result: on a re-run, **the next campaign will fail loudly rather
  than retry silently**. That is the desired behaviour for diagnosing
  root cause; it is *not* a regression to land.

## Cost of the silent retry while it was active

(Computed from the table above against
`results_v3/events.jsonl` directly.)

- 9 retried jobs × first-pass wall ≈ 24-34 s ≈ **~250 s wasted compute**
  spread across the session.
- `dispatch.total` on the 2-GPU retry runs ≈ 1.6-2× the 4-GPU
  successful equivalent (compare fs304 / fs402 OK rows to retried
  rows). At 9 retried jobs, that is the difference between ~130 s and
  ~55 s per job — call it **~10 min lost** out of ~26 min session.

## Open questions

1. **What exception class actually fires inside the worker?** The
   on-disk log cannot tell us — it predates retry instrumentation, and
   the `_serve_worker` path laundres the type through `repr` before it
   ever reaches a structured field. Need either a fresh campaign on
   `>= 04bd525` *and* a typed-error pipe, or a one-off worker hook
   that logs the typed name directly.
2. **What is the order-dependent state that lets `fs304` / `fs402`
   succeed?** Hypotheses worth disproving:
   - CuPy pinned pool grew enough on early deformations that later
     ones reuse instead of allocating.
   - Worker `spawn` warm-up cost amortises after the OS finishes
     paging the parent's data out.
   - Specific to fs004/fs104/fs204 deformed-volume content — but
     deformations don't change FFT shape, batch sizing, or worker
     allocation, so this would have to come through a value-dependent
     CuPy code path (unlikely).
3. **Why does the smaller `n_admitted` fail on the early deformations
   while the larger succeeds on those same deformations?**
   Counter-intuitive enough that it probably eliminates "more POIs =
   more buffer" as a primary cause. Could be Z-shard distribution:
   high mask threshold concentrates admitted POIs spatially, leaving
   some Z-block workers with sparse / empty shards. Verifiable from
   `_shard_admitted_indices` against the admitted indices for each
   `mt` threshold.

## Recommended next step

Ship a typed-error pipe so the next campaign's
`correlate_retry.exception_type` field is meaningful:

1. In `_worker_entry` send `type(exc).__name__` alongside the
   `repr / traceback / gpu_id` tuple.
2. In `_gather_shards`, look at the class name on receive:
   - If it is a known OOM type (`MemoryError`, `OutOfMemoryError`,
     `PinnedMemoryError`, `cudaErrorMemoryAllocation`), re-raise as
     `MemoryError(message)`.
   - Otherwise re-raise as `RuntimeError(message)` as today.

That preserves the existing parent-side contract (workers send
something, gather raises), unblocks `except MemoryError` from ever
actually firing on a worker-origin OOM, and gives
`correlate_retry.exception_type` a meaningful value. It does not fix
the underlying OOM — plan §9 (non-pinned host staging in
`gpu.dispatch`) still owns root cause — but it is a prerequisite for
distinguishing OOM-class failures from bug-class failures in the
fallback.
