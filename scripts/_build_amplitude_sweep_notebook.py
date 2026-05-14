"""Generate ``notebooks/amplitude-sweep.ipynb``.

Throwaway builder. Mirrors the pattern of ``_append_debug_ladder.py``: keep
the notebook JSON reproducible by editing a Python script instead of
hand-editing escaped JSON. Re-run after every edit.

    uv run python scripts/_build_amplitude_sweep_notebook.py
"""

# Builder embeds Unicode math typography and long markdown lines.
# ruff: noqa: E501, RUF001

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.rstrip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.rstrip() + "\n")


CELLS: list[nbf.NotebookNode] = []

CELLS.append(
    md(
        """# Amplitude-swept DVC error analysis — fixed settings bundle, varied displacement magnitude

Companion to `auto-dispatch-temporal-showcase.ipynb`. Where that notebook holds a
velocity field constant and sweeps `t`, here we hold a single DVC settings bundle
constant and sweep the **displacement magnitude** directly.

For each target amplitude `A` we warp the same reference under a pure rigid shift
`u(x) = A · d_hat` and run a single-pair DVC. Because the shift is constant across
the volume, the target amplitude equals the per-POI ground-truth amplitude exactly
— no spatial-average calibration is needed.

### Why this parameterization
- **Amplitude is the only knob.** Texture, direction, settings bundle, and reference
  volume are all frozen across the sweep.
- **GT is trivial.** `u_GT(POI) = A · d_hat` for every POI, every amplitude.
  Recovered vs analytical is one subtract per amplitude.
- **Probes the full operating range.** A log-spaced grid from 0.25 vx to 24 vx covers
  the subvoxel-bias regime, the canonical linear-recovery regime, and the high-amplitude
  regime that approaches the search radius (`window // 2 = 32` vx by default).

### Plot outputs
1. MAE / RMSE / p95 vs amplitude (log-log).
2. POI status fractions (OK / OUTLIER / OUT_OF_RANGE / …) vs amplitude.
3. POI-level scatter of `||u_rec − u_GT||` against amplitude, with per-amplitude
   median and p95 overlaid — exposes per-POI variability hidden by the aggregate.
4. Summary table.

### Sign convention
Pull-back: `deformed(x) = reference(x − u(x))`. `correlate()` recovers the same `u`
that was applied here, so recovered vs analytical compares directly without sign flips.
"""
    )
)

CELLS.append(
    code(
        """from __future__ import annotations

import time
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np

from mamba_dvc.types import POIStatus
from mamba_dvc.validate.synthetic import make_texture, rigid_shift, warp

try:
    import cupy as _cp  # noqa: F401
    from mamba_dvc.gpu.dispatch import MultiGPUDispatcher
    _HAS_CUPY = True
except ImportError:
    MultiGPUDispatcher = None  # type: ignore[assignment]
    _HAS_CUPY = False

print(f"CuPy available: {_HAS_CUPY}")
"""
    )
)

CELLS.append(
    md(
        """## 1. Setup — fixed reference, fixed settings, swept amplitude

One reference texture (`make_texture(..., seed=0)`), one unit direction, one DVC
settings bundle. Only `TARGET_AMPLITUDES_VX` varies across the sweep; everything
else is locked.

The shift direction `(1.0, 1.6, 0.4)` is non-axis-aligned so every axis carries
signal — a unit-magnitude shift is split unevenly across z / y / x.
"""
    )
)

CELLS.append(
    code(
        """# --- Volume / texture ----------------------------------------------------
VOLUME_SHAPE: tuple[int, int, int] = (192, 192, 192)
TEXTURE_SIGMA = 1.5
TEXTURE_SEED = 0
WARP_ORDER = 3                          # cubic spline; order=1 inflates the subvoxel floor

# --- Amplitude sweep -----------------------------------------------------
TARGET_AMPLITUDES_VX: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0)
SHIFT_DIRECTION: tuple[float, float, float] = (1.0, 1.6, 0.4)   # raw direction, normalized below

# --- DVC settings bundle (FIXED across the sweep) ------------------------
WINDOW = 64
OVERLAP = 0.5
MASK_THRESHOLD = 0.9
SEARCH_RADIUS = None                    # None -> window // 2 == 32 vx
TUKEY_ALPHA = None                      # None -> mode-dependent default
BATCH_SIZE: int | str = "auto"
EPS = 1e-12

# --- Dispatcher ----------------------------------------------------------
USE_DISPATCHER = _HAS_CUPY
DEVICE_IDS: list[int] | None = None     # None -> all visible CUDA devices

# Derived: unit direction vector
_dir = np.asarray(SHIFT_DIRECTION, dtype=np.float32)
DIRECTION_UNIT = _dir / float(np.linalg.norm(_dir))

print("Volume shape       :", VOLUME_SHAPE)
print("Target amplitudes  :", TARGET_AMPLITUDES_VX, "vx")
print("Direction (raw)    :", SHIFT_DIRECTION)
print(
    "Direction (unit)   : "
    f"({DIRECTION_UNIT[0]:+.4f}, {DIRECTION_UNIT[1]:+.4f}, {DIRECTION_UNIT[2]:+.4f})"
)
print("Window / overlap   :", WINDOW, "/", OVERLAP)
print("Search radius      :", SEARCH_RADIUS, "(None -> window//2)")
print("Use dispatcher     :", USE_DISPATCHER, "  devices:", DEVICE_IDS)
"""
    )
)

CELLS.append(
    md(
        """## 2. Reference texture

Same band-limited noise reference the temporal showcase uses (`sigma=1.5`,
`seed=0`). One reference is reused for every amplitude in the sweep.
"""
    )
)

CELLS.append(
    code(
        """reference = make_texture(VOLUME_SHAPE, sigma=TEXTURE_SIGMA, seed=TEXTURE_SEED)
print(
    f"reference  shape={reference.shape}  dtype={reference.dtype}  "
    f"mean={reference.mean():+.3f}  std={reference.std():.3f}"
)


def show_orthoslices(volume: np.ndarray, title: str) -> None:
    z_mid, y_mid, x_mid = (s // 2 for s in volume.shape)
    vmin, vmax = np.percentile(volume, [1, 99])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, plane, label in zip(
        axes,
        [volume[z_mid], volume[:, y_mid], volume[:, :, x_mid]],
        [f"z={z_mid}", f"y={y_mid}", f"x={x_mid}"],
        strict=True,
    ):
        im = ax.imshow(plane, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    fig.suptitle(title)
    plt.show()


show_orthoslices(reference, "Reference texture (mid-plane orthoslices)")
"""
    )
)

CELLS.append(
    md(
        """## 3. Synthesize amplitude-swept frames

For each `A` in `TARGET_AMPLITUDES_VX`, build the pure-shift field
`rigid_shift(A · d_hat)` and warp the reference under it (pull-back convention).
The result is one deformed frame per amplitude, all sharing the same reference.

`order=3` cubic spline matches the temporal showcase — lower orders inflate the
subvoxel error floor enough to confound the measurement (see
`validate/synthetic.py` docstring).
"""
    )
)

CELLS.append(
    code(
        """shift_vectors: list[tuple[float, float, float]] = []
gt_fields = []
deformed_frames: list[np.ndarray] = []

t0 = time.perf_counter()
for A in TARGET_AMPLITUDES_VX:
    shift = tuple(float(c) for c in (A * DIRECTION_UNIT))
    shift_vectors.append(shift)
    field_gt = rigid_shift(shift)
    gt_fields.append(field_gt)
    deformed_frames.append(
        warp(reference, field_gt, order=WARP_ORDER, convention="pull_back")
    )
t_synth = time.perf_counter() - t0
print(f"Synthesized {len(deformed_frames)} deformed frames in {t_synth:.1f} s")
for A, shift, frame in zip(TARGET_AMPLITUDES_VX, shift_vectors, deformed_frames, strict=True):
    print(
        f"  A={A:6.2f} vx  shift=({shift[0]:+.3f},{shift[1]:+.3f},{shift[2]:+.3f})  "
        f"std={frame.std():.3f}"
    )
"""
    )
)

CELLS.append(
    md(
        """### 3a. Visual sanity — reference vs low / mid / high amplitude

Mid-z slice at the smallest, middle, and largest amplitude alongside the
reference. The largest-amplitude pair should look visibly translated; the
smallest should look nearly identical to the reference.
"""
    )
)

CELLS.append(
    code(
        """preview_idx = [0, len(TARGET_AMPLITUDES_VX) // 2, len(TARGET_AMPLITUDES_VX) - 1]
z_mid = VOLUME_SHAPE[0] // 2

fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
vmin, vmax = np.percentile(reference, [1, 99])
axes[0].imshow(reference[z_mid], cmap="gray", vmin=vmin, vmax=vmax)
axes[0].set_title(f"reference\\n(z={z_mid})")
axes[0].set_axis_off()
for ax, idx in zip(axes[1:], preview_idx, strict=True):
    A = TARGET_AMPLITUDES_VX[idx]
    ax.imshow(deformed_frames[idx][z_mid], cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"|u|={A:g} vx\\n(z={z_mid})")
    ax.set_axis_off()
plt.tight_layout()
plt.show()
"""
    )
)

CELLS.append(
    md(
        """## 4. Run the DVC sweep — fixed settings, swept amplitude

Open `MultiGPUDispatcher` with `anchored_reference=reference` so the reference
is uploaded to every GPU once at `__enter__` and reused for every amplitude
(skip the per-pair re-upload). Then `correlate(None, deformed[i])` runs each
pair against the resident reference.

If CuPy is unavailable the cell falls back to the host-side `correlate()`. The
metric code below does not care which path produced the displacement fields.
"""
    )
)

CELLS.append(
    code(
        """def open_dispatcher():
    if USE_DISPATCHER and MultiGPUDispatcher is not None:
        return MultiGPUDispatcher(
            device_ids=DEVICE_IDS,
            volume_shape=VOLUME_SHAPE,
            anchored_reference=reference,
            window=WINDOW,
            overlap=OVERLAP,
            mask_threshold=MASK_THRESHOLD,
            tukey_alpha=TUKEY_ALPHA,
            search_radius=SEARCH_RADIUS,
            batch_size=BATCH_SIZE,
            eps=EPS,
        )
    return nullcontext(None)


def correlate_one(dispatcher, deformed):
    if dispatcher is not None:
        return dispatcher.correlate(None, deformed)
    from mamba_dvc.pipeline.correlate import correlate as _correlate

    return _correlate(
        reference,
        deformed,
        mask=None,
        window=WINDOW,
        overlap=OVERLAP,
        mask_threshold=MASK_THRESHOLD,
        tukey_alpha=TUKEY_ALPHA,
        search_radius=SEARCH_RADIUS,
        batch_size=BATCH_SIZE if isinstance(BATCH_SIZE, int) else 1,
        eps=EPS,
    )


results_fields: list = []
timings: list[float] = []
with open_dispatcher() as dispatcher:
    if dispatcher is not None:
        print(f"Dispatcher opened on devices {dispatcher.device_ids}")
        print(f"  multiprocess     : {dispatcher.is_multiprocess}")
        print(f"  anchored reference: True (uploaded once at __enter__)")
    else:
        print("Dispatcher: none (host-only correlate fallback)")

    for A, deformed in zip(TARGET_AMPLITUDES_VX, deformed_frames, strict=True):
        t0 = time.perf_counter()
        field = correlate_one(dispatcher, deformed)
        elapsed = time.perf_counter() - t0
        n_ok = int(np.count_nonzero(field.status == int(POIStatus.OK)))
        n_total = int(field.positions.shape[0])
        print(
            f"  A={A:6.2f} vx  ok={n_ok:3d}/{n_total:3d}  elapsed={elapsed:.2f} s"
        )
        results_fields.append(field)
        timings.append(elapsed)
"""
    )
)

CELLS.append(
    md(
        """## 5. Per-amplitude error metrics

`u_GT(POI) = A · d_hat` is constant across POIs, so the per-POI error is
`u_rec − u_GT` and the per-amplitude stats reduce to MAE / RMSE / median / p95
of `||u_rec − u_GT||` over the valid (OK) POIs.

Stored arrays (`status_counts`, `err_mag_by_amp`) feed the plots below.
"""
    )
)

CELLS.append(
    code(
        """amplitude_arr = np.array(TARGET_AMPLITUDES_VX, dtype=np.float64)
mae = np.full(amplitude_arr.shape, np.nan, dtype=np.float64)
rmse = np.full(amplitude_arr.shape, np.nan, dtype=np.float64)
median = np.full(amplitude_arr.shape, np.nan, dtype=np.float64)
p95 = np.full(amplitude_arr.shape, np.nan, dtype=np.float64)
n_valid = np.zeros(amplitude_arr.shape, dtype=np.int64)
n_total = np.zeros(amplitude_arr.shape, dtype=np.int64)
err_mag_by_amp: list[np.ndarray] = []

status_keys = (
    POIStatus.OK,
    POIStatus.MASKED,
    POIStatus.OUTLIER,
    POIStatus.LOW_CONF,
    POIStatus.OUT_OF_RANGE,
)
status_counts = np.zeros((len(amplitude_arr), len(status_keys)), dtype=np.int64)

for i, (A, field, fgt) in enumerate(
    zip(TARGET_AMPLITUDES_VX, results_fields, gt_fields, strict=True)
):
    gt = fgt(field.positions)
    err = field.displacements - gt
    err_mag = np.linalg.norm(err, axis=1)
    err_mag_by_amp.append(err_mag)

    valid = field.valid
    n_total[i] = int(valid.size)
    n_valid[i] = int(valid.sum())
    if valid.any():
        v = err_mag[valid]
        mae[i] = float(v.mean())
        rmse[i] = float(np.sqrt((v ** 2).mean()))
        median[i] = float(np.median(v))
        p95[i] = float(np.percentile(v, 95))

    for k, s in enumerate(status_keys):
        status_counts[i, k] = int(np.count_nonzero(field.status == int(s)))

print(
    f"{'|u| [vx]':>9s}  {'n_ok/total':>10s}  "
    f"{'MAE':>7s}  {'RMSE':>7s}  {'median':>7s}  {'p95':>7s}"
)
for i, A in enumerate(TARGET_AMPLITUDES_VX):
    print(
        f"{A:9.3f}  {n_valid[i]:5d}/{n_total[i]:<4d}  "
        f"{mae[i]:7.4f}  {rmse[i]:7.4f}  {median[i]:7.4f}  {p95[i]:7.4f}"
    )
"""
    )
)

CELLS.append(
    code(
        """fig, axes = plt.subplots(1, 3, figsize=(13, 4))
panels = [(mae, "MAE"), (rmse, "RMSE"), (p95, "p95")]
for ax, (y, label) in zip(axes, panels, strict=True):
    ax.plot(amplitude_arr, y, "o-", color="tab:blue")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("target |u| [voxels]")
    ax.set_ylabel(f"{label} [voxels]")
    ax.set_title(f"{label} vs displacement amplitude")
    ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
"""
    )
)

CELLS.append(
    md(
        """## 6. POI status fractions vs amplitude

Stacked bars showing the fraction of grid POIs classified as OK, OUTLIER,
OUT_OF_RANGE, LOW_CONF, MASKED at each amplitude. Mass shifting from OK to
OUTLIER / OUT_OF_RANGE as `A` grows tells you the search-radius / outlier
gates are biting; mass shifting to LOW_CONF tells you the NCC peak itself
is degrading.
"""
    )
)

CELLS.append(
    code(
        """status_labels = [s.name for s in status_keys]
status_fractions = status_counts / status_counts.sum(axis=1, keepdims=True)
status_colors = ["tab:green", "tab:gray", "tab:red", "tab:orange", "tab:purple"]

fig, ax = plt.subplots(figsize=(8.5, 4.2))
x_pos = np.arange(len(amplitude_arr))
bottom = np.zeros(len(amplitude_arr))
for k, (label, color) in enumerate(zip(status_labels, status_colors, strict=True)):
    ax.bar(x_pos, status_fractions[:, k], bottom=bottom, label=label, color=color, width=0.7)
    bottom += status_fractions[:, k]
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{A:g}" for A in amplitude_arr])
ax.set_xlabel("target |u| [voxels]")
ax.set_ylabel("fraction of POIs")
ax.set_ylim(0.0, 1.0)
ax.set_title(f"POI status fractions vs amplitude  (n_total = {int(n_total[0])})")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
plt.tight_layout()
plt.show()
"""
    )
)

CELLS.append(
    md(
        """## 7. POI-level error scatter

One point per (amplitude, POI). x-axis is the target amplitude; y-axis is
`||u_rec(POI) − u_GT(POI)||` for that POI. Per-amplitude median and p95 are
overlaid. This view exposes the per-POI spread that the aggregate MAE / p95
hides — e.g. whether a small number of outlier POIs drag the p95 trace upward
or whether the entire POI distribution drifts.
"""
    )
)

CELLS.append(
    code(
        """fig, ax = plt.subplots(figsize=(8.5, 5))
rng = np.random.default_rng(0)
for i, A in enumerate(amplitude_arr):
    valid = results_fields[i].valid
    if not valid.any():
        continue
    err_v = err_mag_by_amp[i][valid]
    # multiplicative jitter so points spread visibly on the log x-axis
    jitter = rng.uniform(0.92, 1.08, size=err_v.shape)
    ax.scatter(A * jitter, err_v, s=6, alpha=0.25, color="tab:blue")

ax.plot(amplitude_arr, median, "o-", color="tab:red", label="median (valid)")
ax.plot(amplitude_arr, p95, "s-", color="tab:orange", label="p95 (valid)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("target |u| [voxels]")
ax.set_ylabel("||u_rec − u_GT|| per POI [voxels]")
ax.set_title("POI-level error vs displacement amplitude")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()
plt.show()
"""
    )
)

CELLS.append(
    md(
        """## 8. Summary table

Compact (amplitude, ok counts, MAE / RMSE / p95, elapsed) view. Combined with
the §5–7 plots this is the per-amplitude report card for the fixed settings
bundle declared in §1.
"""
    )
)

CELLS.append(
    code(
        """header = (
    f"{'|u| [vx]':>9s}  {'n_ok':>5s}  {'n_outlier':>9s}  {'n_oor':>5s}  "
    f"{'MAE':>7s}  {'RMSE':>7s}  {'p95':>7s}  {'elapsed':>8s}"
)
print(header)
print("-" * len(header))
for i, A in enumerate(TARGET_AMPLITUDES_VX):
    n_ok = int(status_counts[i, 0])
    n_out = int(status_counts[i, 2])
    n_oor = int(status_counts[i, 4])
    print(
        f"{A:9.3f}  {n_ok:5d}  {n_out:9d}  {n_oor:5d}  "
        f"{mae[i]:7.4f}  {rmse[i]:7.4f}  {p95[i]:7.4f}  {timings[i]:7.2f}s"
    )
"""
    )
)

CELLS.append(
    md(
        """## Next steps

- **Direction sweep.** Repeat with `SHIFT_DIRECTION` set along each cardinal axis
  to expose anisotropy in the windowing or peak-fit.
- **Multi-seed statistical floor.** Loop the whole sweep over
  `TEXTURE_SEED ∈ {0, 1, 2, …}` and report mean ± std bands per amplitude.
  Needed if you want confidence intervals rather than point estimates.
- **Settings-bundle sweep.** Wrap §1–§8 in a function and loop over
  `(WINDOW, OVERLAP, NCC mode / norm)` tuples to build the full
  amplitude × settings heatmap.
  `mamba_dvc.validate.series_error.evaluate_synthetic` is the production version —
  use it when going from exploration to a permanent benchmark.
- **Search-radius edge.** Add amplitudes ≥ 28 vx (with default `window=64`) to
  map the OUT_OF_RANGE cliff. The current grid stops at 24 vx to keep the error
  curve in the regime where the gate is not yet binding.
"""
    )
)

nb = nbf.v4.new_notebook()
nb.cells = CELLS
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.14",
    },
}

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "amplitude-sweep.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print(f"wrote {OUT}  ({len(CELLS)} cells)")
