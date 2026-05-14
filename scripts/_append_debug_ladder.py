"""One-shot helper: append §9 debug-ladder cells to the temporal showcase notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat

NB_PATH = Path("notebooks/auto-dispatch-temporal-showcase.ipynb")

cells_to_append: list = []


def md(src: str) -> None:
    cells_to_append.append(nbformat.v4.new_markdown_cell(src))


def code(src: str) -> None:
    cells_to_append.append(nbformat.v4.new_code_cell(src))


md(
    "## §9 Debug ladder — small-displacement error\n"
    "\n"
    "Adapted from `notebooks/dvc-debug-texture.ipynb`. The error pattern we observed\n"
    "(MAE 6+ vx at `t=1`, 0.05 vx at `t=15`) is inverted relative to the\n"
    "`|u|/W` shrinkage bug that the prior debug session diagnosed, so the prior\n"
    "fix is not regressed — but the methodology is. Five focused bisections\n"
    "below."
)

code(
    "from mamba_dvc.pipeline.correlate import correlate\n"
    "from mamba_dvc.pipeline._internal import NCCMode, NCCNormalization\n"
    "\n"
    "grid_positions = results[PairingStrategy.SEQUENTIAL].grid.positions\n"
    "gt_small = series.field_at(grid_positions, 1.0) - series.field_at(grid_positions, 0.0)\n"
    "gt_big = series.field_at(grid_positions, 15.0) - series.field_at(grid_positions, 0.0)\n"
    "print(f'GT small (t=0->1)  mean ||u||: {float(np.linalg.norm(gt_small, axis=1).mean()):.4f} vx')\n"  # noqa: E501
    "print(f'GT big   (t=0->15) mean ||u||: {float(np.linalg.norm(gt_big, axis=1).mean()):.4f} vx')\n"  # noqa: E501
    "print(f'GT small per-axis mean: {gt_small.mean(axis=0)}')\n"
    "print(f'GT big   per-axis mean: {gt_big.mean(axis=0)}')\n"
    "\n"
    "def stat_row(label, field, gt):\n"
    "    err = field.displacements - gt\n"
    "    valid = field.valid\n"
    "    if not bool(valid.any()):\n"
    "        return f'  {label:38s}  no valid POIs'\n"
    "    mag = np.linalg.norm(err[valid], axis=1)\n"
    "    rec_mean = field.displacements[valid].mean(axis=0)\n"
    "    return (\n"
    "        f'  {label:38s}  MAE={float(mag.mean()):7.4f}  p95={float(np.percentile(mag,95)):7.4f}  '\n"  # noqa: E501
    "        f'<u_rec>=({rec_mean[0]:+.3f},{rec_mean[1]:+.3f},{rec_mean[2]:+.3f})  '\n"
    "        f'n_valid={int(valid.sum())}/{field.positions.shape[0]}'\n"
    "    )"
)

md(
    "### Step 1 — Multi-GPU dispatcher vs single-GPU in-process\n"
    "\n"
    "`device_ids=[0]` skips multiprocessing entirely; same algorithm,\n"
    "different process model. If results agree, the bug is algorithmic, not in\n"
    "the multi-process pool."
)

code(
    "print('=== Small displacement: pair (frames[0], frames[1]) ===')\n"
    "print(stat_row('multi-GPU correlate_series[0]',\n"
    "               results[PairingStrategy.SEQUENTIAL].fields[0], gt_small))\n"
    "with MultiGPUDispatcher(device_ids=[0], volume_shape=VOLUME_SHAPE,\n"
    "                        window=WINDOW, overlap=OVERLAP) as d1:\n"
    "    f_small_d1 = d1.correlate(series.frames[0], series.frames[1])\n"
    "    f_big_d1 = d1.correlate(series.frames[0], series.frames[15])\n"
    "print(stat_row('single-GPU dispatcher [0]', f_small_d1, gt_small))\n"
    "\n"
    "print()\n"
    "print('=== Large displacement: pair (frames[0], frames[15]) ===')\n"
    "print(stat_row('multi-GPU correlate_series[-1]',\n"
    "               results[PairingStrategy.REFERENCE_ANCHORED].fields[-1], gt_big))\n"
    "print(stat_row('single-GPU dispatcher [0]', f_big_d1, gt_big))"
)

md(
    "### Step 2 — Integer-shift sanity via `np.roll`\n"
    "\n"
    "If even a hand-built `np.roll((2,-3,1))` pair fails, the bug is below the\n"
    "warp (search-radius gate, outlier test, or sign convention)."
)

code(
    "int_shift = (2, -3, 1)\n"
    "ref_int = series.frames[0]\n"
    "def_int = np.roll(ref_int, shift=int_shift, axis=(0, 1, 2))\n"
    "\n"
    "with MultiGPUDispatcher(device_ids=[0], volume_shape=VOLUME_SHAPE,\n"
    "                        window=WINDOW, overlap=OVERLAP) as d1:\n"
    "    f_int = d1.correlate(ref_int, def_int)\n"
    "\n"
    "gt_int = np.zeros_like(grid_positions)\n"
    "gt_int[:, 0] = int_shift[0]\n"
    "gt_int[:, 1] = int_shift[1]\n"
    "gt_int[:, 2] = int_shift[2]\n"
    "print(f'=== integer shift {int_shift} via np.roll ===')\n"
    "print(stat_row('single-GPU dispatcher [0]', f_int, gt_int))"
)

md(
    "### Step 3 — Outlier status histograms across pairs\n"
    "\n"
    "If many POIs at small `t` are flagged `OUTLIER` while large `t` is clean,\n"
    "the host-side outlier test is misfiring when the median displacement is\n"
    "near zero (dispersion estimate dominated by noise)."
)

code(
    "from mamba_dvc.types import POIStatus\n"
    "\n"
    "def status_row(label, fields):\n"
    "    print(f'  {label}:')\n"
    "    print(f'    pair   ' + '  '.join(f'{s.name:>10s}' for s in POIStatus))\n"
    "    for i, f in enumerate(fields):\n"
    "        counts = [int(np.count_nonzero(f.status == s)) for s in POIStatus]\n"
    "        print(f'    t={i+1:2d}   ' + '  '.join(f'{c:10d}' for c in counts))\n"
    "\n"
    "status_row('REFERENCE_ANCHORED', results[PairingStrategy.REFERENCE_ANCHORED].fields)\n"
    "print()\n"
    "status_row('SEQUENTIAL',         results[PairingStrategy.SEQUENTIAL].fields)"
)

md(
    "### Step 4 — Search-radius sweep at small displacement\n"
    "\n"
    "Default `SEARCH_RADIUS = window//2 = 32` admits peaks up to 32 vx off zero.\n"
    "If the texture has secondary autocorrelation lobes within ~6 vx of zero,\n"
    "the peakfit can latch onto a sibling at small true displacement. Tightening\n"
    "the basin to ±4 vx should collapse the error if peak ambiguity is the\n"
    "cause."
)

code(
    "radii = [32, 8, 4, 2]\n"
    "print('=== pair (frames[0], frames[1])  small |u| ~ 0.97 vx ===')\n"
    "for r in radii:\n"
    "    with MultiGPUDispatcher(device_ids=[0], volume_shape=VOLUME_SHAPE,\n"
    "                            window=WINDOW, overlap=OVERLAP,\n"
    "                            search_radius=r) as d1:\n"
    "        f = d1.correlate(series.frames[0], series.frames[1])\n"
    "    print(stat_row(f'search_radius={r:2d}', f, gt_small))"
)

md(
    "### Step 5 — NCC kernel A/B at small displacement\n"
    "\n"
    "Re-runs the same A/B from §8 of `dvc-debug-texture.ipynb` on the failing\n"
    "small-`t` pair: `linear`+`overlap`+`alpha=0` (current default) vs\n"
    "`cyclic`+`global`+`alpha=0.25` (legacy). If linear-overlap is the worse one\n"
    "on this case, the fix that landed for static synthetic pairs is somehow\n"
    "miscalibrated on the (frames[0], frames[1]) data path."
)

code(
    "modes = [\n"
    "    ('linear-overlap', NCCMode.LINEAR,  NCCNormalization.OVERLAP, None),\n"
    "    ('cyclic-global',  NCCMode.CYCLIC,  NCCNormalization.GLOBAL,  None),\n"
    "]\n"
    "print('=== pair (frames[0], frames[1])  small |u| ===')\n"
    "for label, mode, norm, alpha in modes:\n"
    "    with MultiGPUDispatcher(device_ids=[0], volume_shape=VOLUME_SHAPE,\n"
    "                            window=WINDOW, overlap=OVERLAP,\n"
    "                            ncc_mode=mode, ncc_normalization=norm,\n"
    "                            tukey_alpha=alpha) as d1:\n"
    "        f = d1.correlate(series.frames[0], series.frames[1])\n"
    "    print(stat_row(label, f, gt_small))\n"
    "\n"
    "print()\n"
    "print('=== pair (frames[0], frames[15])  large |u| ===')\n"
    "for label, mode, norm, alpha in modes:\n"
    "    with MultiGPUDispatcher(device_ids=[0], volume_shape=VOLUME_SHAPE,\n"
    "                            window=WINDOW, overlap=OVERLAP,\n"
    "                            ncc_mode=mode, ncc_normalization=norm,\n"
    "                            tukey_alpha=alpha) as d1:\n"
    "        f = d1.correlate(series.frames[0], series.frames[15])\n"
    "    print(stat_row(label, f, gt_big))"
)

md(
    "### Step 6 — Verdict\n"
    "\n"
    "Interpretation appended after execution, based on which probe lights up."
)


def main() -> None:
    nb = nbformat.read(NB_PATH, as_version=4)
    nb.cells.extend(cells_to_append)
    nbformat.write(nb, NB_PATH)
    print(f"Appended {len(cells_to_append)} cells to {NB_PATH}")


if __name__ == "__main__":
    main()
