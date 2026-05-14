"""Visualize a displacement field saved by ``single_gpu_dvf.py``.

Reshapes the sparse ``(points, 3)`` arrays back onto the regular POI
lattice (``grid_shape``) with NaN where ``valid`` is False, then renders
three orthogonal mid-slices for magnitude, per-axis components, quiver
overlays, confidence, and residuals (if GT was saved). Also produces a
single-figure histogram panel.

All output is matplotlib PNG.

Usage
-----
::

    uv run python scripts/visualize_dvf.py \
        --npz C:/.../out/103L_fs004_mt09/displacement_field.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_AXIS_NAMES = ("dz", "dy", "dx")
_SLICE_LABELS = ("XY (z=Z/2)", "XZ (y=Y/2)", "YZ (x=X/2)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--npz", type=Path, required=True, help="displacement_field.npz")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="figures dir (default: <npz parent>/figures)",
    )
    p.add_argument(
        "--quiver-stride",
        type=int,
        default=None,
        help="POI stride for quiver subsampling per axis (default: auto)",
    )
    return p.parse_args()


def _reshape_to_grid(
    flat: np.ndarray, grid_shape: tuple[int, int, int], last_dim: int | None
) -> np.ndarray:
    """``(points, *)`` → ``(*grid_shape, *)`` with grid_shape ordering (z, y, x)."""
    if last_dim is None:
        return flat.reshape(grid_shape)
    return flat.reshape((*grid_shape, last_dim))


def _three_mid_slices(volume: np.ndarray) -> list[np.ndarray]:
    """Return mid-slices in XY (axis=z), XZ (axis=y), YZ (axis=x) order."""
    nz, ny, nx = volume.shape[:3]
    return [
        volume[nz // 2, :, :],  # XY
        volume[:, ny // 2, :],  # XZ
        volume[:, :, nx // 2],  # YZ
    ]


def _plot_orthogonal_panels(
    slices: list[np.ndarray],
    title: str,
    out_path: Path,
    *,
    cmap: str = "viridis",
    symmetric: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if symmetric:
        finite = [s[np.isfinite(s)] for s in slices]
        bound = float(max((np.max(np.abs(f)) for f in finite if f.size > 0), default=1.0))
        vmin, vmax = -bound, bound
    else:
        finite = np.concatenate([s[np.isfinite(s)].ravel() for s in slices])
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
    for ax, sl, label in zip(axes, slices, _SLICE_LABELS, strict=True):
        im = ax.imshow(sl, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_per_axis(components: np.ndarray, title_prefix: str, out_path: Path) -> None:
    """Components shape ``(nz, ny, nx, 3)``; one row per slice plane, one col per axis."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for row_i, (sl_idx, _label) in enumerate(zip((0, 1, 2), _SLICE_LABELS, strict=True)):
        for col_i, axis_name in enumerate(_AXIS_NAMES):
            vol_axis = components[..., col_i]
            slices = _three_mid_slices(vol_axis)
            sl = slices[sl_idx]
            finite = sl[np.isfinite(sl)]
            bound = float(np.max(np.abs(finite))) if finite.size > 0 else 1.0
            ax = axes[row_i, col_i]
            im = ax.imshow(
                sl,
                origin="lower",
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
            )
            ax.set_title(f"{_SLICE_LABELS[sl_idx]} : {axis_name}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{title_prefix}: per-axis components on orthogonal mid-slices")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_quiver(disp: np.ndarray, valid: np.ndarray, out_path: Path, *, stride: int) -> None:
    """3 panels: subsampled (u, v) overlay on magnitude background per mid-slice."""
    mag = np.linalg.norm(np.where(valid[..., None], disp, np.nan), axis=-1)
    nz, ny, nx = mag.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XY mid-plane (z fixed). u = dx (axis 2), v = dy (axis 1).
    zi = nz // 2
    sl_mag = mag[zi]
    sl_disp = disp[zi]
    sl_valid = valid[zi]
    _quiver_panel(
        axes[0],
        sl_mag,
        u=sl_disp[..., 2],
        v=sl_disp[..., 1],
        valid=sl_valid,
        stride=stride,
        title=_SLICE_LABELS[0],
    )

    # XZ mid-plane (y fixed). u = dx (axis 2), v = dz (axis 0).
    yi = ny // 2
    sl_mag = mag[:, yi]
    sl_disp = disp[:, yi]
    sl_valid = valid[:, yi]
    _quiver_panel(
        axes[1],
        sl_mag,
        u=sl_disp[..., 2],
        v=sl_disp[..., 0],
        valid=sl_valid,
        stride=stride,
        title=_SLICE_LABELS[1],
    )

    # YZ mid-plane (x fixed). u = dy (axis 1), v = dz (axis 0).
    xi = nx // 2
    sl_mag = mag[:, :, xi]
    sl_disp = disp[:, :, xi]
    sl_valid = valid[:, :, xi]
    _quiver_panel(
        axes[2],
        sl_mag,
        u=sl_disp[..., 1],
        v=sl_disp[..., 0],
        valid=sl_valid,
        stride=stride,
        title=_SLICE_LABELS[2],
    )

    fig.suptitle("Displacement field: |u| background + subsampled vectors")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _quiver_panel(
    ax,
    background: np.ndarray,
    *,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    stride: int,
    title: str,
) -> None:
    im = ax.imshow(background, origin="lower", cmap="viridis")
    ny, nx = background.shape
    ys = np.arange(0, ny, stride)
    xs = np.arange(0, nx, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    sub_u = u[yy, xx]
    sub_v = v[yy, xx]
    sub_valid = valid[yy, xx]
    sub_u = np.where(sub_valid, sub_u, np.nan)
    sub_v = np.where(sub_valid, sub_v, np.nan)
    ax.quiver(
        xx,
        yy,
        sub_u,
        sub_v,
        color="white",
        scale_units="xy",
        angles="xy",
        width=0.003,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_histograms(
    disp_mag_flat: np.ndarray,
    conf_flat: np.ndarray,
    resid_mag_flat: np.ndarray | None,
    out_path: Path,
) -> None:
    has_resid = resid_mag_flat is not None
    n_cols = 3 if has_resid else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    axes = np.atleast_1d(axes)

    axes[0].hist(disp_mag_flat[np.isfinite(disp_mag_flat)], bins=80, color="steelblue")
    axes[0].set_title("|displacement| (vox)")
    axes[0].set_yscale("log")

    axes[1].hist(conf_flat[np.isfinite(conf_flat)], bins=80, color="seagreen")
    axes[1].set_title("confidence")
    axes[1].set_yscale("log")

    if has_resid:
        assert resid_mag_flat is not None
        axes[2].hist(
            resid_mag_flat[np.isfinite(resid_mag_flat)],
            bins=80,
            color="indianred",
        )
        axes[2].set_title("|residual| (vox)")
        axes[2].set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _auto_stride(grid_shape: tuple[int, int, int]) -> int:
    longest = max(grid_shape)
    # Aim for ~30 arrows along the longest in-plane axis on a 2D panel.
    return max(1, longest // 30)


def main() -> int:
    """Render orthogonal mid-slices + histograms for a saved displacement field."""
    args = _parse_args()
    npz_path: Path = args.npz
    out_dir = args.out_dir if args.out_dir is not None else npz_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path) as data:
        displacements = data["displacements"]
        valid = data["valid"].astype(bool)
        confidence = data["confidence"]
        grid_shape = tuple(int(x) for x in data["grid_shape"])
        residuals = data["residuals"] if "residuals" in data.files else None

    assert len(grid_shape) == 3
    # Mask non-valid POIs to NaN so colormaps and stats ignore them.
    disp_masked = np.where(valid[:, None], displacements, np.nan).astype(np.float32)
    conf_masked = np.where(valid, confidence, np.nan).astype(np.float32)

    disp_grid = _reshape_to_grid(disp_masked, grid_shape, last_dim=3)
    valid_grid = _reshape_to_grid(valid, grid_shape, last_dim=None)
    conf_grid = _reshape_to_grid(conf_masked, grid_shape, last_dim=None)

    mag_grid = np.linalg.norm(disp_grid, axis=-1)
    mag_slices = _three_mid_slices(mag_grid)
    _plot_orthogonal_panels(
        mag_slices,
        title="|displacement| (vox)",
        out_path=out_dir / "magnitude.png",
        cmap="viridis",
        symmetric=False,
    )

    _plot_per_axis(disp_grid, "displacement", out_dir / "components.png")

    stride = args.quiver_stride if args.quiver_stride is not None else _auto_stride(grid_shape)
    _plot_quiver(disp_grid, valid_grid, out_dir / "quiver.png", stride=stride)

    conf_slices = _three_mid_slices(conf_grid)
    _plot_orthogonal_panels(
        conf_slices,
        title="confidence (peak NCC)",
        out_path=out_dir / "confidence.png",
        cmap="magma",
        symmetric=False,
    )

    resid_mag_flat: np.ndarray | None = None
    if residuals is not None and np.any(np.isfinite(residuals)):
        resid_grid = _reshape_to_grid(residuals.astype(np.float32), grid_shape, last_dim=3)
        resid_mag_grid = np.linalg.norm(resid_grid, axis=-1)
        _plot_orthogonal_panels(
            _three_mid_slices(resid_mag_grid),
            title="|residual| = |pred - GT| (vox)",
            out_path=out_dir / "residual_magnitude.png",
            cmap="inferno",
            symmetric=False,
        )
        _plot_per_axis(resid_grid, "residual", out_dir / "residual_components.png")
        resid_mag_flat = resid_mag_grid.ravel()

    _plot_histograms(
        disp_mag_flat=mag_grid.ravel(),
        conf_flat=conf_grid.ravel(),
        resid_mag_flat=resid_mag_flat,
        out_path=out_dir / "histograms.png",
    )

    print(f"wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
