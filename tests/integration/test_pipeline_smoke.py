"""End-to-end smoke test of the v1 single-pass DVC pipeline.

Calls :func:`mamba_dvc.pipeline.correlate.correlate` against synthetic
textured pairs with known displacements. Asserts that the recovered
displacement at every interior, valid POI matches the analytical
ground truth within plan §2's 0.1-voxel target.

Replaces the earlier inline composition once
:mod:`mamba_dvc.pipeline.correlate` landed; the orchestrator owns the
six-stage chain now.
"""

from __future__ import annotations

import numpy as np
import pytest
from mamba_dvc.core.grid import build_grid
from mamba_dvc.pipeline.correlate import correlate
from mamba_dvc.types import POIStatus
from mamba_dvc.validate.synthetic import make_pair, rigid_shift, sample_on_grid


def _interior_mask(
    starts: np.ndarray,
    window: tuple[int, int, int],
    volume_shape: tuple[int, int, int],
    *,
    margin: int,
) -> np.ndarray:
    """Boolean mask of POIs whose subvolume sits ``margin`` voxels off every face.

    The synthetic warp uses ``mode="reflect"``; POIs whose subvolume
    touches the face see reflected (not displaced) content. Excluding a
    margin equal to the maximum displacement plus a buffer keeps the
    assertion clean (plan §9 calls this out under tier-1 validation).
    """
    z0 = starts[:, 0]
    y0 = starts[:, 1]
    x0 = starts[:, 2]
    wz, wy, wx = window
    sz, sy, sx = volume_shape
    return (
        (z0 >= margin)
        & (z0 + wz <= sz - margin)
        & (y0 >= margin)
        & (y0 + wy <= sy - margin)
        & (x0 >= margin)
        & (x0 + wx <= sx - margin)
    )


@pytest.mark.slow
class TestPipelineSmoke:
    @pytest.mark.parametrize(
        "shift",
        [
            (0.3, -0.7, 1.4),
            (-1.2, 1.8, -0.4),
            (2.4, -0.6, -2.7),
        ],
    )
    def test_recovers_rigid_fractional_shift(self, shift: tuple[float, float, float]):
        # Window 64 (rather than the plan's default 96) keeps the test
        # fast while preserving SNR headroom against peak-locking bias
        # at integer shifts. Shifts are chosen with non-trivial
        # fractional parts on every axis -- see plan §2 and
        # ``docs/plans/ncc.md`` for the rationale.
        shape: tuple[int, int, int] = (160, 160, 160)
        window = 64
        overlap = 0.5
        pair = make_pair(shape=shape, field=rigid_shift(shift), seed=17)

        field = correlate(
            pair.reference,
            pair.deformed,
            window=window,
            overlap=overlap,
            search_radius=window // 2,
        )
        grid = build_grid(shape, window=window, overlap=overlap)
        ground_truth = sample_on_grid(pair.field, grid)

        max_shift = int(np.ceil(max(abs(s) for s in shift)))
        margin = max_shift + 4
        interior = _interior_mask(grid.starts, grid.window, shape, margin=margin)
        usable = interior & field.valid
        assert usable.sum() >= 8, "need enough interior+valid POIs to be meaningful"

        error = np.abs(field.displacements[usable] - ground_truth[usable])
        assert np.all(error < 0.1), (
            f"max error per axis: {error.max(axis=0)} (target < 0.1); "
            f"recovered example={field.displacements[usable][0]} "
            f"expected={ground_truth[usable][0]}"
        )

    def test_recovers_zero_shift(self):
        # Identity warp -- recovered displacement at every POI must be
        # numerically zero modulo float noise.
        shape: tuple[int, int, int] = (64, 64, 64)
        pair = make_pair(shape=shape, field=rigid_shift((0.0, 0.0, 0.0)), seed=23)

        field = correlate(
            pair.reference,
            pair.deformed,
            window=32,
            overlap=0.5,
            search_radius=8,
        )

        assert (field.status == POIStatus.OK).all()
        np.testing.assert_allclose(field.displacements, 0.0, atol=0.05)
