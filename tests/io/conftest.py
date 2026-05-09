"""Shared fixtures for ``tests/io``.

Builds zarr stores conforming to ``bone_screw_synchrotron_v1`` from
``mamba_dvc.validate.synthetic`` outputs. ``make_store`` returns an
in-memory store; ``make_disk_store`` writes one to ``tmp_path`` so
``DvcDataset.open(path)`` can read it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import zarr
from mamba_dvc.io.profiles import StoreProfile
from mamba_dvc.validate.synthetic import (
    DisplacementFunction,
    make_texture,
    rigid_shift,
    warp,
)


@dataclass(frozen=True)
class _SyntheticEntry:
    """One synthetic-deformation entry to materialize into the fixture."""

    name: str
    field: DisplacementFunction


def _populate_store(
    root: zarr.Group,
    *,
    profile: StoreProfile,
    shape: tuple[int, int, int],
    masks: Sequence[str],
    real_entries: Sequence[str],
    synthetic_entries: Sequence[_SyntheticEntry],
    seed: int,
) -> None:
    """Populate ``root`` with the layout described by ``profile``."""
    reference = make_texture(shape, sigma=1.5, seed=seed)

    base = root.create_group(profile.base_group)
    ref_arr = base.create_array(
        name=profile.reference_name,
        shape=shape,
        dtype="float32",
        chunks=shape,
    )
    ref_arr[:] = reference

    rng = np.random.default_rng(seed)
    for mask_name in masks:
        m = base.create_array(name=mask_name, shape=shape, dtype="bool", chunks=shape)
        # Roughly half-True random mask; deterministic by seed.
        m[:] = rng.random(shape) > 0.5

    real = root.create_group(profile.real_group)
    for entry_name in real_entries:
        entry_group = real.create_group(entry_name)
        deformed = entry_group.create_array(
            name=profile.deformed_name,
            shape=shape,
            dtype="float32",
            chunks=shape,
        )
        # Use a small rigid shift via warp so the deformed differs.
        deformed[:] = warp(reference, rigid_shift((0.25, 0.0, 0.0)), order=3)

    syn = root.create_group(profile.synthetic_group)
    axis_order = profile.flow_defaults.axis_order
    for entry in synthetic_entries:
        entry_group = syn.create_group(entry.name)
        deformed = entry_group.create_array(
            name=profile.deformed_name,
            shape=shape,
            dtype="float32",
            chunks=shape,
        )
        deformed[:] = warp(reference, entry.field, order=3)

        # Densely sample the analytical field on every voxel.
        sampled = _sample_field(shape, entry.field)
        if axis_order == "3_zyx":
            flow_arr = entry_group.create_array(
                name=profile.flow_name,
                shape=(3, *shape),
                dtype="float32",
                chunks=(3, *shape),
            )
            flow_arr[:] = np.moveaxis(sampled, -1, 0)
        else:
            flow_arr = entry_group.create_array(
                name=profile.flow_name,
                shape=(*shape, 3),
                dtype="float32",
                chunks=(*shape, 3),
            )
            flow_arr[:] = sampled


def _sample_field(shape: tuple[int, int, int], field: DisplacementFunction) -> np.ndarray:
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)
    disp = field(coords)
    return disp.reshape((*shape, 3)).astype(np.float32)


@pytest.fixture
def synthetic_entry() -> _SyntheticEntry:
    """A default synthetic entry: small rigid shift named ``fs004``."""
    return _SyntheticEntry(name="fs004", field=rigid_shift((0.5, -0.25, 1.0)))


def _default_synthetic_entries() -> tuple[_SyntheticEntry, ...]:
    return (_SyntheticEntry(name="fs004", field=rigid_shift((0.5, -0.25, 1.0))),)


@pytest.fixture
def make_store():
    """Factory: build an in-memory zarr store conforming to ``profile``.

    Returns the root :class:`zarr.Group`.
    """

    def _factory(
        *,
        profile: StoreProfile,
        shape: tuple[int, int, int] = (8, 16, 16),
        masks: Sequence[str] = ("mask",),
        real_entries: Sequence[str] = ("016",),
        synthetic_entries: Sequence[_SyntheticEntry] | None = None,
        seed: int = 0,
    ) -> zarr.Group:
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store)
        _populate_store(
            root,
            profile=profile,
            shape=shape,
            masks=masks,
            real_entries=real_entries,
            synthetic_entries=(
                synthetic_entries
                if synthetic_entries is not None
                else _default_synthetic_entries()
            ),
            seed=seed,
        )
        return root

    return _factory


@pytest.fixture
def make_disk_store(tmp_path: Path):
    """Factory: build a zarr store on disk under ``tmp_path``.

    Returns the path to the on-disk store. Used by tests that exercise
    ``DvcDataset.open(path)``.
    """

    def _factory(
        *,
        profile: StoreProfile,
        shape: tuple[int, int, int] = (8, 16, 16),
        masks: Sequence[str] = ("mask",),
        real_entries: Sequence[str] = ("016",),
        synthetic_entries: Sequence[_SyntheticEntry] | None = None,
        seed: int = 0,
        name: str = "scan",
    ) -> Path:
        store_path = tmp_path / f"{name}.zarr"
        root = zarr.open_group(str(store_path), mode="w")
        _populate_store(
            root,
            profile=profile,
            shape=shape,
            masks=masks,
            real_entries=real_entries,
            synthetic_entries=(
                synthetic_entries
                if synthetic_entries is not None
                else _default_synthetic_entries()
            ),
            seed=seed,
        )
        return store_path

    return _factory
