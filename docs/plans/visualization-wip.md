# Visualization submodule — WIP status

Companion to `visualization.md`. Tracks which steps from §9 are landed
and what is intentionally deferred. Update with each viz pass.

## Current checkpoint: end of step 4 (foundation landed)

Date: 2026-04-30.

Goal of this checkpoint: a notebook-driven sanity check where the user
renders reference, deformed, and recovered displacement field on a
synthetic pair, and compares against the analytical ground-truth field.
Ground-truth field rendering and diagnostics overlays come in the next
pass (steps 5–6).

## Landed

| Step (§9) | Item | Path |
|---|---|---|
| 1 | `viz` extra (folds in former `reactive-viz`) | `pyproject.toml` |
| 1 | `viz` pytest marker | `pyproject.toml` |
| 2 | `VoxelSpacing` + `PhysicalUnit` | `mamba_dvc/types.py` |
| 3 | `_conversion.py` + unit math (`unify_spacing`) | `mamba_dvc/viz/_conversion.py` |
| 3 | `_conversion` pure-Python tests | `tests/viz/test_conversion_units.py`, `tests/viz/test_types_spacing.py` |
| 3 | `_conversion` VTK-bridge structural tests | `tests/viz/test_conversion_vtk.py` |
| 4 | `backend.plotter` + `screenshot` | `mamba_dvc/viz/backend.py` |
| 4 | `volume.render_volume` + `render_isosurface` | `mamba_dvc/viz/volume.py` |
| 4 | Backend + volume smoke tests | `tests/viz/test_backend.py`, `tests/viz/test_volume.py` |

Test guard pattern: each VTK-touching test module begins with
`pv = pytest.importorskip("pyvista")` and a module-level
`pytestmark = pytest.mark.viz`. On environments without the `viz` extra
the tests are silently skipped at collection time. No image diffing —
only structural assertions (actor count, array names, spacing, dtype).

## Deferred

### `ValidationReport` and `viz/validate.py`

`docs/plans/visualization.md` §6.2 introduces a `ValidationReport`
dataclass. Decision (2026-04-30): defer until `validate/known_fields.py`
lands as the producer. Defining a carrier without a producer risks
drifting from the comparator's actual output shape. Tracking note —
when this lands:

1. Add `ValidationReport` to `mamba_dvc/types.py` and re-export.
2. Implement `validate/known_fields.py` to populate it.
3. Implement `viz/validate.py` (`render_error_glyphs`,
   `render_error_histogram`, `render_error_vs_mask_distance`).

### `viz/field.py`, `viz/diagnostics.py`, `viz/slices.py`

Steps 5–7 in §9. Land in the next pass once the human review on the
volume-only checkpoint confirms the backend + spacing wiring.
`render_streamlines` will ship as a `NotImplementedError` stub keyed to
the `densify` v2 hook (§6.1).

### `viz/export.py`

Step 10. The `screenshot` helper already lives in `backend.py`;
animation export waits until field renderers exist.

### Image-diff CI tests

§8 mentions reference PNGs and image diffing. Per the 2026-04-30
decision, structural smoke tests cover the foundation; pixel
comparisons are out of scope until a stable headless render
configuration is chosen. Re-evaluate after step 6.

### Qt backend (`backend.plotter(qt=True)`)

§9 step 12. Wired into the API today (so renderers don't change when
the path is enabled), but no real-volume measurement has been done.
Re-evaluate only if the trame backend stutters interactively on the
full `(960, 1280, 1280)` volume.

## Open seams

- `io/volume.py` (not landed) is the future producer of
  `VoxelSpacing`. Until then, callers either pass `None` (treated as
  voxel units) or build a `VoxelSpacing` by hand.
- `densify` (v2 from `overview.md` §10) blocks `render_streamlines`
  and any per-voxel field coloring.

## Non-negotiables (recap from `visualization.md` §7)

- No new types in `viz/`.
- Renderers return the `Plotter` / `Figure`; never call `.show()`.
- `viz` is optional; `core/`, `pipeline/`, `gpu/`, `io/` may not import
  from it.
- Pure-host arrays in.
- Unit math lives only in `_conversion.unify_spacing`.
- Status colors are configurable (lands with `diagnostics.py`).
