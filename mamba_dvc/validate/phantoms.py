"""Bone-like geometric phantoms for DVC visualization showcases.

Produces a multi-region float32 volume that gives the eye visible
landmarks (cortical shell silhouette, perpendicular implant cylinder,
asymmetric defect cavity) while preserving the high-frequency texture
the FFT NCC pipeline needs to lock onto sub-voxel displacements.

Composition
-----------
1. **Background** -- low-intensity matrix.
2. **Trabecular core** -- noise-thresholded porous structure inside the
   cortical interior, mimicking cancellous bone.
3. **Cortical shell** -- thick cylindrical wall along the bone axis.
4. **Implant** -- high-contrast cylinder along an axis perpendicular to
   the bone, mimicking the screw.
5. **Defect** (optional) -- spherical cavity carved out of the bone to
   break symmetry.
6. **Multiplicative texture** -- band-limited noise modulating the
   piecewise-constant scaffold so every POI window has a unique
   speckle pattern.
7. **Final smoothing** -- small Gaussian to soften region boundaries
   and prevent Gibbs ringing under cubic resampling in
   :func:`mamba_dvc.validate.synthetic.warp`.

The output is host-side numpy float32; no CuPy. Intended consumers are
the showcase notebook and visualization smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float32
from scipy.ndimage import gaussian_filter

from mamba_dvc.validate.synthetic import make_texture

__all__ = ["PhantomAxis", "PhantomSpec", "default_phantom", "render_phantom"]


PhantomAxis = Literal["z", "y", "x"]


@dataclass(frozen=True)
class PhantomSpec:
    """Declarative description of a bone-like phantom.

    All length-style parameters are in voxels. Optional ``*_center``
    fields default to the volume center; optional radii default to a
    fraction of the smaller in-plane extent. Resolution happens inside
    :func:`render_phantom` so a caller can construct a ``PhantomSpec``
    with shape only and let everything else be derived.

    Parameters
    ----------
    shape
        ``(z, y, x)`` voxel shape.
    bone_axis
        Axis the cortical shaft runs along. Defaults to ``"z"``.
    bone_center
        ``(z, y, x)`` center of the bone shaft. ``None`` means the
        volume center.
    cortical_outer_radius
        Outer radius of the cortical shell (in-plane). ``None`` means
        ``min(in_plane_extent) / 2 - 6`` voxels.
    cortical_thickness
        Cortical wall thickness.
    cortical_intensity
        Scaffold intensity in the cortical region.
    implant_axis
        Axis the implant cylinder runs along. Defaults to ``"x"``
        (perpendicular to a z-axis bone).
    implant_center
        ``(z, y, x)`` center of the implant. ``None`` means the volume
        center.
    implant_radius
        Cylinder radius for the implant.
    implant_intensity
        Scaffold intensity for the implant.
    trabecular_intensity
        Scaffold intensity for trabecular struts.
    trabecular_scale
        Gaussian sigma (voxels) used to smooth the trabecular noise
        field. Larger values → coarser struts.
    trabecular_porosity
        Fraction of the cortical interior that should be marrow
        (zero / background) rather than bone. ``0.55`` corresponds to
        a 45% bone-volume-fraction trabecular mesh.
    defect_center
        ``(z, y, x)`` center of the optional defect cavity. ``None``
        with ``defect_radius > 0`` defaults to an offset position so
        the defect breaks symmetry.
    defect_radius
        Defect cavity radius. ``0.0`` disables the defect.
    background_intensity
        Scaffold intensity outside any bone / implant region.
    texture_sigma
        Gaussian sigma for the band-limited texture noise. Forwarded
        to :func:`mamba_dvc.validate.synthetic.make_texture`.
    texture_strength
        Multiplicative texture amplitude. The textured scaffold is
        ``scaffold * (1 + texture_strength * texture)``; values around
        ``0.25`` keep the scaffold dominant for the eye while still
        giving NCC enough local variance.
    smoothing_sigma
        Final Gaussian smoothing applied to the textured volume.
    seed
        RNG seed. The texture and trabecular noise use independent
        offsets derived from this seed so changing one component does
        not require regenerating the other.
    """

    shape: tuple[int, int, int]
    bone_axis: PhantomAxis = "z"
    bone_center: tuple[float, float, float] | None = None
    cortical_outer_radius: float | None = None
    cortical_thickness: float = 6.0
    cortical_intensity: float = 0.9
    implant_axis: PhantomAxis = "x"
    implant_center: tuple[float, float, float] | None = None
    implant_radius: float = 4.0
    implant_intensity: float = 1.2
    trabecular_intensity: float = 0.55
    trabecular_scale: float = 4.0
    trabecular_porosity: float = 0.55
    defect_center: tuple[float, float, float] | None = None
    defect_radius: float = 0.0
    background_intensity: float = 0.05
    texture_sigma: float = 1.5
    texture_strength: float = 0.25
    smoothing_sigma: float = 0.6
    seed: int = 0


_AXIS_INDEX: dict[PhantomAxis, int] = {"z": 0, "y": 1, "x": 2}


def _coord_grids(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return broadcasting ``(z, y, x)`` coordinate grids."""
    nz, ny, nx = shape
    zz = np.arange(nz, dtype=np.float32).reshape(nz, 1, 1)
    yy = np.arange(ny, dtype=np.float32).reshape(1, ny, 1)
    xx = np.arange(nx, dtype=np.float32).reshape(1, 1, nx)
    return zz, yy, xx


def _radial_distance_sq(
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    center: tuple[float, float, float],
    along_axis: PhantomAxis | None,
) -> np.ndarray:
    """Squared distance from a line (cylinder) or point (sphere).

    Returns a fully broadcast ``(z, y, x)`` array so callers can use
    the result directly as a boolean index into a 3D scaffold without
    having to materialize the broadcast themselves.
    """
    zz, yy, xx = grids
    cz, cy, cx = center
    dz2 = (zz - cz) ** 2
    dy2 = (yy - cy) ** 2
    dx2 = (xx - cx) ** 2
    if along_axis is None:
        r2 = dz2 + dy2 + dx2
    else:
        ax = _AXIS_INDEX[along_axis]
        if ax == 0:
            r2 = dy2 + dx2
        elif ax == 1:
            r2 = dz2 + dx2
        else:
            r2 = dz2 + dy2
    full_shape = (zz.shape[0], yy.shape[1], xx.shape[2])
    return np.broadcast_to(r2, full_shape)


def _cylinder_mask(
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    center: tuple[float, float, float],
    radius: float,
    axis: PhantomAxis,
) -> Bool[np.ndarray, "z y x"]:
    """Boolean mask of a cylinder of ``radius`` along ``axis``."""
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    r2 = _radial_distance_sq(grids, center, axis)
    return r2 <= radius * radius


def _cylindrical_shell_mask(
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    center: tuple[float, float, float],
    inner_radius: float,
    outer_radius: float,
    axis: PhantomAxis,
) -> Bool[np.ndarray, "z y x"]:
    """Boolean mask of an annular cylindrical shell."""
    if inner_radius < 0.0:
        raise ValueError(f"inner_radius must be non-negative, got {inner_radius}")
    if outer_radius <= inner_radius:
        raise ValueError(
            f"outer_radius {outer_radius} must exceed inner_radius {inner_radius}"
        )
    r2 = _radial_distance_sq(grids, center, axis)
    return (r2 >= inner_radius * inner_radius) & (r2 <= outer_radius * outer_radius)


def _sphere_mask(
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    center: tuple[float, float, float],
    radius: float,
) -> Bool[np.ndarray, "z y x"]:
    """Boolean mask of a sphere."""
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    r2 = _radial_distance_sq(grids, center, along_axis=None)
    return r2 <= radius * radius


def _trabecular_mask(
    shape: tuple[int, int, int],
    *,
    scale: float,
    porosity: float,
    seed: int,
) -> Bool[np.ndarray, "z y x"]:
    """Generate a porous trabecular bone-volume mask via thresholded noise.

    The output ``True`` voxels are bone struts; ``False`` are marrow.
    Bone-volume fraction is ``1 - porosity``.
    """
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    if not 0.0 < porosity < 1.0:
        raise ValueError(f"porosity must be in (0, 1), got {porosity}")

    rng = np.random.default_rng(seed)
    white = rng.standard_normal(shape, dtype=np.float32)
    smoothed = gaussian_filter(white, sigma=scale, mode="reflect")
    threshold = float(np.quantile(smoothed, porosity))
    return smoothed > threshold


def _resolve_center(
    center: tuple[float, float, float] | None,
    shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    if center is not None:
        return center
    nz, ny, nx = shape
    return (nz / 2.0, ny / 2.0, nx / 2.0)


def _resolve_cortical_outer_radius(
    outer_radius: float | None,
    shape: tuple[int, int, int],
    bone_axis: PhantomAxis,
) -> float:
    if outer_radius is not None:
        return outer_radius
    ax = _AXIS_INDEX[bone_axis]
    in_plane = [shape[i] for i in range(3) if i != ax]
    return float(min(in_plane)) / 2.0 - 6.0


def _resolve_defect_center(
    center: tuple[float, float, float] | None,
    shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    if center is not None:
        return center
    nz, ny, nx = shape
    # Off-center so the defect breaks rotational and translational symmetry.
    return (nz / 2.0 + nz / 8.0, ny / 2.0 - ny / 8.0, nx / 2.0 + nx / 8.0)


def _build_scaffold(spec: PhantomSpec) -> Float32[np.ndarray, "z y x"]:
    """Compose the piecewise-constant intensity scaffold."""
    grids = _coord_grids(spec.shape)
    bone_center = _resolve_center(spec.bone_center, spec.shape)
    implant_center = _resolve_center(spec.implant_center, spec.shape)
    cortical_outer = _resolve_cortical_outer_radius(
        spec.cortical_outer_radius, spec.shape, spec.bone_axis
    )
    cortical_inner = cortical_outer - spec.cortical_thickness
    if cortical_inner <= 0.0:
        raise ValueError(
            f"cortical_thickness {spec.cortical_thickness} too large for "
            f"outer radius {cortical_outer}"
        )

    cortical_mask = _cylindrical_shell_mask(
        grids, bone_center, cortical_inner, cortical_outer, spec.bone_axis
    )
    interior_mask = _cylinder_mask(grids, bone_center, cortical_inner, spec.bone_axis)
    implant_mask = _cylinder_mask(
        grids, implant_center, spec.implant_radius, spec.implant_axis
    )
    trabecular_struts = _trabecular_mask(
        spec.shape,
        scale=spec.trabecular_scale,
        porosity=spec.trabecular_porosity,
        seed=spec.seed + 1,
    )
    trabecular_mask = interior_mask & trabecular_struts & ~implant_mask

    scaffold = np.full(spec.shape, spec.background_intensity, dtype=np.float32)
    scaffold[trabecular_mask] = spec.trabecular_intensity
    scaffold[cortical_mask] = spec.cortical_intensity
    scaffold[implant_mask] = spec.implant_intensity

    if spec.defect_radius > 0.0:
        defect_center = _resolve_defect_center(spec.defect_center, spec.shape)
        defect_mask = _sphere_mask(grids, defect_center, spec.defect_radius)
        # Carve cavity but preserve the implant -- a real screw drilling
        # into a defect would still appear in the X-ray.
        carve = defect_mask & ~implant_mask
        scaffold[carve] = spec.background_intensity

    return scaffold


def render_phantom(spec: PhantomSpec) -> Float32[np.ndarray, "z y x"]:
    """Render a textured bone-like phantom from ``spec``.

    Parameters
    ----------
    spec
        Phantom description; see :class:`PhantomSpec`.

    Returns
    -------
    numpy.ndarray
        ``(z, y, x)`` float32 volume. Mean intensity is dominated by
        ``background_intensity`` away from any bone region; the
        cortical shell, trabecular struts, and implant ring up to
        their configured intensities, with multiplicative texture and
        a final Gaussian smoothing applied on top.

    Raises
    ------
    ValueError
        For inconsistent geometry (e.g. cortical thickness exceeding
        the outer radius) or out-of-range parameters.
    """
    if len(spec.shape) != 3:
        raise ValueError(f"shape must have length 3, got {len(spec.shape)}")
    if any(s <= 0 for s in spec.shape):
        raise ValueError(f"shape entries must be positive, got {spec.shape}")

    scaffold = _build_scaffold(spec)
    texture = make_texture(spec.shape, sigma=spec.texture_sigma, seed=spec.seed + 2)
    textured = scaffold * (1.0 + spec.texture_strength * texture)

    if spec.smoothing_sigma > 0.0:
        smoothed = gaussian_filter(textured, sigma=spec.smoothing_sigma, mode="reflect")
    else:
        smoothed = textured
    return smoothed.astype(np.float32, copy=False)


def default_phantom(
    shape: tuple[int, int, int],
    *,
    seed: int = 0,
) -> Float32[np.ndarray, "z y x"]:
    """Render a phantom with sensible defaults for ``shape``.

    Convenience wrapper used by the showcase notebook. Builds a
    cylindrical cortical shell along ``z``, a perpendicular implant
    along ``x``, a porous trabecular mesh inside the shell, and an
    off-center defect cavity. Use :func:`render_phantom` directly when
    you need to override individual knobs.

    Parameters
    ----------
    shape
        ``(z, y, x)`` voxel shape.
    seed
        RNG seed forwarded to :class:`PhantomSpec`.

    Returns
    -------
    numpy.ndarray
        ``(z, y, x)`` float32 phantom volume.
    """
    _nz, ny, nx = shape
    in_plane = float(min(ny, nx))
    spec = PhantomSpec(
        shape=shape,
        defect_radius=in_plane / 12.0,
        seed=seed,
    )
    return render_phantom(spec)
