r"""Ephemeral SSD→host read-bandwidth benchmark for raw and Zarr layouts.

Writes a temporary float32 volume into a :class:`tempfile.TemporaryDirectory`
on the chosen drive, then times one cold and one warm read for each
selected backend. The temp dir is removed on exit unless ``--keep`` is
passed.

Run with::

    uv run python scripts/bench_io.py --size-gb 2.0
    uv run python scripts/bench_io.py --size-gb 8.0 --backend zarr \\
        --chunks 64,256,256 --tmpdir D:/tmp

Cache caveat (Windows)
----------------------
The OS file cache cannot be flushed from userspace without admin rights,
so the "cold" read after writing a small file usually still hits RAM.
For an honest SSD-bound number, set ``--size-gb`` larger than your free
RAM (check with Task Manager) so the tail of the file must be re-read
from disk.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import zarr


def _fmt_bw(num_bytes: int, seconds: float) -> str:
    """Return a ``GB/s`` bandwidth string (decimal GB, matches vendor specs)."""
    if seconds <= 0.0:
        return "   inf GB/s"
    gbps = num_bytes / seconds / 1e9
    return f"{gbps:7.3f} GB/s"


def _fmt_size(num_bytes: int) -> str:
    """Return a human-friendly size string."""
    gb = num_bytes / 1e9
    return f"{gb:.2f} GB"


@contextmanager
def _timed(label: str, num_bytes: int) -> Iterator[None]:
    """Print a single ``label  bandwidth  (elapsed)`` line."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"  {label:<20s} {_fmt_bw(num_bytes, dt)}  ({dt:6.2f} s)")


def _shape_for(size_gb: float) -> tuple[int, int, int]:
    """Pick a (z, y, x) float32 shape with ~``size_gb`` total bytes.

    Uses the project's nominal y/x = 1280 and scales z to hit the target.
    """
    bytes_per = 4
    target_voxels = size_gb * 1e9 / bytes_per
    y, x = 1280, 1280
    z = max(1, round(target_voxels / (y * x)))
    return (z, y, x)


def _bench_raw(tmpdir: Path, arr: np.ndarray) -> None:
    """Benchmark a flat binary read via ``np.fromfile``."""
    path = tmpdir / "volume.bin"
    nbytes = arr.nbytes

    print("raw .bin")
    with _timed("write", nbytes):
        arr.tofile(path)

    for label in ("read (cold)", "read (warm)"):
        with _timed(label, nbytes):
            out = np.fromfile(path, dtype=arr.dtype)
        if out.size != arr.size:
            raise RuntimeError(f"short read: got {out.size}, expected {arr.size}")
        del out


def _bench_zarr(tmpdir: Path, arr: np.ndarray, chunks: tuple[int, int, int]) -> None:
    """Benchmark a Zarr v3 round-trip with explicit chunking."""
    path = tmpdir / "volume.zarr"
    nbytes = arr.nbytes

    print(f"zarr v3  chunks={chunks}")
    with _timed("write", nbytes):
        z = zarr.create_array(
            store=str(path),
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=chunks,
            overwrite=True,
        )
        z[:] = arr

    for label in ("read (cold)", "read (warm)"):
        with _timed(label, nbytes):
            z = zarr.open_array(str(path), mode="r")
            out = z[:]
        if out.shape != arr.shape:
            raise RuntimeError(f"shape mismatch: {out.shape} vs {arr.shape}")
        del out


def _parse_chunks(s: str) -> tuple[int, int, int]:
    """Parse a ``z,y,x`` chunk spec."""
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("chunks must be 'z,y,x'")
    return (parts[0], parts[1], parts[2])


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark with the parsed CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--size-gb", type=float, default=2.0, help="target volume size in GB (default 2.0)"
    )
    parser.add_argument(
        "--backend",
        choices=("raw", "zarr", "both"),
        default="both",
        help="which I/O path to benchmark (default both)",
    )
    parser.add_argument(
        "--chunks",
        type=_parse_chunks,
        default=(64, 256, 256),
        help="zarr chunk shape z,y,x (default 64,256,256 ~= 16 MB float32)",
    )
    parser.add_argument(
        "--tmpdir",
        type=Path,
        default=None,
        help=(
            "parent dir for the temp store (defaults to system tmp;"
            " pick the drive you want to benchmark)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="rng seed for the synthetic volume"
    )
    parser.add_argument(
        "--keep", action="store_true", help="do not delete the temp dir on exit"
    )
    args = parser.parse_args(argv)

    shape = _shape_for(args.size_gb)
    nbytes = math.prod(shape) * 4

    parent = args.tmpdir
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)

    tmpdir = Path(tempfile.mkdtemp(prefix="bench_io_", dir=str(parent) if parent else None))
    try:
        drive = tmpdir.drive or str(tmpdir)
        print(f"tmpdir   : {tmpdir}")
        print(f"drive    : {drive}")
        print(f"shape    : {shape}  ({_fmt_size(nbytes)} float32)")
        print()

        rng = np.random.default_rng(args.seed)
        # Generate in float32 directly to avoid an extra 2x peak in host RAM.
        arr = rng.standard_normal(shape, dtype=np.float32)

        if args.backend in ("raw", "both"):
            _bench_raw(tmpdir, arr)
            print()
        if args.backend in ("zarr", "both"):
            _bench_zarr(tmpdir, arr, args.chunks)
            print()
    finally:
        if args.keep:
            print(f"kept     : {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
