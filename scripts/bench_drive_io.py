r"""Head-to-head, cache-honest read-bandwidth benchmark for two+ drives.

Why this exists
---------------
``scripts/bench_io.py`` measures one drive at a time and — as its own
docstring admits — cannot defeat the Windows file cache from userspace,
so its "cold" read after a fresh write usually still hits RAM. That makes
it useless for the question we actually care about: *is reading a volume
from ``D:`` genuinely slower off the platter than from ``C:``, or is it
just cache warmth?*

This script answers that by reading the test file back through a Win32
handle opened with ``FILE_FLAG_NO_BUFFERING``, which bypasses the cache
manager entirely and forces every byte to come off the physical device.
That is the *reliable filesystem-to-main-memory throughput* number. It
runs the same synthetic volume on every requested drive and prints a
side-by-side summary.

For reference it also does one ordinary buffered read per drive
(``np.fromfile``), which — because the data was just written — is served
from RAM and therefore reports roughly the cache/copy ceiling, not the
disk. The gap between the two columns is the cache's contribution.

Run with::

    uv run python scripts/bench_drive_io.py                       # %TEMP% (C:) vs D:/tmp
    uv run python scripts/bench_drive_io.py --size-gb 16 --repeats 5
    uv run python scripts/bench_drive_io.py --paths D:/tmp G:/scratch

Notes
-----
* Unbuffered reads are immune to RAM size, so ``--size-gb`` only needs to
  be large enough to average out spin-up / queue-warm transients; 8 GB is
  plenty. The buffered reference read *will* allocate that much host RAM.
* The realistic pipeline path (Zarr v3, possibly compressed) is *not*
  measured here — point ``scripts/bench_io.py --backend zarr --tmpdir``
  at each drive for that.
* Windows only: the no-buffering trick has no portable POSIX twin worth
  the maintenance here.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import statistics
import sys
import tempfile
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Win32 unbuffered I/O plumbing
# --------------------------------------------------------------------------- #
_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_OPEN_EXISTING = 3
_FILE_FLAG_NO_BUFFERING = 0x20000000
_FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

if sys.platform == "win32":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _CreateFileW = _kernel32.CreateFileW
    _CreateFileW.restype = wintypes.HANDLE
    _CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )

    _ReadFile = _kernel32.ReadFile
    _ReadFile.restype = wintypes.BOOL
    _ReadFile.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    )

    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.restype = wintypes.BOOL
    _CloseHandle.argtypes = (wintypes.HANDLE,)

    _GetDiskFreeSpaceW = _kernel32.GetDiskFreeSpaceW
    _GetDiskFreeSpaceW.restype = wintypes.BOOL
    _GetDiskFreeSpaceW.argtypes = (
        wintypes.LPCWSTR,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
    )


def _sector_size(any_path: Path) -> int:
    """Return the volume's logical sector size for ``any_path`` (default 4096).

    ``FILE_FLAG_NO_BUFFERING`` requires both the user buffer and the read
    length to be a multiple of this; we additionally floor it at 4096 so
    Advanced-Format drives reporting 512 logical bytes still satisfy the
    physical-sector constraint.
    """
    root = os.path.splitdrive(str(any_path.resolve()))[0] + "\\"
    spc = wintypes.DWORD(0)
    bps = wintypes.DWORD(0)
    free = wintypes.DWORD(0)
    total = wintypes.DWORD(0)
    ok = _GetDiskFreeSpaceW(
        root, ctypes.byref(spc), ctypes.byref(bps), ctypes.byref(free), ctypes.byref(total)
    )
    if not ok or bps.value == 0:
        return 4096
    return max(int(bps.value), 4096)


def _read_unbuffered(path: Path, *, chunk_bytes: int, align: int) -> int:
    """Read ``path`` start-to-finish bypassing the OS cache; return bytes read.

    Reads in ``chunk_bytes`` blocks (a multiple of ``align``) into a single
    re-used, ``align``-aligned host buffer, so peak host RAM is one chunk
    regardless of file size.

    Raises
    ------
    OSError
        On any ``CreateFileW`` / ``ReadFile`` failure.
    ValueError
        If the file size is not a multiple of ``align`` (cannot be served
        through a no-buffering handle).
    """
    size = path.stat().st_size
    if size % align != 0:
        raise ValueError(
            f"{path} is {size} B, not a multiple of the {align} B sector size; "
            "unbuffered read cannot handle the trailing partial sector"
        )
    if chunk_bytes % align != 0:
        raise ValueError(f"chunk size {chunk_bytes} is not a multiple of {align}")

    # Over-allocate by one alignment unit and slide to the first aligned byte.
    raw = np.empty(chunk_bytes + align, dtype=np.uint8)
    base_addr = raw.ctypes.data
    pad = (-base_addr) % align
    buf_ptr = ctypes.c_void_p(base_addr + pad)

    handle = _CreateFileW(
        str(path),
        _GENERIC_READ,
        _FILE_SHARE_READ,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_SEQUENTIAL_SCAN,
        None,
    )
    if not handle or handle == _INVALID_HANDLE_VALUE:
        raise OSError(ctypes.get_last_error(), f"CreateFileW failed for {path}")

    nread = wintypes.DWORD(0)
    total = 0
    try:
        remaining = size
        while remaining > 0:
            want = chunk_bytes if remaining >= chunk_bytes else remaining
            ok = _ReadFile(handle, buf_ptr, want, ctypes.byref(nread), None)
            if not ok:
                raise OSError(ctypes.get_last_error(), f"ReadFile failed on {path}")
            got = nread.value
            if got == 0:
                break
            total += got
            remaining -= got
    finally:
        _CloseHandle(handle)
    # Keep ``raw`` referenced until the handle is closed.
    del raw
    return total


# --------------------------------------------------------------------------- #
# Benchmark core
# --------------------------------------------------------------------------- #
@dataclass
class DriveResult:
    """Timings collected for one benchmarked directory."""

    label: str
    path: Path
    nbytes: int
    write_s: float
    unbuffered_s: list[float] = field(default_factory=list)
    buffered_s: float | None = None

    def _bw(self, seconds: float) -> float:
        return self.nbytes / seconds / 1e9 if seconds > 0 else float("inf")

    @property
    def unbuffered_bw(self) -> list[float]:
        """Per-repeat unbuffered read bandwidths in GB/s (decimal)."""
        return [self._bw(s) for s in self.unbuffered_s]

    @property
    def write_bw(self) -> float:
        """Write+fsync bandwidth in GB/s (decimal)."""
        return self._bw(self.write_s)

    @property
    def buffered_bw(self) -> float | None:
        """Buffered (cache-served) read bandwidth in GB/s, if measured."""
        return None if self.buffered_s is None else self._bw(self.buffered_s)


def _shape_for(size_gb: float) -> tuple[int, int, int]:
    """Pick a ``(z, 1280, 1280)`` float32 shape near ``size_gb``.

    ``1280 * 1280 * 4 == 6_553_600 == 1600 * 4096`` bytes per z-slice, so the
    total file size is always a multiple of any plausible sector size — which
    keeps the unbuffered read path off its partial-sector edge case.
    """
    y = x = 1280
    z = max(1, round(size_gb * 1e9 / (y * x * 4)))
    return (z, y, x)


def _write_volume(path: Path, arr: np.ndarray) -> float:
    """Write ``arr`` to ``path`` and ``fsync`` it; return wall seconds.

    The ``fsync`` is deliberately inside the timed region: without it we
    would be timing a memcpy into the write-back cache, not the device, and
    the subsequent unbuffered read could even race ahead of the flush.
    """
    t0 = time.perf_counter()
    with open(path, "wb") as f:
        arr.tofile(f)
        f.flush()
        os.fsync(f.fileno())
    return time.perf_counter() - t0


def _read_buffered(path: Path, dtype: np.dtype, expected_size: int) -> float:
    """Ordinary ``np.fromfile`` read (cache-served here); return wall seconds."""
    t0 = time.perf_counter()
    out = np.fromfile(path, dtype=dtype)
    dt = time.perf_counter() - t0
    if out.size != expected_size:
        raise RuntimeError(f"short read on {path}: {out.size} != {expected_size}")
    del out
    return dt


def _bench_one(
    parent: Path,
    label: str,
    arr: np.ndarray,
    *,
    chunk_bytes: int,
    repeats: int,
    do_buffered: bool,
    keep: bool,
) -> DriveResult:
    """Run write + unbuffered/buffered reads for one parent directory."""
    tmpdir = Path(tempfile.mkdtemp(prefix="bench_drive_", dir=str(parent)))
    vol = tmpdir / "volume.bin"
    align = _sector_size(tmpdir)
    if chunk_bytes % align != 0:
        chunk_bytes = ((chunk_bytes + align - 1) // align) * align

    result = DriveResult(label=label, path=tmpdir, nbytes=arr.nbytes, write_s=0.0)
    try:
        print(f"[{label}] {tmpdir}  (sector {align} B, read chunk {chunk_bytes // 1024} KiB)")
        result.write_s = _write_volume(vol, arr)
        print(f"  write+fsync          {result.write_bw:7.3f} GB/s  ({result.write_s:6.2f} s)")

        for i in range(repeats):
            t0 = time.perf_counter()
            got = _read_unbuffered(vol, chunk_bytes=chunk_bytes, align=align)
            dt = time.perf_counter() - t0
            if got != arr.nbytes:
                raise RuntimeError(f"short unbuffered read: {got} != {arr.nbytes}")
            result.unbuffered_s.append(dt)
            bw = arr.nbytes / dt / 1e9 if dt > 0 else float("inf")
            print(f"  read unbuffered #{i + 1}   {bw:7.3f} GB/s  ({dt:6.2f} s)")

        if do_buffered:
            result.buffered_s = _read_buffered(vol, arr.dtype, arr.size)
            assert result.buffered_bw is not None
            print(
                f"  read buffered (warm) {result.buffered_bw:7.3f} GB/s  "
                f"({result.buffered_s:6.2f} s)  <- np.fromfile, cache-served (not disk)"
            )
    finally:
        if keep:
            print(f"  kept: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
    print()
    return result


def _check_writable(parent: Path) -> Path | None:
    """Return ``parent`` if it exists and accepts a probe file, else ``None``."""
    try:
        parent.mkdir(parents=True, exist_ok=True)
        probe = parent / f".bench_drive_probe_{os.getpid()}"
        probe.write_bytes(b"x")
        probe.unlink()
    except OSError as exc:  # report and skip, don't crash the whole run
        print(f"skip {parent}: not writable ({exc})", file=sys.stderr)
        return None
    return parent


def _label_for(path: Path) -> str:
    """Short label for a parent dir — the drive letter when there is one."""
    drive = os.path.splitdrive(str(path.resolve()))[0]
    return drive or str(path)


def _print_summary(results: list[DriveResult], chunk_bytes: int, repeats: int) -> None:
    """Print the side-by-side comparison table."""
    if not results:
        return
    size_gb = results[0].nbytes / 1e9
    print(
        f"Summary  (size {size_gb:.2f} GB, read chunk {chunk_bytes // 1024} KiB, "
        f"{repeats} unbuffered repeats; GB/s = decimal 1e9 B/s)"
    )
    header = (
        f"  {'drive':<8}{'fs->RAM min':>13}{'fs->RAM med':>13}"
        f"{'fs->RAM max':>13}{'write':>10}{'cached read':>14}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        bws = sorted(r.unbuffered_bw)
        lo = bws[0]
        hi = bws[-1]
        med = statistics.median(bws)
        cached = "    n/a" if r.buffered_bw is None else f"{r.buffered_bw:7.3f}"
        print(
            f"  {r.label:<8}{lo:>13.3f}{med:>13.3f}{hi:>13.3f}{r.write_bw:>10.3f}{cached:>14}"
        )
    if len(results) >= 2:
        fastest = max(results, key=lambda r: statistics.median(r.unbuffered_bw))
        slowest = min(results, key=lambda r: statistics.median(r.unbuffered_bw))
        if fastest.label != slowest.label:
            ratio = statistics.median(fastest.unbuffered_bw) / statistics.median(
                slowest.unbuffered_bw
            )
            print(
                f"\n  {fastest.label} is {ratio:.2f}x faster than {slowest.label} "
                "on the unbuffered (true filesystem) read."
            )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the head-to-head benchmark."""
    if sys.platform != "win32":
        print("bench_drive_io.py needs Windows (FILE_FLAG_NO_BUFFERING).", file=sys.stderr)
        return 2

    default_paths = [Path(tempfile.gettempdir()), Path("D:/tmp")]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--paths",
        type=Path,
        nargs="+",
        default=default_paths,
        help="parent directories to benchmark, one per drive (default: %(default)s)",
    )
    parser.add_argument(
        "--size-gb", type=float, default=8.0, help="synthetic volume size in GB (default 8.0)"
    )
    parser.add_argument(
        "--read-mb",
        type=int,
        default=8,
        help="unbuffered read chunk size in MiB (default 8; rounded up to the sector size)",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="unbuffered read repeats per drive (default 3)"
    )
    parser.add_argument(
        "--no-buffered",
        dest="buffered",
        action="store_false",
        help="skip the buffered (cache-served) reference read",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="rng seed for the synthetic volume (default 0)"
    )
    parser.add_argument(
        "--keep", action="store_true", help="do not delete the temp dirs on exit"
    )
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.read_mb < 1:
        parser.error("--read-mb must be >= 1")

    parents: list[Path] = []
    for p in args.paths:
        ok = _check_writable(p)
        if ok is not None:
            parents.append(ok)
    if not parents:
        print("no writable parent directories — nothing to do", file=sys.stderr)
        return 1

    shape = _shape_for(args.size_gb)
    chunk_bytes = args.read_mb * 1024 * 1024
    print(f"volume shape : {shape}  ({np.prod(shape) * 4 / 1e9:.2f} GB float32)")
    print(f"drives       : {', '.join(_label_for(p) for p in parents)}")
    print(f"repeats      : {args.repeats}   buffered reference: {args.buffered}\n")

    rng = np.random.default_rng(args.seed)
    arr = rng.standard_normal(shape, dtype=np.float32)

    results: list[DriveResult] = []
    for parent in parents:
        results.append(
            _bench_one(
                parent,
                _label_for(parent),
                arr,
                chunk_bytes=chunk_bytes,
                repeats=args.repeats,
                do_buffered=args.buffered,
                keep=args.keep,
            )
        )

    _print_summary(results, chunk_bytes, args.repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
