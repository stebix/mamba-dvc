# mamba-dvc

FFT-based digital volume correlation (DVC) for synchrotron µCT of rat bone with
screw implant. Target volumes are `(960, 1280, 1280)` float32; compute runs on
4× NVIDIA A6000 via CuPy.

The architecture and algorithmic decisions are documented in
[`docs/plans/overview.md`](docs/plans/overview.md). The zarr reader contract —
profile / manifest schema, verifier rules, ground-truth field handling — lives
in [`docs/plans/zarr-interface.md`](docs/plans/zarr-interface.md).

## Install

Python 3.14 is required (CuPy 14 / CUDA 13.x). Dependencies are managed with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Inspecting a zarr store: `mamba-dvc inspect`

```bash
uv run mamba-dvc inspect /path/to/scan.zarr
```

Opens the store, resolves its `StoreProfile` (and any sidecar
`StoreManifest`), runs the verifier, and prints:

1. A header listing the resolved profile and the manifest source
   (sidecar YAML, root `.zattrs`, `--manifest` flag, `--profile` flag, or
   `none`).
2. A validity panel — green `OK` on a clean store, red `FAILED` with one
   bullet per verifier error otherwise.
3. A tree of every parsed slot: reference (`[reference]`), masks
   (`[default]` tag on the resolved default), real-deformation entries
   under `iterations/`, and synthetic entries under `synthetic/` with
   their flow arrays tagged by axis order and convention.

The store is opened with `strict=False`, so a non-conforming store still
renders the tree of whatever could be parsed alongside the error list.

### Common invocations

```bash
# Add chunks, codecs, and on-disk size per array leaf.
uv run mamba-dvc inspect /path/to/scan.zarr --verbose

# Machine-readable inventory + verifier report (skips rich rendering).
uv run mamba-dvc inspect /path/to/scan.zarr --json

# Use an explicit manifest YAML; bypass sidecar / .zattrs discovery.
uv run mamba-dvc inspect /path/to/scan.zarr --manifest /path/to/manifest.yaml

# Force a specific registered profile (synthesizes / merges a manifest).
uv run mamba-dvc inspect /path/to/scan.zarr --profile bone_screw_synchrotron_v1
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Verifier passed; store conforms to its profile + manifest. |
| `1` | Verifier reported errors. The tree of what was parsed is still rendered to stdout, and `--json` payloads carry `validity.ok = false`. |
| `2` | Could not open: missing path, malformed manifest YAML, or unknown profile name. The error message goes to stderr. |

The `1` path is the load-bearing one for scripting — CI can gate uploads on
`mamba-dvc inspect --json <path>` returning `validity.ok == true`.

## Development

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pyright
uv run pytest -m "not gpu"   # skip GPU-only tests on a CPU-only machine
uv run pre-commit run --all-files
```

See [`CLAUDE.md`](CLAUDE.md) for the full contributor guide.
