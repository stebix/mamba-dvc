"""Shared fixtures for ``tests/cli``.

Re-exports the disk-store fixture from ``tests/io/conftest`` so the
CLI tests can reuse the existing ``bone_screw_synchrotron_v1`` builder
without duplicating it.
"""

from __future__ import annotations

from tests.io.conftest import (
    make_disk_store,  # noqa: F401  (re-exported as a pytest fixture)
    make_store,  # noqa: F401
    synthetic_entry,  # noqa: F401
)
