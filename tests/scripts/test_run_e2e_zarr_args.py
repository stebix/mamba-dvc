"""Unit tests for ``scripts/run_e2e_zarr.py`` argument parsing.

The script is a manual perf harness, not part of the public API. Only
the new ``--batch-size`` parser is covered here -- the rest of the
script is exercised by the manual e2e workflow.
"""

from __future__ import annotations

import argparse
from types import ModuleType

import pytest


class TestParseBatchSize:
    """Mirror the contract on
    :func:`mamba_dvc.gpu.dispatch.correlate_multi_gpu`: positive int or
    the literal string ``"auto"``.
    """

    def test_auto_returns_string(self, run_e2e_zarr: ModuleType):
        assert run_e2e_zarr._parse_batch_size("auto") == "auto"

    def test_positive_int_returns_int(self, run_e2e_zarr: ModuleType):
        assert run_e2e_zarr._parse_batch_size("64") == 64

    def test_zero_raises(self, run_e2e_zarr: ModuleType):
        with pytest.raises(argparse.ArgumentTypeError, match="positive int or 'auto'"):
            run_e2e_zarr._parse_batch_size("0")

    def test_negative_raises(self, run_e2e_zarr: ModuleType):
        with pytest.raises(argparse.ArgumentTypeError, match="positive int or 'auto'"):
            run_e2e_zarr._parse_batch_size("-8")

    def test_unknown_string_raises(self, run_e2e_zarr: ModuleType):
        with pytest.raises(argparse.ArgumentTypeError, match="positive int or 'auto'"):
            run_e2e_zarr._parse_batch_size("big")

    def test_float_string_raises(self, run_e2e_zarr: ModuleType):
        # "32.0" is not a valid int; reject rather than truncate.
        with pytest.raises(argparse.ArgumentTypeError, match="positive int or 'auto'"):
            run_e2e_zarr._parse_batch_size("32.0")


class TestArgparserIntegration:
    """End-to-end integration through ``_build_argparser``.

    Ensures the custom ``type=`` callable is wired correctly and that
    the default value still parses as ``int``.
    """

    def test_argparser_accepts_auto(self, run_e2e_zarr: ModuleType):
        p = run_e2e_zarr._build_argparser()
        ns = p.parse_args(["--store", "x", "--deformation", "d", "--batch-size", "auto"])
        assert ns.batch_size == "auto"

    def test_argparser_accepts_int(self, run_e2e_zarr: ModuleType):
        p = run_e2e_zarr._build_argparser()
        ns = p.parse_args(["--store", "x", "--deformation", "d", "--batch-size", "128"])
        assert ns.batch_size == 128

    def test_argparser_default_is_64(self, run_e2e_zarr: ModuleType):
        # Plan §11.2: default stays at 64 for one PR cycle. Pinning the
        # value in a test forces an explicit conversation when we flip
        # the default to "auto".
        p = run_e2e_zarr._build_argparser()
        ns = p.parse_args(["--store", "x", "--deformation", "d"])
        assert ns.batch_size == 64

    def test_argparser_rejects_garbage(self, run_e2e_zarr: ModuleType):
        p = run_e2e_zarr._build_argparser()
        with pytest.raises(SystemExit):
            p.parse_args(["--store", "x", "--deformation", "d", "--batch-size", "big"])
