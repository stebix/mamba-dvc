"""Tests for ``mamba_dvc.run.config``: parsing + variant expansion."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from mamba_dvc.run.config import BatchSpec, slug_value


def _spec(**overrides) -> BatchSpec:
    base: dict = {
        "campaign": "c",
        "stores": ["a.zarr"],
    }
    base.update(overrides)
    return BatchSpec.from_mapping(base)


class TestBasicParsing:
    def test_minimal_config_has_one_base_variant(self) -> None:
        spec = _spec()
        assert spec.campaign == "c"
        assert spec.stores == (Path("a.zarr"),)
        assert len(spec.variants) == 1
        v = spec.variants[0]
        assert v.name == "base"
        assert v.variant_id == f"base-{v.params_hash}"
        assert v.params["window"] == 96
        assert v.params["mask_threshold"] == pytest.approx(0.9)

    def test_campaign_dir(self) -> None:
        spec = _spec(out_dir="results")
        assert spec.campaign_dir == Path("results") / "c"

    def test_devices_and_distance_bins(self) -> None:
        spec = _spec(devices=[0, 2], distance_bins=[0, 5, 10])
        assert spec.devices == (0, 2)
        assert spec.distance_bins == (0.0, 5.0, 10.0)

    def test_devices_default_none(self) -> None:
        assert _spec().devices is None

    def test_strict_defaults_true(self) -> None:
        assert _spec().strict is True

    def test_strict_can_be_disabled(self) -> None:
        assert _spec(strict=False).strict is False

    def test_strict_must_be_bool(self) -> None:
        with pytest.raises(ValueError, match="'strict' must be a bool"):
            _spec(strict="no")

    def test_from_yaml_round_trips(self, tmp_path: Path) -> None:
        cfg = tmp_path / "campaign.yaml"
        cfg.write_text(
            yaml.safe_dump({"campaign": "c", "stores": ["x.zarr"]}), encoding="utf-8"
        )
        spec = BatchSpec.from_yaml(cfg)
        assert spec.source_path == cfg
        assert spec.campaign == "c"


class TestSelect:
    def test_defaults_to_all(self) -> None:
        spec = _spec()
        assert spec.select_real == "all"
        assert spec.select_synthetic == "all"

    def test_explicit_select(self) -> None:
        spec = _spec(select={"real": "none", "synthetic": ["fs104", "fs204"]})
        assert spec.select_real == "none"
        assert spec.select_synthetic == ("fs104", "fs204")

    def test_select_names_resolution(self) -> None:
        spec = _spec(select={"real": "all", "synthetic": ["b", "missing"]})
        names = spec.select_names(real=["r1"], synthetic=["a", "b", "c"])
        assert names == [("r1", "real"), ("b", "synthetic")]


class TestExpandSweep:
    def test_cartesian_product(self) -> None:
        spec = _spec(sweep={"mask_threshold": [0.7, 0.5], "window": [64, 96]})
        # 2 x 2 = 4 variants.
        assert len(spec.variants) == 4
        thresholds = sorted({v.params["mask_threshold"] for v in spec.variants})
        assert thresholds == pytest.approx([0.5, 0.7])

    def test_sweep_value_equal_to_default_named_base(self) -> None:
        # mask_threshold default is 0.9 → that sweep point differs in nothing.
        spec = _spec(sweep={"mask_threshold": [0.9, 0.5]})
        names = {v.name for v in spec.variants}
        assert names == {"base", "mt0.5"}

    def test_explicit_variants_merged(self) -> None:
        spec = _spec(
            sweep={"mask_threshold": [0.7, 0.5]},
            variants=[{"name": "hires", "window": 128, "mask_threshold": 0.3}],
        )
        assert len(spec.variants) == 3
        hires = next(v for v in spec.variants if v.name == "hires")
        assert hires.params["window"] == 128
        assert hires.params["mask_threshold"] == pytest.approx(0.3)
        assert hires.variant_id == f"hires-{hires.params_hash}"

    def test_dedup_collapses_identical(self) -> None:
        spec = _spec(
            sweep={"mask_threshold": [0.5, 0.5]},
            variants=[{"mask_threshold": 0.5}],
        )
        assert len(spec.variants) == 1

    def test_auto_slug_multi_axis(self) -> None:
        spec = _spec(sweep={"mask_threshold": [0.3], "window": [64]})
        (v,) = spec.variants
        # Slug order follows the knob declaration order: window before mask_threshold.
        assert v.name == "w64-mt0.3"


class TestVariantIdStability:
    def test_same_config_same_ids(self) -> None:
        cfg = {"campaign": "c", "stores": ["a.zarr"], "sweep": {"mask_threshold": [0.7, 0.5]}}
        a = BatchSpec.from_mapping(cfg)
        b = BatchSpec.from_mapping(cfg)
        assert [v.variant_id for v in a.variants] == [v.variant_id for v in b.variants]

    def test_one_changed_value_one_new_id(self) -> None:
        a = _spec(sweep={"mask_threshold": [0.9, 0.7, 0.5]})
        b = _spec(sweep={"mask_threshold": [0.9, 0.7, 0.3]})  # last point changed
        ids_a = {v.variant_id for v in a.variants}
        ids_b = {v.variant_id for v in b.variants}
        assert len(ids_a & ids_b) == 2  # the 0.9 and 0.7 points are shared
        assert len(ids_b - ids_a) == 1  # only the 0.3 point is new

    def test_int_window_stays_int(self) -> None:
        spec = _spec(defaults={"window": 64})
        assert spec.variants[0].params["window"] == 64

    def test_list_window_becomes_tuple(self) -> None:
        spec = _spec(defaults={"window": [64, 80, 80]})
        assert spec.variants[0].params["window"] == (64, 80, 80)


class TestTierSplit:
    def test_tiers_partition_params(self) -> None:
        spec = _spec(
            sweep={"mask_threshold": [0.5]}, defaults={"flow_convention": "push_forward"}
        )
        (v,) = spec.variants
        assert set(v.load_params) == {"mask", "dry_shape", "flow_convention"}
        assert set(v.grid_params) == {"window", "overlap"}
        assert set(v.compute_params) == {
            "mask_threshold",
            "batch_size",
            "ncc_mode",
            "ncc_normalization",
            "search_radius",
            "tukey_alpha",
        }
        assert v.load_params["flow_convention"] == "push_forward"
        assert dict(v.params) == {**v.load_params, **v.grid_params, **v.compute_params}


class TestNormalizationAndErrors:
    def test_dry_shape_list(self) -> None:
        spec = _spec(defaults={"dry_shape": [192, 256, 256]})
        assert spec.variants[0].load_params["dry_shape"] == (192, 256, 256)

    def test_dry_shape_scalar_rejected(self) -> None:
        with pytest.raises(ValueError, match="dry_shape"):
            _spec(defaults={"dry_shape": 192})

    def test_batch_size_auto_or_int(self) -> None:
        assert (
            _spec(defaults={"batch_size": "auto"}).variants[0].params["batch_size"] == "auto"
        )
        assert _spec(defaults={"batch_size": 64}).variants[0].params["batch_size"] == 64
        with pytest.raises(ValueError, match="batch_size"):
            _spec(defaults={"batch_size": 0})

    def test_flow_convention_validated(self) -> None:
        with pytest.raises(ValueError, match="flow_convention"):
            _spec(defaults={"flow_convention": "sideways"})

    def test_unknown_top_key(self) -> None:
        with pytest.raises(ValueError, match="unknown campaign config key"):
            _spec(store="typo.zarr")

    def test_missing_campaign(self) -> None:
        with pytest.raises(ValueError, match="campaign"):
            BatchSpec.from_mapping({"stores": ["a.zarr"]})

    def test_empty_stores(self) -> None:
        with pytest.raises(ValueError, match="stores"):
            BatchSpec.from_mapping({"campaign": "c", "stores": []})

    def test_unknown_knob_in_defaults(self) -> None:
        with pytest.raises(ValueError, match="unknown parameter knob"):
            _spec(defaults={"not_a_knob": 1})

    def test_sweep_axis_must_be_list(self) -> None:
        with pytest.raises(ValueError, match="sweep"):
            _spec(sweep={"mask_threshold": 0.5})

    def test_distance_bins_must_be_sorted(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            _spec(distance_bins=[10, 5])


class TestSlugValue:
    def test_renders(self) -> None:
        assert slug_value(None) == "none"
        assert slug_value(0.5) == "0.5"
        assert slug_value(5.0) == "5"
        assert slug_value(96) == "96"
        assert slug_value("auto") == "auto"
        assert slug_value((64, 64, 64)) == "64x64x64"
