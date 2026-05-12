"""Tests for ``mamba_dvc.run.batch``: the execution loop, resume, isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from mamba_dvc.run import NullObserver, plan_jobs, run_batch
from mamba_dvc.run.config import BatchSpec
from mamba_dvc.types import DisplacementField

# --------------------------------------------------------------- fakes


class _CountingGT:
    """Callable ground-truth field recording how often it is evaluated."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.zeros_like(positions)


class _Pair:
    def __init__(self, name: str, kind: str, gt: _CountingGT | None) -> None:
        self.reference = np.zeros((4, 4, 4), dtype=np.float32)
        self.deformed = np.zeros((4, 4, 4), dtype=np.float32)
        self.mask = np.ones((4, 4, 4), dtype=np.bool_)
        self.gt_field = gt
        self.name = name
        self.kind = kind
        self.spacing = None


class _FakeDataset:
    def __init__(self, real: list[str], synthetic: list[str]) -> None:
        self._real = list(real)
        self._synthetic = list(synthetic)
        self.load_pair_calls: list[tuple[str, object, object]] = []
        # One GT instance per synthetic entry, reused across load_pair calls.
        self.gt: dict[str, _CountingGT] = {n: _CountingGT() for n in synthetic}

    def list_real(self) -> list[str]:
        return list(self._real)

    def list_synthetic(self) -> list[str]:
        return list(self._synthetic)

    def load_pair(self, deformation: str, *, mask=None, dry_shape=None) -> _Pair:
        self.load_pair_calls.append((deformation, mask, dry_shape))
        kind = "synthetic" if deformation in self._synthetic else "real"
        gt = self.gt[deformation] if kind == "synthetic" else None
        return _Pair(deformation, kind, gt)


def _make_field(window: int = 96, n: int = 4) -> DisplacementField:
    return DisplacementField(
        positions=np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        displacements=np.zeros((n, 3), dtype=np.float32),
        valid=np.ones(n, dtype=np.bool_),
        confidence=np.ones(n, dtype=np.float32),
        status=np.zeros(n, dtype=np.uint8),
        grid_shape=(n, 1, 1),
        spacing=(1, 1, 1),
        window=(window, window, window),
    )


def _fake_correlate(reference, deformed, mask=None, *, device_ids=None, window=96, **kwargs):
    w = window if isinstance(window, int) else int(window[0])
    return _make_field(window=w)


def _opener_for(ds_by_store: dict[Path, _FakeDataset]):
    def _open(store: Path, flow_convention):
        return ds_by_store[store]

    return _open


def _spec(tmp_path: Path, *, stores: list[str], **overrides) -> BatchSpec:
    cfg: dict = {
        "campaign": "camp",
        "stores": [str(tmp_path / s) for s in stores],
        "out_dir": str(tmp_path / "out"),
        "defaults": {"flow_convention": "push_forward"},  # avoid the unpinned warning
    }
    cfg.update(overrides)
    return BatchSpec.from_mapping(cfg)


# --------------------------------------------------------------- tests


class TestLoopReuse:
    def test_one_load_pair_per_deformation_truth_cached_per_grid(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1", "fs2"])
        spec = _spec(
            tmp_path,
            stores=["scanA.zarr"],
            # compute-tier sweep: same grid, three thresholds → cache hit.
            sweep={"mask_threshold": [0.9, 0.7, 0.5]},
            defaults={"flow_convention": "push_forward"},
        )
        results = run_batch(
            spec, correlate_fn=_fake_correlate, open_dataset=_opener_for({store: ds})
        )
        # 2 deformations x 3 variants = 6 ok results.
        assert len(results) == 6
        assert all(r.status == "ok" for r in results)
        # load_pair: once per (deformation, mask, dry_shape) — masks/dry_shape
        # don't vary here → once per deformation.
        assert len(ds.load_pair_calls) == 2
        # GT evaluated once per (pair, grid-spec) — one grid → once per deformation.
        assert ds.gt["fs1"].calls == 1
        assert ds.gt["fs2"].calls == 1

    def test_grid_sweep_reevaluates_truth_per_grid(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        spec = _spec(tmp_path, stores=["scanA.zarr"], sweep={"window": [64, 96]})
        run_batch(spec, correlate_fn=_fake_correlate, open_dataset=_opener_for({store: ds}))
        assert len(ds.load_pair_calls) == 1  # window is grid-tier, not load-tier
        assert ds.gt["fs1"].calls == 2  # two distinct grids

    def test_real_entry_correlated_but_not_scored(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=["it016"], synthetic=[])
        spec = _spec(tmp_path, stores=["scanA.zarr"])
        results = run_batch(
            spec, correlate_fn=_fake_correlate, open_dataset=_opener_for({store: ds})
        )
        (r,) = results
        assert r.status == "ok"
        assert "mae" not in r.summary  # no GT → no error metrics
        assert (r.npz_path is not None) and r.npz_path.exists()
        sidecar = json.loads(r.sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["scored"] is False


class TestOutputArtifacts:
    def test_writes_npz_sidecar_manifest_variants_snapshot(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        spec = _spec(tmp_path, stores=["scanA.zarr"], sweep={"mask_threshold": [0.7, 0.5]})
        results = run_batch(
            spec, correlate_fn=_fake_correlate, open_dataset=_opener_for({store: ds})
        )
        campaign_dir = spec.campaign_dir
        assert (campaign_dir / "config.snapshot.yaml").exists()
        assert (campaign_dir / "variants.json").exists()
        manifest_lines = (
            (campaign_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        )
        assert len(manifest_lines) == 2
        rows = [json.loads(line) for line in manifest_lines]
        assert {r["mask_threshold"] for r in rows} == {0.7, 0.5}
        assert all(r["status"] == "ok" for r in rows)
        assert all(r["n_ok"] == 4 for r in rows)

        variants = json.loads((campaign_dir / "variants.json").read_text(encoding="utf-8"))
        assert len(variants) == 2
        for r in results:
            loaded = DisplacementField.load_npz(r.npz_path)
            assert loaded.positions.shape == (4, 3)
            sidecar = json.loads(r.sidecar_path.read_text(encoding="utf-8"))
            assert sidecar["params_hash"] == r.job.variant.params_hash
            assert sidecar["params"]["mask_threshold"] in (0.7, 0.5)


class TestResume:
    def test_done_jobs_skipped_force_reincludes(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        spec = _spec(tmp_path, stores=["scanA.zarr"], sweep={"mask_threshold": [0.7, 0.5]})
        opener = _opener_for({store: ds})

        # First run does both.
        first = run_batch(spec, correlate_fn=_fake_correlate, open_dataset=opener)
        assert len(first) == 2

        # Plan now finds nothing pending; a re-run produces no new results.
        assert plan_jobs(spec, open_dataset=opener) == []
        second = run_batch(spec, correlate_fn=_fake_correlate, open_dataset=opener)
        assert second == []

        # force=True re-runs everything.
        forced = run_batch(spec, correlate_fn=_fake_correlate, open_dataset=opener, force=True)
        assert len(forced) == 2
        assert len(plan_jobs(spec, open_dataset=opener, force=True)) == 2

    def test_hash_mismatch_reincludes(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        spec = _spec(tmp_path, stores=["scanA.zarr"])
        opener = _opener_for({store: ds})
        run_batch(spec, correlate_fn=_fake_correlate, open_dataset=opener)
        assert plan_jobs(spec, open_dataset=opener) == []

        # Corrupt the sidecar's params_hash → the variant is "not done".
        sidecar = next(
            p for p in spec.campaign_dir.rglob("*.json") if p.name != "variants.json"
        )
        sidecar.write_text(json.dumps({"params_hash": "deadbeef"}), encoding="utf-8")
        assert len(plan_jobs(spec, open_dataset=opener)) == 1


class TestFailureIsolation:
    def test_one_failing_variant_recorded_rest_complete(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        spec = _spec(tmp_path, stores=["scanA.zarr"], sweep={"mask_threshold": [0.9, 0.5]})

        def flaky_correlate(
            reference, deformed, mask=None, *, device_ids=None, mask_threshold=0.9, **kw
        ):
            if mask_threshold == 0.5:
                raise RuntimeError("boom")
            return _make_field()

        results = run_batch(
            spec, correlate_fn=flaky_correlate, open_dataset=_opener_for({store: ds})
        )
        assert len(results) == 2
        by_status = {r.status for r in results}
        assert by_status == {"ok", "failed"}
        failed = next(r for r in results if r.status == "failed")
        assert "boom" in (failed.error or "")
        assert failed.npz_path is None
        # The failure is in the manifest with its phase.
        rows = [
            json.loads(line)
            for line in (spec.campaign_dir / "manifest.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        failed_rows = [r for r in rows if r["status"] == "failed"]
        assert len(failed_rows) == 1
        assert failed_rows[0]["phase"] == "correlate"

    def test_store_open_failure_recorded(self, tmp_path: Path) -> None:
        spec = _spec(tmp_path, stores=["broken.zarr"])

        def bad_opener(store, flow_convention):
            raise FileNotFoundError("no such store")

        results = run_batch(spec, correlate_fn=_fake_correlate, open_dataset=bad_opener)
        assert len(results) == 1
        assert results[0].status == "failed"
        assert results[0].job.deformation == "*"


class TestOnlyFilter:
    def test_only_restricts_variants(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        spec = _spec(
            tmp_path, stores=["scanA.zarr"], sweep={"mask_threshold": [0.9, 0.7, 0.5]}
        )
        opener = _opener_for({store: ds})
        jobs = plan_jobs(spec, only={"mask_threshold": "0.5"}, open_dataset=opener)
        assert len(jobs) == 1
        assert jobs[0].variant.params["mask_threshold"] == pytest.approx(0.5)

        results = run_batch(
            spec,
            only={"mask_threshold": "0.7"},
            correlate_fn=_fake_correlate,
            open_dataset=opener,
        )
        assert len(results) == 1

    def test_only_unknown_knob_raises(self, tmp_path: Path) -> None:
        spec = _spec(tmp_path, stores=["scanA.zarr"])
        with pytest.raises(ValueError, match="unknown knob"):
            plan_jobs(spec, only={"nope": "1"})


class _RecordingObserver(NullObserver):
    """Logs a tag tuple for every event so tests can assert ordering and counts."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_batch_start(self, *, n_jobs: int, n_variants: int) -> None:
        self.events.append(("batch_start", n_jobs, n_variants))

    def on_pair_load_start(self, store, deformation: str, *, n_variants: int) -> None:
        self.events.append(("pair_load_start", store.name, deformation, n_variants))

    def on_job_start(self, job) -> None:
        self.events.append(("job_start", job.deformation, job.variant.variant_id))

    def on_job_end(self, result) -> None:
        self.events.append(("job_end", result.job.deformation, result.status))

    def on_batch_end(self, results) -> None:
        self.events.append(("batch_end", len(results)))

    def tags(self) -> list[str]:
        return [e[0] for e in self.events]


class TestObserver:
    def test_event_sequence_for_a_small_campaign(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1", "fs2"])
        spec = _spec(tmp_path, stores=["scanA.zarr"], sweep={"mask_threshold": [0.7, 0.5]})
        obs = _RecordingObserver()
        results = run_batch(
            spec,
            correlate_fn=_fake_correlate,
            open_dataset=_opener_for({store: ds}),
            observer=obs,
        )
        assert len(results) == 4
        assert obs.events[0] == ("batch_start", 4, 2)
        assert obs.events[-1] == ("batch_end", 4)
        # one load per deformation; one job_start/job_end pair per job; serial work.
        assert obs.tags() == [
            "batch_start",
            "pair_load_start", "job_start", "job_end", "job_start", "job_end",
            "pair_load_start", "job_start", "job_end", "job_start", "job_end",
            "batch_end",
        ]  # fmt: skip

    def test_failed_variant_still_bracketed(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        spec = _spec(tmp_path, stores=["scanA.zarr"], sweep={"mask_threshold": [0.9, 0.5]})

        def flaky(
            reference, deformed, mask=None, *, device_ids=None, mask_threshold=0.9, **kw
        ):
            if mask_threshold == 0.5:
                raise RuntimeError("boom")
            return _make_field()

        obs = _RecordingObserver()
        run_batch(
            spec, correlate_fn=flaky, open_dataset=_opener_for({store: ds}), observer=obs
        )
        assert obs.tags().count("job_start") == obs.tags().count("job_end") == 2
        assert ("job_end", "fs1", "failed") in obs.events
        assert ("job_end", "fs1", "ok") in obs.events

    def test_load_pair_failure_brackets_each_variant(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        spec = _spec(tmp_path, stores=["scanA.zarr"], sweep={"mask_threshold": [0.9, 0.5]})

        def boom_load(deformation, *, mask=None, dry_shape=None):
            raise OSError("disk gone")

        ds.load_pair = boom_load  # break load_pair after the dataset opens fine
        obs = _RecordingObserver()
        run_batch(
            spec,
            correlate_fn=_fake_correlate,
            open_dataset=_opener_for({store: ds}),
            observer=obs,
        )
        assert obs.tags().count("pair_load_start") == 1
        assert obs.tags().count("job_start") == 2  # one per variant in the broken load group
        assert all(e[2] == "failed" for e in obs.events if e[0] == "job_end")

    def test_store_open_failure_is_one_bracketed_job(self, tmp_path: Path) -> None:
        spec = _spec(tmp_path, stores=["broken.zarr"])

        def bad_opener(store, flow_convention):
            raise FileNotFoundError("nope")

        obs = _RecordingObserver()
        results = run_batch(
            spec, correlate_fn=_fake_correlate, open_dataset=bad_opener, observer=obs
        )
        assert obs.events[0] == ("batch_start", 1, 1)
        assert ("job_start", "*", results[0].job.variant.variant_id) in obs.events
        assert ("job_end", "*", "failed") in obs.events
        assert obs.events[-1] == ("batch_end", 1)
        assert "pair_load_start" not in obs.tags()  # never opened → no load attempted

    def test_null_observer_accepted_as_default(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        spec = _spec(tmp_path, stores=["scanA.zarr"])
        results = run_batch(
            spec,
            correlate_fn=_fake_correlate,
            open_dataset=_opener_for({store: ds}),
            observer=NullObserver(),
        )
        assert [r.status for r in results] == ["ok"]


class TestStrictForwarding:
    def test_default_opener_forwards_strict_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mamba_dvc.run.batch import _make_default_opener

        captured: dict[str, object] = {}

        class _StubDataset:
            @classmethod
            def open(cls, store, *, manifest=None, strict=True):
                captured["strict"] = strict
                return object()

        monkeypatch.setattr("mamba_dvc.run.batch.DvcDataset", _StubDataset)
        spec = _spec(tmp_path, stores=["s.zarr"], strict=False)
        _make_default_opener(spec)(tmp_path / "s.zarr", None)
        assert captured["strict"] is False


class TestConventionWarning:
    def test_unpinned_convention_warns(self, tmp_path: Path) -> None:
        store = tmp_path / "scanA.zarr"
        ds = _FakeDataset(real=[], synthetic=["fs1"])
        # No flow_convention in defaults → built-in None → warn.
        spec = BatchSpec.from_mapping(
            {"campaign": "c", "stores": [str(store)], "out_dir": str(tmp_path / "out")}
        )
        with pytest.warns(UserWarning, match="flow_convention is unset"):
            run_batch(
                spec, correlate_fn=_fake_correlate, open_dataset=_opener_for({store: ds})
            )
