import csv
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import archive_series
import critical_study
from native_wolff import Chain


def invoke(main, *args):
    with patch.object(sys, "argv", ["test", *map(str, args)]), redirect_stdout(io.StringIO()):
        main()


class CriticalStudyArchiveTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.output = root / "results"
        self.cache = root / "cache"
        library = root / "fixture-library"
        library.write_bytes(b"test fixture")

        # Four 2x2 trajectories of eight configurations. Identical spin signs or
        # stratum means would hide errors in signed M or hot/cold stratification.
        self.m = np.array([
            [4, -4, 4, 2, -2, 4, 0, -2],
            [-2, 0, 0, 0, 2, 0, 0, 0],
            [4, -4, -4, 4, 2, -2, -2, 4],
            [0, 0, 2, -2, 2, -2, 0, -4],
        ], dtype=np.int32)
        self.e = np.where(abs(self.m) == 4, -8, 0).astype(np.int32)
        self.e[(self.m == 0) & (np.arange(8) % 2 == 0)] = 8
        self.seeds = [int(np.random.SeedSequence([17, 2, i]).generate_state(1, dtype=np.uint64)[0])
                      for i in range(4)]

        def synthetic_chain(L, T, *, warmup, samples, seed, initial, stride):
            i = self.seeds.index(seed)
            self.assertEqual((L, T, warmup, samples, stride), (2, critical_study.CRITICAL_T, 12, 8, 1))
            self.assertEqual(initial, ("random", "up")[i % 2])
            return Chain(self.e[i], self.m[i], i + 1.0)

        with patch("native_wolff.build_library", return_value=library), \
             patch("native_wolff.sample_chain", side_effect=synthetic_chain) as sample:
            invoke(critical_study.main, "--sizes", 2, "--replicas", 4, "--samples", 8,
                   "--warmup", 12, "--seed", 17, "--workers", 2, "--output", self.output,
                   "--series-dir", self.cache)
        self.assertEqual(sample.call_count, 4)
        invoke(archive_series.main, "--results", self.output, "--series-dir", self.cache)
        self.path = self.output / "series" / "L002.npz"

    def test_signed_series_roundtrip_and_stratified_standard_errors(self):
        # Verification requires only the public files; give it a nonexistent cache.
        invoke(archive_series.main, "--verify", "--results", self.output,
               "--series-dir", self.output / "absent")
        with np.load(self.path, allow_pickle=False) as f:
            np.testing.assert_array_equal(f["E"], self.e, strict=True)
            np.testing.assert_array_equal(f["M"], self.m, strict=True)
            np.testing.assert_array_equal(f["seed"], np.asarray(self.seeds, dtype=np.uint64), strict=True)
        manifest = json.loads((self.output / "critical_run.json").read_text(encoding="utf-8"))
        for name in ("critical_replicas.csv", "critical_diagnostics.csv"):
            self.assertEqual(manifest["sha256"][name], hashlib.sha256((self.output / name).read_bytes()).hexdigest())
        series_manifest = json.loads((self.path.parent / "sha256.json").read_text(encoding="utf-8"))
        self.assertEqual(series_manifest, {self.path.name: hashlib.sha256(self.path.read_bytes()).hexdigest()})
        with (self.output / "critical_replicas.csv").open(encoding="utf-8", newline="") as f:
            replicas = list(csv.DictReader(f))
        with (self.output / "critical_diagnostics.csv").open(encoding="utf-8", newline="") as f:
            diagnostics, = csv.DictReader(f)
        for key, matrix in (("e", self.e / 4), ("abs_m", abs(self.m) / 4), ("m2", (self.m / 4)**2)):
            means = matrix.mean(axis=1)
            np.testing.assert_array_equal([float(r[key]) for r in replicas], means)
            if key == "m2":
                continue
            # For two replicas per stratum, SE(stratum mean) is half their separation.
            hot_se = abs(means[0] - means[2]) / 2
            cold_se = abs(means[1] - means[3]) / 2
            contrast_se = np.hypot(hot_se, cold_se)
            self.assertAlmostEqual(float(diagnostics[key]), means.mean())
            self.assertAlmostEqual(float(diagnostics[key + "_se"]), contrast_se / 2)
            self.assertAlmostEqual(float(diagnostics[key + "_hot_minus_cold"]), means[::2].mean() - means[1::2].mean())
            self.assertAlmostEqual(float(diagnostics[key + "_hot_minus_cold_se"]), contrast_se)
            self.assertAlmostEqual(float(diagnostics[key + "_late_minus_early"]),
                                   matrix[:, 4:].mean() - matrix[:, :4].mean())
        self.assertEqual([int(r["seed"]) for r in replicas], self.seeds)

    def test_rejects_exchanged_seeds(self):
        np.savez_compressed(self.path, E=self.e, M=self.m, seed=np.asarray(self.seeds[::-1], dtype=np.uint64))
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            invoke(archive_series.main, "--verify", "--results", self.output)

    def test_checksum_detects_sign_change_that_preserves_all_archived_means(self):
        modified = self.m.copy()
        modified[0, 0] *= -1
        np.savez_compressed(self.path, E=self.e, M=modified, seed=np.asarray(self.seeds, dtype=np.uint64))
        with self.assertRaisesRegex(ValueError, "checksum"):
            invoke(archive_series.main, "--verify", "--results", self.output)
