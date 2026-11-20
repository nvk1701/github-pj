"""Kiem tra output policy, nhan d512 va bao toan publication; khong training."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments import sensitivity


class SensitivityExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "source"
        self.publication = self.repo / "docs/results/cp04-cp05"
        self.publication.mkdir(parents=True)
        original = sensitivity.PROJECT_ROOT / "docs/results/cp04-cp05"
        shutil.copytree(original / "data", self.publication / "data")
        shutil.copyfile(original / "publication_manifest.json",
                        self.publication / "publication_manifest.json")
        self.addCleanup(patch.stopall)
        patch.object(sensitivity, "PROJECT_ROOT", self.repo).start()

    def snapshot(self):
        return {str(path.relative_to(self.publication)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.publication.rglob("*") if path.is_file()}

    def test_new_export_filters_d512_and_keeps_source_bytes(self):
        before = self.snapshot()
        target = self.base / "derived"
        result = sensitivity.generate_sensitivity_figures_and_tables(target)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(len(result["d512_groups"]), 3)
        self.assertEqual({row["dimension"] for row in result["d512_groups"]}, {512})
        self.assertEqual({row["method"] for row in result["d512_groups"]},
                         {"convex", "constant", "inverse_sqrt"})
        recorded = json.loads((target / "provenance.json").read_text())
        self.assertEqual(recorded, result)
        self.assertEqual(len(recorded["files_sha256"]), 6)
        for name, expected in recorded["files_sha256"].items():
            self.assertEqual(hashlib.sha256((target / name).read_bytes()).hexdigest(), expected)
        self.assertTrue((target / "d512_method_summary.png").is_file())

    def test_existing_root_fails_without_any_overwrite(self):
        target = self.base / "existing"
        target.mkdir()
        sentinel = target / "drift_sensitivity.png"
        sentinel.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            sensitivity.generate_sensitivity_figures_and_tables(target)
        self.assertEqual(sentinel.read_bytes(), b"preserve")
        self.assertEqual(list(target.iterdir()), [sentinel])

    def test_even_empty_existing_root_is_rejected(self):
        target = self.base / "empty"
        target.mkdir()
        with self.assertRaises(FileExistsError):
            sensitivity.generate_sensitivity_figures_and_tables(target)
        self.assertEqual(list(target.iterdir()), [])

    def test_output_inside_or_above_publication_is_rejected(self):
        before = self.snapshot()
        for target in (self.publication / "new-figures", self.publication, self.repo):
            with self.subTest(target=target), self.assertRaises(ValueError):
                sensitivity.generate_sensitivity_figures_and_tables(target)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((self.publication / "new-figures").exists())

    def test_changed_csv_fails_before_creating_output(self):
        path = self.publication / "data/cp05_condition_summary.csv"
        path.write_bytes(path.read_bytes() + b"corrupt\n")
        target = self.base / "invalid-input"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            sensitivity.generate_sensitivity_figures_and_tables(target)
        self.assertFalse(target.exists())

    def test_default_root_stays_outside_source_and_cannot_be_reused(self):
        before = self.snapshot()
        sensitivity.generate_sensitivity_figures_and_tables()
        self.assertTrue((self.repo / "outputs/sensitivity-derived/provenance.json").is_file())
        with self.assertRaises(FileExistsError):
            sensitivity.generate_sensitivity_figures_and_tables()
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
