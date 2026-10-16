import ast
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import feature_utils as utilities


class ExplicitInputTests(unittest.TestCase):
    def test_public_functions_have_no_default_arguments(self):
        for name in utilities.__all__:
            signature = inspect.signature(getattr(utilities, name))
            for parameter in signature.parameters.values():
                with self.subTest(function=name, parameter=parameter.name):
                    self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_example_scripts_have_no_numeric_presets(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / "examples").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            numbers = [
                node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and type(node.value) in (int, float)
            ]
            self.assertEqual(numbers, [], path.name)

    def test_input_templates_contain_only_unset_values(self):
        root = Path(__file__).resolve().parents[1]
        templates = list((root / "examples" / "inputs").glob("*.json"))
        self.assertEqual(len(templates), 3)
        for path in templates:
            values = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(values)
            self.assertTrue(all(value is None for value in values.values()))

    def test_statistics_use_requested_percentiles_and_ddof(self):
        image = np.array([[2.0, 4.0]])
        mask = np.ones_like(image)
        for correction, expected in ((0, 1), (1, np.sqrt(2))):
            result = utilities.intensity_statistics(image, mask, percentiles=[0, 50, 100], ddof=correction)
            self.assertEqual(result["percentiles"], {0.0: 2.0, 50.0: 3.0, 100.0: 4.0})
            self.assertAlmostEqual(result["std"], expected)
        result = utilities.intensity_statistics(image, mask, percentiles=[], ddof=2)
        self.assertTrue(np.isnan(result["std"]))
        self.assertEqual(result["percentiles"], {})

    def test_invalid_statistic_parameters_are_rejected(self):
        for percentiles, ddof in (([-1], 0), ([101], 0), ([50, 50], 0), ([np.nan], 0), ([], -1), ([], 0.5)):
            with self.subTest(percentiles=percentiles, ddof=ddof), self.assertRaises(ValueError):
                utilities.intensity_statistics([[1]], [[1]], percentiles=percentiles, ddof=ddof)

    def test_iqr_factor_and_side_change_the_result(self):
        reference = pd.DataFrame({"value": [0, 1, 2, 3, 4]})
        candidate = pd.DataFrame({"value": [-2, 6]})
        narrow = utilities.fit_iqr_bounds(reference, ["value"], factor=1)
        wide = utilities.fit_iqr_bounds(reference, ["value"], factor=2)
        actual = utilities.flag_iqr_outliers(candidate, narrow, side="both")
        self.assertEqual(actual["value"].tolist(), [True, True])
        actual = utilities.flag_iqr_outliers(candidate, narrow, side="upper")
        self.assertEqual(actual["value"].tolist(), [False, True])
        actual = utilities.flag_iqr_outliers(candidate, wide, side="both")
        self.assertEqual(actual["value"].tolist(), [False, False])


class CommandLineTests(unittest.TestCase):
    def invoke(self, script, payload):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
            process = subprocess.run(
                [sys.executable, str(root / "examples" / script), "--input", str(path)],
                capture_output=True,
                text=True,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def test_region_example_uses_external_measurements(self):
        for label, spacing in ((2, [1, 2]), (3, [2, 3])):
            result = self.invoke("region_measurements.py", {
                "labels": [[0, label], [0, label]],
                "label": label,
                "spacing_mm": spacing,
                "perimeter_pixels": 6,
                "major_axis_pixels": 2,
                "minor_axis_pixels": 1,
                "slice_areas_mm2": [10, 20],
                "slice_thickness_mm": 3,
            })
            self.assertEqual(result["area_pixels"], 2)
            self.assertEqual(result["area_mm2"], 2 * spacing[0] * spacing[1])
            self.assertAlmostEqual(result["volume_ml"], 0.09)

    def test_signal_example_uses_external_statistics_options(self):
        result = self.invoke("signal_measurements.py", {
            "mask": [[1, 1, 1]],
            "water": [[2, 4, None]],
            "fat": [[2, 4, 1]],
            "percentiles": [50],
            "ddof": 1,
        })
        self.assertEqual(result["water_statistics"]["percentiles"], {"50.0": 3.0})
        self.assertAlmostEqual(result["water_statistics"]["std"], np.sqrt(2))
        self.assertEqual(result["fat_fraction_map"], [[0.5, 0.5, None]])

    def test_table_example_uses_external_columns_and_thresholds(self):
        result = self.invoke("table_quality.py", {
            "first_table": [{"key": "b", "x": 2}, {"key": "a", "x": 1}],
            "second_table": [{"key": "a", "y": 3}, {"key": "b", "y": 4}],
            "reference_table": [{"x": value} for value in range(5)],
            "candidate_table": [{"x": value} for value in [1, 6, None]],
            "id_column": "key",
            "selected_columns": ["key", "x"],
            "feature_columns": ["x"],
            "iqr_factor": 1,
            "outlier_side": "upper",
        })
        self.assertEqual(result["aligned_tables"][1][0]["key"], "b")
        self.assertEqual(result["bounds"]["x"]["upper"], 5)
        self.assertEqual(result["outlier_flags"], [{"x": False}, {"x": True}, {"x": None}])

    def test_examples_require_input_and_reject_unfilled_templates(self):
        root = Path(__file__).resolve().parents[1]
        for script, template in (
            ("region_measurements.py", "region.template.json"),
            ("signal_measurements.py", "signals.template.json"),
            ("table_quality.py", "tables.template.json"),
        ):
            with self.subTest(script=script):
                base = [sys.executable, str(root / "examples" / script)]
                result = subprocess.run(base, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--input", result.stderr)
                path = root / "examples" / "inputs" / template
                result = subprocess.run(base + ["--input", str(path)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Required values are missing", result.stderr)


if __name__ == "__main__":
    unittest.main()

