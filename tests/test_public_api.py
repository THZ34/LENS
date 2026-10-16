import unittest
import numpy as np
import pandas as pd
from feature_utils import align_tables, area_to_volume_ml, aspect_ratio, circularity, fat_fraction_map, fat_fraction_summary, fit_iqr_bounds, flag_iqr_outliers, intensity_statistics, missingness_summary, pixels_to_area_mm2, region_area, region_centroid, select_columns, select_label

class GeometryTests(unittest.TestCase):

    def test_label_selection_area_and_coordinate_order(self):
        labels = np.array([[0, 1, 1], [0, 0, 1]])
        mask = select_label(labels, 1)
        self.assertEqual(region_area(mask), 3)
        np.testing.assert_allclose(region_centroid(mask), (1 / 3, 5 / 3))
        self.assertEqual(labels[0, 1], 1)

    def test_absent_label_and_empty_centroid(self):
        mask = select_label(np.zeros((2, 3), dtype=int), 1)
        self.assertEqual(region_area(mask), 0)
        self.assertTrue(np.isnan(region_centroid(mask)).all())

    def test_invalid_label_images_are_rejected(self):
        for labels in (np.array([[0.5]]), np.array([[np.nan]]), np.array([[-1]])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                select_label(labels, 1)

    def test_multiclass_and_nonfinite_masks_are_rejected(self):
        for mask in (np.array([[0, 2]]), np.array([[np.nan]])):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                region_area(mask)

    def test_geometry_requires_two_dimensions(self):
        with self.assertRaises(ValueError):
            region_area(np.ones((2, 2, 2)))

    def test_circularity_is_invariant_to_consistent_unit_scaling(self):
        self.assertAlmostEqual(circularity(100.0, 40.0), np.pi / 4)
        self.assertAlmostEqual(circularity(400.0, 80.0), circularity(100.0, 40.0))

    def test_undefined_shape_ratios_are_missing(self):
        self.assertTrue(np.isnan(circularity(0, 0)))
        self.assertTrue(np.isnan(circularity(np.nan, 1)))
        self.assertTrue(np.isnan(aspect_ratio(2, 0)))

    def test_axis_order_is_explicit(self):
        self.assertEqual(aspect_ratio(6, 3), 2)
        with self.assertRaises(ValueError):
            aspect_ratio(3, 6)

class SignalTests(unittest.TestCase):

    def test_statistics_exclude_nonfinite_values_and_use_population_std(self):
        image = np.array([[2.0, 4.0], [np.nan, np.inf]])
        result = intensity_statistics(image, np.ones((2, 2)), percentiles=[], ddof=0)
        self.assertEqual(result['region_count'], 4)
        self.assertEqual(result['valid_count'], 2)
        self.assertEqual(result['sum'], 6)
        self.assertEqual(result['mean'], 3)
        self.assertEqual(result['std'], 1)
        self.assertEqual(result['iqr'], 1)

    def test_empty_signal_regions_have_missing_statistics(self):
        result = intensity_statistics(np.ones((2, 2)), np.zeros((2, 2)), percentiles=[], ddof=0)
        self.assertEqual(result['valid_count'], 0)
        self.assertTrue(np.isnan(result['sum']))
        self.assertTrue(np.isnan(result['mean']))

    def test_ratio_of_sums_is_distinct_from_mean_fraction(self):
        water = np.array([[9.0, 1.0]])
        fat = np.array([[1.0, 1.0]])
        result = fat_fraction_summary(water, fat, np.ones((1, 2)))
        self.assertAlmostEqual(result['ratio_of_sums'], 1 / 6)
        self.assertAlmostEqual(result['mean_fraction'], 0.3)

    def test_paired_signal_support_excludes_invalid_values(self):
        water = np.array([[9.0, np.nan, -1.0, 0.0, 4.0]])
        fat = np.array([[1.0, 2.0, 2.0, 0.0, 1.0]])
        mask = np.array([[1, 1, 1, 1, 0]])
        result = fat_fraction_summary(water, fat, mask)
        fractions = fat_fraction_map(water, fat, mask)
        self.assertEqual(result['region_count'], 4)
        self.assertEqual(result['valid_count'], 1)
        self.assertAlmostEqual(result['ratio_of_sums'], 0.1)
        self.assertAlmostEqual(fractions[0, 0], 0.1)
        self.assertTrue(np.isnan(fractions[0, 1:]).all())

    def test_zero_and_one_fractions_are_valid(self):
        actual = fat_fraction_map([[1.0, 0.0]], [[0.0, 1.0]], [[1, 1]])
        np.testing.assert_array_equal(actual, [[0.0, 1.0]])

    def test_all_invalid_fractions_are_missing(self):
        result = fat_fraction_summary([[0.0]], [[0.0]], [[1]])
        self.assertEqual(result['valid_count'], 0)
        self.assertTrue(np.isnan(result['ratio_of_sums']))
        self.assertTrue(np.isnan(result['mean_fraction']))

    def test_signals_require_matching_shapes(self):
        with self.assertRaises(ValueError):
            fat_fraction_map(np.ones((2, 2)), np.ones((1, 2)), np.ones((2, 2)))

    def test_three_dimensional_signal_arrays_are_supported(self):
        result = fat_fraction_summary(np.ones((2, 2, 2)), np.ones((2, 2, 2)), np.ones((2, 2, 2)))
        self.assertEqual(result['valid_count'], 8)
        self.assertEqual(result['ratio_of_sums'], 0.5)

class UnitTests(unittest.TestCase):

    def test_anisotropic_pixel_area(self):
        self.assertAlmostEqual(pixels_to_area_mm2(6, (2, 3)), 36)
        np.testing.assert_allclose(pixels_to_area_mm2([0, 5], (2, 3)), [0, 30])

    def test_volume_uses_cubic_millimetres_to_millilitres(self):
        self.assertAlmostEqual(area_to_volume_ml([100, 200], 5), 1.5)

    def test_invalid_spacing_and_counts_are_rejected(self):
        for spacing in ((0, 1), (np.nan, 1), (1,), (-1, 1)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                pixels_to_area_mm2(2, spacing)
        with self.assertRaises(ValueError):
            pixels_to_area_mm2(1.5, (1, 1))

    def test_missing_slice_areas_are_not_silently_summed(self):
        for areas in ([1, np.nan], [], [-1, 1]):
            with self.subTest(areas=areas), self.assertRaises(ValueError):
                area_to_volume_ml(areas, 2)
        with self.assertRaises(ValueError):
            area_to_volume_ml([1, 2], 0)

class TableTests(unittest.TestCase):

    def test_alignment_preserves_first_table_order_and_leading_zeros(self):
        first = pd.DataFrame({'sample_id': ['002', '001', '003'], 'x': [2, 1, 3]})
        second = pd.DataFrame({'sample_id': ['003', ' 002 '], 'y': [30, 20]})
        saved = second.copy(deep=True)
        a, b = align_tables(first, second, id_column='sample_id')
        self.assertEqual(a['sample_id'].tolist(), ['002', '003'])
        self.assertEqual(b['y'].tolist(), [20, 30])
        pd.testing.assert_frame_equal(second, saved)

    def test_alignment_supports_multiple_tables(self):
        tables = [pd.DataFrame({'key': values}) for values in (['b', 'a'], ['a', 'b'], ['b'])]
        for table in align_tables(*tables, id_column='key'):
            self.assertEqual(table['key'].tolist(), ['b'])

    def test_no_common_ids_returns_empty_tables(self):
        a, b = align_tables(pd.DataFrame({'sample_id': ['a']}), pd.DataFrame({'sample_id': ['b']}), id_column='sample_id')
        self.assertTrue(a.empty and b.empty)
        self.assertEqual(a.columns.tolist(), ['sample_id'])

    def test_invalid_ids_are_rejected(self):
        second = pd.DataFrame({'sample_id': ['a']})
        for ids in (['a', ' a '], ['a', None], ['a', ' ']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                align_tables(pd.DataFrame({'sample_id': ids}), second, id_column='sample_id')

    def test_column_selection_is_ordered_and_independent(self):
        table = pd.DataFrame({'a': [1], 'b': [2]})
        result = select_columns(table, ['b', 'a'])
        self.assertEqual(result.columns.tolist(), ['b', 'a'])
        result.loc[0, 'a'] = 10
        self.assertEqual(table.loc[0, 'a'], 1)
        with self.assertRaises(ValueError):
            select_columns(table, ['missing'])

class QualityTests(unittest.TestCase):

    def test_missingness_reports_counts_and_fractions(self):
        table = pd.DataFrame({'a': [1, None], 'b': [None, None]})
        result = missingness_summary(table)
        self.assertEqual(result.loc['a', 'missing_count'], 1)
        self.assertEqual(result.loc['a', 'missing_fraction'], 0.5)
        self.assertEqual(result.loc['b', 'missing_fraction'], 1)

    def test_empty_table_missingness_fraction_is_undefined(self):
        result = missingness_summary(pd.DataFrame({'a': pd.Series(dtype=float)}))
        self.assertEqual(result.loc['a', 'missing_count'], 0)
        self.assertTrue(np.isnan(result.loc['a', 'missing_fraction']))

    def test_fixed_reference_bounds_and_nullable_flags(self):
        bounds = fit_iqr_bounds(pd.DataFrame({'x': [0, 1, 2, 3, 4, np.inf]}), ['x'], factor=1)
        self.assertEqual(bounds.loc['x', 'valid_count'], 5)
        self.assertEqual(bounds.loc['x', 'lower'], -1)
        self.assertEqual(bounds.loc['x', 'upper'], 5)
        candidate = pd.DataFrame({'x': [-2, -1, 5, 6, np.nan, np.inf]})
        flags = flag_iqr_outliers(candidate, bounds, side='both')
        self.assertEqual(flags['x'].iloc[:4].tolist(), [True, False, False, True])
        self.assertTrue(flags['x'].iloc[4:].isna().all())
        upper = flag_iqr_outliers(candidate, bounds, side='upper')
        self.assertEqual(upper['x'].iloc[:4].tolist(), [False, False, False, True])

    def test_all_missing_reference_has_unknown_flags(self):
        bounds = fit_iqr_bounds(pd.DataFrame({'x': [np.nan, np.nan]}), ['x'], factor=1)
        flags = flag_iqr_outliers(pd.DataFrame({'x': [1.0, np.nan]}), bounds, side='both')
        self.assertTrue(flags['x'].isna().all())

    def test_constant_reference_has_zero_width_interval(self):
        bounds = fit_iqr_bounds(pd.DataFrame({'x': [2, 2, 2]}), ['x'], factor=1)
        flags = flag_iqr_outliers(pd.DataFrame({'x': [1, 2, 3]}), bounds, side='both')
        self.assertEqual(flags['x'].tolist(), [True, False, True])

    def test_nullable_numeric_columns_are_supported(self):
        table = pd.DataFrame({'x': pd.Series([1, 2, pd.NA], dtype='Int64')})
        bounds = fit_iqr_bounds(table, ['x'], factor=1)
        self.assertEqual(bounds.loc['x', 'valid_count'], 2)
        self.assertTrue(pd.isna(flag_iqr_outliers(table, bounds, side='both').loc[2, 'x']))

    def test_invalid_thresholds_and_nonnumeric_features_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_iqr_bounds(pd.DataFrame({'x': [1, 2]}), ['x'], factor=-1)
        with self.assertRaises(ValueError):
            fit_iqr_bounds(pd.DataFrame({'x': ['a', 'b']}), ['x'], factor=1)
        bounds = pd.DataFrame({'lower': [2.0], 'upper': [1.0]}, index=['x'])
        with self.assertRaises(ValueError):
            flag_iqr_outliers(pd.DataFrame({'x': [1]}), bounds, side='both')

    def test_duplicate_columns_are_rejected(self):
        table = pd.DataFrame([[1, 2]], columns=['x', 'x'])
        with self.assertRaises(ValueError):
            fit_iqr_bounds(table, ['x'], factor=1)
if __name__ == '__main__':
    unittest.main()
