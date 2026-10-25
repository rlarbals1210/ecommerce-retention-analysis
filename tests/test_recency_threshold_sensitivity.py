"""Boundary, population and statistical checks for the added sensitivity analysis."""
import math
import unittest

from src.recency_threshold_sensitivity import analyze, counts, validate_distribution


def cell(day, n=100, active=False, buyer=False):
    return dict(recency_at_ref=day, users=n, april_active=active,
                is_buyer_at_reference=buyer, recency_state="valid")


class ThresholdSensitivityTest(unittest.TestCase):
    def test_inclusive_boundaries_and_nested_partition(self):
        rows = [cell(day, day+1) for day in [0, 24, 25, 29, 30, 34, 35, 39, 40, 182]]
        self.assertEqual(counts(rows, 25, 29), (26+30, 0))
        self.assertEqual(counts(rows, 30, 34), (31+35, 0))
        self.assertEqual(counts(rows, 35, 39), (36+40, 0))
        self.assertEqual(counts(rows, 30)[0], counts(rows, 30, 34)[0]+counts(rows, 35)[0])

    def test_detects_invalid_and_duplicate_cells_and_lost_population(self):
        good = cell(30)
        validate_distribution([good], 100)
        bad_values = [dict(good, recency_at_ref=30.5), dict(good, recency_at_ref=-1),
                      dict(good, recency_at_ref=183), dict(good, users=-1),
                      dict(good, april_active=1), dict(good, recency_state="negative")]
        for bad in bad_values:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_distribution([bad], 100)
        with self.assertRaises(ValueError):
            validate_distribution([good, good], 200)
        with self.assertRaises(ValueError):
            validate_distribution([good], 101)

    def test_known_difference_and_equal_rate_control(self):
        rows = [cell(25, 100), cell(25, 100, True), cell(30, 800), cell(30, 200, True),
                cell(35, 900), cell(35, 100, True)]
        bands, comparisons, thresholds = analyze(rows)
        primary = comparisons[0]
        self.assertAlmostEqual(primary['difference_pp'], 10)
        self.assertGreater(primary['ci95_low_pp'], 0)
        self.assertTrue(primary['nominal_significant'])
        self.assertEqual(thresholds[1]['removed_vs_30'], 1000)
        self.assertEqual(thresholds[1]['removed_share_of_30'], 0.5)
        rows = [cell(day, n, active) for day in [25, 30, 35]
                for n, active in [(800, False), (200, True)]]
        primary = analyze(rows)[1][0]
        self.assertAlmostEqual(primary['difference_pp'], 0)
        self.assertAlmostEqual(primary['p_value'], 1)
        self.assertLess(primary['ci95_low_pp'], 0)
        self.assertGreater(primary['ci95_high_pp'], 0)

    def test_extreme_p_value_is_not_reported_as_exact_zero(self):
        rows = [cell(day, n, active) for day in [25, 30, 35]
                for n, active in ([(800_000, False), (200_000, True)] if day != 35
                                  else [(900_000, False), (100_000, True)])]
        primary = analyze(rows)[1][0]
        self.assertEqual(primary['p_value'], '<1e-300')
        self.assertTrue(math.isfinite(primary['log10_p_value']))
        self.assertLess(primary['log10_p_value'], -300)


if __name__ == '__main__':
    unittest.main()
