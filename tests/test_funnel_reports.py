import unittest
import pandas as pd
from src.funnel_reports import summarize


class SessionSetTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame([
            ['a', 100, 20, 15, 10], ['b', 900, 90, 80, 45]
        ], columns=['month', 'total_sessions', 'cart_sessions', 'purchase_sessions', 'cart_and_purchase'])

    def test_weighted_totals_and_intersection(self):
        monthly, overall = summarize(self.data)
        self.assertAlmostEqual(overall.iloc[1].pct_of_total_sessions, .11)
        self.assertAlmostEqual(overall.iloc[2].pct_of_total_sessions, .055)
        self.assertAlmostEqual(overall.iloc[2].share_of_parent_set, .5)
        self.assertEqual(monthly.purchase_no_cart.sum(), 40)
        # All purchases (95) must not replace the intersection (55).
        self.assertEqual(overall.iloc[2].sessions, 55)

    def test_reject_impossible_intersection_and_duplicate_month(self):
        invalid = self.data.copy()
        invalid.loc[0, 'cart_and_purchase'] = 16
        with self.assertRaises(ValueError):
            summarize(invalid)
        with self.assertRaises(ValueError):
            summarize(pd.concat([self.data, self.data]))

    def test_recompute_rates_instead_of_trusting_stale_export(self):
        self.data['cart_share_of_sessions'] = .99
        monthly, _ = summarize(self.data)
        self.assertAlmostEqual(monthly.iloc[0].cart_share_of_sessions, .2)
        self.assertAlmostEqual(monthly.iloc[1].cart_share_of_sessions, .1)


if __name__ == '__main__':
    unittest.main()
