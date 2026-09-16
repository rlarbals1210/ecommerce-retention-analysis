import unittest
import pandas as pd
from src.recency_segments import with_totals, check_risk_alignment, comparison_markdown


class SegmentTest(unittest.TestCase):
    def test_weighted_total_and_alignment(self):
        rows = pd.DataFrame([
            dict(recency_band='15-29', is_buyer_at_reference=True, users=2, returned_users=1, baseline=.5),
            dict(recency_band='15-29', is_buyer_at_reference=False, users=8, returned_users=1, baseline=.125)])
        table = with_totals(rows)
        self.assertEqual(table.iloc[0].baseline, .2)
        summary = table.drop(columns='recency_band')
        check_risk_alignment(table, summary)
        summary.loc[0, 'users'] += 1
        with self.assertRaises(ValueError):
            check_risk_alignment(table, summary)

    def test_zero_denominator_and_dormant_label(self):
        rows = pd.DataFrame([dict(recency_band=b, is_buyer_at_reference=s, users=0, returned_users=0, baseline=None)
                             for b in ['0-7', '8-14', '15-29', '30+'] for s in [False, True]])
        table = with_totals(rows)
        self.assertTrue(table.baseline.isna().all())
        rendered = comparison_markdown(table)
        self.assertIn('휴면 기준 충족군의 복귀', rendered)
        self.assertIn('해당 없음', rendered)


if __name__ == '__main__':
    unittest.main()
