from unittest.mock import patch
import unittest
import json
import tempfile
from pathlib import Path
import pandas as pd
from src.recency_segments import with_totals, check_risk_alignment, comparison_markdown, load_comparison


class SegmentTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("config.risk_bounds", return_value=(14, 34))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pending_independent_review_cannot_load_analysis_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'validation.json'
            path.write_text(json.dumps({'decision_status':'pending_independent_review'}))
            with patch('config.RECENCY_VALIDATION_PATH',path), patch('src.recency_segments.pd.read_parquet', side_effect=AssertionError('No data read before independent approval')):
                with self.assertRaises(ValueError):load_comparison()

    def test_weighted_total_and_alignment(self):
        rows = pd.DataFrame([
            dict(recency_band='14-34', is_buyer_at_reference=True, users=2, returned_users=1, baseline=.5),
            dict(recency_band='14-34', is_buyer_at_reference=False, users=8, returned_users=1, baseline=.125)])
        table = with_totals(rows)
        self.assertEqual(table.iloc[0].baseline, .2)
        summary = table.drop(columns='recency_band')
        check_risk_alignment(table, summary)
        summary.loc[0, 'users'] += 1
        with self.assertRaises(ValueError):
            check_risk_alignment(table, summary)

    def test_zero_denominator_and_dormant_label(self):
        rows = pd.DataFrame([dict(recency_band=b, is_buyer_at_reference=s, users=0, returned_users=0, baseline=None)
                             for b in ['0-7', '8-13', '14-34', '35+'] for s in [False, True]])
        table = with_totals(rows)
        self.assertTrue(table.baseline.isna().all())
        rendered = comparison_markdown(table)
        self.assertIn('휴면 기준 충족군의 복귀', rendered)
        self.assertIn('해당 없음', rendered)


if __name__ == '__main__':
    unittest.main()
