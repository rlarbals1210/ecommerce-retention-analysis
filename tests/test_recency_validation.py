import copy
import unittest

import pandas as pd

import config
from src.recency_validation import daily_distribution, interval_counts, risk_summary, validate_frames, published_matches
from src.ab_design import build_risk_population, summarize_risk


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.old = pd.DataFrame({'user_id': range(1, 9), 'recency_at_ref': [8, 9, 15, 20, 21, 22, 0, 182]})
        self.labels = pd.DataFrame({'user_id': range(1, 9), 'status': ['retained', 'retained', 'retained', 'churned', 'retained', 'churned', 'churned', 'retained']})
        self.buyers = pd.DataFrame({'user_id': [3, 4], 'reference_date': [config.AB_REFERENCE_DATE] * 2})
        self.audit = {'parse_errors': 0, 'cutoff_excluded_rows': 0, 'input_changed': False,
                      'files': [{'rows': 8, 'included_rows': 8, 'parse_errors': 0, 'cutoff_excluded_rows': 0, 'valid_timestamp_rows': 8, 'input_unchanged_during_scan': True}]}

    def validate(self, candidate, audit=None, synthetic=True):
        events = pd.DataFrame({'user_id': candidate.user_id,
                               'event_time': [(pd.Timestamp('2020-03-31', tz='UTC') - pd.Timedelta(days=int(d))).isoformat() for d in candidate.recency_at_ref]})
        return validate_frames(self.old, candidate, self.labels, self.buyers,
                               candidate.user_id, events, audit or self.audit, synthetic_passed=synthetic)

    def test_equal_distribution_with_user_swaps(self):
        candidate = self.old.copy()
        candidate.loc[0:1, 'recency_at_ref'] = [9, 8]
        report = self.validate(candidate)
        self.assertEqual(report['decision_branch'], 1)
        self.assertEqual(report['user_comparison']['different_users'], 2)

    def test_day_change_inside_band_is_branch_two(self):
        candidate = self.old.copy()
        candidate.loc[0, 'recency_at_ref'] = 10
        self.assertEqual(self.validate(candidate)['decision_branch'], 2)

    def test_cross_band_outside_risk_is_branch_three(self):
        candidate = self.old.copy()
        candidate.loc[0, 'recency_at_ref'] = 30
        self.assertEqual(self.validate(candidate)['decision_branch'], 3)

    def test_synthetic_failure_holds_even_when_equal(self):
        self.assertEqual(self.validate(self.old, synthetic=False)['decision_branch'], 4)

    def test_diagnostic_failure_holds(self):
        audit = copy.deepcopy(self.audit)
        audit['input_changed'] = True
        self.assertEqual(self.validate(self.old, audit)['decision_branch'], 4)

    def test_invalid_candidate_holds(self):
        candidate = self.old.copy()
        candidate.loc[0, 'recency_at_ref'] = 183
        self.assertEqual(self.validate(candidate)['decision_branch'], 4)

    def test_invalid_old_rows_not_silently_lost(self):
        old = self.old.astype({'recency_at_ref': float})
        old.loc[0:3, 'recency_at_ref'] = [float('nan'), -1, 0.5, 183]
        dist = daily_distribution(old, self.labels, self.buyers.user_id, 182)
        self.assertEqual(sum(r['users'] for r in dist), 8)
        self.assertEqual(sum(r['users'] for r in dist if r['recency_state'] != 'valid'), 4)

    def test_weighted_overall_not_average_of_rates(self):
        intervals = [{'recency_band': '15-29', 'is_buyer_at_reference': True, 'users': 2, 'returned_users': 1},
                     {'recency_band': '15-29', 'is_buyer_at_reference': False, 'users': 8, 'returned_users': 1}]
        summary = risk_summary(intervals)
        self.assertEqual(summary[0]['baseline'], 0.2)

    def test_band_boundaries(self):
        dist = [{'recency_at_ref': d, 'is_buyer_at_reference': False, 'april_active': True, 'users': 1, 'recency_state': 'valid'} for d in [0, 7, 8, 14, 15, 29, 30, 182]]
        self.assertEqual([r['users'] for r in interval_counts(dist) if not r['is_buyer_at_reference']], [2, 2, 2, 2])

    def test_legacy_summary_and_published_counts(self):
        result = self.validate(self.old)['risk_summaries']['candidate']
        legacy = summarize_risk(build_risk_population(self.old, self.labels, self.buyers.user_id))
        self.assertTrue(published_matches(result, legacy))
        legacy.loc[0, 'returned_users'] += 1
        self.assertFalse(published_matches(result, legacy))

    def test_sample_disagreement_holds(self):
        events = pd.DataFrame({'user_id': [1], 'event_time': ['2020-03-31T00:00:00+00:00']})
        result = validate_frames(self.old, self.old, self.labels, self.buyers, [1], events, self.audit, synthetic_passed=True)
        self.assertEqual(result['decision_branch'], 4)
        self.assertEqual(result['checks']['c_sample']['mismatched_users'], 1)

    def test_population_mismatch_holds(self):
        candidate = self.old.iloc[:-1].copy()
        self.assertEqual(self.validate(candidate)['decision_branch'], 4)


if __name__ == '__main__':
    unittest.main()
