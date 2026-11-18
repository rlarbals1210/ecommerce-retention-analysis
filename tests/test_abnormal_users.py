import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.abnormal_users import (
    ScanRejected, aligned_indices, duck_connection, scan_month, score_scenarios,
    validate_population,
)


class AbnormalUsersTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.db = duck_connection(self.root / "spill")
        self.addCleanup(self.db.close)

    def fixture(self, *lines):
        path = self.root / "2020-Mar.csv"
        path.write_text("user_id,event_time\n" + "\n".join(lines) + "\n")
        return path

    def scan(self, path):
        results = []
        diag = scan_month(path, "2020-03", self.root / "month", self.db,
                          lambda ids, counts, peaks: results.extend(zip(ids, counts, peaks)),
                          chunksize=1)
        return sorted(results), diag

    def test_split_minute_and_utc_cutoff(self):
        data = self.fixture(
            "1,2020-03-26 23:59:01 UTC",
            "2,2020-03-26 23:58:59 UTC",
            "1,2020-03-27T08:59:59+09:00",
            "1,2020-03-27 00:00:00 UTC",
        )
        rows, diag = self.scan(data)
        self.assertEqual(rows, [(1, 2, 2), (2, 1, 1)])
        self.assertEqual(diag["raw_rows"], 4)
        self.assertEqual(diag["cutoff_excluded_rows"], 1)
        self.assertEqual(diag["max_time_utc"], "2020-03-27T00:00:00+00:00")

    def test_month_boundary_in_middle_is_rejected_before_reduction(self):
        data = self.fixture("1,2020-03-01 00:00:00 UTC",
                            "1,2020-04-01 00:00:00 UTC",
                            "1,2020-03-31 23:59:59 UTC")
        with self.assertRaises(ScanRejected) as rejected:
            self.scan(data)
        self.assertEqual(rejected.exception.diagnostic["outside_month_rows"], 1)
        self.assertEqual(rejected.exception.diagnostic["max_time_utc"], "2020-04-01T00:00:00+00:00")

    def test_malformed_and_missing_times_are_rejected(self):
        data = self.fixture("1,bad", "1,", "1,2020-03-26 23:59:59 UTC")
        with self.assertRaises(ScanRejected) as rejected:
            self.scan(data)
        self.assertEqual(rejected.exception.diagnostic["parse_errors"], 2)

    def test_strict_threshold_with_ties_and_union(self):
        ratios = np.array([1.0, 1.0, 2.0, 3.0])
        peaks = np.array([1, 2, 9, 9])
        out, masks = score_scenarios(ratios, peaks, np.array([1, 2, 4, 9]))
        self.assertEqual(out["scenarios"]["S1"]["threshold"], 2.9699999999999998)
        self.assertEqual(out["scenarios"]["S3"]["threshold"], 9.0)
        self.assertEqual(out["scenarios"]["S3"]["ties"], 2)
        self.assertEqual(out["scenarios"]["S3"]["users"], 0)
        self.assertEqual(masks["S4"].tolist(), [False, False, False, True])
        self.assertEqual(out["scenarios"]["S4"]["event_rows"], 9)

    def test_zero_denominator_and_both_directions_of_population_mismatch(self):
        with self.assertRaisesRegex(ValueError, "zero"):
            validate_population(np.array([0, 2]), np.array([2, 3]))
        with self.assertRaisesRegex(ValueError, "absent"):
            validate_population(np.array([1, 2]), np.array([2, 0]))
        with self.assertRaisesRegex(ValueError, "extra"):
            aligned_indices(np.array([1, 2]), np.array([0, 3]))
        self.assertEqual(aligned_indices(np.array([1, 2]), np.array([2, 1])).tolist(), [1, 0])
