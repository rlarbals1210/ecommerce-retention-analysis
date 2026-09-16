import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.recency_reference import (
    independent_sample_recency, prepare_sample, sample_digest, scan_prepared,
    scan_recency, select_sample,
)


class RecencyReferenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def source(self, rows, name="events.csv"):
        path = self.root / name
        pd.DataFrame(rows, columns=["user_id", "event_time"]).to_csv(path, index=False)
        return path

    def scan(self, rows, chunksize=2, sample_ids=()):
        return scan_recency([self.source(rows)], "2020-03-31", sample_ids, chunksize)

    def labels(self):
        labels = pd.DataFrame({"user_id": [5, 2, 3, 1, 4],
                               "status": ["censored", "retained", "churned", "retained", "churned"]})
        path = self.root / "labels.parquet"
        labels.to_parquet(path, index=False)
        return path

    def test_utc_boundary_and_future_only_exclusion(self):
        result, audit, _ = self.scan([
            (1, "2020-03-31 23:59:59 UTC"), (1, "2020-04-01 00:00:00 UTC"),
            (2, "2020-04-01 00:00:00 UTC"), (3, "2020-04-30 12:00:00 UTC"),
            (4, "2020-04-01T08:59:59+09:00"), (5, "2020-04-01T09:00:00+09:00"),
            (6, "2020-03-31T23:00:00-02:00"), (7, "2019-10-01 00:00:00 UTC"),
        ])
        self.assertEqual(result.set_index("user_id")["recency_at_ref"].to_dict(), {1: 0, 4: 0, 7: 182})
        self.assertEqual(audit["cutoff_excluded_rows"], 5)
        self.assertEqual(audit["parse_errors"], 0)

    def test_order_chunks_duplicates_and_files_are_invariant(self):
        rows = [(u, f"2020-03-{day:02} 12:00:00 UTC")
                for u in range(1, 8) for day in [1, 3, 15, 31]]
        expected = self.scan(rows, 100)[0]
        shuffled = pd.DataFrame(rows * 2).sample(frac=1, random_state=18).to_records(index=False).tolist()
        for chunk in [1, 2, 3, 7, 17, 100]:
            with self.subTest(chunk=chunk):
                pd.testing.assert_frame_equal(self.scan(shuffled, chunk)[0], expected)
        paths = [self.source(rows[:13], "one.csv"), self.source(rows[13:], "two.csv")]
        pd.testing.assert_frame_equal(scan_recency(paths[::-1], "2020-03-31", chunksize=3)[0], expected)

    def test_matches_historical_activity_day_algorithm_without_out_of_range_rows(self):
        rows = [(3, "2019-10-01 10:00:00 UTC"), (1, "2020-02-29 00:00:00 UTC"),
                (1, "2020-03-20 23:59:59 UTC"), (1, "2020-03-20 23:59:59 UTC"),
                (2, "2020-03-31 12:00:00 UTC"), (3, "2019-11-15 00:00:00 UTC")]
        # Independent literal reference to the historical cell, including deduplication.
        data = pd.DataFrame(rows, columns=["user_id", "event_time"])
        data["date"] = pd.to_datetime(data["event_time"], utc=True).dt.normalize()
        daily = data[["user_id", "date"]].drop_duplicates()
        last = daily.groupby("user_id")["date"].max()
        expected = (pd.Timestamp("2020-03-31", tz="UTC") - last).dt.days.rename("recency_at_ref").reset_index()
        pd.testing.assert_frame_equal(self.scan(rows, 2)[0], expected)

    def test_pre_filter_diagnostics_include_invalid_and_excluded_rows(self):
        result, audit, events = self.scan([
            (1, "2020-03-01 00:00:00 UTC"), (1, "not-a-timestamp"),
            (1, None), (1, "2020-04-02 00:00:00 UTC"),
            (2, "2019-09-30 23:59:59 UTC"),
        ], sample_ids=[1])
        diag = audit["files"][0]
        self.assertEqual(diag["rows"], 5)
        self.assertEqual(diag["parse_errors"], 2)
        self.assertEqual(diag["cutoff_excluded_rows"], 1)
        self.assertEqual(diag["included_rows"], 2)
        self.assertEqual(diag["rows"], diag["parse_errors"] + diag["cutoff_excluded_rows"] + diag["included_rows"])
        self.assertTrue(diag["min_time_utc"].startswith("2019-09-30"))
        self.assertTrue(diag["max_time_utc"].startswith("2020-04-02"))
        self.assertEqual(len(events), 4)  # Bad/future sampled rows remain in the evidence.
        self.assertEqual(events["source_row"].tolist(), [1, 2, 3, 4])
        self.assertEqual(result.set_index("user_id").loc[2, "recency_at_ref"], 183)
        self.assertEqual(audit["state"], "candidate_only_not_validated")

    def test_empty_pre_reference_population(self):
        result, audit, _ = self.scan([(1, "2020-04-01 00:00:00 UTC")])
        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), ["user_id", "recency_at_ref"])
        self.assertEqual(audit["cutoff_excluded_rows"], 1)

    def test_sample_is_fixed_order_independent_and_eligible_only(self):
        labels = pd.read_parquet(self.labels())
        first, count = select_sample(labels, size=3, seed=42)
        second, _ = select_sample(labels.iloc[::-1], size=3, seed=42)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(count, 4)
        self.assertEqual(len(first), 3)
        self.assertNotIn(5, first["user_id"].tolist())
        self.assertEqual(sample_digest(first["user_id"]), sample_digest(second["user_id"]))

    def test_invalid_label_populations_are_rejected(self):
        labels = pd.read_parquet(self.labels())
        for bad in [pd.concat([labels, labels.iloc[:1]], ignore_index=True),
                    labels.assign(user_id=[1, 2, 3, 4, None]),
                    labels.assign(status=["unknown"] * 5)]:
            with self.assertRaises(ValueError):
                select_sample(bad, size=2)
        with self.assertRaises(ValueError):
            select_sample(labels, size=5)

    def test_preparation_does_not_read_raw_and_refuses_overwrite(self):
        directory = self.root / "run"
        with patch("src.recency_reference.pd.read_csv", side_effect=AssertionError("Raw scan forbidden")):
            manifest = prepare_sample(self.labels(), directory, size=3)
        self.assertEqual(manifest["state"], "sample_prepared_raw_scan_not_started")
        self.assertEqual(sorted(p.name for p in directory.iterdir()), ["preparation.json", "sample_users.parquet"])
        original = (directory / "preparation.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_sample(self.labels(), directory, size=3)
        self.assertEqual((directory / "preparation.json").read_bytes(), original)

    def test_independent_sample_oracle_and_missing_sample_failure(self):
        rows = [(1, "2020-03-01 00:00:00 UTC"), (1, "2020-03-31T15:00:00+09:00"),
                (1, "2020-04-01 00:00:00 UTC"), (2, "2020-03-02 20:00:00 UTC"),
                (2, "2020-03-10 20:00:00 UTC"), (3, "2020-03-31 00:00:00 UTC")]
        candidate, _, events = self.scan(rows, 1, [1, 2])
        with patch("src.recency_reference.utc_reference", side_effect=AssertionError("Production helper forbidden")), \
                patch("src.recency_reference.pd.to_datetime", side_effect=AssertionError("Production parser forbidden")):
            actual = independent_sample_recency(events, [1, 2], "2020-03-31")
        self.assertEqual(actual, candidate.set_index("user_id")["recency_at_ref"].loc[[1, 2]].to_dict())
        with self.assertRaises(ValueError):
            independent_sample_recency(events, [1, 2, 3], "2020-03-31")
        invalid = pd.DataFrame({"user_id": [1], "event_time": ["bad"]})
        with self.assertRaises(ValueError):
            independent_sample_recency(invalid, [1], "2020-03-31")

    def test_explicit_scan_writes_only_candidate_and_preserves_existing_outputs(self):
        directory = self.root / "run"
        prepare_sample(self.labels(), directory, size=3)
        source = self.source([(i, "2020-03-20 00:00:00 UTC") for i in range(1, 5)])
        with patch("src.recency_reference.config.RAW_PATHS", [source]):
            audit = scan_prepared(directory, chunksize=1)
            self.assertEqual(audit["state"], "candidate_only_not_validated")
            with self.assertRaises(FileExistsError):
                scan_prepared(directory, chunksize=1)
        self.assertTrue((directory / "recency_at_reference.candidate.parquet").is_file())
        self.assertFalse((directory / "recency_at_reference.parquet").exists())

    def test_tampered_sample_is_rejected_before_scan(self):
        directory = self.root / "run"
        prepare_sample(self.labels(), directory, size=3)
        pd.DataFrame({"user_id": [999, 998, 997]}).to_parquet(directory / "sample_users.parquet", index=False)
        with patch("src.recency_reference.scan_recency", side_effect=AssertionError("Must not scan")):
            with self.assertRaises(ValueError):
                scan_prepared(directory)

    def test_invalid_scan_arguments(self):
        with self.assertRaises(ValueError):
            scan_recency([], "2020-03-31")
        path = self.source([(1, "2020-03-01 00:00:00 UTC")])
        with self.assertRaises(ValueError):
            scan_recency([path, path], "2020-03-31")
        with self.assertRaises(ValueError):
            scan_recency([path], "2020-03-31", chunksize=0)


if __name__ == "__main__":
    unittest.main()
