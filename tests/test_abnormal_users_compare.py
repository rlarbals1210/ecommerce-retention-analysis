import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import duckdb
import pandas as pd

import config
from src.ab_design import build_risk_population, summarize_risk
from src.abnormal_users_compare import (
    cohort_cells, cohort_summary, collect_gaps, comparison_row, interval_summary,
    user_median_summary, verify_baseline, verify_membership,
)
from src.churn_features import gap_summary
from src.recency_reference import sha256_file


def make_activity(path):
    frame = pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 5],
        "date": pd.to_datetime([
            "2020-03-01", "2020-03-03", "2020-04-01", "2020-03-22",
            "2020-03-26", "2020-03-11", "2020-04-01", "2020-04-02",
            "2020-03-16", "2020-03-18", "2020-03-20",
        ], utc=True),
    })
    frame.to_parquet(path, index=False)
    return frame


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("config.risk_bounds", return_value=(14, 34))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_gap_exclusion_and_user_medians_across_batches(self):
        with TemporaryDirectory() as tmp, duckdb.connect() as db:
            path = Path(tmp) / "activity.parquet"
            make_activity(path)
            membership = pd.DataFrame({"user_id": [1], "bits": [1]})
            gaps = collect_gaps(db, path, membership, batch_rows=1)
        self.assertEqual(interval_summary(gaps["full"], 0)["gap_count"], 6)
        full = interval_summary(gaps["full"], 1)
        self.assertEqual(full["gap_count"], 4)
        for key, value in {"p75": 2.5, "p90": 3.4, "p95": 3.7}.items():
            self.assertAlmostEqual(full[key], value)
        self.assertEqual(interval_summary(gaps["pre"], 1)["gap_count"], 3)
        full_users = user_median_summary(gaps["full"], 0)
        self.assertEqual(full_users["contributing_users"], 4)
        self.assertEqual(full_users["one_gap_users"], 2)
        self.assertEqual(full_users["one_gap_share"], .5)
        self.assertAlmostEqual(full_users["p90"], 12.05)
        self.assertEqual(user_median_summary(gaps["pre"], 1),
                         {"contributing_users": 2, "one_gap_users": 1,
                          "one_gap_share": .5, "p90": 3.8})

    def test_no_exclusion_gaps_match_public_pandas_function(self):
        with TemporaryDirectory() as tmp, duckdb.connect() as db:
            path = Path(tmp) / "activity.parquet"
            frame = make_activity(path)
            empty = pd.DataFrame({"user_id": pd.Series(dtype="int64"),
                                  "bits": pd.Series(dtype="uint8")})
            gaps = collect_gaps(db, path, empty, batch_rows=3)
        pre = frame.loc[frame.date < pd.Timestamp(config.AB_FEATURE_END_EXCLUSIVE)]
        for name, activity in (("full", frame), ("pre", pre)):
            public = gap_summary(activity)
            self.assertEqual(interval_summary(gaps[name], 0),
                             {field: public[field] for field in ("gap_count", "p75", "p90", "p95")})

    def test_cohort_exclusion_and_power_match_public_calculation(self):
        ids = list(range(1, 11))
        labels = pd.DataFrame({"user_id": ids,
                               "status": ["retained", "retained", "churned", "churned",
                                          "retained", "churned", "retained", "retained",
                                          "churned", "censored"]})
        candidate = pd.DataFrame({"user_id": ids[:-1],
                                  "recency_at_ref": [14, 14, 34, 20, 17, 18, 7, 13, 35]})
        buyers = pd.DataFrame({"user_id": [1, 2, 3]})
        membership = pd.DataFrame({"user_id": [1, 8], "bits": [1, 1]})
        with TemporaryDirectory() as tmp, duckdb.connect() as db:
            paths = {}
            for name, frame in (("labels", labels), ("candidate", candidate), ("buyers", buyers)):
                paths[name] = Path(tmp) / f"{name}.parquet"
                frame.to_parquet(paths[name], index=False)
            cells, counts = cohort_cells(db, paths, membership)
        self.assertEqual(counts, {"S1": 2, "S2": 0, "S3": 0, "S4": 0})
        baseline, result = cohort_summary(cells, 0), cohort_summary(cells, 1)
        self.assertEqual(result["eligible_users"], 7)
        self.assertEqual(result["status_counts"], {"churned": 4, "retained": 3, "censored": 1})
        self.assertEqual(result["recency_bands"]["8-13"]["users"], 0)
        self.assertEqual(result["recency_bands"]["14-34"]["users"], 5)
        self.assertEqual(result["risk"]["buyer_before_reference"]["returned_users"], 1)
        self.assertEqual(result["risk"]["nonbuyer_before_reference"]["users"], 3)
        public = summarize_risk(build_risk_population(candidate, labels, buyers.user_id))
        self.assertEqual(list(baseline["risk"].values()), public.to_dict("records"))
        with self.assertRaisesRegex(ValueError, "No-exclusion reproduction mismatch"):
            verify_baseline(baseline, {})

    def test_comparison_rates_and_p75_use_no_exclusion_denominators(self):
        before = cohort_summary([("retained", 14, True, 0, 3), ("churned", 14, True, 0, 3)], 0)
        after = cohort_summary([("retained", 14, True, 0, 1), ("churned", 14, True, 0, 3)], 0)
        prior_intervals = {"pre": {"gap_count": 10, "p75": 13, "p90": 31, "p95": 48}}
        intervals = {"pre": {"gap_count": 7, "p75": 14, "p90": 31, "p95": 47}}
        row = comparison_row(after, intervals, {}, before, prior_intervals)
        self.assertEqual(row["cohort"]["inactivity_delta_pp"], 25)
        self.assertEqual(row["cohort"]["risk"]["overall"]["return_delta_pp"], -25)
        self.assertEqual(row["completed_gaps"]["pre"]["excluded_gap_share"], .3)
        self.assertEqual(row["completed_gaps"]["pre"]["delta_days"], {"p75": 1, "p90": 0, "p95": -1})
        self.assertTrue(row["pre_p75_changed"])

    def test_start_gate_rejects_hash_and_count_mismatch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            members = root / "scenario_membership.parquet"
            pd.DataFrame({"user_id": [1, 2], "S1": [True, False], "S2": [True, False],
                          "S3": [False, True], "S4": [True, True]}).to_parquet(members, index=False)
            scores, activity, module = (root / name for name in
                                        ("user_scores.parquet", "activity", "module"))
            for path in (scores, activity, module):
                path.write_bytes(b"fixture")
            expected = {"S1": 1, "S2": 1, "S3": 1, "S4": 2}
            report = root / "first.json"
            report.write_text(json.dumps({"scenarios": {key: {"users": n}
                                          for key, n in expected.items()}}))
            manifest = root / "membership_manifest.json"
            manifest.write_text(json.dumps({
                "files_sha256": {path.name: sha256_file(path) for path in (scores, members)},
                "activity_day_cache_sha256": sha256_file(activity),
                "module_sha256": sha256_file(module),
            }))
            args = (root, report, activity, module, expected)
            _, verified = verify_membership(*args)
            self.assertEqual(verified["first_stop_sha256"], sha256_file(report))
            report.write_text(json.dumps({"scenarios": {key: {"users": 3} for key in expected}}))
            with self.assertRaisesRegex(ValueError, "candidate count mismatch"):
                verify_membership(*args)
            for path in (scores, members, activity, module):
                with self.subTest(path=path.name):
                    original = path.read_bytes()
                    path.write_bytes(original + b"changed")
                    with self.assertRaisesRegex(ValueError, "hash mismatch"):
                        verify_membership(*args)
                    path.write_bytes(original)
