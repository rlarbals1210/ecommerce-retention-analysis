import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ab_design import build_risk_population, purchase_users_before_reference, summarize_risk


class ReferenceStrataTest(unittest.TestCase):
    def scan(self, rows, chunksize=2):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "events.csv"
            pd.DataFrame(rows, columns=["user_id", "event_type", "event_time"]).to_csv(p, index=False)
            return purchase_users_before_reference([p], "2020-03-31", chunksize)

    def test_full_utc_day_and_timezone_boundary(self):
        rows = [(1, "purchase", "2020-03-31 23:59:59 UTC"),
                (2, "purchase", "2020-04-01 00:00:00 UTC"),
                (3, "view", "2020-03-20 12:00:00 UTC"),
                (4, "purchase", "2020-04-01T08:59:59+09:00"),
                (5, "purchase", "2020-04-01T09:00:00+09:00")]
        self.assertEqual(self.scan(rows).tolist(), [1, 4])

    def test_future_purchase_mutation_cannot_change_strata(self):
        past = [(1, "purchase", "2020-03-01 12:00:00 UTC"),
                (2, "view", "2020-03-02 12:00:00 UTC")]
        future = [(2, "purchase", "2020-04-01 00:00:00 UTC"),
                  (999, "purchase", "2020-04-30 23:59:59 UTC")]
        ids = self.scan(past)
        self.assertEqual(ids.tolist(), self.scan(past + future).tolist())
        changed = [(u + 500, t, d) for u, t, d in future]
        self.assertEqual(ids.tolist(), self.scan(past + changed).tolist())
        r = pd.DataFrame({"user_id": [1, 2], "recency_at_ref": [15, 29]})
        y = pd.DataFrame({"user_id": [1, 2], "status": ["churned", "retained"]})
        self.assertEqual(build_risk_population(r, y, ids)["is_buyer_at_reference"].tolist(), [True, False])

    def test_chunks_and_duplicates_do_not_change_membership(self):
        rows = [(1, "purchase", "2020-03-01 12:00:00 UTC")] * 3
        self.assertEqual(self.scan(rows, 1).tolist(), self.scan(rows, 5).tolist())
        self.assertEqual(self.scan(rows).tolist(), [1])

    def test_risk_boundary_and_missing_outcomes(self):
        r = pd.DataFrame({"user_id": [1, 2, 3, 4], "recency_at_ref": [14, 15, 29, 30]})
        y = pd.DataFrame({"user_id": [1, 2, 3, 4], "status": ["retained", "churned", "retained", "churned"]})
        self.assertEqual(build_risk_population(r, y, [3])["user_id"].tolist(), [2, 3])
        with self.assertRaises(ValueError):
            build_risk_population(r, y.loc[y["user_id"].ne(2)], [3])
        y.loc[y["user_id"].eq(3), "status"] = "censored"
        with self.assertRaises(ValueError):
            build_risk_population(r, y, [3])

    def test_duplicate_users_are_rejected(self):
        r = pd.DataFrame({"user_id": [1, 1], "recency_at_ref": [15, 15]})
        y = pd.DataFrame({"user_id": [1], "status": ["retained"]})
        with self.assertRaises(ValueError):
            build_risk_population(r, y, [1])

    def test_summary_counts_and_known_power_calculation(self):
        # Rounded 25% baseline has a known two-sided +5pp design of 1,250 per arm.
        risk = pd.DataFrame({"status": ["retained", "churned", "churned", "churned"] * 2,
                             "is_buyer_at_reference": [True] * 4 + [False] * 4})
        summary = summarize_risk(risk).set_index("segment")
        self.assertEqual(summary.loc["overall", "users"], 8)
        self.assertEqual(summary.loc["overall", "returned_users"], 2)
        self.assertEqual(summary.loc["overall", "n_per_arm"], 1250)
        self.assertEqual(summary.iloc[1:]["users"].sum(), summary.loc["overall", "users"])
        self.assertEqual(summary.iloc[1:]["returned_users"].sum(), summary.loc["overall", "returned_users"])


if __name__ == "__main__":
    unittest.main()
