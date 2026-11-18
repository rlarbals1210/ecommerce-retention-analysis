import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import config
from src.load import load_month
from src.churn_features import label_activity_days, gap_summary, pre_reference_evidence

class ChurnFeaturesTest(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            pd.DataFrame(rows, columns=["user_id", "event_time"]).to_csv(path, index=False)
            return load_month(path, usecols=["user_id", "event_time"], utc=True)

    def test_actual_04_loader_utc_boundary_and_labels(self):
        raw = self.load([(1,"2020-03-26 23:59:59 UTC"),(1,"2020-03-27 00:00:00 UTC"),
                         (2,"2020-03-27T08:59:59+09:00"),(3,"2020-03-27T09:00:00+09:00"),
                         (4,"2020-04-30 23:59:59 UTC")])
        raw["date"] = raw.event_time.dt.normalize()
        actual = label_activity_days(raw).status.to_dict()
        self.assertEqual(actual, {1:"retained",2:"churned",3:"censored",4:"censored"})
        self.assertEqual(config.MAX_RECENCY_DAYS, 177)
        self.assertEqual(config.AB_FEATURE_END_EXCLUSIVE, "2020-03-27T00:00:00Z")
        self.assertEqual(config.AB_OUTCOME_END_EXCLUSIVE, "2020-05-01T00:00:00Z")
        self.assertEqual((pd.Timestamp(config.AB_OUTCOME_END_EXCLUSIVE)-pd.Timestamp(config.AB_FEATURE_END_EXCLUSIVE)).days,35)

    def test_loader_rejects_invalid_missing_and_end_mismatch(self):
        for value in [None,"bad"]:
            with self.assertRaises((ValueError, TypeError)):
                self.load([(1,value)])
        raw=self.load([(1,"2020-05-01 00:00:00 UTC")]);raw["date"]=raw.event_time.dt.normalize()
        with self.assertRaises(ValueError):label_activity_days(raw)

    def test_pre_gaps_exclude_future_events_and_use_completed_intervals(self):
        daily=pd.DataFrame({"user_id":[1,1,1,2,2,3],"date":pd.to_datetime(["2020-03-01","2020-03-11","2020-03-30","2020-03-10","2020-03-26","2020-04-30"],utc=True)})
        full=gap_summary(daily)
        evidence=pre_reference_evidence(daily,full)
        self.assertEqual(evidence["pre_reference"]["gap_count"],2)
        self.assertEqual(evidence["pre_reference"]["contributing_users"],2)
        self.assertEqual(evidence["pre_reference"]["p75"],14.5)
        self.assertEqual(evidence["proposed_risk_lower_days"],15.5)
        self.assertTrue(evidence["existing_bands_require_decision"])
        self.assertEqual(evidence["review_status"],"pending_user_confirmation_stage3")
        self.assertEqual(full["gap_count"],3)

    def test_approval_requires_unchanged_evidence_and_matching_bounds(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"evidence.json"
            approval_path = Path(tmp)/"approval.json"
            evidence = {"run_id":config.RUN_ID,"reference_date":config.AB_REFERENCE_DATE,
                        "review_status":"pending_user_confirmation_stage3","pre_reference":{"p75":13}}
            original = json.dumps(evidence)
            path.write_text(original)
            approval = {"run_id":config.RUN_ID,"reference_date":config.AB_REFERENCE_DATE,
                        "status":"approved_by_user","evidence_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                        "decision":{"lower_days":14,"upper_days":34,"n_days":35,"recency_bands":["0-7","8-13","14-34","35+"]}}
            with patch("config.RISK_BASIS_PATH",path), patch("config.RISK_APPROVAL_PATH",approval_path):
                with self.assertRaises(FileNotFoundError):config.risk_bounds()
                approval_path.write_text(json.dumps(approval))
                self.assertEqual(config.risk_bounds(),(14,34))
                self.assertEqual(config.recency_bands(),("0-7","8-13","14-34","35+"))
                self.assertEqual(path.read_text(),original)
                path.write_text(original+" ")
                with self.assertRaises(ValueError):config.risk_bounds()
                path.write_text(original)
                approval["decision"]["lower_days"]=15;approval_path.write_text(json.dumps(approval))
                with self.assertRaises(ValueError):config.risk_bounds()
                approval["decision"]["lower_days"]=14;approval["decision"]["recency_bands"][1]="8-14"
                approval_path.write_text(json.dumps(approval))
                with self.assertRaises(ValueError):config.risk_bounds()
