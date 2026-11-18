import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import config
from src.ab_design import load_risk_population, main
from src.recency_segments import load_comparison
from src.recency_validation import daily_distribution


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IndependentReviewTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.review = root / "independent_review_record.json"
        self.bundle = root / "independent_review_bundle.json"
        self.validation = root / "recency_reference_validation.json"
        self.buyers = root / "purchase_users_at_reference.parquet"
        self.candidate = root / "candidate.parquet"
        labels = root / "churn_labels.parquet"
        approval, basis = root / "approval.json", root / "basis.json"
        provenance = root / "purchase_cache_provenance.json"
        patcher = patch.multiple(config, PROJECT_ROOT=root, RUN_REPORTS_DIR=root, RUN_PROC_DIR=root,
                                 RECENCY_VALIDATION_PATH=self.validation, RECENCY_CANDIDATE_PATH=self.candidate,
                                 RISK_APPROVAL_PATH=approval, RISK_BASIS_PATH=basis)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("config.risk_bounds", return_value=(14, 34))
        patcher.start()
        self.addCleanup(patcher.stop)

        r = pd.DataFrame({"user_id": [1, 2], "recency_at_ref": [14, 34]})
        y = pd.DataFrame({"user_id": [1, 2], "status": pd.Categorical(["retained", "churned"])})
        r.to_parquet(self.candidate, index=False)
        y.to_parquet(labels, index=False)
        pd.DataFrame({"user_id": [1], "reference_date": config.AB_REFERENCE_DATE}).to_parquet(self.buyers, index=False)
        for p in (approval, basis):
            p.write_text("{}")
        provenance.write_text(json.dumps({"buyers": 1, "source_files": []}))
        pd.DataFrame({"user_id": [1, 2], "is_buyer": [True, False]}).to_parquet(root / "user_features.parquet")
        identity = {"run_id": config.RUN_ID, "reference_date": config.AB_REFERENCE_DATE}
        checks = ("a_synthetic", "b_candidate", "buyer_input", "d_diagnostics", "c_sample")
        self.validation.write_text(json.dumps({
            **identity, "decision_status": "pending_independent_review", "checks": {k: {"passed": True} for k in checks},
            "input_files": {k: {"sha256": sha(p)} for k, p in (("candidate", self.candidate), ("labels", labels),
                                                              ("buyers", self.buyers))},
            "risk_bound_approval": {"approval_sha256": sha(approval), "evidence_sha256": sha(basis)},
            "purchase_cache_provenance_sha256": sha(provenance),
            "daily_distributions": {"candidate": daily_distribution(r, y, [1], config.MAX_RECENCY_DAYS)},
        }))
        self.bundle.write_text(json.dumps({
            **identity, "artifact_files": {p.name: {"sha256": sha(p)} for p in
                (self.validation, self.candidate, labels, self.buyers, approval, basis, provenance)},
        }))
        self.review.write_text(json.dumps({
            **identity, "decision_status": f"jointly_confirmed_{config.RUN_ID}",
            "review_conclusion": "passed_with_condition", "condition_satisfied": True,
            "validation_sha256": sha(self.validation), "bundle_sha256": sha(self.bundle),
        }))

    def assert_blocked(self):
        with patch("pandas.read_parquet", side_effect=AssertionError("Read before review")):
            for loader in (load_risk_population, load_comparison):
                with self.subTest(loader=loader.__name__), self.assertRaises(ValueError):
                    loader()

    def test_missing_record_or_status_only_does_not_authorize(self):
        self.review.unlink()
        self.assert_blocked()
        self.validation.write_text(json.dumps({"decision_status": f"jointly_confirmed_{config.RUN_ID}"}))
        self.assert_blocked()

    def test_bundle_and_validation_hashes_must_match(self):
        original = self.bundle.read_text()
        self.bundle.write_text(original + " ")
        self.assert_blocked()
        self.bundle.write_text(original)
        self.validation.write_text(self.validation.read_text() + " ")
        self.assert_blocked()

    def test_changed_parquet_is_rejected_before_loading(self):
        self.buyers.write_bytes(b"changed")
        self.assert_blocked()

    def test_both_paths_read_the_same_reviewed_population(self):
        _, risk = load_comparison()
        pd.testing.assert_frame_equal(load_risk_population(), risk)

    def test_default_cli_uses_reviewed_cache_without_rebuilding(self):
        before = sha(self.buyers)
        summary = pd.DataFrame()
        with (patch("sys.argv", ["ab_design"]), patch("src.ab_design.purchase_users_before_reference") as scan,
              patch("src.ab_design.summarize_risk", return_value=summary),
              patch("src.ab_design.write_summary") as write, patch("src.ab_design.summary_markdown", return_value="")):
            main()
        scan.assert_not_called()
        write.assert_called_once_with(summary)
        self.assertEqual(sha(self.buyers), before)
