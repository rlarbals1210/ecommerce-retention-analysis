"""Load analysis inputs after the current run's independent review is confirmed."""
import hashlib
import json

import config
from src.recency_reference import sha256_file


def load_reviewed_inputs():
    directory = config.RUN_REPORTS_DIR
    try:
        record = json.loads((directory / "independent_review_record.json").read_bytes())
        bundle_path = directory / "independent_review_bundle.json"
        bundle_bytes = bundle_path.read_bytes()
        validation_bytes = config.RECENCY_VALIDATION_PATH.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError("Candidate requires a confirmed independent review record") from exc
    bundle = json.loads(bundle_bytes)
    report = json.loads(validation_bytes)
    if any(item.get("run_id") != config.RUN_ID or
           item.get("reference_date") != config.AB_REFERENCE_DATE
           for item in (record, bundle, report)):
        raise ValueError("Review, bundle or validation belongs to another run")
    if (record.get("decision_status") != f"jointly_confirmed_{config.RUN_ID}"
            or record.get("review_conclusion") != "passed_with_condition"
            or record.get("condition_satisfied") is not True):
        raise ValueError("Independent review condition requires confirmation")
    report_hash = hashlib.sha256(validation_bytes).hexdigest()
    report_key = config.RECENCY_VALIDATION_PATH.relative_to(config.PROJECT_ROOT).as_posix()
    bundle_files = bundle.get("artifact_files", {})
    if (record.get("bundle_sha256") != hashlib.sha256(bundle_bytes).hexdigest()
            or record.get("validation_sha256") != report_hash
            or bundle_files.get(report_key, {}).get("sha256") != report_hash):
        raise ValueError("Independent review evidence hash mismatch")
    expected_checks = {"a_synthetic", "b_candidate", "buyer_input", "d_diagnostics", "c_sample"}
    checks = report.get("checks", {})
    if (report.get("decision_status") != "pending_independent_review"
            or report.get("error") or report.get("decision_branch") is not None
            or not expected_checks <= checks.keys()
            or not all(check.get("passed") is True for check in checks.values())):
        raise ValueError("Original automatic validation must have passed")

    paths = {
        "candidate": config.RECENCY_CANDIDATE_PATH,
        "labels": config.RUN_PROC_DIR / "churn_labels.parquet",
        "buyers": config.RUN_PROC_DIR / "purchase_users_at_reference.parquet",
        "approval": config.RISK_APPROVAL_PATH,
        "basis": config.RISK_BASIS_PATH,
        "purchase_provenance": directory / "purchase_cache_provenance.json",
    }
    risk_approval = report.get("risk_bound_approval", {})
    for name, path in paths.items():
        if name in {"approval", "basis"}:
            expected = risk_approval.get({"approval": "approval_sha256", "basis": "evidence_sha256"}[name])
        elif name == "purchase_provenance":
            expected = report.get("purchase_cache_provenance_sha256")
        else:
            expected = report["input_files"][name]["sha256"]
        key = path.relative_to(config.PROJECT_ROOT).as_posix()
        if expected != bundle_files.get(key, {}).get("sha256") or sha256_file(path) != expected:
            raise ValueError(f"Reviewed input changed: {name}")
    return report, paths
