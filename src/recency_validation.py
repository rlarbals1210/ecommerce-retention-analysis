"""Audit the isolated recency candidate without replacing caches/public summaries.

python -m src.recency_validation
The report carries exact counts and both daily distributions for independent
review. A branch is provisional until the separate reviewer confirms it.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

import config
from src.recency_reference import (
    file_identity, independent_sample_recency, sample_digest, sha256_file,
    utc_reference, write_json_new,
)


def recency_checks(frame, eligible, upper_days):
    ids, raw = frame["user_id"], frame["recency_at_ref"]
    values = pd.to_numeric(raw, errors="coerce")
    finite = values.notna() & np.isfinite(values)
    valid = finite & values.ge(0) & values.le(upper_days) & values.mod(1).eq(0)
    result = {
        "rows": len(frame), "missing_ids": int(ids.isna().sum()),
        "duplicate_ids": int(ids.duplicated().sum()),
        "integer_id_dtype": bool(pd.api.types.is_integer_dtype(ids)),
        "missing_recency": int(raw.isna().sum()),
        "nonnumeric_recency": int((raw.notna() & values.isna()).sum()),
        "nonfinite_recency": int((values.notna() & ~np.isfinite(values)).sum()),
        "negative_recency": int((finite & values.lt(0)).sum()),
        "noninteger_recency": int((finite & values.mod(1).ne(0)).sum()),
        "above_max_recency": int((finite & values.gt(upper_days)).sum()),
        "invalid_recency_rows": int((~valid).sum()),
        "missing_eligible_users": int((~eligible.isin(ids)).sum()),
        "extra_users": int((~ids.isin(eligible)).sum()),
        "minimum": float(values[finite].min()) if finite.any() else None,
        "maximum": float(values[finite].max()) if finite.any() else None,
    }
    result["joinable"] = (result["missing_ids"] == result["duplicate_ids"] == 0
                          and result["integer_id_dtype"])
    result["passed"] = (result["joinable"] and result["invalid_recency_rows"] == 0
                        and result["missing_eligible_users"] == result["extra_users"] == 0)
    return result


def daily_distribution(frame, labels, buyer_ids, upper_days):
    """Preserve invalid rows as explicit buckets; no silent groupby dropna loss."""
    joined = frame.merge(labels[["user_id", "status"]], on="user_id", how="left",
                         validate="one_to_one")
    if not joined["status"].isin(["retained", "churned"]).all():
        raise ValueError("Distribution has missing/censored outcome labels")
    values = pd.to_numeric(joined["recency_at_ref"], errors="coerce")
    finite = values.notna() & np.isfinite(values)
    valid = finite & values.ge(0) & values.le(upper_days) & values.mod(1).eq(0)
    joined["is_buyer_at_reference"] = joined["user_id"].isin(buyer_ids)
    joined["april_active"] = joined["status"].eq("retained")
    cols = ["recency_at_ref", "is_buyer_at_reference", "april_active"]
    groups = joined.loc[valid].groupby(cols, observed=True, dropna=False).size()
    rows = [{"recency_at_ref": int(day), "is_buyer_at_reference": bool(buyer),
             "april_active": bool(active), "users": int(count), "recency_state": "valid"}
            for (day, buyer, active), count in groups.items()]
    if (~valid).any():
        bad = joined.loc[~valid, ["is_buyer_at_reference", "april_active"]].copy()
        bad["recency_state"] = np.select(
            [values.loc[~valid].isna(), ~np.isfinite(values.loc[~valid]),
             values.loc[~valid].lt(0), values.loc[~valid].mod(1).ne(0)],
            ["missing_or_nonnumeric", "nonfinite", "negative", "noninteger"], default="above_max")
        for (state, buyer, active), count in bad.groupby(
                ["recency_state", "is_buyer_at_reference", "april_active"], observed=True).size().items():
            rows.append({"recency_at_ref": None, "is_buyer_at_reference": bool(buyer),
                         "april_active": bool(active), "users": int(count), "recency_state": state})
    if sum(row["users"] for row in rows) != len(frame):
        raise ValueError("Distribution does not account for every row")
    return rows


def interval_counts(distribution):
    groups = {(band, buyer): [0, 0] for band in ["0-7", "8-14", "15-29", "30+"]
              for buyer in [False, True]}
    for row in distribution:
        if row["recency_state"] != "valid":
            band = "invalid:" + row["recency_state"]
        else:
            day = row["recency_at_ref"]
            band = "0-7" if day <= 7 else "8-14" if day <= 14 else "15-29" if day <= 29 else "30+"
        pair = groups.setdefault((band, row["is_buyer_at_reference"]), [0, 0])
        pair[0] += row["users"]
        pair[1] += row["users"] if row["april_active"] else 0
    return [{"recency_band": band, "is_buyer_at_reference": buyer, "users": n,
             "returned_users": r, "baseline": r / n if n else None}
            for (band, buyer), (n, r) in groups.items()]


def risk_summary(intervals, alpha=0.05, power=0.8, mde=0.05):
    risk = {r["is_buyer_at_reference"]: r for r in intervals if r["recency_band"] == "15-29"}
    rows = []
    for name, strata in [("overall", [False, True]), ("buyer_before_reference", [True]),
                         ("nonbuyer_before_reference", [False])]:
        n = sum(risk[s]["users"] for s in strata)
        returned = sum(risk[s]["returned_users"] for s in strata)
        baseline = returned / n if n else None
        target = baseline + mde if baseline is not None else None
        per_arm = None
        if baseline is not None and 0 < baseline < target < 1:
            per_arm = math.ceil(NormalIndPower().solve_power(
                effect_size=proportion_effectsize(target, baseline), alpha=alpha,
                power=power, ratio=1, alternative="two-sided"))
        rows.append({"segment": name, "users": n, "returned_users": returned,
                     "baseline": baseline, "target": target,
                     "n_per_arm": per_arm, "n_total": 2 * per_arm if per_arm else None})
    return rows


def decide(validated, distribution_equal, public_equal):
    if not validated:
        return 4
    if not public_equal:
        return 3
    return 1 if distribution_equal else 2


def validate_frames(old, candidate, labels, buyers, sample_ids, sample_events, scan_audit,
                    *, synthetic_passed, upper_days=182):
    if (labels["user_id"].isna().any() or labels["user_id"].duplicated().any()
            or not labels["status"].isin(["retained", "churned", "censored"]).all()):
        raise ValueError("Invalid outcome-label input")
    eligible = labels.loc[labels["status"].isin(["retained", "churned"]), "user_id"]
    buyer_valid = (buyers["user_id"].notna().all() and not buyers["user_id"].duplicated().any()
                   and buyers["reference_date"].eq(config.AB_REFERENCE_DATE).all()
                   and buyers["user_id"].isin(eligible).all())
    candidate_check = recency_checks(candidate, eligible, upper_days)
    old_check = recency_checks(old, eligible, upper_days)
    files = scan_audit["files"]
    diagnostic_pass = (bool(files) and scan_audit["parse_errors"] == 0
                       and not scan_audit["input_changed"]
                       and sum(d["parse_errors"] for d in files) == scan_audit["parse_errors"]
                       and sum(d["cutoff_excluded_rows"] for d in files) == scan_audit["cutoff_excluded_rows"]
                       and all(d["rows"] == d["included_rows"] + d["parse_errors"] + d["cutoff_excluded_rows"]
                               and d["valid_timestamp_rows"] == d["included_rows"] + d["cutoff_excluded_rows"]
                               and d["input_unchanged_during_scan"] for d in files))
    report = {"eligible_users": len(eligible), "old_cache_checks": old_check,
              "checks": {"a_synthetic": {"passed": bool(synthetic_passed)},
                         "b_candidate": candidate_check,
                         "buyer_input": {"passed": bool(buyer_valid)},
                         "d_diagnostics": {"passed": bool(diagnostic_pass)}},
              "scan_diagnostics": scan_audit,
              "decision_branch": 4, "decision_status": "provisional_pending_independent_review",
              "adopted": False, "published": False}
    try:
        oracle = independent_sample_recency(sample_events, sample_ids, config.AB_REFERENCE_DATE)
        selected = candidate.loc[candidate["user_id"].isin(sample_ids)]
        observed = selected.set_index("user_id")["recency_at_ref"].to_dict()
        sample_ok = (len(sample_ids) == len(set(sample_ids)) == len(selected)
                     and observed == oracle)
        report["checks"]["c_sample"] = {"passed": sample_ok, "sample_users": len(sample_ids),
                                          "observed_users": len(selected),
                                          "mismatched_users": sum(observed.get(k) != v for k, v in oracle.items())}
    except (ValueError, TypeError) as exc:
        report["checks"]["c_sample"] = {"passed": False, "error": str(exc)}
    passed = all(c["passed"] for c in report["checks"].values())
    if not candidate_check["joinable"] or not old_check["joinable"] or not buyer_valid:
        return report
    # Population mismatch prevents reliable label joins; retain structural failure evidence.
    if any(c["missing_eligible_users"] or c["extra_users"] for c in [candidate_check, old_check]):
        return report
    aligned = old[["user_id", "recency_at_ref"]].merge(
        candidate, on="user_id", how="outer", validate="one_to_one", suffixes=("_old", "_new"))
    left = pd.to_numeric(aligned["recency_at_ref_old"], errors="coerce")
    right = pd.to_numeric(aligned["recency_at_ref_new"], errors="coerce")
    delta = (left - right).abs()
    finite_delta = delta[np.isfinite(delta)]
    report["user_comparison"] = {
        "compared_users": len(aligned), "different_users": int((~left.eq(right)).sum()),
        "max_absolute_recency_difference": float(finite_delta.max()) if len(finite_delta) else None,
        "nonfinite_comparisons": int((~np.isfinite(delta)).sum())}
    del aligned, left, right, delta, finite_delta
    old_hist = daily_distribution(old, labels, buyers["user_id"], upper_days)
    new_hist = daily_distribution(candidate, labels, buyers["user_id"], upper_days)
    old_intervals, new_intervals = interval_counts(old_hist), interval_counts(new_hist)
    old_summary, new_summary = risk_summary(old_intervals), risk_summary(new_intervals)
    histogram_equal = old_hist == new_hist
    public_equal = old_intervals == new_intervals and old_summary == new_summary
    report.update({"daily_distributions": {"old": old_hist, "candidate": new_hist},
                   "interval_counts": {"old": old_intervals, "candidate": new_intervals},
                   "risk_summaries": {"old": old_summary, "candidate": new_summary},
                   "comparison": {"daily_distribution_equal": histogram_equal,
                                  "interval_and_risk_equal": public_equal},
                   "power_parameters": {"alpha": 0.05, "power": 0.8, "mde_absolute": 0.05,
                                        "alternative": "two-sided", "ratio": 1},
                   "decision_branch": decide(passed, histogram_equal, public_equal)})
    return report


def published_matches(summary, published):
    if len(published) != 3 or published["segment"].duplicated().any():
        return False
    lookup = published.set_index("segment")
    for row in summary:
        if row["segment"] not in lookup.index:
            return False
        record = lookup.loc[row["segment"]]
        if any(record[k] != row[k] for k in ["users", "returned_users", "n_per_arm", "n_total"]):
            return False
        if any(row[k] is None or not math.isclose(float(record[k]), row[k], rel_tol=0, abs_tol=1e-15)
               for k in ["baseline", "target"]):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=config.RECENCY_REBUILD_DIR)
    parser.add_argument("--output", type=Path, default=config.RECENCY_VALIDATION_PATH)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Validation report exists; do not overwrite")
    report = {"decision_branch": 4, "decision_status": "provisional_pending_independent_review",
              "adopted": False, "published": False}
    try:
        run = args.run_dir
        manifest = json.loads((run / "preparation.json").read_text())
        scan = json.loads((run / "scan_diagnostics.json").read_text())
        inputs = {"labels": config.PROC_DIR / "churn_labels.parquet",
                  "old": config.PROC_DIR / "recency_at_reference.parquet",
                  "candidate": run / "recency_at_reference.candidate.parquet",
                  "buyers": config.PROC_DIR / "purchase_users_at_reference.parquet",
                  "sample_ids": run / "sample_users.parquet", "sample_events": run / "sample_events.parquet"}
        identities = {key: {**file_identity(path), "sha256": sha256_file(path)} for key, path in inputs.items()}
        if identities["labels"]["sha256"] != manifest["label_source"]["sha256"]:
            raise ValueError("Labels changed since sample selection")
        if manifest["reference_date"] != config.AB_REFERENCE_DATE or scan["reference_date"] != config.AB_REFERENCE_DATE:
            raise ValueError("Reference date mismatch")
        if scan["cutoff_exclusive_utc"] != (utc_reference(config.AB_REFERENCE_DATE) + pd.Timedelta(days=1)).isoformat():
            raise ValueError("Cutoff mismatch")
        if [d["name"] for d in scan["files"]] != config.RAW_FILES[:6]:
            raise ValueError("Incomplete/unexpected raw input set")
        for path, diag in zip(config.RAW_PATHS[:6], scan["files"]):
            if file_identity(path) != {k: diag[k] for k in ["name", "bytes", "mtime_ns"]}:
                raise ValueError("Raw input metadata changed after scanning")
        test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                               "-p", "test_recency_reference.py", "-v"], cwd=config.PROJECT_ROOT,
                              capture_output=True, text=True, check=False)
        sample = pd.read_parquet(inputs["sample_ids"])["user_id"]
        if len(sample) != manifest["sample_size"] or sample_digest(sample) != manifest["sample_ids_sha256"]:
            raise ValueError("Sample manifest mismatch")
        if scan["sample_ids_sha256"] != manifest["sample_ids_sha256"]:
            raise ValueError("Scan used a different sample")
        producer_hash = sha256_file(config.PROJECT_ROOT / "src" / "recency_reference.py")
        if producer_hash != scan["producer_sha256"]:
            raise ValueError("Tested producer differs from the scanned producer")
        labels = pd.read_parquet(inputs["labels"], columns=["user_id", "status"])
        labels["status"] = labels["status"].astype("category")
        report = validate_frames(pd.read_parquet(inputs["old"], columns=["user_id", "recency_at_ref"]),
                                 pd.read_parquet(inputs["candidate"]), labels, pd.read_parquet(inputs["buyers"]),
                                 sample, pd.read_parquet(inputs["sample_events"]), scan,
                                 synthetic_passed=test.returncode == 0,
                                 upper_days=(utc_reference(config.AB_REFERENCE_DATE) - utc_reference(config.AB_OBSERVATION_START)).days)
        report["checks"]["a_synthetic"].update({"returncode": test.returncode,
                                               "output": test.stdout + test.stderr,
                                               "producer_sha256": producer_hash})
        report["input_files"] = identities
        report["sample_ids_sha256"] = manifest["sample_ids_sha256"]
        report["label_reference_count_matches_preparation"] = report["eligible_users"] == manifest["eligible_users"]
        if not report["label_reference_count_matches_preparation"]:
            report["decision_branch"] = 4
        if "risk_summaries" in report:
            published = pd.read_csv(config.REPORTS_DIR / "ab_design_summary.csv")
            matches = {key: published_matches(value, published) for key, value in report["risk_summaries"].items()}
            report["published_05_summary_matches"] = matches
            if not matches["candidate"] and report["decision_branch"] != 4:
                report["decision_branch"] = 3
        report["input_metadata_unchanged_during_validation"] = all(
            file_identity(path) == {k: identities[key][k] for k in ["name", "bytes", "mtime_ns"]}
            for key, path in inputs.items())
        if not report["input_metadata_unchanged_during_validation"]:
            report["decision_branch"] = 4
        report["validation_code_sha256"] = sha256_file(Path(__file__))
    except Exception as exc:
        report.update({"decision_branch": 4, "error": f"{type(exc).__name__}: {exc}"})
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["limitations"] = ["Original cache provenance is unknown; matching is not independent proof.",
                              "1,000-user sample cannot exclude rare or concentrated defects.",
                              "Raw-file identities use name/size/mtime, not a full content hash.",
                              "Independent review and public release approval are still pending."]
    write_json_new(args.output, report)
    print(json.dumps({"report": str(args.output), "decision_branch": report["decision_branch"],
                      "error": report.get("error"), "checks": report.get("checks"),
                      "user_comparison": report.get("user_comparison"),
                      "comparison": report.get("comparison")}, ensure_ascii=False, indent=2))
    if report["decision_branch"] == 4:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
