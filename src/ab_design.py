"""Rebuild reference-date purchase strata without using outcome-window purchases.

Run from the repository root: python -m src.ab_design
Raw CSVs are read only, in bounded chunks. User-level outputs remain in ignored data/.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

import config


def purchase_users_before_reference(paths, reference_date, chunksize=500_000):
    """Include the full UTC reference day; exclude the next day and later."""
    reference = pd.Timestamp(reference_date)
    reference = (reference.tz_localize("UTC") if reference.tzinfo is None
                 else reference.tz_convert("UTC")).normalize()
    cutoff = reference + pd.Timedelta(days=1)
    buyers = set()
    for path in paths:
        for chunk in pd.read_csv(
            path, usecols=["user_id", "event_type", "event_time"],
            dtype={"user_id": config.DTYPES["user_id"],
                   "event_type": config.DTYPES["event_type"], "event_time": "string"},
            chunksize=chunksize,
        ):
            purchases = chunk.loc[chunk["event_type"].eq("purchase")]
            if purchases.empty:
                continue
            times = pd.to_datetime(purchases["event_time"], utc=True, format="mixed")
            if times.isna().any():
                raise ValueError("Purchase timestamps must not be missing")
            buyers.update(purchases.loc[times.lt(cutoff), "user_id"].unique())
        print(f"{Path(path).name}: cumulative pre-reference buyers {len(buyers):,}", flush=True)
    return pd.Index(sorted(buyers), dtype="int64", name="user_id")


def build_risk_population(recency, labels, buyer_ids):
    """Use only precomputed reference-day recency and pre-reference purchase IDs."""
    if recency["user_id"].duplicated().any() or labels["user_id"].duplicated().any():
        raise ValueError("User IDs must be unique in both inputs")
    values = recency["recency_at_ref"]
    if values.isna().any() or values.lt(0).any() or values.mod(1).ne(0).any():
        raise ValueError("Recency must contain non-negative whole calendar days")
    risk = recency.loc[values.between(15, 29), ["user_id", "recency_at_ref"]].copy()
    risk = risk.merge(labels[["user_id", "status"]], on="user_id", how="left", validate="one_to_one")
    if not risk["status"].isin(["retained", "churned"]).all():
        raise ValueError("Every risk user needs an observed outcome; missing/censored labels are invalid")
    risk["is_buyer_at_reference"] = risk["user_id"].isin(buyer_ids)
    return risk


def summarize_risk(risk, alpha=0.05, power=0.8, mde=0.05):
    rows = []
    groups = [("overall", risk),
              ("buyer_before_reference", risk.loc[risk["is_buyer_at_reference"]]),
              ("nonbuyer_before_reference", risk.loc[~risk["is_buyer_at_reference"]])]
    for name, group in groups:
        if group.empty:
            raise ValueError(f"Empty segment: {name}")
        returned = int(group["status"].eq("retained").sum())
        baseline = returned / len(group)
        target = baseline + mde
        if not 0 < baseline < target < 1:
            raise ValueError("Power calculation requires 0 < baseline < baseline + MDE < 1")
        effect = proportion_effectsize(target, baseline)
        n = math.ceil(NormalIndPower().solve_power(
            effect_size=effect, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"))
        rows.append(dict(segment=name, users=len(group), returned_users=returned,
                         baseline=baseline, target=target, n_per_arm=n, n_total=2*n))
    return pd.DataFrame(rows)


def load_risk_population():
    """Read the existing 2020-03-31 recency cache and explicitly dated buyer cache."""
    recency = pd.read_parquet(config.PROC_DIR / "recency_at_reference.parquet",
                             columns=["user_id", "recency_at_ref"])
    labels = pd.read_parquet(config.PROC_DIR / "churn_labels.parquet", columns=["user_id", "status"])
    labels["status"] = labels["status"].astype("category")
    buyers = pd.read_parquet(config.PROC_DIR / "purchase_users_at_reference.parquet")
    if buyers.empty or not buyers["reference_date"].eq(config.AB_REFERENCE_DATE).all():
        raise ValueError("Buyer cache does not match the fixed recency reference date; rebuild it")
    if buyers["user_id"].duplicated().any():
        raise ValueError("Buyer cache contains duplicate users")
    if not buyers["user_id"].isin(recency["user_id"]).all():
        raise ValueError("Pre-reference buyers must have pre-reference activity")
    # The reference recency cache contains exactly the users whose April outcome is observable.
    eligible = labels.loc[labels["status"].isin(["retained", "churned"]), "user_id"]
    if len(recency) != len(eligible) or not recency["user_id"].isin(eligible).all():
        raise ValueError("Recency population does not match the outcome-label population")
    return build_risk_population(recency, labels, buyers["user_id"])


def summary_markdown(summary):
    names = {"overall": "전체 위험군", "buyer_before_reference": "기준일 이전 구매 경험군",
             "nonbuyer_before_reference": "기준일 이전 비구매군"}
    lines = ["| 구분 | 대상자 수 | 4월 재방문자 수 | baseline | 설계 대립값(+5%p) | 군당 표본 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples(index=False):
        lines.append(f"| {names[r.segment]} | {r.users:,} | {r.returned_users:,} | "
                     f"{r.baseline:.2%} | {r.target:.2%} | {r.n_per_arm:,} |")
    return "\n".join(lines)


def write_summary(summary):
    summary.to_csv(config.REPORTS_DIR / "ab_design_summary.csv", index=False)
    doc_path = config.PROJECT_ROOT / "docs" / "ab_test_design.md"
    doc = doc_path.read_text()
    start, end = "<!-- ab-summary:start -->", "<!-- ab-summary:end -->"
    if doc.count(start) != 1 or doc.count(end) != 1:
        raise ValueError("A/B document must have one generated-summary marker pair")
    before, remainder = doc.split(start)
    _, after = remainder.split(end)
    doc_path.write_text(before + start + "\n" + summary_markdown(summary) + "\n" + end + after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    paths = config.RAW_PATHS[:6]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        parser.error("Set ECOM_DATA_DIR to the monthly CSV directory. Missing: " + ", ".join(missing))
    buyers = purchase_users_before_reference(paths, config.AB_REFERENCE_DATE, args.chunksize)
    pd.DataFrame({"user_id": buyers, "reference_date": config.AB_REFERENCE_DATE}).to_parquet(
        config.PROC_DIR / "purchase_users_at_reference.parquet", index=False)
    risk = load_risk_population()
    summary = summarize_risk(risk)
    write_summary(summary)
    # Diagnostic comparison only: lifetime purchase history is never used to build the new strata.
    old = pd.read_parquet(config.PROC_DIR / "user_features.parquet", columns=["user_id", "is_buyer"])
    comparison = risk[["user_id", "is_buyer_at_reference", "status"]].merge(
        old, on="user_id", how="left", validate="one_to_one")
    if comparison["is_buyer"].isna().any():
        raise ValueError("Missing historical purchase flag in correction audit")
    moved = comparison.loc[comparison["is_buyer"] & ~comparison["is_buyer_at_reference"]]
    if (comparison["is_buyer_at_reference"] & ~comparison["is_buyer"]).any():
        raise ValueError("Pre-reference buyers cannot disappear from the lifetime buyer set")
    audit = {
        "reference_date": config.AB_REFERENCE_DATE,
        "timezone": "UTC",
        "feature_end_exclusive": "2020-04-01T00:00:00Z",
        "outcome_end_exclusive": "2020-05-01T00:00:00Z",
        "pre_reference_buyers_all_users": len(buyers),
        "risk_users": len(risk),
        "risk_returned_users": int(risk["status"].eq("retained").sum()),
        "reclassified_future_only_buyers": len(moved),
        "reclassified_returned_users": int(moved["status"].eq("retained").sum()),
        "source_files": [{"name": p.name, "bytes": p.stat().st_size} for p in paths],
        "activity_source": "Existing recency_at_reference.parquet (2020-03-31, UTC calendar days)",
        "outcome_source": "Existing churn_labels.parquet (April activity)",
    }
    (config.REPORTS_DIR / "ab_design_validation.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(summary_markdown(summary), flush=True)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
