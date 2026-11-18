"""Compare Recency-28 support with raw activity on the corresponding UTC date."""
from __future__ import annotations
import json
import math
from pathlib import Path
import pandas as pd
import config
from src.recency_reference import sha256_file, write_json_new


def main():
    target = config.RUN_REPORTS_DIR / "recency28_support_diagnostics.json"
    if target.exists():
        raise FileExistsError("Do not overwrite a diagnostic run")
    audit = json.loads((config.RECENCY_REBUILD_DIR / "scan_diagnostics.json").read_text())
    validation = json.loads(config.RECENCY_VALIDATION_PATH.read_text())
    if validation.get("decision_status") != "pending_independent_review":
        raise ValueError("Expected successful validation awaiting independent review")
    rows = validation["daily_distributions"]["candidate"]
    totals = {}
    for row in rows:
        if row["recency_state"] != "valid":
            raise ValueError("Unexpected invalid Recency bucket")
        day = row["recency_at_ref"]
        pair = totals.setdefault(day, [0, 0])
        pair[0] += row["users"]
        pair[1] += row["users"] if row["outcome_active"] else 0
    original = pd.read_csv(config.RUN_REPORTS_DIR / "recency_day_outcome_comparison.csv")
    for row in original.itertuples(index=False):
        actual = totals.get(int(row.recency_days), [0, 0])
        if actual != [int(row.users), int(row.returned_users)]:
            raise ValueError("Independent raw Recency scan differs from 04-based descriptive curve")
    raw = audit["recency28_raw_activity"]
    target_date = pd.Timestamp(raw["target_date"], tz="UTC")
    table = []
    for day in raw["days"]:
        recency = (pd.Timestamp(config.AB_REFERENCE_DATE, tz="UTC") - pd.Timestamp(day["date"], tz="UTC")).days
        count, returned = totals.get(recency, [0, 0])
        hours = [h for h in day["hourly_event_rows"] if h["event_rows"]]
        table.append({"date_utc": day["date"], "recency_days": recency,
                      "raw_event_rows": day["event_rows"], "raw_unique_users": day["unique_users"],
                      "active_utc_hours": len(hours), "raw_min_time_utc": day["min_time_utc"],
                      "raw_max_time_utc": day["max_time_utc"],
                      "last_activity_users": count, "returned_users": returned,
                      "return_rate": returned / count if count else None})
    focus = next(r for r in table if r["recency_days"] == 28)
    neighbours = [r for r in table if r["recency_days"] in (27, 29)]
    mean_rows = sum(r["raw_event_rows"] for r in neighbours)/2
    mean_users = sum(r["raw_unique_users"] for r in neighbours)/2
    mean_support = sum(r["last_activity_users"] for r in neighbours)/2
    report = {"run_id": config.RUN_ID,"target_recency_days":28,"target_date_utc":target_date.date().isoformat(),
              "days": table,"hourly_raw_activity":raw["days"],
              "ratios_to_mean_of_adjacent_days":{
                  "raw_event_rows":focus["raw_event_rows"]/mean_rows if mean_rows else None,
                  "raw_unique_users":focus["raw_unique_users"]/mean_users if mean_users else None,
                  "last_activity_users":focus["last_activity_users"]/mean_support if mean_support else None},
              "descriptive_curve_counts_match_independent_scan":True,
              "interpretation":"Raw daily activity and final-pre-reference-activity support are distinct counts. Low raw activity can constrain support, but collection failure versus real behaviour cannot be identified without source history.",
              "limitations":["No causal or threshold-optimality claim.","No bot filtering or source repair performed."],
              "inputs":{str(p.relative_to(config.PROJECT_ROOT)):sha256_file(p) for p in [config.RECENCY_VALIDATION_PATH,config.RECENCY_REBUILD_DIR/"scan_diagnostics.json",config.RUN_REPORTS_DIR/"recency_day_outcome_comparison.csv"]},
              "code_sha256":sha256_file(Path(__file__))}
    pd.DataFrame(table).to_csv(config.RUN_REPORTS_DIR/"recency28_support_comparison.csv",index=False)
    write_json_new(target,report)
    print(json.dumps({"focus":focus,"ratios":report["ratios_to_mean_of_adjacent_days"],"all_curve_counts_match":True},indent=2))


if __name__ == "__main__":
    main()
