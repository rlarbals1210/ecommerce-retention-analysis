"""Compare fixed abnormal-activity scenarios using reviewed local caches."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config
from src.independent_review import load_reviewed_inputs
from src.recency_reference import sha256_file, write_json_new
from src.recency_validation import interval_counts, risk_summary


BASE = "ce81d18"
RUN_NAME = "abnormal_users_ref20200326"
SCENARIOS = ("S1", "S2", "S3", "S4")
EXPECTED_CANDIDATES = dict(zip(SCENARIOS, (133_466, 13_214, 9_818, 21_882)))
EXPECTED_STATUS = {"churned": 10_838_810, "retained": 2_509_166, "censored": 2_291_827}
EXPECTED_BANDS = [(1_158_712, 680_047), (634_624, 269_894), (2_852_344, 726_061),
                  (8_702_296, 833_164)]
EXPECTED_RISK = [(2_852_344, 726_061, 1_263), (376_467, 161_770, 1_556),
                 (2_475_877, 564_291, 1_184)]
EXPECTED_GAPS = {"full": (44_588_010, 14, 35, 55), "pre": (36_655_640, 13, 31, 48)}
GAP_LEVELS = (.75, .90, .95)
QUANTILE_METHOD = "pandas Series.quantile(interpolation='linear')"
LIMITATIONS = [
    "봇 여부를 알려 주는 정답 라벨이 없다.",
    "B는 자동화·중복 기록·수집 방식을 구분하지 못하고 활동량과 연관된다. 사전 활동일 중앙값은 S3 14일, 전체 2일이다.",
    "기준일 이후 처음 관측된 2,291,827명은 판정 대상에서 제외했으며 전체 기간 간격 집계에는 남겼다.",
    "동일 행 중복은 검사하지 않았다.",
    "1차 스캔의 대체 형식 파싱 행 수는 기록되지 않았다. 파싱 오류는 0건이었다.",
    "1차 스캔의 월 간 합계·최댓값 결합은 코드 검토로 확인했다. 이 결합의 단위 테스트는 없다.",
]


def verify_membership(run, report_path, activity_path, first_module_path, expected):
    manifest_bytes = (run / "membership_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    expected_hashes = {
        "user_scores.parquet": manifest["files_sha256"]["user_scores.parquet"],
        "scenario_membership.parquet": manifest["files_sha256"]["scenario_membership.parquet"],
    }
    for name, digest in expected_hashes.items():
        if sha256_file(run / name) != digest:
            raise ValueError(f"Membership manifest hash mismatch: {name}")
    for path, field in ((activity_path, "activity_day_cache_sha256"),
                        (first_module_path, "module_sha256")):
        if sha256_file(path) != manifest[field]:
            raise ValueError(f"Membership manifest hash mismatch: {field}")

    members = pq.read_table(run / "scenario_membership.parquet").to_pandas()
    if members.columns.tolist() != ["user_id", *SCENARIOS]:
        raise ValueError("Unexpected scenario membership schema")
    if (members.isna().any().any() or members.user_id.duplicated().any()
            or not pd.api.types.is_integer_dtype(members.user_id)
            or not all(pd.api.types.is_bool_dtype(members[scenario]) for scenario in SCENARIOS)):
        raise ValueError("Scenario membership rows require unique integer IDs and boolean flags")
    if (not (members.S1 | members.S3).all() or (members.S2 & ~members.S1).any()
            or not (members.S4 == (members.S2 | members.S3)).all()):
        raise ValueError("Scenario membership set relationships differ")
    counts = {scenario: int(members[scenario].sum()) for scenario in SCENARIOS}
    first_stop_counts = {key: report["scenarios"][key]["users"] for key in SCENARIOS}
    if counts != expected or counts != first_stop_counts:
        raise ValueError(f"Scenario candidate count mismatch: {counts}")

    bits = np.zeros(len(members), dtype=np.uint8)
    for index, key in enumerate(SCENARIOS):
        bits |= members[key].to_numpy(dtype=np.uint8) << index
    membership = pd.DataFrame({"user_id": members.user_id, "bits": bits}).sort_values("user_id")
    return membership, {
        "passed": True,
        "first_stop_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "membership_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "files_sha256": {**expected_hashes, **{key: manifest[key] for key in
                          ("activity_day_cache_sha256", "module_sha256")}},
        "candidate_counts": counts,
    }


def cohort_cells(db, paths, membership):
    db.register("membership", membership)
    cells = db.execute("""
        SELECT labels.status, candidate.recency_at_ref AS recency,
               buyers.user_id IS NOT NULL AS buyer, COALESCE(membership.bits, 0) AS bits,
               COUNT(*) AS users
        FROM read_parquet(?) AS labels
        LEFT JOIN read_parquet(?) AS candidate USING (user_id)
        LEFT JOIN read_parquet(?) AS buyers USING (user_id)
        LEFT JOIN membership USING (user_id)
        GROUP BY ALL
    """, [str(paths[key]) for key in ("labels", "candidate", "buyers")]).fetchall()
    matched = dict.fromkeys(SCENARIOS, 0)
    for status, recency, buyer, bits, users in cells:
        if status == "censored":
            if recency is not None or buyer or bits:
                raise ValueError("Post-reference user present in pre-reference inputs")
        elif (status not in ("churned", "retained") or recency is None
              or not 0 <= recency <= config.MAX_RECENCY_DAYS):
            raise ValueError("Invalid status or Recency in reviewed cohort")
        for index, key in enumerate(SCENARIOS):
            if bits & (1 << index):
                matched[key] += users
    return cells, matched


def cohort_summary(cells, exclude_bit):
    status_counts = dict.fromkeys(EXPECTED_STATUS, 0)
    distribution = []
    for status, recency, buyer, bits, users in cells:
        if bits & exclude_bit:
            continue
        status_counts[status] += users
        if status != "censored":
            distribution.append({"recency_state": "valid", "recency_at_ref": recency,
                                 "is_buyer_at_reference": buyer, "outcome_active": status == "retained",
                                 "users": users})
    intervals = interval_counts(distribution)
    bands = {}
    for band in config.recency_bands():
        rows = [row for row in intervals if row["recency_band"] == band]
        users = sum(row["users"] for row in rows)
        returned = sum(row["returned_users"] for row in rows)
        bands[band] = {"users": users, "returned_users": returned,
                       "return_rate": returned / users if users else None}
    eligible = status_counts["churned"] + status_counts["retained"]
    risk = {row["segment"]: row for row in risk_summary(intervals)}
    return {"eligible_users": eligible, "status_counts": status_counts,
            "inactivity_rate": status_counts["churned"] / eligible if eligible else None,
            "recency_bands": bands, "risk": risk}


@dataclass
class GapData:
    days: np.ndarray
    gap_bits: np.ndarray
    medians: np.ndarray
    median_bits: np.ndarray
    single_gap: np.ndarray


def membership_bits(ids, membership):
    keys = membership.user_id.to_numpy()
    indices = np.searchsorted(keys, ids)
    bits = np.zeros(len(ids), dtype=np.uint8)
    inside = np.flatnonzero(indices < len(keys))
    matched = inside[keys[indices[inside]] == ids[inside]]
    bits[matched] = membership.bits.to_numpy()[indices[matched]]
    return bits


def complete_user_blocks(batches):
    """Keep each user's completed gaps together even across Arrow batch boundaries."""
    tail = None
    for batch in batches:
        frame = batch.to_pandas()
        if tail is not None:
            frame = pd.concat([tail, frame], ignore_index=True)
        boundary = frame.user_id.searchsorted(frame.user_id.iat[-1])
        if boundary:
            yield frame.iloc[:boundary]
        tail = frame.iloc[boundary:].copy()
    if tail is not None:
        yield tail


def gap_block(frame, membership):
    days = frame.gap_days.to_numpy(dtype=np.int16)
    if np.any(days <= 0):
        raise ValueError("Completed gaps must be positive")
    by_user = frame.groupby("user_id", sort=False).gap_days
    medians = by_user.median()
    return GapData(days, membership_bits(frame.user_id.to_numpy(), membership),
                   medians.to_numpy(dtype=np.float32),
                   membership_bits(medians.index.to_numpy(), membership),
                   by_user.size().to_numpy() == 1)


def collect_gaps(db, activity_path, membership, *, batch_rows=1_000_000):
    # With UTC, eligibility of a completed gap is determined by its later day.
    gaps = db.execute("""
        SELECT user_id, gap_days, date < ?::TIMESTAMPTZ AS pre
        FROM (
            SELECT user_id, date, CAST(date_diff('day', LAG(date) OVER
                (PARTITION BY user_id ORDER BY date), date) AS SMALLINT) AS gap_days
            FROM read_parquet(?)
        )
        WHERE gap_days IS NOT NULL
        ORDER BY user_id
    """, [config.AB_FEATURE_END_EXCLUSIVE, str(activity_path)]).to_arrow_reader(batch_rows)
    windows = {"full": [], "pre": []}
    for frame in complete_user_blocks(gaps):
        windows["full"].append(gap_block(frame, membership))
        windows["pre"].append(gap_block(frame.loc[frame.pre], membership))
    combined = {}
    for window, blocks in windows.items():
        combined[window] = GapData(*(np.concatenate([getattr(block, field) for block in blocks])
                                     for field in GapData.__dataclass_fields__))
    return combined


def interval_quantiles(days):
    if len(days) == 0:
        return dict.fromkeys(("p75", "p90", "p95"))
    values = pd.Series(days).quantile(list(GAP_LEVELS), interpolation="linear")
    return dict(zip(("p75", "p90", "p95"), map(float, values)))


def interval_summary(gaps, exclude_bit):
    days = gaps.days[(gaps.gap_bits & exclude_bit) == 0] if exclude_bit else gaps.days
    return {"gap_count": len(days), **interval_quantiles(days)}


def user_median_summary(gaps, exclude_bit):
    keep = (gaps.median_bits & exclude_bit) == 0
    medians = gaps.medians[keep]
    count = len(medians)
    single = int(gaps.single_gap[keep].sum())
    return {"contributing_users": count, "one_gap_users": single,
            "one_gap_share": single / count if count else None,
            "p90": float(pd.Series(medians).quantile(.90, interpolation="linear")) if count else None}


def verify_baseline(cohort, gaps):
    status = cohort["status_counts"]
    bands = [(row["users"], row["returned_users"]) for row in cohort["recency_bands"].values()]
    risk = [(row["users"], row["returned_users"], row["n_per_arm"])
            for row in cohort["risk"].values()]
    measured = {window: tuple(row[x] for x in ("gap_count", "p75", "p90", "p95"))
                for window, row in gaps.items()}
    checks = {
        "eligible_and_labels": status == EXPECTED_STATUS and cohort["eligible_users"] == 13_347_976,
        "recency_bands": bands == EXPECTED_BANDS,
        "risk_counts_and_sample_sizes": risk == EXPECTED_RISK,
        "completed_gap_counts_and_quantiles": measured == EXPECTED_GAPS,
    }
    if not all(checks.values()):
        raise ValueError(f"No-exclusion reproduction mismatch: {checks}; "
                         f"measured status={status}, bands={bands}, risk={risk}, gaps={measured}")
    return {"passed": True, "checks": checks}


def difference(value, baseline, *, scale=1):
    return None if value is None or baseline is None else (value - baseline) * scale


def comparison_row(cohort, intervals, medians, baseline_cohort, baseline_intervals):
    for window, row in intervals.items():
        prior = baseline_intervals[window]
        removed = prior["gap_count"] - row["gap_count"]
        row.update(excluded_gaps=removed, excluded_gap_share=removed / prior["gap_count"],
                   delta_days={key: difference(row[key], prior[key]) for key in ("p75", "p90", "p95")})
    cohort["inactivity_delta_pp"] = difference(cohort["inactivity_rate"],
                                               baseline_cohort["inactivity_rate"], scale=100)
    for key, row in cohort["recency_bands"].items():
        row["return_delta_pp"] = difference(row["return_rate"],
                                             baseline_cohort["recency_bands"][key]["return_rate"],
                                             scale=100)
    for key, row in cohort["risk"].items():
        prior = baseline_cohort["risk"][key]
        row.update(return_delta_pp=difference(row["baseline"], prior["baseline"], scale=100),
                   sample_size_delta=difference(row["n_per_arm"], prior["n_per_arm"]))
    return {"cohort": cohort, "completed_gaps": intervals, "user_median_gaps": medians,
            "pre_p75_changed": intervals["pre"]["p75"] != baseline_intervals["pre"]["p75"]}


def markdown(report):
    def number(value, spec):
        return "—" if value is None else format(value, spec)

    def push_table(lines, columns, rows):
        lines.extend(["| " + " | ".join(columns) + " |",
                      "| " + " | ".join(["---"] * len(columns)) + " |",
                      *("| " + " | ".join(row) + " |" for row in rows), ""])

    lines = [
        "# 비정상 활동 후보 제외 전후 지표",
        "",
        "기준일 2020-03-26 UTC · 관찰 기간 2020-03-27~04-30 · 위험군 Recency 14~34일.",
        "S0는 무제외, S1은 A-p99, S2는 A-p99.9, S3는 B-p99.9, S4는 S2와 S3의 합집합이다.",
        "변화량은 S0 대비다. 완료 간격 분위수와 사용자별 중앙값의 p90은 각각 선형 보간했다.",
        "",
        "## 완료 간격",
        "",
        "간격 단위 분위수에서는 완료 간격 한 건이 관측치 하나다. 사용자 단위 p90은 각 사용자의 완료 간격 중앙값을 관측치 하나로 쓴다. 간격 1개인 사용자는 포함하고 0개인 사용자는 제외한다.",
        "",
    ]
    gap_rows, user_rows, status_rows, band_rows, risk_rows = [], [], [], [], []
    for key, scenario in report["scenarios"].items():
        for window, title in (("full", "전체"), ("pre", "사전")):
            gaps = scenario["completed_gaps"][window]
            change = gaps["delta_days"]
            gap_rows.append([key, title, f"{gaps['gap_count']:,}", f"{gaps['excluded_gaps']:,}",
                             number(gaps["excluded_gap_share"], ".4%"),
                             *(number(gaps[p], "g") for p in ("p75", "p90", "p95")),
                             *(number(change[p], "+g") for p in ("p75", "p90", "p95"))])
            user = scenario["user_median_gaps"][window]
            user_rows.append([key, title, f"{user['contributing_users']:,}",
                              f"{user['one_gap_users']:,}", number(user["one_gap_share"], ".4%"),
                              number(user["p90"], "g")])
        cohort = scenario["cohort"]
        status_rows.append([key, f"{cohort['eligible_users']:,}",
                            *(f"{cohort['status_counts'][name]:,}" for name in EXPECTED_STATUS),
                            number(cohort["inactivity_rate"], ".4%"),
                            number(cohort["inactivity_delta_pp"], "+.4f")])
        for band, row in cohort["recency_bands"].items():
            band_rows.append([key, band, f"{row['users']:,}", f"{row['returned_users']:,}",
                              number(row["return_rate"], ".4%"), number(row["return_delta_pp"], "+.4f")])
        names = {"overall": "전체", "buyer_before_reference": "사전 구매", "nonbuyer_before_reference": "사전 비구매"}
        for segment, row in cohort["risk"].items():
            risk_rows.append([key, names[segment], f"{row['users']:,}", f"{row['returned_users']:,}",
                              number(row["baseline"], ".4%"), number(row["return_delta_pp"], "+.4f"),
                              number(row["n_per_arm"], ","), number(row["sample_size_delta"], "+,")])
    push_table(lines, ["시나리오", "기간", "간격 수", "제외 건수", "제외 비중",
                       "p75", "p90", "p95", "Δp75(일)", "Δp90(일)", "Δp95(일)"], gap_rows)
    lines.extend(["## 사용자 단위 p90", ""])
    push_table(lines, ["시나리오", "기간", "기여 사용자", "간격 1개 사용자", "간격 1개 비중",
                       "개인별 중앙값의 p90(일)"], user_rows)
    lines.extend(["## 관찰 기간 미활동", ""])
    push_table(lines, ["시나리오", "평가 대상", "미활동", "재방문", "판단 보류", "미활동 비율", "변화(%p)"],
               status_rows)
    lines.extend(["## Recency 구간", ""])
    push_table(lines, ["시나리오", "구간", "대상", "재방문", "재방문율", "변화(%p)"], band_rows)
    lines.extend(["## 위험군과 군당 표본", "",
                  "Cohen’s h · NormalIndPower · 양측 α=.05 · 검정력 .80 · 1:1 · MDE +5%p · 올림.", ""])
    push_table(lines, ["시나리오", "층", "대상", "재방문", "재방문율", "변화(%p)", "군당 표본", "변화(명)"],
               risk_rows)
    lines.extend(["## 한계", "", *("- " + limitation for limitation in LIMITATIONS)])
    return "\n".join(lines) + "\n"


def main():
    head = subprocess.check_output(
        ["git", "--no-optional-locks", "rev-parse", "--short=7", "HEAD"],
        cwd=config.PROJECT_ROOT, text=True).strip()
    if (head != BASE or config.AB_REFERENCE_DATE != "2020-03-26"
            or config.CHURN_INACTIVITY_DAYS != 35 or config.risk_bounds() != (14, 34)):
        raise ValueError("Unexpected baseline, reference date or risk bounds")
    run = config.PROC_DIR / "runs" / RUN_NAME
    directory = config.REPORTS_DIR / "sensitivity" / "abnormal_users"
    first = directory / "first_stop_ref20200326.json"
    output = directory / "comparison_ref20200326.json"
    summary = output.with_suffix(".md")
    if output.exists() or summary.exists():
        raise FileExistsError("Existing comparison outputs must be preserved")
    activity = config.RUN_PROC_DIR / "user_activity_days.parquet"
    module = Path(__file__).with_name("abnormal_users.py")
    membership, start = verify_membership(run, first, activity, module, EXPECTED_CANDIDATES)
    _, reviewed = load_reviewed_inputs()
    start["reviewed_inputs_passed"] = True
    print("Start gate passed", flush=True)

    with TemporaryDirectory(prefix="abnormal-comparison-") as temp, closing(duckdb.connect()) as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GiB'")
        db.execute("SET TimeZone='UTC'")
        db.execute("SET temp_directory=?", [temp])
        db.execute("SET max_temp_directory_size='4GiB'")
        cells, matched = cohort_cells(db, reviewed, membership)
        if matched != EXPECTED_CANDIDATES:
            raise ValueError(f"Fixed candidates do not match the reviewed population: {matched}")
        cohort = cohort_summary(cells, 0)
        gaps = collect_gaps(db, activity, membership)
    baseline_intervals = {key: interval_summary(data, 0) for key, data in gaps.items()}
    baseline_gate = verify_baseline(cohort, baseline_intervals)
    print("No-exclusion reproduction gate passed", flush=True)

    report = {"public_base": BASE, "reference_date": config.AB_REFERENCE_DATE,
              "risk_range": [14, 34], "start_gate": start, "no_exclusion_gate": baseline_gate,
              "quantile_method": QUANTILE_METHOD,
              "gap_definition": "Days between consecutive observed UTC activity dates for a user",
              "pre_gap_definition": "Both dates on or before the UTC reference date",
              "user_quantile_definition": "p90 of each user's median completed gap, including one-gap users",
              "sample_size_method": {"effect": "Cohen's h", "solver": "statsmodels NormalIndPower",
                                     "alpha": .05, "power": .8, "ratio": 1, "alternative": "two-sided",
                                     "target_increment": .05, "rounding": "ceil"},
              "limitations": LIMITATIONS, "scenarios": {}}
    for scenario, bit in zip(("S0", *SCENARIOS), (0, 1, 2, 4, 8), strict=True):
        scenario_cohort = cohort_summary(cells, bit)
        expected = 0 if scenario == "S0" else EXPECTED_CANDIDATES[scenario]
        if (cohort["eligible_users"] - scenario_cohort["eligible_users"] != expected
                or scenario_cohort["status_counts"]["censored"] != EXPECTED_STATUS["censored"]):
            raise ValueError(f"Unexpected cohort row removal: {scenario}")
        report["scenarios"][scenario] = comparison_row(
            scenario_cohort, {key: interval_summary(data, bit) for key, data in gaps.items()},
            {key: user_median_summary(data, bit) for key, data in gaps.items()}, cohort, baseline_intervals)

    _, end = verify_membership(run, first, activity, module, EXPECTED_CANDIDATES)
    if end != {key: value for key, value in start.items() if key != "reviewed_inputs_passed"}:
        raise ValueError("First-stop inputs changed during comparison")
    load_reviewed_inputs()
    rendered = markdown(report)
    write_json_new(output, report)
    with summary.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    print("Comparison written", flush=True)


if __name__ == "__main__":
    main()
