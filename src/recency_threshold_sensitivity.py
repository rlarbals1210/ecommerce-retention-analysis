"""Post-submission, fixed-landmark sensitivity analysis; no raw CSV scan.

Run: python -m src.recency_threshold_sensitivity --out-dir /path/to/new-replay-directory
Inputs are the previously validated daily counts and their unchanged local caches.
Existing definitions, notebooks and public portfolio text are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import (
    confint_proportions_2indep, proportion_confint, proportions_ztest,
)

import config


ANALYSIS_DATE = "2026-10-26"  # Client date; analysis specification is fixed before execution.
ALPHA = 0.05
BANDS = (("25-29", 25, 29), ("30-34", 30, 34), ("35-39", 35, 39),
         ("35+", 35, None), ("30+", 30, None))
COMPARISONS = (("primary", "30-34", "35-39"),
               ("secondary_exploratory", "30-34", "35+"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_distribution(rows, expected_users):
    require(bool(rows), "Empty daily distribution")
    keys = set()
    for row in rows:
        require(row.get("recency_state") == "valid", "Invalid recency bucket")
        day, count = row["recency_at_ref"], row["users"]
        require(type(day) is int and 0 <= day <= 182, "Invalid day")
        require(type(count) is int and count > 0, "Invalid user count")
        require(type(row["april_active"]) is bool, "Outcome must be Boolean")
        require(type(row["is_buyer_at_reference"]) is bool, "Buyer flag must be Boolean")
        key = (day, row["is_buyer_at_reference"], row["april_active"])
        require(key not in keys, "Duplicate daily cell")
        keys.add(key)
    require(sum(r["users"] for r in rows) == expected_users, "Eligible count mismatch")


def counts(rows, lower, upper=None, buyer=None):
    chosen = [r for r in rows if r["recency_at_ref"] >= lower
              and (upper is None or r["recency_at_ref"] <= upper)
              and (buyer is None or r["is_buyer_at_reference"] == buyer)]
    return (sum(r["users"] for r in chosen),
            sum(r["users"] for r in chosen if r["april_active"]))


def check_existing_table(rows, path):
    """Compare integer numerators/denominators, not rounded percentage strings."""
    bounds = {"0-7": (0, 7), "8-14": (8, 14), "15-29": (15, 29), "30+": (30, None)}
    strata = {"overall": None, "buyer_before_reference": True,
              "nonbuyer_before_reference": False}
    with Path(path).open(newline="") as stream:
        records = list(csv.DictReader(stream))
    expected_keys = {(band, segment) for band in bounds for segment in strata}
    require(len(records) == len(expected_keys), "Existing table has unexpected rows")
    require({(r["recency_band"], r["segment"]) for r in records} == expected_keys,
            "Existing table keys differ")
    for r in records:
        n, returned = counts(rows, *bounds[r["recency_band"]], buyer=strata[r["segment"]])
        require((n, returned) == (int(r["users"]), int(r["returned_users"])),
                f"Existing counts differ: {r['recency_band']}/{r['segment']}")
        require(math.isclose(returned / n, float(r["baseline"]), abs_tol=1e-14, rel_tol=0),
                "Existing rate differs")
    return len(records)


def analyze(rows):
    bands = []
    for label, lo, hi in BANDS:
        n, returned = counts(rows, lo, hi)
        require(n > 0, f"Empty band: {label}")
        low, high = proportion_confint(returned, n, alpha=ALPHA, method="wilson")
        bands.append(dict(recency_band=label, users=n, returned_users=returned,
                          return_rate=returned / n, ci95_low=float(low), ci95_high=float(high)))
    by_band = {r["recency_band"]: r for r in bands}
    for column in ("users", "returned_users"):
        require(by_band["30+"][column] == by_band["30-34"][column] + by_band["35+"][column],
                "Threshold partition mismatch")
    tests = []
    for role, a, b in COMPARISONS:
        left, right = by_band[a], by_band[b]
        successes = np.array([left["returned_users"], right["returned_users"]])
        totals = np.array([left["users"], right["users"]])
        pooled = successes.sum() / totals.sum()
        require(np.all(totals * pooled >= 5) and np.all(totals * (1 - pooled) >= 5),
                "Normal approximation has sparse expected counts")
        z, pvalue = proportions_ztest(successes, totals, alternative="two-sided", prop_var=False)
        low, high = confint_proportions_2indep(
            successes[0], totals[0], successes[1], totals[1],
            compare="diff", method="newcomb", alpha=ALPHA)
        log10_p = (math.log(2) + float(norm.logsf(abs(z)))) / math.log(10)
        tests.append(dict(
            role=role, group_a=a, group_b=b, users_a=int(totals[0]), users_b=int(totals[1]),
            returned_a=int(successes[0]), returned_b=int(successes[1]),
            rate_a=left["return_rate"], rate_b=right["return_rate"],
            difference_pp=100 * (left["return_rate"] - right["return_rate"]),
            ci95_low_pp=100 * float(low), ci95_high_pp=100 * float(high),
            z_statistic=float(z), p_value=float(pvalue) if pvalue > 0 else "<1e-300",
            log10_p_value=log10_p, alpha=ALPHA,
            nominal_significant=bool(log10_p < math.log10(ALPHA)),
            test="two-sided pooled two-proportion z-test",
            ci_method="Newcombe difference of independent proportions",
        ))
    n30 = by_band["30+"]["users"]
    thresholds = []
    for threshold, band in ((30, "30+"), (35, "35+")):
        row = by_band[band]
        thresholds.append(dict(
            threshold_days=threshold, users=row["users"], returned_users=row["returned_users"],
            return_rate=row["return_rate"], removed_vs_30=n30-row["users"],
            removed_share_of_30=(n30-row["users"])/n30))
    return bands, tests, thresholds


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(bands, tests, thresholds, provenance):
    by = {r["recency_band"]: r for r in bands}
    primary = tests[0]
    removed = thresholds[1]
    p = lambda t: "< 1e-300" if isinstance(t["p_value"], str) else f"{t['p_value']:.4g}"
    lines = [
        "# 30일·35일 무활동 기준 추가 비교", "",
        f"분석일: {ANALYSIS_DATE} (Asia/Seoul). 제출 후 수행한 별도 보완 분석이며 기존 기준의 변경 여부는 결과 검토 후 결정한다.", "",
        "## 결과", "",
        f"30~34일 무활동 고객의 4월 활동률은 {by['30-34']['return_rate']:.2%}, 35~39일은 {by['35-39']['return_rate']:.2%}다. "
        f"차이는 {primary['difference_pp']:.2f}%p (95% CI {primary['ci95_low_pp']:.2f}~{primary['ci95_high_pp']:.2f}%p), "
        f"양측 p {p(primary)}다.", "",
        "이는 동일 기준일에서 무활동 기간이 다른 고객들의 이후 활동률 차이다. 30일과 35일의 정확도, 최적 기준, 5일 대기의 효과 또는 캠페인 효과를 검증한 결과로 해석하지 않는다.", "",
        "| 기준일 무활동 일수 | 사용자 수 | 4월 활동자 수 | 4월 활동률 | 95% 신뢰구간 (Wilson) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in bands:
        lines.append(f"| {row['recency_band']} | {row['users']:,} | {row['returned_users']:,} | {row['return_rate']:.2%} | {row['ci95_low']:.2%}~{row['ci95_high']:.2%} |")
    lines += ["", "35+는 35~39를 포함하고 30+는 30~34와 35+를 포함한다. 이 표의 행들은 상호 배타적인 전체 분할이 아니므로 합산하지 않는다.", "",
        f"기준을 30일에서 35일로 올리면 같은 기준일의 대상자는 {by['30+']['users']:,}명에서 {by['35+']['users']:,}명으로 "
        f"{removed['removed_vs_30']:,}명({removed['removed_share_of_30']:.2%}) 감소한다. "
        f"제외되는 30~34일 고객 중 {by['30-34']['returned_users']:,}명은 이후 4월에 활동했고, "
        f"{by['30-34']['users']-by['30-34']['returned_users']:,}명은 4월 활동이 관측되지 않았다.", "",
        "이 감소 규모는 고정 시점의 대상 풀 변화다. 일별 캠페인 유입량·발송량·비용 절감액과 같지 않으며, 4월 활동자는 처치 이력이 확인된 미발송 대조군도 아니다.", "",
        "## 비교 설계", "",
        "- 대상: UTC 2020-03-31 하루 종료까지 한 번이라도 관측된 사용자. 가입 고객을 의미하지 않는다.",
        "- 입력: 2020-04-01 00:00 UTC 이전 마지막 활동일과 2020-03-31의 날짜 차이. 기준일 이후 정보는 Recency에 사용하지 않는다.",
        "- 결과: 2020-04-01~2020-04-30의 활동 유무. 모두 동일한 후속 30일을 관찰한다. 4월 최초 관측자는 기존 라벨의 censored로 제외한다.",
        "- 주 비교: 30~34일 vs 35~39일. 경계 주변의 서로 다른 사용자 집단을 비교한다.",
        "- 보조 탐색 비교: 30~34일 vs 35일 이상. 후자는 오래 비활동한 고객까지 포함하므로 구성 차이가 더 크다.",
        "- 양측 두 비율 z검정, 유의수준 0.05. 구간별 Wilson 95% CI, 비율 차이 Newcombe 95% CI. 보조 비교는 다중검정 보정 없는 탐색 결과이며 독립적인 재현 증거가 아니다.",
        "- 비율 차이는 A−B, 단위는 %p. 극소 p값의 부동소수점 언더플로는 0으로 주장하지 않고 <1e-300과 log10(p)로 기록한다.", "",
        "| 비교 | 구간 A / B | 차이 (%p) | 차이 95% CI (%p) | p값 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in tests:
        lines.append(f"| {row['role']} | {row['group_a']} / {row['group_b']} | {row['difference_pp']:.2f} | {row['ci95_low_pp']:.2f}~{row['ci95_high_pp']:.2f} | {p(row)} |")
    lines += ["", "## 결과를 설명하는 범위", "",
        f"주 비교에서 관측된 차이는 {primary['difference_pp']:.2f}%p이며 95% 신뢰구간은 0을 포함하지 않는다. 따라서 이 두 구간의 이후 활동률에 통계적으로 유의한 차이가 관측됐다고 설명할 수 있다. 다만 특정 날짜에 급격한 단절이 있다고 검정한 설계는 아니므로 30일 또는 35일이 명확한 경계라고 주장하지 않는다.", "",
        "30일을 유지하면 30~34일 고객도 대상에 포함하고, 35일로 높이면 이들을 대상에서 제외한다. 그 선택의 가치는 개입 목표와 비용·실험 효과에 따라 달라진다. 관측된 재방문율만으로 어느 쪽이 낫다고 판정하지 않는다.", "",
        "면접에서 추가 분석의 시점을 포함해 설명할 수 있는 문장:", "",
        f"> 제출 당시에는 재방문 간격의 90백분위수인 35일을 참고해 30일 기준을 설정했습니다. 제출 후에는 기준의 근거를 보완하기 위해 같은 기준일의 무활동 구간별 이후 30일 활동률을 비교했습니다. 30~34일과 35~39일 구간은 각각 {by['30-34']['return_rate']:.2%}, {by['35-39']['return_rate']:.2%}였고 차이는 통계적으로 유의했습니다. 다만 서로 다른 고객 집단의 관찰 비교여서 최적 기준이나 캠페인 효과가 입증된 것은 아닙니다.", "",
        "## 해석과 한계", "",
        "- p값은 비율 차이에 관한 표본모형 하의 수치다. 분석 대상은 이 데이터 내 관측 사용자 전수이며, 다른 시기·서비스로 일반화하거나 원본 로그 오류를 배제하는 근거가 아니다.",
        "- 비유의는 동등성 증명이 아니고, 유의함도 실무적으로 중요한 차이 또는 좋은 이탈 기준의 증명이 아니다. 판정할 실무적 차이의 최소값은 정하지 않았으므로 사후에 임의의 기준을 붙이지 않는다.",
        "- 같은 사용자가 무활동 30일에서 35일까지 어떻게 변하는지를 추적한 설계가 아니다. 30~34일과 35~39일 사용자는 다르며, 구매 이력·유입 시점·활동 빈도·계절성 등 구성 차이가 남는다. 비율 검정은 사용자 간 독립성을 가정하며 상호의존성은 검증하지 않았다.",
        "- 영구 이탈의 정답 라벨이 없으므로 정확도·오분류율·최적 임계값을 계산하지 않았다. 다음 30일 미활동은 그 기간의 미활동일 뿐이다.",
        "- 단일 기준일이며 다른 월의 반복 검증은 없다. 기준일과 관찰창은 고정하고 무활동 구간만 비교했으며, 35일 후속 관찰창으로 재라벨링하지 않았다.",
        "- 기존 지표의 4월 미활동 라벨과 기준일의 Recency≥30 상태는 다르다. 기존 CHURN_INACTIVITY_DAYS=30·라벨·A/B 설계·README·노션·Tableau·제출본을 이 분석으로 수정하지 않는다.",
        "- 기본 결정은 보류다. 관찰 차이만으로 30일 유지 또는 35일 변경을 자동 결정하지 않고, 사용자가 결과를 검토한 뒤 반영 여부를 결정한다.", "",
        "## 검증과 재현", "",
        "```bash", "python -m src.recency_threshold_sensitivity", "python -m unittest discover -s tests -p 'test_recency_threshold_sensitivity.py' -v", "```", "",
        "- 입력은 reports/recency_reference_validation.json의 daily_distributions.candidate. 사용자 ID를 포함하지 않는 집계 732행만 메모리에 읽는다. 원본 CSV를 열거나 사용자 전체를 다시 병합하지 않는다.",
        "- candidate·기존 recency·labels·buyers 파일의 SHA256을 기존 보고서와 대조한다. 파일 해시 일치는 이후 변경 여부 검사이며 원본에서의 독립 재검증이 아니다.",
        f"- 검증된 기존 CSV의 {provenance['existing_table_rows_checked']}개 구간/구매층별 분자·분모와 반올림 전 비율을 대조했다. 전체 {provenance['eligible_users']:,}명이 도수 분포에 보존되는지도 검사했다.",
        "- 기존 보고서의 공동 확인은 candidate 분석 사용과 독립 집계 검토에 관한 것이다. 원본 이벤트 독립 검토 미수행, 기존 캐시 출처 불명확, 일부 과거 파일 불변성 및 시간 측정 한계, 과거 공개 승인 보류 표시는 그대로 유지한다. 이번 분석이 해당 과거 게이트를 변경하지 않는다.",
        "- 입력 식별자·버전·분석 명세·정합성 검사는 reports/recency_threshold_sensitivity_validation.json, 전체 정밀 수치는 같은 이름의 CSV 파일들을 참조한다.", "",
        "통계 방법: [statsmodels 두 비율 검정](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html), "
        "[Newcombe 비율 차이 신뢰구간](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.confint_proportions_2indep.html).", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Fresh replay output directory")
    args = parser.parse_args()
    archive = config.LEGACY_N30_REPORTS_DIR
    source = archive / "recency_reference_validation.json"
    existing_table = archive / "recency_segment_comparison.csv"
    legacy = config.LEGACY_N30_PROC_DIR
    reference_date = config.LEGACY_N30_REFERENCE_DATE
    report = json.loads(source.read_text())
    require(reference_date == "2020-03-31", "Unexpected legacy reference date")
    require(report["decision_status"] == "jointly_confirmed_branch_1", "Unconfirmed candidate")
    require(report["joint_review"]["candidate_approved_for_analysis"], "Candidate not approved for analysis")
    require(report["scan_diagnostics"]["reference_date"] == reference_date,
            "Report reference mismatch")
    require(report["scan_diagnostics"]["cutoff_exclusive_utc"] == "2020-04-01T00:00:00+00:00",
            "Report cutoff mismatch")
    inputs = {
        "candidate": legacy / "recency_rebuild" / "recency_at_reference.candidate.parquet",
        "old": legacy / "recency_at_reference.parquet",
        "labels": legacy / "churn_labels.parquet",
        "buyers": legacy / "purchase_users_at_reference.parquet",
    }
    identities = {}
    for key, path in inputs.items():
        digest = sha256(path)
        require(digest == report["input_files"][key]["sha256"], f"Validated input changed: {key}")
        identities[key] = dict(path=str(path.relative_to(config.PROJECT_ROOT)), sha256=digest)
    rows = report["daily_distributions"]["candidate"]
    validate_distribution(rows, report["eligible_users"])
    require(rows == report["daily_distributions"]["old"], "Old/candidate distribution differs")
    existing_rows = check_existing_table(rows, existing_table)
    bands, tests, thresholds = analyze(rows)
    provenance = dict(
        analysis_date=ANALYSIS_DATE, reference_date=reference_date,
        observation_start="2020-04-01", observation_end="2020-04-30", timezone="UTC",
        analysis="fixed-landmark observational comparison, not a threshold optimization",
        primary_comparison="30-34 vs 35-39", secondary_comparison="30-34 vs 35+ (exploratory)",
        existing_table_rows_checked=existing_rows, distribution_rows=len(rows),
        eligible_users=report["eligible_users"], total_april_active=counts(rows, 0)[1],
        source_report_sha256=sha256(source), source_files=identities,
        existing_table_sha256=sha256(existing_table),
        analysis_code_sha256=sha256(Path(__file__)),
        versions={name: version(name) for name in ["numpy", "scipy", "statsmodels"]},
        python_version=platform.python_version(), alpha=ALPHA,
        raw_events_recomputed=False, threshold_changed=False,
        portfolio_reflection="pending_user_review",
        historical_report_adopted=report["adopted"], historical_report_published=report["published"],
        limitations=report["limitations"], checks_passed=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)
    prefix = args.out_dir / "recency_threshold_sensitivity"
    write_csv(prefix.with_suffix(".csv"), bands)
    write_csv(prefix.with_name(prefix.name + "_tests.csv"), tests)
    write_csv(prefix.with_name(prefix.name + "_thresholds.csv"), thresholds)
    prefix.with_name(prefix.name + "_validation.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    doc = args.out_dir / "recency_threshold_sensitivity.md"
    doc.write_text(markdown_report(bands, tests, thresholds, provenance), encoding="utf-8")
    print(json.dumps(dict(bands=bands, comparisons=tests, thresholds=thresholds,
                          report=str(doc), checks_passed=True), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
