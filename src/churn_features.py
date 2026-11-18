"""UTC calendar-day labels and reproducible completed-gap summaries for notebook 04."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import config

OPERATING_BASIS = '상한은 전체 7개월의 관측 재방문 간격 p90을 참고해 확정한 35일 기준 미만으로 둔다. 하한은 사후 정보를 사용하지 않도록 기준일 이전 분포의 p75 + 1일로 정한다. 전체 기간과 사전 기간의 분위수를 나란히 보고하며, p75·p90의 선택은 운영 기준이고 최적값으로 검증한 것은 아니다.'


def label_activity_days(daily, *, reference_date=None, dataset_end=None):
    reference = pd.Timestamp(reference_date or config.AB_REFERENCE_DATE, tz="UTC")
    end = pd.Timestamp(dataset_end or config.DATASET_END_DATE, tz="UTC")
    if not isinstance(daily["date"].dtype, pd.DatetimeTZDtype) or str(daily["date"].dt.tz) != "UTC":
        raise ValueError("Activity dates must use UTC timestamps")
    if daily["date"].isna().any() or not daily["date"].eq(daily["date"].dt.normalize()).all():
        raise ValueError("Activity dates must be nonmissing UTC calendar days")
    if daily["date"].max() != end:
        raise ValueError("Observed dataset end differs from configured end")
    if (end - reference).days != config.CHURN_INACTIVITY_DAYS:
        raise ValueError("Observation duration differs from N")
    before = pd.Index(daily.loc[daily.date.le(reference), "user_id"].unique())
    after = pd.Index(daily.loc[daily.date.gt(reference) & daily.date.le(end), "user_id"].unique())
    grouped = daily.groupby("user_id")["date"]
    labels = pd.DataFrame({"first_activity_date": grouped.min(), "last_activity_date": grouped.max()})
    labels["status"] = "censored"
    labels.loc[labels.index.isin(before), "status"] = "churned"
    labels.loc[labels.index.isin(before.intersection(after)), "status"] = "retained"
    return labels


def gap_summary(sorted_daily):
    if sorted_daily.duplicated(["user_id", "date"]).any():
        raise ValueError("Duplicate user/calendar-day pairs")
    ordered = sorted_daily.sort_values(["user_id", "date"])
    gaps = ordered.groupby("user_id", sort=False)["date"].diff().dt.days.dropna()
    if gaps.empty or gaps.le(0).any():
        raise ValueError("Completed gaps must be positive")
    counts = gaps.astype("int64").value_counts().sort_index()
    quantiles = gaps.quantile([.75, .90, .95], interpolation="linear")
    return {"gap_count": int(len(gaps)),
            "contributing_users": int(ordered.loc[ordered.groupby("user_id", sort=False).cumcount().gt(0), "user_id"].nunique()),
            "p75": float(quantiles.loc[.75]), "p90": float(quantiles.loc[.9]),
            "p95": float(quantiles.loc[.95]), "quantile_method": "pandas linear",
            "gap_histogram": [{"gap_days": int(k), "count": int(v)} for k, v in counts.items()]}


def pre_reference_evidence(daily, full_summary):
    ref = pd.Timestamp(config.AB_REFERENCE_DATE, tz="UTC")
    pre = gap_summary(daily.loc[daily.date.le(ref), ["user_id", "date"]])
    lower = pre["p75"] + 1
    conflict = not float(lower).is_integer() or int(lower) != 15
    return {"run_id": config.RUN_ID, "reference_date": config.AB_REFERENCE_DATE,
            "feature_end_exclusive": config.AB_FEATURE_END_EXCLUSIVE,
            "review_status": "pending_user_confirmation_stage3", "pre_reference": pre,
            "full_period": full_summary, "proposed_risk_lower_days": lower,
            "risk_upper_days": config.CHURN_INACTIVITY_DAYS - 1,
            "existing_bands_require_decision": conflict,
            "basis": OPERATING_BASIS,
            "limitations": ["Completed gaps only; users who have not returned contribute no unfinished interval.",
                            "Frequent users contribute multiple gaps; these are not user-level return percentiles.",
                            "A shorter observation window may observe fewer long gaps.",
                            "Stage-3 stop is not a validity judgment on N=35."]}
