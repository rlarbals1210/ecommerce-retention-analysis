"""
프로젝트 전역 경로/설정.
모든 노트북·스크립트는 하드코딩 대신 이 파일을 import해서 경로를 참조한다.
"""
import os
import json
import hashlib
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent

# 로컬 전용 값(예: SSD 마운트 경로)은 .env에 저장하고 여기서 읽어들인다.
# .env는 .gitignore 처리되어 있어 사람마다 다른 경로를 코드 변경 없이 반영할 수 있다.
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# 원본 데이터 (외장 SSD, 읽기 전용 — 절대 수정/이동/복사 금지)
# ---------------------------------------------------------------------------
DATA_DIR = Path(
    os.environ.get("ECOM_DATA_DIR", PROJECT_ROOT / "data" / "raw")
).expanduser()

# REES46 multi-category store 데이터셋, 월별 CSV 7개 (2019-10 ~ 2020-04)
RAW_FILES = [
    "2019-Oct.csv",
    "2019-Nov.csv",
    "2019-Dec.csv",
    "2020-Jan.csv",
    "2020-Feb.csv",
    "2020-Mar.csv",
    "2020-Apr.csv",
]

RAW_PATHS = [DATA_DIR / f for f in RAW_FILES]

# ---------------------------------------------------------------------------
# 프로젝트 로컬 경로 (git에는 올라가지 않음, .gitignore 처리됨)
# ---------------------------------------------------------------------------
DATA_LOCAL_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_LOCAL_DIR / "raw"          # 원본을 복사하지 않음. 참조용 디렉토리만 유지
INTERIM_DIR = DATA_LOCAL_DIR / "interim"  # 샘플링/중간 가공 결과 (parquet 등)
PROC_DIR = DATA_LOCAL_DIR / "processed"   # 분석에 바로 쓰는 최종 가공 데이터

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _dir in (INTERIM_DIR, PROC_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 스키마 / dtype (메모리 최적화용). 대용량 CSV를 read_csv 할 때 반드시 지정.
# ---------------------------------------------------------------------------
COLUMNS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]

DTYPES = {
    "event_type": "category",
    "product_id": "int64",
    "category_id": "int64",
    "category_code": "category",
    "brand": "category",
    "price": "float32",
    "user_id": "int64",
    "user_session": "string",
}
# event_time은 파싱 비용 때문에 parse_dates로 별도 처리 (load.py 참고)

EVENT_TYPES = ["view", "cart", "remove_from_cart", "purchase"]

# ---------------------------------------------------------------------------
# 이탈(churn) 정의용 상수
# ---------------------------------------------------------------------------
# N=35 is an operating threshold informed by the full-period gap p90.
# The risk lower bound comes from pre-reference p75 + 1; it is not fixed here.
CHURN_INACTIVITY_DAYS = 35
DATASET_END_DATE = "2020-04-30"
AB_OBSERVATION_START = "2019-10-01"
AB_REFERENCE_DATE = (date.fromisoformat(DATASET_END_DATE) - timedelta(days=CHURN_INACTIVITY_DAYS)).isoformat()
AB_FEATURE_END_EXCLUSIVE = (date.fromisoformat(AB_REFERENCE_DATE) + timedelta(days=1)).isoformat() + "T00:00:00Z"
AB_OUTCOME_END_EXCLUSIVE = (date.fromisoformat(DATASET_END_DATE) + timedelta(days=1)).isoformat() + "T00:00:00Z"
MAX_RECENCY_DAYS = (date.fromisoformat(AB_REFERENCE_DATE) - date.fromisoformat(AB_OBSERVATION_START)).days
RUN_ID = f"n{CHURN_INACTIVITY_DAYS}_ref{AB_REFERENCE_DATE.replace('-', '')}"
RUN_PROC_DIR = PROC_DIR / "runs" / RUN_ID
RUN_FIGURES_DIR = RUN_PROC_DIR / "figures"
RUN_REPORTS_DIR = RUN_PROC_DIR / "reports"
LEGACY_N30_PROC_DIR = PROC_DIR
LEGACY_N30_REFERENCE_DATE = "2020-03-31"
LEGACY_N30_REPORTS_DIR = REPORTS_DIR / "archive" / "n30_ref20200331"
RANDOM_SEED = 42
RECENCY_REBUILD_DIR = RUN_PROC_DIR / "recency_rebuild"
RECENCY_SAMPLE_SIZE = 1_000
RECENCY_CHUNKSIZE = 500_000
RECENCY_VALIDATION_PATH = RUN_REPORTS_DIR / "recency_reference_validation.json"
RECENCY_CANDIDATE_PATH = RECENCY_REBUILD_DIR / "recency_at_reference.candidate.parquet"
RECENCY_COMPARISON_PATH = RUN_REPORTS_DIR / "recency_segment_comparison.csv"
RISK_BASIS_PATH = RUN_REPORTS_DIR / "pre_reference_gap_distribution.json"
RISK_APPROVAL_PATH = RUN_REPORTS_DIR / "risk_bound_user_approval.json"


def risk_bounds():
    """Only apply a user decision bound to the unchanged stage-3 evidence bytes."""
    evidence_bytes = RISK_BASIS_PATH.read_bytes()
    evidence = json.loads(evidence_bytes)
    approval = json.loads(RISK_APPROVAL_PATH.read_text())
    if approval.get("evidence_sha256") != hashlib.sha256(evidence_bytes).hexdigest():
        raise ValueError("Risk evidence hash does not match user approval")
    for obj in (evidence, approval):
        if obj.get("run_id") != RUN_ID or obj.get("reference_date") != AB_REFERENCE_DATE:
            raise ValueError("Risk evidence/approval belongs to another run")
    if approval.get("status") != "approved_by_user":
        raise ValueError("Stage-3 risk lower bound requires user confirmation")
    lower = evidence["pre_reference"]["p75"] + 1
    if not float(lower).is_integer():
        raise ValueError("Noninteger p75+1 requires explicit boundary decision")
    lower, upper = int(lower), CHURN_INACTIVITY_DAYS - 1
    decision = approval["decision"]
    if (decision.get("lower_days"), decision.get("upper_days"), decision.get("n_days")) != (lower, upper, CHURN_INACTIVITY_DAYS):
        raise ValueError("Computed risk bounds differ from approved decision")
    bands = ["0-7", f"8-{lower - 1}", f"{lower}-{upper}", f"{CHURN_INACTIVITY_DAYS}+"]
    if not 8 < lower <= upper or decision.get("recency_bands") != bands:
        raise ValueError("Approved Recency partition is incompatible")
    return lower, upper


def recency_bands():
    lower, upper = risk_bounds()
    return ("0-7", f"8-{lower-1}", f"{lower}-{upper}", f"{CHURN_INACTIVITY_DAYS}+")
