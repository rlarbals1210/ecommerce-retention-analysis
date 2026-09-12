"""
프로젝트 전역 경로/설정.
모든 노트북·스크립트는 하드코딩 대신 이 파일을 import해서 경로를 참조한다.
"""
import os
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
# 04_churn_features에서 7개월 전체 유저 재방문 간격을 재검증해 확정 (90번째 percentile인
# 35일에 근접하면서 운영 시점에 해석하기 쉬운 30일 기준으로 설정).
CHURN_INACTIVITY_DAYS = 30

RANDOM_SEED = 42

# Existing recency cache uses UTC calendar days through this date.
AB_REFERENCE_DATE = "2020-03-31"
