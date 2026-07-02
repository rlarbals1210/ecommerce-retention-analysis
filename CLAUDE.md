# commerce-churn-analysis

이커머스 고객 이탈(churn)·리텐션 분석 포트폴리오 프로젝트.
목적은 "모델 정확도 자랑"이 아니라 코호트 분석·퍼널 분석·A/B 테스트 설계 3가지 방법론으로
**비즈니스 문제를 정의하고 데이터로 푸는 과정**을 보여주는 것.

## 데이터 (중요: 읽기 전용, 절대 수정/이동/복사 금지)
- 위치: `/Volumes/SSD/Mac/이탈분석/` (외장 SSD)
- 파일: `2019-Oct.csv` ~ `2020-Apr.csv` (7개월치 월별 CSV, 총 약 56GB, 파일당 5~9GB)
- REES46 multi-category store 공개 데이터셋
- 컬럼 9개: `event_time, event_type(view/cart/remove_from_cart/purchase), product_id,
  category_id, category_code, brand, price, user_id, user_session`
- 절대 `pd.read_csv`로 통째로 읽지 말 것. 필요한 컬럼만 + dtype 지정(`config.DTYPES`) +
  필요시 `user_id` 기준 5~10% 샘플링으로 처리. 큰 연산 전에는 메모리 전략을 먼저 설명하고
  실행 전에 사용자 확인을 받을 것.

## 경로/설정
- 모든 경로·스키마는 `config.py` 한 곳에서만 관리 (하드코딩 금지).
- `config.DATA_DIR`: SSD 원본 경로. `config.PROC_DIR`: 가공 데이터 저장 위치
  (`data/processed/`, git에는 커밋 안 됨).
- 원본 CSV는 읽기만 하고, 전처리 결과는 `data/processed/`에 저장.

## 방법론 (합의된 정의 — 임의로 바꾸지 말 것)
- **코호트**: 사용자 첫 활동(또는 첫 구매) 월 기준, 경과 개월별 잔존율 히트맵.
- **퍼널**: `user_session` 단위로 view→cart→purchase 단계 도달 여부를 플래그화해서 전환율
  계산. ⚠️ 당초 계획한 `remove_from_cart`는 보유한 7개월 데이터(2019-10~2020-04) 전체에
  존재하지 않음 (01_eda에서 확인). 장바구니 이탈은 "cart 담았지만 같은 세션에서 purchase로
  안 이어진 비율"로 우회 측정.
- **이탈 정의**: 마지막 활동 후 N일 무활동. N은 아직 미정 — `01_eda` 결과(데이터 기간·활동
  주기)를 보고 근거와 함께 확정 (`config.CHURN_INACTIVITY_DAYS`).
- **A/B**: 실제 실험이 아니라, 위험 세그먼트 대상 리텐션 캠페인의 A/B **설계 문서**
  (대상·지표·표본크기·기간·검정방법)를 `05_ab_test_design.ipynb`에 작성.

## 작업 순서 (현재 진행 상황)
1. [완료] 폴더 스캐폴딩 + `config.py` + `.gitignore` + `requirements.txt` + `README.md`
2. [완료] `src/load.py`: `load_month`, `load_months` 구현 (usecols+dtype 지정 로드).
   실행 환경은 `uv venv .venv` + `requirements.txt`로 구성됨.
3. [완료] `01_eda.ipynb`: 2019-10, 2019-11 두 달로 1~5단계 전부 완료
   (구조/기간/이벤트분포/결측/유저·세션 통계).
4. [완료] `02_cohort_retention.ipynb`: 7개월 전체, 첫 활동(view 포함)월 기준 코호트.
   달마다 `user_id`만 읽어 고유유저만 추출하는 방식으로 가볍게 처리 (activity_long
   2,853만 행/456MB, 원본 이벤트 수억 건 대비 훨씬 가벼움).
5. [보류] `03_funnel` ~ `05_ab_test_design`: 이어서 진행

## EDA로 확인된 사실 (중요 — 방법론에 반영됨)
- 7개 파일 모두 파일명이 가리키는 월 전체를 정확히 커버함 (기간 이상 없음)
- 이벤트 타입은 `view/cart/purchase` 3종류뿐. **`remove_from_cart`는 7개월 전체에 없음**
  (당초 계획 대비 데이터 제약 — 퍼널의 장바구니 이탈 측정 방식을 "cart 담았지만 같은
  세션에서 purchase로 이어지지 않은 비율"로 변경하기로 합의함)
- 결측치: `category_code` 32.2%, `brand` 14.0%. 나머지 컬럼은 결측 없음
- 2019-10~11월 이벤트 비율: view 94.9% / cart 3.6% / purchase 1.5%
- 유저당 이벤트 수 중앙값 5회(평균 20.7회, 헤비유저·봇 추정 극단치 있음), 유저당 세션 수
  중앙값 2회, 세션당 이벤트 수 중앙값 2회
- 재방문 간격: 중앙값 3일 / 평균 6.5일 / 75%가 8일 이내 (2달 스냅샷 기준 — 실제 이탈
  기준일 N은 04_churn_features에서 7개월 전체로 재확인 필요)
- `user_session`(UUID, string dtype)은 컬럼 3개만 로드해도 6.6GB로 무거움 → 세션을 다루는
  03_funnel 등에서는 category dtype 전환을 검토할 것 (`config.DTYPES`의 `user_session`은
  현재 "string"으로 되어 있음)
- **코호트 잔존율(02)**: 코호트 크기는 매달 증가(10월 302만→4월 451만)하지만, 1개월차
  잔존율은 코호트가 최신일수록 하락(10월 46%→3월 20%). 11~12월 블랙프라이데이/연말 시즌
  일회성 방문자 유입이 원인으로 추정 — 04_churn_features/05_ab_test_design에서 "일회성
  구매자 vs 반복 구매자" 세그먼트 구분에 참고할 것
- 코호트 히트맵 구현 시 주의: 미래(미관측) 코호트×경과개월 조합은 `unstack(fill_value=0)`
  대신 기본값(NaN)으로 남겨야 함. 0으로 채우면 "아직 관측 안 됨"과 "실제 0% 잔존"이
  구분 안 돼서 오독 소지 (실제로 처음엔 0으로 채웠다가 수정한 이력 있음)

## 작업 방식 (사용자 선호)
- 큰 코드를 작성하기 전에 접근 계획을 먼저 제시하고 확인을 받을 것.
- pandas 중심. 무거운 연산은 실행 전에 메모리 전략(컬럼 선택/dtype/샘플링)을 먼저 설명.
