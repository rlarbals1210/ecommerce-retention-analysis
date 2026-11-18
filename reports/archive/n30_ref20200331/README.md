# 30일 기준 산출물 보관본

기준일은 2020-03-31, 결과 관찰창은 2020-04-01~04-30이다. 아래 산출물은 당시 바이트 그대로 보존했다. 검증 파일 안의 경로와 상태는 당시 실행 기록이다.

- [A/B 요약](ab_design_summary.csv) · [구매 이력 정정 기록](ab_design_validation.json) · [Recency 구간](recency_segment_comparison.csv) · [Recency 검증](recency_reference_validation.json)
- [10/26 민감도 분석 원문](recency_threshold_sensitivity.md) · [구간 CSV](recency_threshold_sensitivity.csv) · [검정 CSV](recency_threshold_sensitivity_tests.csv) · [임계값 CSV](recency_threshold_sensitivity_thresholds.csv) · [검증 JSON](recency_threshold_sensitivity_validation.json)
- Tableau KPI·이탈 집계·워크북은 `tableau/`, 기존 간격 그림은 `figures/`에 있다.

당시 로컬 가공 파일을 가진 경우 `python -m src.recency_threshold_sensitivity --out-dir /path/to/new-replay-directory`로 별도 폴더에 재계산할 수 있다. 기존 파일은 해시 검증 입력으로 사용한다.
