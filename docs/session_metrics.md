# 세션 이벤트 집계 정의

분석 단위는 월별 고유 `user_session`이다. 7개월 값은 월별 고유 세션 수를 합산한다.
월 경계 중복·이벤트 순서·동일 상품 전환·모든 세션의 view 포함 여부는 검증하지 않았다.

| CSV 열 | 분자 / 분모 또는 의미 |
|---|---|
| total_sessions | 해당 월 고유 세션 수 |
| cart_sessions | cart 이벤트가 있는 세션 수 |
| purchase_sessions | purchase 이벤트가 있는 전체 세션 수 |
| cart_and_purchase | 두 이벤트가 모두 있는 세션 수(교집합) |
| purchase_no_cart | purchase는 있으나 cart 기록은 없는 세션 수 |
| cart_share_of_sessions | cart_sessions / total_sessions |
| purchase_share_within_cart | cart_and_purchase / cart_sessions |
| cart_without_purchase_rate | (cart_sessions − cart_and_purchase) / cart_sessions |
| pct_of_total_sessions | 해당 집합의 세션 수 / 전체 세션 수 |
| share_of_parent_set | 전체=1, cart/전체, 교집합/cart |

`funnel_overall.csv`의 stage는 `all_sessions`, `cart_present`, `cart_and_purchase`다.
이전 열 이름 `pct_of_view`, `view_to_cart_rate`, `cart_to_purchase_rate`, `cart_abandonment_rate`, `step_conversion`은 위 정의가 드러나는 이름으로 교체했다. Tableau의 연결 필드와 표시 명칭도 함께 바꿨다.

비율은 월별 분자·분모 합계로 계산한다. 월별 비율의 단순 평균을 사용하지 않는다.
집계 CSV는 원본에서 이미 계산한 카운트의 재현 입력이다. 이를 검증하는 것은 원본 전체를 다시 검증하는 것과 구분한다.

Tableau KPI의 구매 경험자 비율·반복 구매율은 7개월 기술통계, 4월 미활동 비율은 3/31까지 관측된 사용자의 4월 활동 여부다. 서로 분모가 다르며 확정적인 영구 이탈률이 아니다.
