-- REES46 월별 CSV에서 월별 코호트 잔존율을 재현하는 DuckDB 쿼리.
-- 실행 예: python src/run_cohort_sql.py --glob "/path/to/2019-*.csv"
WITH monthly_activity AS (
    SELECT DISTINCT
        user_id,
        CAST(date_trunc('month', CAST(event_time AS TIMESTAMP)) AS DATE) AS activity_month
    FROM read_csv_auto($event_files, header = true, union_by_name = true)
),
cohorted AS (
    SELECT
        user_id,
        activity_month,
        MIN(activity_month) OVER (PARTITION BY user_id) AS cohort_month
    FROM monthly_activity
),
cohort_counts AS (
    SELECT
        cohort_month,
        date_diff('month', cohort_month, activity_month) AS month_number,
        COUNT(DISTINCT user_id) AS active_users
    FROM cohorted
    GROUP BY 1, 2
)
SELECT
    cohort_month,
    month_number,
    active_users,
    active_users::DOUBLE
        / FIRST_VALUE(active_users) OVER (
            PARTITION BY cohort_month ORDER BY month_number
        ) AS retention_rate
FROM cohort_counts
ORDER BY cohort_month, month_number;
