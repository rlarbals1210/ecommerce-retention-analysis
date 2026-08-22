-- REES46 월별 코호트 잔존율 재현 쿼리 (DuckDB)
WITH monthly_activity AS (
    SELECT DISTINCT
        user_id,
        CAST(date_trunc('month', event_time) AS DATE) AS activity_month
    FROM read_parquet($event_glob)
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
