WITH source AS (
    SELECT * FROM raw.bureau
),
aggregated AS (
    SELECT
        sk_id_curr,
        COUNT(*)                                                AS bureau_loan_count,
        SUM(CASE WHEN credit_active='Active' THEN 1 ELSE 0 END) AS bureau_active_loans,
        SUM(CASE WHEN credit_active='Closed' THEN 1 ELSE 0 END) AS bureau_closed_loans,
        SUM(CASE WHEN credit_day_overdue > 0 THEN 1 ELSE 0 END) AS bureau_overdue_count,
        MAX(COALESCE(credit_day_overdue, 0))                    AS bureau_max_overdue_days,
        AVG(COALESCE(credit_day_overdue, 0))                    AS bureau_avg_overdue_days,
        SUM(COALESCE(amt_credit_sum, 0))                        AS bureau_total_credit,
        SUM(COALESCE(amt_credit_sum_debt, 0))                   AS bureau_total_debt,
        SUM(COALESCE(amt_credit_sum_overdue, 0))                AS bureau_total_overdue_amt,
        CASE WHEN SUM(COALESCE(amt_credit_sum,0)) > 0
             THEN ROUND((SUM(COALESCE(amt_credit_sum_debt,0)) / SUM(COALESCE(amt_credit_sum,0)))::NUMERIC, 4)
             ELSE 0
        END                                                     AS bureau_debt_to_credit,
        MIN(days_credit)                                        AS bureau_oldest_credit_days,
        MAX(days_credit)                                        AS bureau_newest_credit_days,
        COUNT(DISTINCT credit_type)                             AS bureau_credit_type_count,
        SUM(COALESCE(cnt_credit_prolong, 0))                    AS bureau_total_prolongations
    FROM source
    GROUP BY sk_id_curr
)
SELECT * FROM aggregated
