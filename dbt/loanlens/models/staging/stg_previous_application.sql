WITH source AS (
    SELECT * FROM raw.previous_application
),
aggregated AS (
    SELECT
        sk_id_curr,
        COUNT(*)                                                        AS prev_app_count,
        SUM(CASE WHEN name_contract_status='Approved' THEN 1 ELSE 0 END) AS prev_approved_count,
        SUM(CASE WHEN name_contract_status='Refused'  THEN 1 ELSE 0 END) AS prev_refused_count,
        SUM(CASE WHEN name_contract_status='Canceled' THEN 1 ELSE 0 END) AS prev_canceled_count,
        CASE WHEN COUNT(*) > 0
             THEN ROUND(SUM(CASE WHEN name_contract_status='Approved' THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)::NUMERIC, 4)
             ELSE 0
        END                                                             AS prev_approval_rate,
        AVG(COALESCE(amt_application, 0))                              AS prev_avg_application_amt,
        MAX(COALESCE(amt_credit, 0))                                   AS prev_max_credit,
        AVG(COALESCE(amt_credit, 0))                                   AS prev_avg_credit,
        MAX(days_decision)                                              AS prev_last_decision_days,
        MIN(days_decision)                                              AS prev_first_decision_days
    FROM source
    GROUP BY sk_id_curr
)
SELECT * FROM aggregated
