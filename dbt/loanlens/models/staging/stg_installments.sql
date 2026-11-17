WITH source AS (
    SELECT * FROM raw.installments_payments
),
payment_behavior AS (
    SELECT
        sk_id_curr,
        COUNT(*)                                                        AS instalment_count,
        AVG(CASE WHEN amt_instalment > 0
                 THEN amt_payment / amt_instalment ELSE NULL END)       AS avg_payment_ratio,
        SUM(CASE WHEN days_entry_payment > days_instalment + 1
                 THEN 1 ELSE 0 END)                                     AS late_payment_count,
        ROUND(
            SUM(CASE WHEN days_entry_payment > days_instalment + 1
                     THEN 1 ELSE 0 END)::NUMERIC / NULLIF(COUNT(*),0)::NUMERIC, 4
        )                                                               AS late_payment_rate,
        AVG(COALESCE(amt_instalment,0) - COALESCE(amt_payment,0))      AS avg_payment_deficit,
        AVG(GREATEST(COALESCE(days_entry_payment,0) - COALESCE(days_instalment,0), 0)) AS avg_days_late
    FROM source
    WHERE amt_instalment IS NOT NULL
    GROUP BY sk_id_curr
)
SELECT * FROM payment_behavior
