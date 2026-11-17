WITH app AS (
    SELECT * FROM {{ ref('stg_application') }}
),
bureau AS (
    SELECT * FROM {{ ref('stg_bureau') }}
),
prev AS (
    SELECT * FROM {{ ref('stg_previous_application') }}
),
instalment AS (
    SELECT * FROM {{ ref('stg_installments') }}
),
joined AS (
    SELECT
        app.sk_id_curr,
        app.target,
        app.name_contract_type,
        app.amt_credit,
        app.amt_annuity,
        app.amt_income_total,
        app.amt_goods_price,
        app.age_years,
        app.cnt_children,
        app.cnt_fam_members,
        app.name_family_status,
        app.name_education_type,
        app.name_income_type,
        app.name_housing_type,
        app.code_gender,
        app.employment_years,
        app.is_unemployed,
        app.occupation_type,
        app.organization_type,
        app.ext_source_1,
        app.ext_source_2,
        app.ext_source_3,
        app.ext_source_mean,
        app.debt_to_income,
        app.credit_to_annuity,
        app.income_per_person,
        app.employed_to_age_ratio,
        app.flag_own_car,
        app.flag_own_realty,
        app.flag_work_phone,
        app.flag_email,
        app.region_rating_client,
        app.region_rating_client_w_city,
        app.reg_city_not_live_city,
        app.reg_city_not_work_city,
        app.amt_req_credit_bureau_hour,
        app.amt_req_credit_bureau_day,
        app.amt_req_credit_bureau_week,
        app.amt_req_credit_bureau_mon,
        app.amt_req_credit_bureau_qrt,
        app.amt_req_credit_bureau_year,
        COALESCE(bureau.bureau_loan_count, 0)           AS bureau_loan_count,
        COALESCE(bureau.bureau_active_loans, 0)         AS bureau_active_loans,
        COALESCE(bureau.bureau_closed_loans, 0)         AS bureau_closed_loans,
        COALESCE(bureau.bureau_overdue_count, 0)        AS bureau_overdue_count,
        COALESCE(bureau.bureau_max_overdue_days, 0)     AS bureau_max_overdue_days,
        COALESCE(bureau.bureau_avg_overdue_days, 0)     AS bureau_avg_overdue_days,
        COALESCE(bureau.bureau_total_credit, 0)         AS bureau_total_credit,
        COALESCE(bureau.bureau_total_debt, 0)           AS bureau_total_debt,
        COALESCE(bureau.bureau_total_overdue_amt, 0)    AS bureau_total_overdue_amt,
        COALESCE(bureau.bureau_debt_to_credit, 0)       AS bureau_debt_to_credit,
        bureau.bureau_oldest_credit_days,
        bureau.bureau_newest_credit_days,
        COALESCE(bureau.bureau_credit_type_count, 0)    AS bureau_credit_type_count,
        COALESCE(bureau.bureau_total_prolongations, 0)  AS bureau_total_prolongations,
        CASE WHEN bureau.sk_id_curr IS NOT NULL THEN 1 ELSE 0 END AS has_bureau_history,
        COALESCE(prev.prev_app_count, 0)                AS prev_app_count,
        COALESCE(prev.prev_approved_count, 0)           AS prev_approved_count,
        COALESCE(prev.prev_refused_count, 0)            AS prev_refused_count,
        COALESCE(prev.prev_canceled_count, 0)           AS prev_canceled_count,
        COALESCE(prev.prev_approval_rate, 0)            AS prev_approval_rate,
        COALESCE(prev.prev_avg_application_amt, 0)      AS prev_avg_application_amt,
        COALESCE(prev.prev_max_credit, 0)               AS prev_max_credit,
        COALESCE(prev.prev_avg_credit, 0)               AS prev_avg_credit,
        prev.prev_last_decision_days,
        prev.prev_first_decision_days,
        CASE WHEN prev.sk_id_curr IS NOT NULL THEN 1 ELSE 0 END AS has_prev_application,
        COALESCE(instalment.instalment_count, 0)        AS instalment_count,
        instalment.avg_payment_ratio,
        COALESCE(instalment.late_payment_count, 0)      AS late_payment_count,
        COALESCE(instalment.late_payment_rate, 0)       AS late_payment_rate,
        COALESCE(instalment.avg_payment_deficit, 0)     AS avg_payment_deficit,
        COALESCE(instalment.avg_days_late, 0)           AS avg_days_late,
        CASE WHEN instalment.sk_id_curr IS NOT NULL THEN 1 ELSE 0 END AS has_instalment_history,
        COALESCE(app.amt_credit, 0) +
        COALESCE(bureau.bureau_total_debt, 0)           AS total_debt_burden,
        COALESCE(app.amt_req_credit_bureau_year, 0) +
        COALESCE(prev.prev_app_count, 0)                AS total_credit_applications,
        COALESCE(bureau.bureau_overdue_count, 0) +
        COALESCE(instalment.late_payment_count, 0)      AS total_delinquency_count
    FROM app
    LEFT JOIN bureau     ON app.sk_id_curr = bureau.sk_id_curr
    LEFT JOIN prev       ON app.sk_id_curr = prev.sk_id_curr
    LEFT JOIN instalment ON app.sk_id_curr = instalment.sk_id_curr
)
SELECT * FROM joined
