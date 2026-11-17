WITH source AS (
    SELECT * FROM raw.application_train
),
cleaned AS (
    SELECT
        sk_id_curr,
        target,
        name_contract_type,
        amt_credit,
        amt_annuity,
        amt_income_total,
        amt_goods_price,
        ROUND((ABS(days_birth) / 365.0)::NUMERIC, 2)                          AS age_years,
        cnt_children,
        cnt_fam_members,
        name_family_status,
        name_education_type,
        name_income_type,
        name_housing_type,
        code_gender,
        CASE WHEN days_employed > 0 THEN 0
             ELSE ROUND(ABS(days_employed) / 365.0, 2)
        END                                                         AS employment_years,
        CASE WHEN days_employed > 0 THEN TRUE ELSE FALSE END        AS is_unemployed,
        occupation_type,
        organization_type,
        ext_source_1,
        ext_source_2,
        ext_source_3,
        CASE WHEN amt_income_total > 0
             THEN ROUND((amt_credit / amt_income_total)::NUMERIC, 4)
        END                                                         AS debt_to_income,
        CASE WHEN amt_annuity > 0
             THEN ROUND((amt_credit / amt_annuity)::NUMERIC, 4)
        END                                                         AS credit_to_annuity,
        CASE WHEN cnt_fam_members > 0
             THEN ROUND((amt_income_total / cnt_fam_members)::NUMERIC, 2)
             ELSE amt_income_total
        END                                                         AS income_per_person,
        CASE WHEN days_birth != 0
             THEN ROUND(ABS(days_employed)::NUMERIC / ABS(days_birth)::NUMERIC, 4)
        END                                                         AS employed_to_age_ratio,
        ROUND(
            ((COALESCE(ext_source_1,0) + COALESCE(ext_source_2,0) + COALESCE(ext_source_3,0)) /
            NULLIF(
                (CASE WHEN ext_source_1 IS NOT NULL THEN 1 ELSE 0 END +
                 CASE WHEN ext_source_2 IS NOT NULL THEN 1 ELSE 0 END +
                 CASE WHEN ext_source_3 IS NOT NULL THEN 1 ELSE 0 END), 0
            ))::NUMERIC, 6
        )                                                           AS ext_source_mean,
        flag_own_car,
        flag_own_realty,
        flag_mobil,
        flag_emp_phone,
        flag_work_phone,
        flag_email,
        region_rating_client,
        region_rating_client_w_city,
        reg_city_not_live_city,
        reg_city_not_work_city,
        amt_req_credit_bureau_hour,
        amt_req_credit_bureau_day,
        amt_req_credit_bureau_week,
        amt_req_credit_bureau_mon,
        amt_req_credit_bureau_qrt,
        amt_req_credit_bureau_year
    FROM source
    WHERE sk_id_curr IS NOT NULL
)
SELECT * FROM cleaned
