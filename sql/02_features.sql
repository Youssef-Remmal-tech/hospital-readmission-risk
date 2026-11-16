-- 02_features.sql
-- Clinical feature engineering: diagnosis groups, a Charlson comorbidity score
-- from ICD-9 codes, the LACE readmission index, and medication-change signals.

-- ICD-9 code as a number where possible (V and E codes stay NULL)
CREATE OR REPLACE MACRO icd_num(code) AS
    CASE WHEN code IS NULL OR code[1] IN ('V', 'E') THEN NULL ELSE TRY_CAST(code AS DOUBLE) END;

-- Primary-diagnosis group, following Strack et al. (2014)
CREATE OR REPLACE MACRO diag_group(code) AS
    CASE
        WHEN code IS NULL THEN 'missing'
        WHEN code[1] IN ('V', 'E') THEN 'other'
        WHEN floor(icd_num(code)) = 250 THEN 'diabetes'
        WHEN icd_num(code) BETWEEN 390 AND 459.99 OR floor(icd_num(code)) = 785 THEN 'circulatory'
        WHEN icd_num(code) BETWEEN 460 AND 519.99 OR floor(icd_num(code)) = 786 THEN 'respiratory'
        WHEN icd_num(code) BETWEEN 520 AND 579.99 OR floor(icd_num(code)) = 787 THEN 'digestive'
        WHEN icd_num(code) BETWEEN 800 AND 999.99 THEN 'injury'
        WHEN icd_num(code) BETWEEN 710 AND 739.99 THEN 'musculoskeletal'
        WHEN icd_num(code) BETWEEN 580 AND 629.99 OR floor(icd_num(code)) = 788 THEN 'genitourinary'
        WHEN icd_num(code) BETWEEN 140 AND 239.99 THEN 'neoplasms'
        ELSE 'other'
    END;

-- Charlson conditions (Deyo ICD-9 definitions) with LACE weights.
-- "family" lets the severe form of a condition replace the mild form.
CREATE OR REPLACE TABLE int_comorbidity AS
WITH codes AS (
    SELECT encounter_id, icd_num(code) AS x
    FROM stg_encounters
    UNPIVOT (code FOR position IN (diag_1, diag_2, diag_3))
),
flagged AS (
    SELECT encounter_id,
        CASE
            WHEN floor(x) IN (410, 412)                                       THEN 'mi'
            WHEN floor(x) = 428                                               THEN 'chf'
            WHEN floor(x) = 441 OR x IN (443.9, 785.4)                        THEN 'pvd'
            WHEN floor(x) BETWEEN 430 AND 438                                 THEN 'cvd'
            WHEN floor(x) = 290                                               THEN 'dementia'
            WHEN floor(x) BETWEEN 490 AND 505 OR x = 506.4                    THEN 'pulmonary'
            WHEN x IN (710.0, 710.1, 710.4) OR x BETWEEN 714.0 AND 714.2
                 OR x = 714.81 OR floor(x) = 725                               THEN 'connective'
            WHEN x IN (571.2, 571.4, 571.5, 571.6)                            THEN 'liver_mild'
            WHEN x BETWEEN 456.0 AND 456.21 OR x BETWEEN 572.2 AND 572.8      THEN 'liver_severe'
            WHEN (x >= 250.0 AND x < 250.4) OR (x >= 250.7 AND x < 250.8)     THEN 'diabetes'
            WHEN x >= 250.4 AND x < 250.7                                     THEN 'diabetes_complicated'
            WHEN floor(x) IN (582, 585, 586, 588) OR x BETWEEN 583.0 AND 583.7 THEN 'renal'
            WHEN x BETWEEN 196 AND 199.1                                      THEN 'metastatic'
            WHEN floor(x) BETWEEN 140 AND 172 OR x BETWEEN 174 AND 195.8
                 OR floor(x) BETWEEN 200 AND 208                              THEN 'tumour'
            WHEN floor(x) BETWEEN 42 AND 44                                   THEN 'aids'
        END AS condition
    FROM codes
    WHERE x IS NOT NULL
),
weighted AS (
    SELECT DISTINCT encounter_id, condition,
        CASE condition
            WHEN 'mi' THEN 1 WHEN 'cvd' THEN 1 WHEN 'pvd' THEN 1 WHEN 'diabetes' THEN 1
            WHEN 'chf' THEN 2 WHEN 'diabetes_complicated' THEN 2 WHEN 'pulmonary' THEN 2
            WHEN 'liver_mild' THEN 2 WHEN 'tumour' THEN 2
            WHEN 'dementia' THEN 3 WHEN 'connective' THEN 3
            WHEN 'aids' THEN 4 WHEN 'liver_severe' THEN 4 WHEN 'renal' THEN 4
            WHEN 'metastatic' THEN 6
        END AS weight,
        CASE WHEN condition LIKE 'diabetes%' THEN 'diabetes'
             WHEN condition LIKE 'liver%' THEN 'liver'
             WHEN condition IN ('tumour', 'metastatic') THEN 'cancer'
             ELSE condition END AS family
    FROM flagged
    WHERE condition IS NOT NULL
)
SELECT encounter_id, SUM(family_weight) AS charlson_score
FROM (SELECT encounter_id, family, MAX(weight) AS family_weight FROM weighted GROUP BY ALL)
GROUP BY encounter_id;

CREATE OR REPLACE TABLE feature_mart AS
WITH base AS (
    SELECT
        e.*,
        COALESCE(c.charlson_score, 0) AS charlson_score,
        -- number of diabetes drugs with a dose change during the stay
        ( (metformin IN ('Up','Down'))::INT + (repaglinide IN ('Up','Down'))::INT + (nateglinide IN ('Up','Down'))::INT
        + (chlorpropamide IN ('Up','Down'))::INT + (glimepiride IN ('Up','Down'))::INT + (acetohexamide IN ('Up','Down'))::INT
        + (glipizide IN ('Up','Down'))::INT + (glyburide IN ('Up','Down'))::INT + (tolbutamide IN ('Up','Down'))::INT
        + (pioglitazone IN ('Up','Down'))::INT + (rosiglitazone IN ('Up','Down'))::INT + (acarbose IN ('Up','Down'))::INT
        + (miglitol IN ('Up','Down'))::INT + (troglitazone IN ('Up','Down'))::INT + (tolazamide IN ('Up','Down'))::INT
        + (insulin IN ('Up','Down'))::INT + (glyburide_metformin IN ('Up','Down'))::INT
        + (glipizide_metformin IN ('Up','Down'))::INT ) AS diabetes_drug_dose_changes,
        ( (metformin <> 'No')::INT + (repaglinide <> 'No')::INT + (nateglinide <> 'No')::INT + (chlorpropamide <> 'No')::INT
        + (glimepiride <> 'No')::INT + (acetohexamide <> 'No')::INT + (glipizide <> 'No')::INT + (glyburide <> 'No')::INT
        + (tolbutamide <> 'No')::INT + (pioglitazone <> 'No')::INT + (rosiglitazone <> 'No')::INT + (acarbose <> 'No')::INT
        + (miglitol <> 'No')::INT + (troglitazone <> 'No')::INT + (tolazamide <> 'No')::INT + (insulin <> 'No')::INT
        + (glyburide_metformin <> 'No')::INT + (glipizide_metformin <> 'No')::INT ) AS diabetes_drugs_prescribed
    FROM stg_encounters e
    LEFT JOIN int_comorbidity c USING (encounter_id)
)
SELECT
    encounter_id,
    patient_id,
    -- model inputs
    age_band_start,
    length_of_stay,
    num_lab_procedures,
    num_procedures,
    num_medications,
    number_diagnoses,
    prior_outpatient,
    prior_emergency,
    prior_inpatient,
    prior_outpatient + prior_emergency + prior_inpatient          AS prior_visits_total,
    charlson_score,
    diabetes_drug_dose_changes,
    diabetes_drugs_prescribed,
    CASE WHEN insulin = 'No' THEN 'none' ELSE lower(insulin) END AS insulin_status,
    CASE WHEN a1c_result IS NULL OR a1c_result = 'None' THEN 'not_tested'
         WHEN a1c_result = 'Norm' THEN 'normal' ELSE 'high' END   AS a1c_status,
    CASE WHEN max_glu_serum IS NULL OR max_glu_serum = 'None' THEN 'not_tested'
         WHEN max_glu_serum = 'Norm' THEN 'normal' ELSE 'high' END AS glucose_status,
    (med_change = 'Ch')::INT                                     AS any_medication_change,
    diag_group(diag_1)                                           AS primary_diagnosis_group,
    CASE WHEN admission_type_id = 1 THEN 'emergency'
         WHEN admission_type_id = 2 THEN 'urgent'
         WHEN admission_type_id = 3 THEN 'elective'
         ELSE 'other_or_unknown' END                             AS admission_type,
    CASE WHEN discharge_disposition_id = 1 THEN 'home'
         ELSE 'disposition_' || discharge_disposition_id END     AS discharge_disposition,
    CASE WHEN medical_specialty IS NULL THEN 'missing'
         WHEN medical_specialty IN ('InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
              'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics',
              'Orthopedics-Reconstructive', 'Radiologist', 'Pulmonology')
             THEN medical_specialty
         ELSE 'other' END                                        AS admitting_specialty,
    -- LACE index (van Walraven et al. 2010), the benchmark used by many hospitals
    CASE WHEN length_of_stay < 1 THEN 0 WHEN length_of_stay <= 3 THEN length_of_stay
         WHEN length_of_stay <= 6 THEN 4 WHEN length_of_stay <= 13 THEN 5 ELSE 7 END
  + CASE WHEN admission_type_id = 1 THEN 3 ELSE 0 END
  + CASE WHEN charlson_score >= 4 THEN 5 ELSE charlson_score END
  + LEAST(prior_emergency, 4)                                    AS lace_score,
    -- monitoring only
    race,
    gender,
    readmitted_30d
FROM base
ORDER BY encounter_id;
