-- 01_staging.sql
-- Load the raw encounters, convert "?" to NULL, and keep only discharges
-- where a readmission was possible (patients who died or went to hospice are removed).

CREATE OR REPLACE TABLE raw_encounters AS
SELECT * FROM read_csv(getvariable('raw_path'), header = true, all_varchar = true);

CREATE OR REPLACE TABLE stg_encounters AS
SELECT
    CAST(encounter_id AS BIGINT)                         AS encounter_id,
    CAST(patient_nbr AS BIGINT)                          AS patient_id,
    NULLIF(race, '?')                                    AS race,
    NULLIF(gender, 'Unknown/Invalid')                    AS gender,
    -- age arrives as "[70-80)"; keep the lower bound of the 10-year band
    CAST(regexp_extract(age, '\[(\d+)-', 1) AS INTEGER) AS age_band_start,
    CAST(admission_type_id AS INTEGER)                   AS admission_type_id,
    CAST(discharge_disposition_id AS INTEGER)            AS discharge_disposition_id,
    CAST(admission_source_id AS INTEGER)                 AS admission_source_id,
    CAST(time_in_hospital AS INTEGER)                    AS length_of_stay,
    NULLIF(medical_specialty, '?')                       AS medical_specialty,
    CAST(num_lab_procedures AS INTEGER)                  AS num_lab_procedures,
    CAST(num_procedures AS INTEGER)                      AS num_procedures,
    CAST(num_medications AS INTEGER)                     AS num_medications,
    CAST(number_outpatient AS INTEGER)                   AS prior_outpatient,
    CAST(number_emergency AS INTEGER)                    AS prior_emergency,
    CAST(number_inpatient AS INTEGER)                    AS prior_inpatient,
    NULLIF(diag_1, '?') AS diag_1,
    NULLIF(diag_2, '?') AS diag_2,
    NULLIF(diag_3, '?') AS diag_3,
    CAST(number_diagnoses AS INTEGER)                    AS number_diagnoses,
    max_glu_serum, "A1Cresult" AS a1c_result,
    metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide,
    glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol,
    troglitazone, tolazamide, examide, citoglipton, insulin,
    "glyburide-metformin" AS glyburide_metformin, "glipizide-metformin" AS glipizide_metformin,
    "glimepiride-pioglitazone" AS glimepiride_pioglitazone,
    "metformin-rosiglitazone" AS metformin_rosiglitazone,
    "metformin-pioglitazone" AS metformin_pioglitazone,
    "change"    AS med_change,
    "diabetesMed" AS on_diabetes_med,
    CASE WHEN readmitted = '<30' THEN 1 ELSE 0 END       AS readmitted_30d
FROM raw_encounters
-- 11, 19, 20, 21 = expired; 13, 14 = hospice. These patients cannot be readmitted.
WHERE CAST(discharge_disposition_id AS INTEGER) NOT IN (11, 13, 14, 19, 20, 21);
