-- 03_checks.sql  Each check returns 0 failures when the data is fit to model.
SELECT 'duplicate_encounters' AS check_name, COUNT(*) - COUNT(DISTINCT encounter_id) AS failures FROM feature_mart
UNION ALL
SELECT 'expired_or_hospice_kept', COUNT(*) FROM stg_encounters WHERE discharge_disposition_id IN (11, 13, 14, 19, 20, 21)
UNION ALL
SELECT 'mart_row_mismatch', ABS((SELECT COUNT(*) FROM feature_mart) - (SELECT COUNT(*) FROM stg_encounters))
UNION ALL
SELECT 'lace_out_of_range', COUNT(*) FROM feature_mart WHERE lace_score NOT BETWEEN 0 AND 19
UNION ALL
SELECT 'null_numeric_inputs', COUNT(*) FROM feature_mart
WHERE age_band_start IS NULL OR length_of_stay IS NULL OR num_medications IS NULL OR prior_inpatient IS NULL;
