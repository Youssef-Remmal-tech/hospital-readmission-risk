"""Every project setting in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "diabetic_data.csv.gz"   # gzip keeps the repo small; DuckDB reads it directly
RAW_SHA256 = "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97"
PROCESSED = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
DOCS = ROOT / "docs"

SEED = 42
TEST_SIZE = 0.30
TARGET = "readmitted_30d"
GROUP = "patient_id"

NUMERIC = [
    "age_band_start", "length_of_stay", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses", "prior_outpatient", "prior_emergency",
    "prior_inpatient", "prior_visits_total", "charlson_score",
    "diabetes_drug_dose_changes", "diabetes_drugs_prescribed", "any_medication_change",
]
CATEGORICAL = [
    "insulin_status", "a1c_status", "glucose_status", "primary_diagnosis_group",
    "admission_type", "discharge_disposition", "admitting_specialty",
]
# Race and gender are not model inputs. They are used only to check outcomes by group.
MONITORING_ONLY = ["race", "gender"]

# Follow-up capacity scenarios: share of discharges the care team can contact
CAPACITY_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_CAPACITY = 0.20

# Effect of transitional-care interventions on 30-day readmission
# Leppin et al. (2014), JAMA Internal Medicine, 42 RCTs: pooled RR 0.82 (95% CI 0.73 to 0.91)
INTERVENTION_RR = 0.82
INTERVENTION_RR_LOW = 0.73
INTERVENTION_RR_HIGH = 0.91

N_BOOTSTRAP = 1000
LACE_HIGH_RISK = 10
