# Data

`raw/diabetic_data.csv.gz` is the UCI **Diabetes 130-US Hospitals for Years 1999–2008** dataset (Strack et al., 2014), also distributed on Kaggle.

- 101,766 hospital encounters for 71,518 patients with diabetes, 50 columns
- Licence: CC BY 4.0
- SHA-256 of the decompressed CSV: `0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97` (checked on every run)
- De-identified by the original authors: no names, dates or locations

| Column group | Used as |
|---|---|
| Admission and discharge codes, length of stay | Model inputs. Discharges ending in death or hospice (codes 11, 13, 14, 19, 20, 21) are removed |
| Prior-year outpatient, emergency and inpatient visits | Model inputs and LACE "E" component |
| `diag_1`–`diag_3` (ICD-9) | Diagnosis groups and a Charlson comorbidity score |
| Lab results, 23 diabetes medication columns | HbA1c and glucose status, insulin and dose-change features |
| `race`, `gender` | Not model inputs. Used only for equity monitoring |
| `readmitted` | Target: `<30` = readmitted within 30 days |

`processed/` is created by the pipeline and is not committed.

Strack, B. et al. (2014). Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. *BioMed Research International*, 2014, 781670.
