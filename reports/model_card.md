# Model Card: 30-day Readmission Risk (Diabetes Inpatients)

| | |
|---|---|
| **Model** | Histogram gradient boosting, 21 inputs (14 numeric, 7 categorical) |
| **Alternatives evaluated** | Logistic regression (interpretable), LACE index (current practice) |
| **Output** | Probability of an emergency readmission within 30 days of discharge |
| **Training data** | UCI Diabetes 130-US Hospitals, 1999–2008: 69,519 discharges |
| **Evaluation data** | 29,824 discharges from 20,997 different patients, held out by patient |

## Intended use

**Use it to:** rank a day's discharges so a transitional-care team contacts the highest-risk patients first, within the capacity it has.

**Do not use it to:**
- make clinical diagnoses
- decide treatment
- deny or delay any service
- judge individual clinicians or hospitals

A person on the care team reviews every list before any contact.

**Who uses it:** discharge coordinators, transitional-care nurses, and service managers planning follow-up capacity.

## Inputs

These are recorded during the stay or taken from the previous year's records:

- Age band, length of stay and admission type
- Discharge destination and admitting specialty
- Primary diagnosis group and Charlson comorbidity score (from ICD-9 codes)
- Number of diagnoses, lab tests, procedures and medicines
- Outpatient, emergency and inpatient visits in the previous year
- HbA1c and glucose results, insulin status, and diabetes drugs and dose changes

**Deliberately excluded:** race and gender, which are used only for monitoring, and payer.

## Performance on the hold-out set

| Metric | Gradient boosting | Logistic regression | LACE |
|---|---|---|---|
| AUC (95% CI) | 0.675 (0.663–0.687) | 0.665 (0.654–0.677) | 0.568 (0.557–0.579) |
| Average precision | 0.231 | 0.211 | 0.138 |
| Readmissions reached when contacting 20% | 38.6% | 38.0% | 27.0% |
| Calibration | within 1 point in every decile | within 1.6 points | n/a |

Confidence intervals come from a patient-clustered bootstrap with 1,000 resamples.

## Equity

At a 20% contact rate, mean predicted risk is within 1.2 percentage points of the observed rate for every race, gender and age group.

The share of each group's readmissions that the model catches varies:

| Group | Readmissions caught |
|---|---|
| Women | 41.5% |
| Men | 35.1% |
| Caucasian | 39.2% |
| African American | 38.5% |
| Hispanic (n = 608) | 31.8% |
| Asian (n = 217) | 26.3% |

**Recommendation:** report capture by group every month during any pilot. If the gap for smaller groups persists on larger numbers, investigate before scaling up.

## Known limitations

- Historical US data from 1999–2008, so it must be retrained and validated locally before any use.
- Only three diagnosis codes per stay, which under-counts comorbidity.
- No social, frailty or post-discharge information.
- Readmissions to hospitals outside the source network may be missing.
- The impact estimates assume the pooled trial effect (RR 0.82; Leppin et al., 2014) applies equally to every contacted patient.

## Monitoring plan

| Check | Frequency | Action threshold |
|---|---|---|
| Capture at agreed capacity vs LACE | Monthly | Model no better than LACE for 2 consecutive months |
| Calibration (predicted vs observed by decile) | Quarterly | Any decile off by more than 3 points |
| Capture and calibration by race, gender, age | Monthly | Gap of more than 10 points sustained for a quarter |
| Input drift (PSI per feature) | Monthly | PSI above 0.25 |
