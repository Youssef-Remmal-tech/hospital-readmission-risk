"""End-to-end pipeline: SQL features -> models -> validation -> capacity and impact analysis.

Usage:  python run_pipeline.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src import config, plots
from src.build_features import build_feature_mart
from src.evaluate import (bootstrap_auc_diff, calibration_table, capacity_table, equity_table,
                          impact_per_1000, summary)
from src.models import fit_gbm, fit_logistic, odds_ratios, predict

LABELS = {
    "prior_inpatient": "Inpatient stays, previous year",
    "discharge_disposition": "Discharge destination",
    "prior_emergency": "Emergency visits, previous year",
    "prior_visits_total": "All visits, previous year",
    "number_diagnoses": "Number of diagnoses",
    "num_medications": "Medicines during stay",
    "num_lab_procedures": "Lab tests during stay",
    "length_of_stay": "Length of stay",
    "primary_diagnosis_group": "Primary diagnosis group",
    "admitting_specialty": "Admitting specialty",
    "age_band_start": "Age band",
    "prior_outpatient": "Outpatient visits, previous year",
    "charlson_score": "Charlson comorbidity score",
    "num_procedures": "Procedures during stay",
    "admission_type": "Admission type",
    "insulin_status": "Insulin dose change",
    "diabetes_drugs_prescribed": "Diabetes drugs prescribed",
    "diabetes_drug_dose_changes": "Diabetes drug dose changes",
    "a1c_status": "HbA1c result",
    "glucose_status": "Glucose result",
    "any_medication_change": "Any medication change",
}


def main():
    for d in (config.FIGURES, config.TABLES, config.DOCS):
        d.mkdir(parents=True, exist_ok=True)

    print("1/6  Building features in DuckDB")
    mart = build_feature_mart()

    # Patients can have several admissions. Split by patient so nobody appears in both samples.
    split = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.SEED)
    tr_idx, te_idx = next(split.split(mart, groups=mart[config.GROUP]))
    train, test = mart.iloc[tr_idx].reset_index(drop=True), mart.iloc[te_idx].reset_index(drop=True)
    y_te = test[config.TARGET].to_numpy()

    print("2/6  Fitting logistic regression and gradient boosting (patient-grouped CV)")
    lr, lr_info = fit_logistic(train)
    gbm, gbm_info = fit_gbm(train)

    print("3/6  Scoring the hold-out patients")
    p_lr, p_gbm = predict(lr, test), predict(gbm, test)
    # LACE is an integer score; a tiny fixed-seed jitter breaks ties at random for ranking
    lace_rank = test["lace_score"].to_numpy() + np.random.default_rng(config.SEED).uniform(0, 1e-3, len(test))
    m = {"gradient_boosting": summary(y_te, p_gbm), "logistic_regression": summary(y_te, p_lr),
         "lace_index": summary(y_te, test["lace_score"].to_numpy())}
    m["lace_index"]["brier"] = None
    boot_gbm_lace = bootstrap_auc_diff(y_te, lace_rank, p_gbm, test[config.GROUP], config.N_BOOTSTRAP, config.SEED)
    boot_gbm_lr = bootstrap_auc_diff(y_te, p_lr, p_gbm, test[config.GROUP], config.N_BOOTSTRAP, config.SEED)

    # Leakage check: what if we had split encounters at random instead of by patient?
    a, b = train_test_split(mart, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=mart[config.TARGET])
    naive = clone(gbm).fit(a[config.NUMERIC + config.CATEGORICAL], a[config.TARGET])
    leakage = {"naive_split_auc": float(summary(b[config.TARGET], predict(naive, b))["auc"]),
               "patient_split_auc": float(m["gradient_boosting"]["auc"]),
               "naive_test_patients_seen_in_training": float(b[config.GROUP].isin(a[config.GROUP]).mean())}

    print("4/6  Capacity, impact and equity analysis")
    rng = np.random.default_rng(config.SEED)
    scores = {"Risk model (gradient boosting)": p_gbm, "Logistic regression": p_lr,
              "LACE index": lace_rank, "Random selection": rng.uniform(size=len(test))}
    cap = capacity_table(y_te, scores, config.CAPACITY_LEVELS)
    n = len(test)
    cap["avoided_per_1000"] = impact_per_1000(cap, n, config.INTERVENTION_RR)
    cap["avoided_per_1000_low"] = impact_per_1000(cap, n, config.INTERVENTION_RR_HIGH)
    cap["avoided_per_1000_high"] = impact_per_1000(cap, n, config.INTERVENTION_RR_LOW)
    cap["contacts_per_readmission_avoided"] = cap["patients_contacted"] / (cap["readmissions_in_contacted"] * (1 - config.INTERVENTION_RR))
    # Random selection is the expectation, not one draw: share caught = capacity
    rnd = cap["method"] == "Random selection"
    cap.loc[rnd, "readmissions_in_contacted"] = (cap.loc[rnd, "capacity"] * y_te.sum()).round().astype(int)
    cap.loc[rnd, "share_of_all_readmissions"] = cap.loc[rnd, "capacity"]
    cap.loc[rnd, "readmission_rate_in_contacted"] = y_te.mean()
    cap.loc[rnd, "lift_vs_random"] = 1.0
    for col, rr in [("avoided_per_1000", config.INTERVENTION_RR), ("avoided_per_1000_low", config.INTERVENTION_RR_HIGH),
                    ("avoided_per_1000_high", config.INTERVENTION_RR_LOW)]:
        cap.loc[rnd, col] = cap.loc[rnd, "readmissions_in_contacted"] / n * 1000 * (1 - rr)
    cap.loc[rnd, "contacts_per_readmission_avoided"] = cap.loc[rnd, "patients_contacted"] / (cap.loc[rnd, "readmissions_in_contacted"] * (1 - config.INTERVENTION_RR))

    equity = equity_table(test, p_gbm, config.DEFAULT_CAPACITY)
    cal_gbm, cal_lr = calibration_table(y_te, p_gbm), calibration_table(y_te, p_lr)

    perm = permutation_importance(gbm, test[config.NUMERIC + config.CATEGORICAL], y_te, scoring="roc_auc",
                                  n_repeats=5, random_state=config.SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": config.NUMERIC + config.CATEGORICAL, "auc_drop": perm.importances_mean,
                        "auc_drop_sd": perm.importances_std}).sort_values("auc_drop", ascending=False)
    imp["label"] = imp["feature"].map(LABELS)
    ors = odds_ratios(lr)

    print("5/6  Writing tables and charts")
    cap.to_csv(config.TABLES / "capacity_and_impact.csv", index=False)
    equity.to_csv(config.TABLES / "equity_monitoring.csv", index=False)
    pd.concat([cal_gbm.assign(model="gradient_boosting"), cal_lr.assign(model="logistic_regression")]) \
        .to_csv(config.TABLES / "calibration.csv", index=False)
    imp.to_csv(config.TABLES / "permutation_importance.csv", index=False)
    ors.to_csv(config.TABLES / "logistic_odds_ratios.csv", index=False)
    lace_bands = test.assign(lace_high=test["lace_score"] >= config.LACE_HIGH_RISK).groupby("lace_high")[config.TARGET] \
        .agg(encounters="size", readmission_rate="mean").reset_index()
    lace_bands.to_csv(config.TABLES / "lace_high_risk_check.csv", index=False)

    plots.gains_chart(y_te, {"Risk model (gradient boosting)": p_gbm, "LACE index": lace_rank},
                      config.FIGURES / "01_gains_chart.png", mark=config.DEFAULT_CAPACITY)
    impact_plot = cap[cap["method"].isin(["Risk model (gradient boosting)", "LACE index", "Random selection"])].rename(
        columns={"avoided_per_1000": "avoided", "avoided_per_1000_low": "low", "avoided_per_1000_high": "high"})
    plots.impact_chart(impact_plot[impact_plot["capacity"].isin([0.10, 0.20, 0.30])],
                       config.FIGURES / "02_impact_per_1000.png")
    plots.drivers_chart(imp, config.FIGURES / "03_risk_drivers.png")
    plots.calibration_chart(cal_gbm, cal_lr, config.FIGURES / "04_calibration.png")

    # Data for the interactive capacity planner (1% steps)
    fine = capacity_table(y_te, {"model": p_gbm, "lace": lace_rank}, [round(k, 2) for k in np.arange(0.01, 0.51, 0.01)])
    planner = {"n": n, "readmissions": int(y_te.sum()),
               "model": fine[fine.method == "model"][["capacity", "readmissions_in_contacted"]].values.tolist(),
               "lace": fine[fine.method == "lace"][["capacity", "readmissions_in_contacted"]].values.tolist()}
    (config.DOCS / "planner_data.js").write_text("window.PLANNER = " + json.dumps(planner) + ";\n")

    print("6/6  Saving metrics")
    metrics = {
        "data": {"raw_encounters": mart.attrs.get("raw_encounters"), "eligible_encounters": len(mart),
                 "patients": int(mart[config.GROUP].nunique()), "readmission_rate": float(mart[config.TARGET].mean()),
                 "train_encounters": len(train), "test_encounters": n,
                 "test_patients": int(test[config.GROUP].nunique()), "test_readmissions": int(y_te.sum())},
        "hold_out": m,
        "tuning": {"logistic_regression": lr_info, "gradient_boosting": gbm_info},
        "bootstrap_gbm_minus_lace": boot_gbm_lace,
        "bootstrap_gbm_minus_logistic": boot_gbm_lr,
        "leakage_check": leakage,
    }
    (config.ROOT / "outputs" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    at = cap[cap["capacity"] == config.DEFAULT_CAPACITY].set_index("method")
    print(json.dumps(metrics["hold_out"], indent=1, default=float))
    print(at[["readmissions_in_contacted", "share_of_all_readmissions", "lift_vs_random",
              "avoided_per_1000", "contacts_per_readmission_avoided"]].round(3).to_string())
    print(json.dumps({"gbm_vs_lace": boot_gbm_lace, "gbm_vs_lr": boot_gbm_lr, "leakage": leakage}, indent=1))


if __name__ == "__main__":
    main()
