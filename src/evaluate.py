"""Discrimination, calibration, capacity-constrained targeting, intervention impact and equity checks."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config


def summary(y, p) -> dict:
    return {"auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
            "brier": brier_score_loss(y, p) if np.all((p >= 0) & (p <= 1)) else None}


def bootstrap_auc_diff(y, p_a, p_b, groups, n=1000, seed=42) -> dict:
    """Patient-clustered paired bootstrap of AUC(b) - AUC(a)."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b, groups = map(np.asarray, (y, p_a, p_b, groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_by_group = np.split(np.argsort(inv), np.cumsum(np.bincount(inv))[:-1])
    diffs, auc_a, auc_b = [], [], []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_group[g] for g in pick])
        if y[idx].min() == y[idx].max():
            continue
        a, b = roc_auc_score(y[idx], p_a[idx]), roc_auc_score(y[idx], p_b[idx])
        auc_a.append(a); auc_b.append(b); diffs.append(b - a)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"auc_a_ci": ci(auc_a), "auc_b_ci": ci(auc_b), "diff": float(np.mean(diffs)), "diff_ci": ci(diffs)}


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False) + 1
    return df.groupby("decile").agg(n=("y", "size"), predicted=("p", "mean"),
                                    observed=("y", "mean")).reset_index()


def capacity_table(y, scores: dict, levels) -> pd.DataFrame:
    """For each targeting method, what happens if the team can only contact the top k% of discharges."""
    y = np.asarray(y)
    base = y.mean()
    rows = []
    for name, s in scores.items():
        order = np.argsort(-np.asarray(s), kind="stable")
        for k in levels:
            n_target = int(round(k * len(y)))
            hit = y[order[:n_target]]
            rows.append({"method": name, "capacity": k, "patients_contacted": n_target,
                         "readmissions_in_contacted": int(hit.sum()),
                         "share_of_all_readmissions": hit.sum() / y.sum(),
                         "readmission_rate_in_contacted": hit.mean(),
                         "lift_vs_random": hit.mean() / base})
    return pd.DataFrame(rows)


def impact_per_1000(cap: pd.DataFrame, n_test: int, rr: float) -> pd.Series:
    """Readmissions avoided per 1,000 discharges if the intervention cuts risk by (1 - rr) for contacted patients."""
    return cap["readmissions_in_contacted"] / n_test * 1000 * (1 - rr)


def equity_table(test: pd.DataFrame, p: np.ndarray, capacity: float) -> pd.DataFrame:
    """Share flagged, readmissions caught and calibration by race, gender and age band at a capacity cut-off."""
    cutoff = np.quantile(p, 1 - capacity)
    df = test.assign(p=p, flagged=(p >= cutoff).astype(int),
                     age_group=pd.cut(test["age_band_start"], [-1, 49, 69, 200], labels=["under 50", "50-69", "70+"]),
                     race=test["race"].fillna("Unknown"), gender=test["gender"].fillna("Unknown"))
    out = []
    for attr in ["race", "gender", "age_group"]:
        g = df.groupby(attr, observed=True)
        t = pd.DataFrame({
            "encounters": g.size(),
            "readmission_rate": g[config.TARGET].mean(),
            "mean_predicted_risk": g["p"].mean(),
            "share_flagged": g["flagged"].mean(),
            "readmissions_caught": g.apply(lambda d: d.loc[d[config.TARGET] == 1, "flagged"].mean(), include_groups=False),
        }).reset_index().rename(columns={attr: "group"})
        t.insert(0, "attribute", attr)
        out.append(t)
    return pd.concat(out, ignore_index=True)
