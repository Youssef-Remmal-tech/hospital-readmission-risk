"""Logistic regression (interpretable), gradient boosting (challenger) and the LACE benchmark."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config

COUNT_COLS = ["num_lab_procedures", "num_procedures", "num_medications", "prior_outpatient",
              "prior_emergency", "prior_inpatient", "prior_visits_total"]


def _logistic_pipeline(C=1.0) -> Pipeline:
    other_numeric = [c for c in config.NUMERIC if c not in COUNT_COLS]
    pre = ColumnTransformer([
        ("counts", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("scale", StandardScaler())]), COUNT_COLS),
        ("numeric", StandardScaler(), other_numeric),
        ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                                      drop="first", sparse_output=False), config.CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(C=C, max_iter=2000))])


def _gbm_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("numeric", "passthrough", config.NUMERIC),
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), config.CATEGORICAL),
    ])
    cat_mask = [False] * len(config.NUMERIC) + [True] * len(config.CATEGORICAL)
    gbm = HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=config.SEED,
                                         early_stopping=False)
    return Pipeline([("pre", pre), ("model", gbm)])


def fit_logistic(train: pd.DataFrame):
    cv = GroupKFold(n_splits=5)
    search = GridSearchCV(_logistic_pipeline(), {"model__C": [0.01, 0.1, 1.0]},
                          scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(train[config.NUMERIC + config.CATEGORICAL], train[config.TARGET], groups=train[config.GROUP])
    return search.best_estimator_, {"best_params": search.best_params_, "cv_auc": search.best_score_}


def fit_gbm(train: pd.DataFrame, grouped: bool = True):
    grid = {"model__learning_rate": [0.03, 0.06], "model__max_depth": [3, 5],
            "model__max_iter": [300, 600], "model__min_samples_leaf": [100]}
    X, y = train[config.NUMERIC + config.CATEGORICAL], train[config.TARGET]
    if grouped:
        search = GridSearchCV(_gbm_pipeline(), grid, scoring="roc_auc", cv=GroupKFold(n_splits=5), n_jobs=-1)
        search.fit(X, y, groups=train[config.GROUP])
    else:
        search = GridSearchCV(_gbm_pipeline(), grid, scoring="roc_auc", cv=5, n_jobs=-1)
        search.fit(X, y)
    return search.best_estimator_, {"best_params": search.best_params_, "cv_auc": search.best_score_}


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[config.NUMERIC + config.CATEGORICAL])[:, 1]


def odds_ratios(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["pre"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_[0]
    out = pd.DataFrame({"feature": names, "coefficient": coefs, "odds_ratio": np.exp(coefs)})
    out["feature"] = out["feature"].str.replace(r"^(counts|numeric|categorical)__", "", regex=True)
    return out.reindex(out["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
