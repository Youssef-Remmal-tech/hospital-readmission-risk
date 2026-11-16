"""Tests for the SQL layer, clinical scores and the capacity logic.  Run with:  pytest -q"""
import duckdb
import numpy as np
import pandas as pd
import pytest

from src import config
from src.build_features import build_feature_mart
from src.evaluate import bootstrap_auc_diff, capacity_table, impact_per_1000


@pytest.fixture(scope="module")
def mart():
    return build_feature_mart()


def test_row_counts_match_source(mart):
    # 101,766 encounters in the UCI file; 2,423 end in death or hospice and are excluded
    assert mart.attrs["raw_encounters"] == 101_766
    assert len(mart) == 99_343
    assert mart["encounter_id"].is_unique


def test_no_expired_or_hospice_discharges(mart):
    assert not mart["discharge_disposition"].isin(
        [f"disposition_{c}" for c in (11, 13, 14, 19, 20, 21)]).any()


def test_protected_characteristics_not_inputs():
    assert not set(config.MONITORING_ONLY) & set(config.NUMERIC + config.CATEGORICAL)


def test_lace_score_within_published_range(mart):
    # L max 7 + A 3 + C max 5 + E max 4 = 19
    assert mart["lace_score"].between(0, 19).all()


@pytest.mark.parametrize("codes, expected", [
    (("428", "250.01", None), 3),      # heart failure 2 + uncomplicated diabetes 1
    (("250.4", "250.01", None), 2),    # complicated diabetes replaces uncomplicated
    (("197", "174", None), 6),         # metastatic replaces primary tumour
    (("V57", "E909", None), 0),        # V and E codes carry no Charlson weight
])
def test_charlson_scoring(codes, expected):
    con = duckdb.connect()
    con.execute("CREATE TABLE stg_encounters AS SELECT 1::BIGINT AS encounter_id, "
                "?::VARCHAR AS diag_1, ?::VARCHAR AS diag_2, ?::VARCHAR AS diag_3", list(codes))
    sql = (config.SQL_DIR / "02_features.sql").read_text().split("CREATE OR REPLACE TABLE feature_mart")[0]
    con.execute(sql)
    got = con.execute("SELECT COALESCE(MAX(charlson_score), 0) FROM int_comorbidity").fetchone()[0]
    assert got == expected


def test_capacity_table_full_capacity_catches_everything():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    s = np.linspace(1, 0, 10)
    t = capacity_table(y, {"m": s}, [1.0])
    assert t["share_of_all_readmissions"].iloc[0] == pytest.approx(1.0)
    assert t["lift_vs_random"].iloc[0] == pytest.approx(1.0)


def test_impact_scales_with_effect_size():
    cap = pd.DataFrame({"readmissions_in_contacted": [100]})
    assert impact_per_1000(cap, 10_000, 0.82).iloc[0] == pytest.approx(1.8)
    assert impact_per_1000(cap, 10_000, 1.0).iloc[0] == pytest.approx(0.0)


def test_bootstrap_identical_scores_give_zero_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.uniform(size=500)
    out = bootstrap_auc_diff(y, p, p, np.arange(500), n=50)
    assert out["diff"] == pytest.approx(0.0)
