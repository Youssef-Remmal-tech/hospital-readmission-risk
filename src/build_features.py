"""Run the DuckDB SQL layer and return the feature mart."""
import gzip
import hashlib

import duckdb
import pandas as pd

from . import config


def build_feature_mart() -> pd.DataFrame:
    if not config.RAW_CSV.exists():
        raise FileNotFoundError(f"Raw data not found at {config.RAW_CSV}. See data/README.md.")
    digest = hashlib.sha256(gzip.decompress(config.RAW_CSV.read_bytes())).hexdigest()  # checksum of the original CSV
    if digest != config.RAW_SHA256:
        raise ValueError(f"Checksum {digest} does not match the expected UCI file.")

    con = duckdb.connect()
    con.execute("SET VARIABLE raw_path = ?", [str(config.RAW_CSV)])
    for name in ("01_staging.sql", "02_features.sql"):
        con.execute((config.SQL_DIR / name).read_text())
    checks = con.execute((config.SQL_DIR / "03_checks.sql").read_text()).df()
    failed = checks[checks["failures"] != 0]
    if not failed.empty:
        raise ValueError(f"Data-quality checks failed:\n{failed}")
    mart = con.execute("SELECT * FROM feature_mart").df()
    stats = con.execute("""
        SELECT (SELECT COUNT(*) FROM raw_encounters) AS raw_encounters,
               (SELECT COUNT(*) FROM stg_encounters) AS eligible_encounters""").df().iloc[0].to_dict()
    con.close()
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    mart.to_csv(config.PROCESSED / "feature_mart.csv", index=False)
    mart.attrs.update({k: int(v) for k, v in stats.items()})
    return mart
