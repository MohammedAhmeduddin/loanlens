"""
Load raw Home Credit CSVs into PostgreSQL raw schema.
Run once after docker compose up.

Usage:
    python scripts/load_data.py
"""

import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loanlens.config import get_settings
from loanlens.utils.logging import logger

settings = get_settings()


def get_engine():
    """Create synchronous SQLAlchemy engine for bulk loading."""
    return create_engine(settings.database_url_sync)


def load_csv_to_postgres(
    csv_path: Path,
    table_name: str,
    schema: str = "raw",
    chunksize: int = 10_000,
    nrows: int = None
) -> int:
    if not csv_path.exists():
        logger.warning(f"File not found: {csv_path} — skipping")
        return 0

    logger.info(f"Loading {csv_path.name} → {schema}.{table_name}")
    engine = get_engine()
    total_rows = 0

    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        nrows=nrows,
        low_memory=False
    )

    for i, chunk in enumerate(reader):
        chunk.columns = chunk.columns.str.lower()
        chunk.to_sql(
            name=table_name,
            schema=schema,
            con=engine,
            if_exists="replace" if i == 0 else "append",  # ← KEY CHANGE
            index=False,
            method="multi"
        )
        total_rows += len(chunk)
        if i % 10 == 0:
            logger.info(f"  Loaded {total_rows:,} rows...")

    logger.info(f"  Done — {total_rows:,} total rows loaded")
    return total_rows

def main():
    """Load all Home Credit CSV files into PostgreSQL."""

    raw_dir = Path(settings.raw_data_dir)

    files_to_load = [
        {
            "csv": raw_dir / "application_train.csv",
            "table": "application_train",
            "nrows": None  # Load all 307K rows
        },
        {
            "csv": raw_dir / "bureau.csv",
            "table": "bureau",
            "nrows": None  # Load all 1.7M rows
        },
        {
            "csv": raw_dir / "previous_application.csv",
            "table": "previous_application",
            "nrows": None
        },
        {
            "csv": raw_dir / "installments_payments.csv",
            "table": "installments_payments",
            "nrows": 500_000  # Sample — full 13M rows causes OOM
        },
    ]

    engine = get_engine()
    total = 0

    for file_config in files_to_load:
        rows = load_csv_to_postgres(
            csv_path=file_config["csv"],
            table_name=file_config["table"],
            nrows=file_config.get("nrows")
        )
        total += rows

    logger.info(f"Data loading complete — {total:,} total rows across all tables")

    # Verify row counts
    logger.info("Verifying row counts...")
    with engine.connect() as conn:
        for table in ["application_train", "bureau",
                      "previous_application", "installments_payments"]:
            result = conn.execute(
                text(f"SELECT COUNT(*) FROM raw.{table}")
            )
            count = result.scalar()
            logger.info(f"  raw.{table}: {count:,} rows")


if __name__ == "__main__":
    main()
