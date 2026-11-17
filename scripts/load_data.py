"""
Load raw Home Credit CSVs into PostgreSQL raw schema.
Run once after docker compose up.

Usage:
    python scripts/load_data.py                                  # all tables
    python scripts/load_data.py --tables installments_payments   # one table

Rows are streamed in chunks and written with PostgreSQL COPY, so memory
stays flat and the full 13.6M-row installments file loads in minutes.
"""

import argparse
import csv
import io

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loanlens.config import get_settings
from loanlens.utils.logging import logger

settings = get_settings()


def get_engine():
    """Create synchronous SQLAlchemy engine for bulk loading."""
    return create_engine(settings.database_url_sync)


def copy_insert(table, conn, keys, data_iter):
    """pandas.to_sql method that bulk-loads with PostgreSQL COPY."""
    buf = io.StringIO()
    csv.writer(buf).writerows(data_iter)
    buf.seek(0)
    columns = ", ".join(f'"{k}"' for k in keys)
    name = f"{table.schema}.{table.name}" if table.schema else table.name
    with conn.connection.cursor() as cur:
        cur.copy_expert(f"COPY {name} ({columns}) FROM STDIN WITH CSV", buf)


def load_csv_to_postgres(
    csv_path: Path,
    table_name: str,
    schema: str = "raw",
    chunksize: int = 200_000,
    nrows: int = None
) -> int:
    if not csv_path.exists():
        logger.warning(f"File not found: {csv_path} — skipping")
        return 0

    logger.info(f"Loading {csv_path.name} → {schema}.{table_name}")
    engine = get_engine()
    total_rows = 0

    # Reloading: empty the existing table instead of dropping it, because
    # dbt staging views depend on it and Postgres refuses to drop it.
    exists = inspect(engine).has_table(table_name, schema=schema)
    if exists:
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE TABLE {schema}.{table_name}"))
        logger.info(f"  Emptied existing {schema}.{table_name} (dependent views kept)")

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
            if_exists="append" if exists or i > 0 else "replace",
            index=False,
            method=copy_insert,
        )
        total_rows += len(chunk)
        if i % 5 == 0:
            logger.info(f"  Loaded {total_rows:,} rows...")

    logger.info(f"  Done — {total_rows:,} total rows loaded")
    return total_rows

def main():
    """Load Home Credit CSV files into PostgreSQL."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="*", help="Only load these tables")
    args = parser.parse_args()

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
            "nrows": None  # All 13.6M rows (was a 500K head sample, which
                           # gave only 23.5% of applicants any payment history)
        },
    ]

    engine = get_engine()
    total = 0

    if args.tables:
        files_to_load = [f for f in files_to_load if f["table"] in args.tables]

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
        for table in [f["table"] for f in files_to_load]:
            result = conn.execute(
                text(f"SELECT COUNT(*) FROM raw.{table}")
            )
            count = result.scalar()
            logger.info(f"  raw.{table}: {count:,} rows")


if __name__ == "__main__":
    main()
