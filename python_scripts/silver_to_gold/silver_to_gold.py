"""Build the gold layer from silver.

Mirrors the other two jobs: one table per loop iteration, failures collected so
one bad table does not abort the rest, and the logical batch date threaded in
as a parameter.

--full forces a complete rebuild instead of recomputing only the event dates
this batch touched. Needed after changing a transform, since an incremental run
would only correct the dates that happen to be in scope.
"""

import argparse
from datetime import datetime

from python_scripts.common.config_loader import load_config
from python_scripts.common.helpers import parse_data_date, yesterday_utc
from python_scripts.common.models.config_models import AppConfig
from python_scripts.silver_to_gold.gold_write import write_gold
from python_scripts.silver_to_gold.silver_read import (
    read_silver_inputs, silver_table_exists,
)
from python_scripts.silver_to_gold.transforms import get_transform
from python_scripts.spark.spark_session import create_spark_session


def run_silver_to_gold(spark, appConfig: AppConfig, batch_date: datetime,
                       tables: list[str], full: bool = False):
  failures = {}

  for table in tables:
    try:
      goldConfig = appConfig.get_gold_config(table)

      missing = [t for t in goldConfig.depends_on
                 if not silver_table_exists(spark, appConfig, t)]
      if missing:
        raise ValueError(f"silver tables not built yet: {missing}")

      inputs = read_silver_inputs(spark, appConfig, goldConfig, batch_date, full)

      result = get_transform(table)(inputs, batch_date)

      outcome = write_gold(spark, result, appConfig, table, goldConfig)
      scope = "full rebuild" if full or not goldConfig.event_time_column else "incremental"
      print(f"{table}: {outcome} ({scope})")
    except Exception as exc:
      # Rendered to text while the JVM is still alive -- a Py4JJavaError read
      # after the JVM dies raises ConnectionRefusedError instead of reporting
      # what went wrong.
      failures[table] = f"{type(exc).__name__}: {exc}"

  if failures:
    for name, message in failures.items():
      print(f"  {name}: {message}")
    raise RuntimeError(f"silver_to_gold failed for: {sorted(failures)}")


def main() -> None:
  parser = argparse.ArgumentParser(description="Build the gold layer from silver")
  parser.add_argument("--env", required=True, choices=["local", "prod"])
  parser.add_argument("--config-dir",
                      help="directory holding {env}.yaml; defaults to the repo config/. Pass '.' on Glue -- the wheel installs into site-packages where that path does not exist, and --extra-files lands in the working directory")
  parser.add_argument("--period", type=parse_data_date,
                      default=yesterday_utc(),
                      help="Batch period 'YYYY-MM-DD'; defaults to yesterday UTC "
                           "so a scheduled run needs no date computed for it")
  parser.add_argument("--full", action="store_true",
                      help="rebuild every grain instead of only those this batch touched")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--tables", help="gold tables to build", nargs="+")
  group.add_argument("--all", action="store_true", help="build every gold table")
  args, _unknown_args = parser.parse_known_args()

  appConfig = load_config(args.env, args.config_dir)
  batch_date = args.period

  tables = list(appConfig.gold) if args.all else args.tables

  unknown = set(tables) - set(appConfig.gold)
  if unknown:
    raise ValueError(
        f"unknown gold table(s) {sorted(unknown)}; config defines {sorted(appConfig.gold)}")

  spark, _s3Protocol = create_spark_session(appConfig)

  try:
    run_silver_to_gold(spark, appConfig, batch_date, tables, args.full)
  finally:
    try:
      spark.stop()
    except Exception:
      pass


if __name__ == "__main__":
  main()
