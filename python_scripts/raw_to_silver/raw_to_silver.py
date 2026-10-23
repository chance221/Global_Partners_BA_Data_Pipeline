import argparse

from datetime import datetime
from python_scripts.io_utils.paths import build_path
from python_scripts.common.config_loader import load_config
from python_scripts.common.models.config_models import AppConfig 


from python_scripts.spark.spark_session import create_spark_session
from python_scripts.common.helpers import parse_data_date, check_business_keys_present_in_table
from python_scripts.raw_to_silver.raw_transforms import (
    apply_dynamic_transform, silver_reject_reasons,
)
from python_scripts.io_utils.quarantine import split_rejects, write_quarantine
from python_scripts.raw_to_silver.silver_write import write_silver


def run_raw_to_silver(spark, s3Protocol, appConfig:AppConfig, batch_date:datetime, tables:list[str]):
  failures = {}

  for table in tables:
    try:
      tableConfig = appConfig.get_table_config(table)

      source_path = build_path(s3Protocol, appConfig.bucket, appConfig.env, appConfig.layer("raw"), table, batch_date, tableConfig.load_type)

      source_df = spark.read.parquet(source_path)

      check_business_keys_present_in_table(tableConfig.business_keys, table, source_df)

      # Checked on the raw frame, before any conversion, so quarantine holds
      # the values the source actually contained.
      accepted, rejected = split_rejects(
          source_df, silver_reject_reasons(source_df, tableConfig)
      )

      held = write_quarantine(rejected, s3Protocol, appConfig, table,
                              batch_date, "raw_to_silver")

      updated_df = apply_dynamic_transform(accepted, tableConfig, batch_date)

      outcome = write_silver(spark, updated_df, appConfig, table, tableConfig)
      quarantined = f" ({held} quarantined)" if held else ""
      print(f"{table}: {outcome}{quarantined}")
    except Exception as exc:
       # Rendered to text here, while the JVM is still alive. A Py4JJavaError
       # fetches its message from the JVM when it is formatted, so holding the
       # object and printing it after the loop loses the message entirely if
       # the JVM has since died -- the print raises ConnectionRefusedError and
       # the only record of what actually failed is gone.
       failures[table] = f"{type(exc).__name__}: {exc}"

  if failures:
      for name, message in failures.items():
        print(f"  {name}: {message}")
      raise RuntimeError(f"raw_to_silver failed for: {sorted(failures)}")

    

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the raw ingest job")
    parser.add_argument("--env", required=True, choices=["local", "prod"])
    parser.add_argument("--config-dir",
                        help="directory holding {env}.yaml; defaults to the repo config/. Pass '.' on Glue -- the wheel installs into site-packages where that path does not exist, and --extra-files lands in the working directory")
    parser.add_argument("--period", required=True, type=parse_data_date, help="Daily batch perioc in 'YYYY-MM-DD' format")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tables", help="tables to ingest from db into raw layer", nargs="+")
    group.add_argument("--all", action="store_true", help="run every dataset config")
    args, _unknown_args = parser.parse_known_args()  
    
    env = args.env  
    appConfig = load_config(env, args.config_dir)
    batch_date = args.period
    
    if args.all:
        #get all of the appData values
        tables = list(appConfig.tables.keys())
    else:
        tables = args.tables 
  
    unknown = set(tables) - set(appConfig.tables)
    
    if unknown:
      raise ValueError(f"unknown dataset(s) {sorted(unknown)}; config defines {sorted(appConfig.tables)}")
  
    spark, s3Protocol = create_spark_session(appConfig)

    try:
      run_raw_to_silver(spark, s3Protocol, appConfig, batch_date, tables)
    finally:
      # A dead JVM makes stop() raise as well, which would replace the real
      # error with a connection error. Nothing to be done either way.
      try:
        spark.stop()
      except Exception:
        pass


if __name__ == "__main__":
  main()
