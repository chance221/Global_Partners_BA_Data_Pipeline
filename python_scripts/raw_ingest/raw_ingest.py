import argparse
from datetime import datetime, timedelta
from pyspark.sql import DataFrame

from python_scripts.io_utils.jdbc_source import get_source_table_data 
from python_scripts.io_utils.extract_query import build_extract_table
from python_scripts.io_utils.s3_link import write_to_s3_bucket
from python_scripts.io_utils.quarantine import (
    split_rejects, null_key_reasons, write_quarantine,
)
from python_scripts.io_utils.paths import build_path
from python_scripts.spark.spark_session import create_spark_session
from python_scripts.io_utils.get_db_secret import get_rds_credentials
from python_scripts.common.config_loader import load_config, AppConfig
from python_scripts.common.helpers import parse_data_date, yesterday_utc, check_business_keys_present_in_table




def run_raw_ingest(spark, s3Protocol, appConfig:AppConfig, batch_date:datetime, tables:list[str]):
    creds = get_rds_credentials(appConfig.secret_name, appConfig.region)
    bucket = appConfig.bucket
    env = appConfig.env
    db = appConfig.database
    rawLayer = appConfig.layers['raw']
    mode = "overwrite"
    failures = {}


    for table in  tables:
      try:
        tableConfig = appConfig.get_table_config(table)
        path = build_path(s3Protocol, bucket, env, rawLayer, table, batch_date, tableConfig.load_type)

        # For an incremental table this is a derived table carrying the window
        # predicate, so SQL Server filters instead of Spark.
        source = build_extract_table(tableConfig, batch_date, batch_date + timedelta(days=1))

        # Cached because both halves of the split derive from it, and without
        # it the JDBC source would be read twice.
        df = get_source_table_data(spark, creds, source, db).cache()

        check_business_keys_present_in_table(tableConfig.business_keys, table, df)

        # A null business key is the only row-level defect visible this early:
        # such a row can never be merged, so it is held rather than landed.
        accepted, rejected = split_rejects(
            df, null_key_reasons(tableConfig.business_keys)
        )

        held = write_quarantine(rejected, s3Protocol, appConfig, table,
                                batch_date, "raw_ingest")

        write_to_s3_bucket(accepted, path, mode)

        rows = spark.read.parquet(path).count()
        window = (f" [{batch_date:%Y-%m-%d} on {tableConfig.watermark_column}]"
                  if tableConfig.load_type == "incremental" else "")
        quarantined = f", {held} quarantined" if held else ""
        print(f"{table}: {rows} rows -> {path}{window}{quarantined}")
        df.unpersist()

      except Exception as exc:
         # Rendered to text while the JVM is still alive -- see the same note in
         # run_raw_to_silver. A Py4JJavaError read after the JVM dies raises
         # ConnectionRefusedError instead of reporting what went wrong.
         failures[table] = f"{type(exc).__name__}: {exc}"
    if failures:
        for name, message in failures.items():
          print(f"  {name}: {message}")
        raise RuntimeError(f"ingest failed for: {sorted(failures)}")
    
      

  

def main() -> None:
  parser = argparse.ArgumentParser(description="Run the raw ingest job")
  parser.add_argument("--env", required=True, choices=["local", "prod"])
  parser.add_argument("--config-dir",
                      help="directory holding {env}.yaml; defaults to the repo config/. Pass '.' on Glue -- the wheel installs into site-packages where that path does not exist, and --extra-files lands in the working directory")
  parser.add_argument("--period", type=parse_data_date,
                      default=yesterday_utc(),
                      help="Batch period 'YYYY-MM-DD'; defaults to yesterday UTC so a scheduled run needs no date computed for it")
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
    run_raw_ingest(spark, s3Protocol, appConfig, batch_date, tables)
  finally:
    # A dead JVM makes stop() raise as well, which would replace the real
    # error with a connection error. Nothing to be done either way.
    try:
      spark.stop()
    except Exception:
      pass


if __name__ == "__main__":
  main()
