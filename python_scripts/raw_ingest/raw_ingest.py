import argparse
from datetime import datetime
from pyspark.sql import DataFrame

from python_scripts.io_utils.jdbc_source import get_source_table_data 
from python_scripts.io_utils.s3_link import write_to_s3_bucket
from python_scripts.io_utils.paths import build_path
from python_scripts.spark.spark_session import create_spark_session
from python_scripts.io_utils.get_db_secret import get_rds_credentials
from python_scripts.common.config_loader import load_config, AppConfig

# region Parse Arguments
#**********************************************
def parse_data_date(value: str) -> datetime:
    """Parse and validate a YYYY-MM-DD batch date."""
    return datetime.strptime(value, "%Y-%m-%d")

# endregion ********************************



# region Check Data
#**********************************************

def check_business_keyss_present_in_table(businessKeys:list[str], table:str, df:DataFrame):
   missing = set(businessKeys) - set(df.columns)

   if missing:
      raise ValueError(
          f"{table}: business_keys missing from source: {sorted(missing)}; "
          f"source has {sorted(df.columns)}"
      )

# endregion


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

        df = get_source_table_data(spark, creds, tableConfig.source_table, db)

        check_business_keyss_present_in_table(tableConfig.business_keys, table, df)
        
        write_to_s3_bucket(df, path, mode)

        rows = spark.read.parquet(path).count()
        print(f"{table}: {rows} rows -> {path}")

      except Exception as exc:
         failures[table] = exc
    if failures:
        for name, exc in failures.items():
          print(f"  {name}: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"ingest failed for: {sorted(failures)}")
    
      

  

def main() -> None:
  parser = argparse.ArgumentParser(description="Run the raw ingest job")
  parser.add_argument("--env", required=True, choices=["local", "prod"])
  parser.add_argument("--period", required=True, type=parse_data_date, help="Daily batch perioc in 'YYYY-MM-DD' format")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--tables", help="tables to ingest from db into raw layer", nargs="+")
  group.add_argument("--all", action="store_true", help="run every dataset config")
  args, _unknown_args = parser.parse_known_args()  
  
  env = args.env  
  appConfig = load_config(env)
  batch_date = args.period

  if args.all:
      #get all of the appData values
      tables = list(appConfig.tables.keys())
  else:
      tables = args.tables 

  unknown = set(tables) - set(appConfig.tables)
    
  if unknown:
    raise ValueError(f"unknown dataset(s) {sorted(unknown)}; config defines {sorted(appConfig.tables)}")

  spark, s3Protocol = create_spark_session()

  try:
    run_raw_ingest(spark, s3Protocol, appConfig, batch_date, tables)
  finally:
     spark.stop()  


if __name__ == "__main__":
  main()