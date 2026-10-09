from pyspark.sql import  DataFrame
from datetime import datetime, timezone
from typing import Optional

from python_scripts.spark.spark_session import create_spark_session
from python_scripts.io_utils.get_db_secret import get_rds_credentials, RdsSecret


def build_path(protocol:str, bucket:str, env:str, layer:str, db_table:str,  write_date:datetime) -> str:
  
  effective = write_date or datetime.now(timezone.utc)
  year, month, day = effective.strftime("%Y"), effective.strftime("%m"), write_date.strftime("%d")

  no_prefix_table= db_table.removeprefix('dbo.')
  path = f'{protocol}://{bucket}/{env}/{layer}/{no_prefix_table}/year={year}/month={month}/day={day}'

  return path



def create_driver_properties(user: str, pw: str,):
  db_properties = {
    "user" : f'{user}',
    "password": f'{pw}',
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
  }

  return db_properties



def get_source_table_data(sparkSession, secret: RdsSecret, src_table: str, db: str) -> DataFrame:
  """
  Reads a SQL Server Table

  Args:
    spark: Active Spark Session
    secret: RDS secret dict containing all info needed to read using the jdbs driver
    src_table: Source table
    db: Database name
  """
  jdbc_url = f"jdbc:sqlserver://{secret['host']}:{secret.get('port', 1433)};databaseName={db};encrypt=true;trustServerCertificate=true"

  returned_df= sparkSession.read.jdbc(url=jdbc_url, table=src_table, properties=create_driver_properties(secret["username"], secret["password"]))

  return returned_df


  
def write_to_s3_bucket(
    
    data:DataFrame,
    s3_target_path:str,
    mode:str
    ):
  """
    Writes data to S3 bucket
  
    Args:
      data: spark.sql DataFrame
      s3_target_path: s3 path to be updated
      src_table: Source table
    """
  print(f"💾 Writing raw batch to S3: {s3_target_path}...")

  data.write\
    .mode(mode)\
    .parquet(s3_target_path)




  