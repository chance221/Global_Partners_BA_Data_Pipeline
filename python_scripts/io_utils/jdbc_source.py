from python_scripts.io_utils.get_db_secret import  RdsSecret
from pyspark.sql import  DataFrame


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

  returned_df= sparkSession\
    .read.jdbc(
      url=jdbc_url, 
      table=src_table, 
      properties=create_driver_properties(secret["username"], secret["password"])
    )

  return returned_df