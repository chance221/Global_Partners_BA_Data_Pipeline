import os
from dotenv import load_dotenv
from datetime import datetime


from python_scripts.io_utils.io_utils import get_source_table_data, write_to_s3_bucket, build_path
from python_scripts.spark.spark_session import create_spark_session
from python_scripts.io_utils.get_db_secret import get_rds_credentials

load_dotenv(".env.local")
spark, s3Protocol = create_spark_session()

dbSecretName = os.environ['secretName']
region = os.environ.get('region', 'us-east-1')
db = os.environ['dbName']
creds = get_rds_credentials(dbSecretName, region)
bucket = os.environ['bucketName']
env = os.environ['envName']
rawLayer = os.environ['rawLayerName']
table = os.environ['dbTable']

df = get_source_table_data(spark, creds, "dbo.date_dim", db)

print(df.count())
rando_date = datetime(2026, 3, 15)

path = build_path(s3Protocol, bucket, env, rawLayer, table, rando_date )

write_to_s3_bucket(df, path, "overwrite")
