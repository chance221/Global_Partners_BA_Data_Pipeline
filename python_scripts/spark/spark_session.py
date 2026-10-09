import os
from pyspark.sql import SparkSession


def create_spark_session():
    # Detect if running on AWS Glue
    is_glue = "GLUE_COMMAND_ARGUMENTS" in os.environ

    if is_glue:
        # Glue handles dependencies and S3 credentials automatically
        return SparkSession.builder.getOrCreate(), "s3"
    else:
        # Local configuration using the Default AWS Credential Chain
        spark = SparkSession.builder \
            .appName("RDS-to-S3-Batch") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                    "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")\
            .config("spark.jars.packages", "com.microsoft.sqlserver:mssql-jdbc:12.8.1.jre11,"
                    "org.apache.hadoop:hadoop-aws:3.3.4")\
            .getOrCreate()
        return spark, "s3a"

