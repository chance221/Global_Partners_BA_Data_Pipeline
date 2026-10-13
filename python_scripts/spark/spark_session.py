import os
from pyspark.sql import SparkSession


def create_spark_session():
    # Detect if running on AWS Glue
    is_glue = "GLUE_COMMAND_ARGUMENTS" in os.environ

    if is_glue:
        # Glue handles dependencies and S3 credentials automatically
        return SparkSession.builder.getOrCreate(), "s3"
    else:
        # Local configuration mirroring the Glue 6.0 runtime: Spark 4.1.1,
        # Hadoop 3.4.2, Scala 2.13.17, Java 17.
        #
        # hadoop-aws MUST match the Hadoop version bundled in pyspark -- check with
        #   ls .venv/Lib/site-packages/pyspark/jars/ | grep hadoop-client-api
        # A mismatch fails at runtime with NoSuchMethodError, not at install.
        #
        # Hadoop 3.4+ uses AWS SDK for Java v2, so the credentials provider is the
        # software.amazon.awssdk class, not the v1 com.amazonaws one (Glue 6.0 also
        # ships v2 only). S3A's own default chain omits the profile provider, so this
        # setting is what lets it read ~/.aws/credentials.
        spark = SparkSession.builder \
            .appName("RDS-to-S3-Batch") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.sql.sources.partitionOverwriteMode", "dynamic")\
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                    "software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider")\
            .config("spark.jars.packages", "com.microsoft.sqlserver:mssql-jdbc:12.8.1.jre11,"
                    "org.apache.hadoop:hadoop-aws:3.4.2")\
            .getOrCreate()
        return spark, "s3a"

