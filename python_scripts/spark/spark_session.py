from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import SparkSession

if TYPE_CHECKING:  
    from python_scripts.common.models.config_models import AppConfig


# switching between catalog backends is a config change rather than a code
# change -- and so a table identifier means the same thing everywhere.
ICEBERG_CATALOG = "iceberg"

# Catalog backends this factory can configure switches for local vs cloud runs
# only includes config values and packages needed for each environment

VALID_CATALOG_TYPES = frozenset({"hadoop", "glue"})

_GLUE_CATALOG_IMPL = "org.apache.iceberg.aws.glue.GlueCatalog"
_S3_FILE_IO = "org.apache.iceberg.aws.s3.S3FileIO"

_BASE_PACKAGES = (
    "com.microsoft.sqlserver:mssql-jdbc:12.8.1.jre11",
    "org.apache.hadoop:hadoop-aws:3.4.2",
    "org.apache.iceberg:iceberg-spark-runtime-4.1_2.13:1.11.0",
)


_ICEBERG_AWS_BUNDLE = "org.apache.iceberg:iceberg-aws-bundle:1.11.0"


def resolve_catalog(appConfig: AppConfig | None = None) -> tuple[str, str]:
    """Decide which catalog backend to use and where its warehouse lives.
    """
    repo_warehouse = (
        f"file:///{Path(__file__).resolve().parents[2].as_posix()}/warehouse")

    catalog_type = os.environ.get("ICEBERG_CATALOG_TYPE")
    if catalog_type is None:
        catalog_type = appConfig.catalog_type if appConfig else "hadoop"

    if catalog_type not in VALID_CATALOG_TYPES:
        raise ValueError(
            f"catalog type {catalog_type!r} is not one of "
            f"{sorted(VALID_CATALOG_TYPES)}"
        )

    warehouse = os.environ.get("ICEBERG_WAREHOUSE")
    if warehouse is None:
        warehouse = (appConfig.warehouse if appConfig else None) or repo_warehouse

    if catalog_type == "glue" and not warehouse.startswith(("s3://", "s3a://")):
        raise ValueError(
            f"catalog type 'glue' needs an s3:// warehouse, got {warehouse!r}; "
            f"set it under `catalog:` in config or via ICEBERG_WAREHOUSE"
        )

    return catalog_type, warehouse


def _catalog_settings(catalog_type: str, warehouse: str) -> dict[str, str]:
    """The spark.sql.catalog.* keys that define the Iceberg catalog.

    Returned as a mapping rather than applied here, because the two code paths
    apply it differently: locally it goes onto the builder before the JVM
    starts, while on Glue it is set on the session Glue has already created.
    One definition either way, so the environments cannot drift.
    """
    prefix = f"spark.sql.catalog.{ICEBERG_CATALOG}"

    settings = {
        prefix: "org.apache.iceberg.spark.SparkCatalog",
        f"{prefix}.warehouse": warehouse,
    }

    if catalog_type == "hadoop":
        settings[f"{prefix}.type"] = "hadoop"
    else:
        # catalog-impl rather than type: `type` only names the backends Iceberg
        # ships shorthands for, and glue is not one of them.
        #
        # S3FileIO rather than letting Iceberg fall back to the Hadoop
        # filesystem: it talks to S3 through the AWS SDK directly, which is what
        # PyIceberg also does, so both readers resolve credentials the same way
        # -- the profile locally, the instance role on EC2.
        settings[f"{prefix}.catalog-impl"] = _GLUE_CATALOG_IMPL
        settings[f"{prefix}.io-impl"] = _S3_FILE_IO

    return settings


def create_spark_session(appConfig: AppConfig | None = None):
    """Build the Spark session the jobs run on.

    `appConfig` supplies the catalog backend and warehouse. 

    """
    # Detect if running on AWS Glue
    is_glue = "GLUE_COMMAND_ARGUMENTS" in os.environ

    if is_glue:
        # Glue supplies the session, jars and credentials but does NOT
        # register a catalog named `iceberg` -- --datalake-formats only adds
        # the jars. Settable on a live session because Spark resolves
        # catalogs lazily. These two cannot be, so they stay job parameters:
        #   --datalake-formats iceberg      jars cannot join a running JVM
        #   --conf spark.sql.extensions=... static, read at session creation
        spark = SparkSession.builder.getOrCreate()
        catalog_type, warehouse_path = resolve_catalog(appConfig)
        for key, value in _catalog_settings(catalog_type, warehouse_path).items():
            spark.conf.set(key, value)
        return spark, "s3"

    catalog_type, warehouse_path = resolve_catalog(appConfig)

    packages = list(_BASE_PACKAGES)
    if catalog_type == "glue":
        packages.append(_ICEBERG_AWS_BUNDLE)

    # Local config mirrors Glue 6.0 and all required dependencies
    builder = SparkSession.builder \
        .appName("RDS-to-S3-Batch") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")\
        .config("spark.driver.memory", "4g")\
        .config("spark.sql.shuffle.partitions", "16")\
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider")\
        .config("spark.jars.packages", ",".join(packages))\
        .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")

    for key, value in _catalog_settings(catalog_type, warehouse_path).items():
        builder = builder.config(key, value)

    return builder.getOrCreate(), "s3a"
