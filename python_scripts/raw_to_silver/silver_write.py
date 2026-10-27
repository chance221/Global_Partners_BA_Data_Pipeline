"""Apply a transformed batch to a silver Iceberg table.

Iceberg writes address a catalog identifier, never an S3 path, so `build_path`
stays a raw-layer concern. 
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from python_scripts.common.models.config_models import AppConfig, TableConfig
from python_scripts.io_utils import iceberg_write
from python_scripts.spark.spark_session import ICEBERG_CATALOG


def qualified_table_name(appConfig: AppConfig, table: str,
                         catalog: str = ICEBERG_CATALOG) -> str:
    """
    Build the three-part Iceberg identifier for a dataset.
    """
    return f"{catalog}.{appConfig.layer('silver')}.{table}"


def write_silver(spark: SparkSession, df: DataFrame, appConfig: AppConfig,
                 table: str, tableConfig: TableConfig,
                 catalog: str = ICEBERG_CATALOG) -> str:
    """
    Apply one transformed batch to its silver table.
    """
    fqn = qualified_table_name(appConfig, table, catalog)

    # Create the silver namespace if this is the first table ever written.
    iceberg_write.ensure_namespace(spark, catalog, appConfig.layer("silver"))

    return iceberg_write.write_iceberg(
        spark, df, fqn,
        strategy=tableConfig.silver_write,
        merge_keys=tableConfig.business_keys,
        scope_keys=tableConfig.silver_scope_keys,
        partition_by=tableConfig.silver_partition_by,
        view_name=f"_silver_batch_{table}",
    )
