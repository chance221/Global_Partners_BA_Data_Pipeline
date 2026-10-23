"""Apply a transformed batch to a silver Iceberg table.

Iceberg writes address a catalog identifier, never an S3 path, so `build_path`
stays a raw-layer concern. The catalog decides where bytes land, which is what
lets the same job code run locally and on Glue with no branching.

The write strategies live in io_utils.iceberg_write, since gold needs the same
three. What remains here is silver-specific: resolving a dataset name to an
identifier, and reading the strategy and keys out of its TableConfig.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from python_scripts.common.models.config_models import AppConfig, TableConfig
from python_scripts.io_utils import iceberg_write
from python_scripts.spark.spark_session import ICEBERG_CATALOG


def qualified_table_name(appConfig: AppConfig, table: str,
                         catalog: str = ICEBERG_CATALOG) -> str:
    """Build the three-part Iceberg identifier for a dataset.

    The namespace reuses the configured silver layer name, so the catalog
    identifier and the raw S3 prefix cannot drift apart.
    """
    return f"{catalog}.{appConfig.layer('silver')}.{table}"


def write_silver(spark: SparkSession, df: DataFrame, appConfig: AppConfig,
                 table: str, tableConfig: TableConfig,
                 catalog: str = ICEBERG_CATALOG) -> str:
    """Apply one transformed batch to its silver table.

    `df` must already be projected, cast and have rejected rows removed.
    Returns a short description of what happened, for the run log.
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
