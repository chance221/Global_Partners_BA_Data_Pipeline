from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession

from python_scripts.common.models.config_models import AppConfig, GoldTableConfig
from python_scripts.io_utils import iceberg_write
from python_scripts.spark.spark_session import ICEBERG_CATALOG


def gold_table_name(appConfig: AppConfig, table: str,
                    catalog: str = ICEBERG_CATALOG) -> str:
    return f"{catalog}.{appConfig.layer('gold')}.{table}"


def write_gold(spark: SparkSession, df: DataFrame, appConfig: AppConfig,
               table: str, goldConfig: GoldTableConfig,
               catalog: str = ICEBERG_CATALOG) -> str:
    """Apply one recomputed batch to its gold table.

    `df` must hold the complete result for every scope value it covers --
    delete_insert removes all existing rows for those values.

    Returns a short description of what happened, for the run log.
    """
    fqn = gold_table_name(appConfig, table, catalog)

    # Create the gold namespace if this is the first table ever written.
    iceberg_write.ensure_namespace(spark, catalog, appConfig.layer("gold"))

    return iceberg_write.write_iceberg(
        spark, df, fqn,
        strategy=goldConfig.write,
        merge_keys=goldConfig.grain_keys,
        scope_keys=goldConfig.scope_keys,
        partition_by=goldConfig.partition_by,
        view_name=f"_gold_batch_{table}",
    )
