from pyspark.sql import functions as F
from pyspark.sql import Column, DataFrame
from datetime import datetime

from python_scripts.common.models.config_models import TableConfig
from python_scripts.io_utils.quarantine import (
    cast_failure_reasons, duplicate_key_reasons, null_key_reasons,
)


def silver_reject_reasons(df: DataFrame, tableConfig: TableConfig) -> list[Column]:
  """Every check a raw row must pass before it can be written to silver.
  """
  actual = {c.lower(): c for c in df.columns}

  # A configured cast whose column is absent is not a reject -- it is a broken
  # contract, which apply_dynamic_transform raises on by design.
  transformations = {
      actual[column.lower()]: data_type
      for column, data_type in tableConfig.silver_transforms.items()
      if column.lower() in actual
  }
  keys = [actual[key.lower()] for key in tableConfig.business_keys]

  reasons = cast_failure_reasons(transformations) + null_key_reasons(keys)

  # Only merge requires uniqueness. delete_insert replaces whole parent scopes
  # and replace rewrites everything, so duplicates are legitimate for both --
  # which is the whole reason order_item_options uses delete_insert.
  if tableConfig.silver_write == "merge":
    reasons += duplicate_key_reasons(keys)

  return reasons


def apply_dynamic_transform(df: DataFrame, tableConfig: TableConfig, batch_date: datetime) -> DataFrame:
  req_columns = {k.lower(): v for k, v in tableConfig.required_columns.items()}
  transformations = {k.lower(): v for k, v in tableConfig.silver_transforms.items()}
  
  select_exprs = []
  #loop theough the existing columns
  for col_name in df.columns:
    
    col_lower = col_name.lower()

    if col_lower in req_columns:      
      
      if col_lower in transformations:
        target_type = transformations[col_lower]

        select_exprs.append(F.col(col_name).cast(target_type).alias(col_lower))

      else:
          select_exprs.append(F.col(col_name).alias(col_lower))

  # A column the contract promises but raw does not have would otherwise be
  # dropped silently, which is somethign that should fail loudly
  missing = set(req_columns) - {c.lower() for c in df.columns}
  if missing:
    raise ValueError(
        f"required_columns missing from raw: {sorted(missing)}; "
        f"raw has {sorted(c.lower() for c in df.columns)}"
    )

  # updated_at is wall-clock processing time; batch_date is the logical period
  # the row belongs to.
  select_exprs.append(F.current_timestamp().alias("updated_at"))
  select_exprs.append(
      F.lit(batch_date.strftime("%Y-%m-%d")).cast("date").alias("batch_date")
  )

  final_df = df.select(select_exprs)

  # Value-level cleaning, after the casts so column names are already
  # lowercased. Rules are applied in the order config lists them, each seeing
  # the output of the one before it -- which is why URL excision is listed
  # first for item_category: a repaired "Kid's" must not then match ^Kids$.
  for column, rules in tableConfig.silver_cleaners.items():
    col_lower = column.lower()

    # Unreachable via config validation, which rejects a cleaner on a column
    # absent from required_columns. Kept because that check is skipped when
    # required_columns is empty, and a silent no-op is the failure mode.
    if col_lower not in final_df.columns:
      raise ValueError(
          f"silver_cleaners names {column!r}, which is not carried into "
          f"silver; carried: {sorted(final_df.columns)}"
      )

    cleaned = F.col(col_lower)
    for pattern, replacement in rules:
      cleaned = F.regexp_replace(cleaned, pattern, replacement)

    final_df = final_df.withColumn(col_lower, cleaned)

  return final_df