"""Build the source-side object that `spark.read.jdbc` reads from.

`spark.read.jdbc(table=...)` interpolates its argument into `FROM <table>`, so
passing an aliased derived table pushes a predicate down to SQL Server instead
of pulling the whole table across the wire and filtering in Spark.

Which predicate depends on how silver applies the batch:

  merge          A row-level window is enough. Each row is identified by its
                 business keys, so MERGE can upsert whatever arrives.
  delete_insert  The window must be widened to whole parents. delete_insert
                 removes every target row for the scope keys in the batch, so a
                 batch holding one new option row without its siblings would
                 delete the siblings and not put them back. The watermark picks
                 *which parents changed*; the outer query pulls those parents
                 whole.

Identifiers and load types interpolated here are config-owned and already
validated by TableConfig, so this module trusts them rather than re-checking.
"""

from __future__ import annotations

from datetime import datetime, timezone

from python_scripts.common.models.config_models import TableConfig


def sql_timestamp_literal(moment: datetime) -> str:
    """Render a datetime as an ISO-8601 literal SQL Server reads unambiguously.

    The `yyyy-MM-ddThh:mm:ss.fff` form is language- and DATEFORMAT-neutral,
    unlike the space-separated form, so the predicate means the same thing
    whatever the server's locale is set to.

    Converted to UTC and the offset dropped, because `datetime2` carries no
    offset and `inserted_at` is stamped by `SYSUTCDATETIME()`. Callers get UTC
    from `parse_data_date`, so the conversion is a no-op in practice and is
    here only so a value from another zone still compares correctly.
    """
    as_utc = moment.astimezone(timezone.utc)
    return as_utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{as_utc.microsecond // 1000:03d}"


def watermark_predicate(column: str, start: datetime, end: datetime) -> str:
    """Half-open range on the watermark column, `[start, end)`.

    Half-open so consecutive periods neither overlap nor leave a gap: a row
    stamped exactly on the boundary belongs to one window only.
    """
    return (
        f"{column} >= '{sql_timestamp_literal(start)}' "
        f"AND {column} < '{sql_timestamp_literal(end)}'"
    )


def build_extract_table(table_config: TableConfig, start: datetime,
                        end: datetime) -> str:
    """Return the `table` argument for `spark.read.jdbc`.

    For an unfiltered load that is the source table name. For an incremental
    load it is an aliased derived table carrying the window predicate, widened
    to whole parents when silver applies the batch by delete_insert.

    Args:
        start: inclusive start of the window, timezone-aware.
        end: exclusive end of the window, timezone-aware.
    """
    source = table_config.source_table

    if table_config.load_type != "incremental":
        return source

    if table_config.silver_write != "delete_insert":
        predicate = watermark_predicate(table_config.watermark_column, start, end)
        return f"(SELECT * FROM {source} WHERE {predicate}) AS src"

    # Widen to every row of each parent the window touched.
    #
    predicate = watermark_predicate(f"w.{table_config.watermark_column}", start, end)
    correlation = " AND ".join(
        f"w.{key} = o.{key}" for key in table_config.silver_scope_keys
    )

    return (
        f"(SELECT o.* FROM {source} o WHERE EXISTS ("
        f"SELECT 1 FROM {source} w WHERE {correlation} AND {predicate}"
        f")) AS src"
    )
