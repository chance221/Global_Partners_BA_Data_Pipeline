from datetime import datetime, timedelta, timezone

from pyspark.sql import DataFrame


# Parse Date Arguments
#**********************************************
def yesterday_utc() -> datetime:
    """Default batch period for a scheduled daily run.

    A job firing after midnight processes the day that just closed, not the few
    hours of the day it started in: the extract window is
    [period, period + 1 day), so yesterday captures a complete day.

    Midnight-aligned and UTC so an implicit period is indistinguishable from
    the same date passed explicitly -- the partition segments build_path emits
    and the watermark predicate both come out identical.
    """
    return (datetime.now(timezone.utc) - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0)


def parse_data_date(value: str) -> datetime:
    
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)



# check config keys
#**********************************************
def check_business_keys_present_in_table(businessKeys:list[str], table:str, df:DataFrame):
   missing = {k.lower() for k in businessKeys} - {c.lower() for c in df.columns}

   if missing:
      raise ValueError(
          f"{table}: business_keys missing from source: {sorted(missing)}; "
          f"source has {sorted(df.columns)}"
      )
