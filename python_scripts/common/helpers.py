from datetime import datetime, timezone

from pyspark.sql import DataFrame


# Parse Date Arguments
#**********************************************
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
