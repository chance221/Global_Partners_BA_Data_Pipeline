from datetime import datetime, timezone


def build_path(protocol:str, bucket:str, env:str, layer:str, db_table:str,  write_date:datetime, load_type:str) -> str:  
  
  path = f'{protocol}://{bucket}/{env}/{layer}/{db_table}'
  
  if load_type == 'static':
    return path
  
  # 'incremental' shares the shape of 'full': one partition per batch period.
  # Raw is partitioned by when rows ARRIVED, not when the orders were created --
  # an incremental window selects by inserted_at, so a 2021 order arriving today
  # belongs in today's partition. Event-time partitioning happens in silver,
  # which partitions by days(creation_time_utc).
  if load_type in ('full', 'incremental'):
    path += f'/batch_year={write_date:%Y}/batch_month={write_date:%m}/batch_day={write_date:%d}'
    return path

  raise ValueError(f'unknown load_type {load_type}; expected static, full or incremental')