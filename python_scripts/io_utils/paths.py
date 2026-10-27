from datetime import datetime, timezone


def build_path(protocol:str, bucket:str, env:str, layer:str, db_table:str,  write_date:datetime, load_type:str) -> str:  
  
  path = f'{protocol}://{bucket}/{env}/{layer}/{db_table}'
  
  if load_type == 'static':
    return path
  
  if load_type in ('full', 'incremental'):
    path += f'/batch_year={write_date:%Y}/batch_month={write_date:%m}/batch_day={write_date:%d}'
    return path

  raise ValueError(f'unknown load_type {load_type}; expected static, full or incremental')