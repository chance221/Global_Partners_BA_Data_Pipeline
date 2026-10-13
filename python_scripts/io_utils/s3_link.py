from pyspark.sql import  DataFrame


  
def write_to_s3_bucket(
    
    data:DataFrame,
    s3_target_path:str,
    mode:str
    ):
  """
    Writes data to S3 bucket
  
    Args:
      data: spark.sql DataFrame
      s3_target_path: s3 path to be updated
      src_table: Source table
    """
  print(f"💾 Writing raw batch to S3: {s3_target_path}...")

  data.write\
    .mode(mode)\
    .parquet(s3_target_path)




  