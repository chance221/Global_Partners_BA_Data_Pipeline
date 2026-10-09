import json 
import boto3

from typing import TypedDict

from botocore.exceptions import ClientError

class RdsSecret(TypedDict):
    username: str
    password: str
    engine: str
    host: str
    port: int
    dbInstanceIdentifier: str


def get_rds_credentials(secret_name, region_name='us-east-1') -> RdsSecret:
  session = boto3.session.Session()
  client = session.client(
    service_name="secretsmanager",
    region_name= region_name
  )

  try:
    response = client.get_secret_value(SecretId = secret_name)

  except ClientError as e:
    print(f"Error retrieving secret:{e}")
    raise e

  if "SecretString" in response:
    secret_dict = json.loads(response['SecretString'])

    return secret_dict

  else:
      raise ValueError("Was not able to retrieve secret")


