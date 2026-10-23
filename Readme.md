# Notes on Architecture

Because this will bea daily batch job that may includes variable amounts of data having a fully scalable pipeline where maintenance cost is one of the primary concerns. Using AWS presented a lot of options but there are a few things I wanted to call out in thie architecture and why the choices were made

There is a split on the type of data format. A data lake was created for the raww lay that holds parquet files in S3 object storage maually partitioned on run date to represent the daily nature of this batch job. The current data size would not require partitioning but this pipeline assims a much larger volume of data would come through for a national company. 

The silver, and gold layers are each partitioned by month whiich also is not necessary and may create additional overhead but used to show the pipeline can scale to handle a much larger amount of data. Both layers are considerred data lakehouses because they store data using an iceberg format in an s3 bucket which holds not only the parquet files but metadata that allows for iceberg to operate much more effficiently than pure parquet reads through pyspark. 

</br>

## Why Iceberg vs Strictly Parquet Updates for Silver and Gold Layers

After the database tables are ingested into flat parquet files in the raw layer AWS Iceberg is implemented through a Glue job in Silver and Gold Layers for multiple reasons. Iceberg features the elimination of object scans by using Hidden Partitioning and column level statistics by maintining metadata that lists where warehouse  files are located, and what data is reelvant, stale or unecessary. This means when performing SCD 1 updates, data can be safely overwritten using standard SQL UPDATE and MERGE commands without worry of destroying data dut to a crashed run because of its snapshot retrival capability. Because of this, it is also cost efficient reducing merge operations by 20-50% for write workloads. If a file doesn't contain the data you're looking for Iceberg would skip it entirely, so S3 request cost can be reduced by up to 90% due to its metadata strategy.  

# Secretes Manager:

No credentials will be kept in the repository. This is imnportant to make sure that the data remain secure throughout the process. Because the jobs will be dispatched and run through glue the execution account associated with that AWS operation will reference all necessary secrets and provide permissions for each environment respectually 

</br>

## AWS Bucket

Because the existing data will sit in a SQL server bucket versioning is unecessarilyy duplicative for this daily batch job, unless auditing becomes a major concern. Also this would duplicate what the snapshots already achieve with iceberg handles gracefullly with metadata. The Buckets are set up with basic SSE-S3 managed key encryption while SSE-KMS is preferred in production corporate environments to maintain explicit audit capability. Because this is a basic project I wanted to eliminate the unecessary AWS KMS API request cost for this daily batch job. S3 Data Tables instead of S3 Buckets were considered but because these are meant for high throughput for streaming operations, and this is daily batch data that will grow over time and most likely require raw to be archived as well as applying a expiration policy to each iceberg warehouse. Storage can also be coupled with Glue Table Optomizer to automatically compact and retain snapshots relatively easily.

</br>

## AWS RDS MSSQL Server

Conncection is made by generatinga connection string that is built using the secret returned from the AWS RDS Secrets object directly. The certificate for RDS is automatically accepted for dev purposes but in a live production environment should have a downloaded root certificate and placed in a java trust store.

</br>

## Medallion Architecture Using SCD1

As there is no business need to maintain old records using SCD1 is the best choice to keep storage cost down as well as the most performant choice when implementing an OLAP layer. Because we are not adding unecessary columns, aggregations are much faster, queries stay lean and scan fewer rows comparatively quickly. The only issue is that the data will only reflect the current state meaning if a client or product moved locations only the most recent record entry for that sale will be reflected, but considering we are not focusing on sales data trends that are not heavily reliant on product or customer locations, the trade-off is considered acceptable for this task. If we wanted to use machine learning to make inferences on ever changing historical data SCD2 would bethe most appropriate 



</br>

## AWS Glue

The job will run daily on a schedule started by AWS Glue. It fetches the most recent code sotred in an S3 bucket, as well as orchestrates the job from start to finish. If there is a specific date that needs to be re-run or re-processed you can bypass the sceduled event altogether and pass in a parameter for the date you are seeking to run directly into the Step function and trigger specific stages of the PySpark job based on parameters. Quarantened data is kept in a a quarantene folder in the S3 bucket and allows the job to complete, so failed records can be reviewed for reprocessing, after cleaned. The jobs are set to run based on batch date so picking up older orders would require this to be changed to the current daily run to be processed through all three stages, or at the stage of failure. 

</br>

## Streamlit

one of the most flexible tools when it comes to visualizing data from .parquet files in a way that makes decision making easier for the consumers of the pipeline. It uses python to create robust visualizations that are relatively easy to manipulate for the end user, and if setup correctly can help simplify complicated business decisions that involves large amounts of data down to a glance.
