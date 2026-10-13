# Notes on Architecture

Because this will bea daily batch job that may includes variable amounts of data having a fully scalable data pipeline where maintenance cost is one of the primary concerns is important. Using AWS presented a lot of options but there are a few things I wanted to call out in thie architecture and why the choices were made

# Secretes Manager:

No credentials will be kept in the repository

</br>

## AWS Bucket

Because the data will sit in a SQL server bucket versioning is unecessary for this daily batch job, unless auditing becomes a major concern.
The Buckets are set up with basic SSE-S3 managed key encryption while SSE-KMS is preferred in production corporate environments to maintain explicit audits. Because this is a basic project I wanted to eliminate the unecessary AWS KMS API request cost for this daily batch job. S3 Data Tables instead of S3 Buckets were considered but because these are meant for high throughput for streaming operations, and since this daily data will grow over time and most likely rell require it to be archived, and can be coupled with Glue Table Optomizer to automatically compact and retain snapshots relatively easily.

</br>

## AWS RDS MSSQL Server

Conncection is made by generatinga connection string that is built using the secret returned from the AWS RDS Secrets object directly. The certificate for RDS is automatically accepted for dev purposes but in a live production environment should have a downloaded root certificate and placed in a java trust store.

</br>

## Medallion Architecture Using SCD1

As there is no business need to maintain old records using SCD1 is the best choice to keep storage cost down as well as the most performant choice when implementing an OLAP layer. Because we are not adding unecessary columns, aggregations are much faster, queries stay lean and scan fewer rows comparatively quickly. The only issue is that the data will only reflect the current state meaning if a client or product moved locations only the most recent record entry for that sale will be reflected, but considering we are not focusing on sales data trends that are not heavily reliant on product or customer locations, the trade-off is considered acceptable for this task. If we wanted to use machine learning to make inferences on ever changing historical data SCD2 would bethe most appropriate 

</br>

## Iceberg vs Strictly Parquet Updates for Silver and Golc Layers

After the database tables are ingested into flat parquet files AWS Iceberg is implemented in Silver and Gold Layers for multiple reasons. Iceberg uses a table format that tracks entire collections of parquet files using metadata, so when performing SCD 1 updates, data can be safely overwritten using standard SQL UPDATE and MERGE commands without worry of destroying data because of a crashed run. Because it uses metadata to maintain where parquet files are located, merge operations can be 20-50% less for write workloads, and S3 request cost can be reduced by up to 90% due to the elimination of object scans by using Hidden Partitioning and column level statistics. By maintaining metadata if a file ddoesn't contain the data you're looking for Iceberg would skip it entirely. It also prevents the problem where you have to trade of by overwriting partitions completely and making the overwrite dynamic for safety. Because ice

</br>

## AWS Glue

The job will run daily on a schedule started by AWS Glue. It fetches the most recent code sotred in an S3 bucket, as well as orchestrates the job from start to finish. If there is a specific date that needs to be re-run or re-processed you can bypass the sceduled event altogether and pass in a parameter for the date you are seeking to run directly into the Step function and trigger specific stages of the PySpark job based on parameters. Quarantened data is kept in a separate S3 bucket but allows the job to complete until failed records can be reviewed for reprocessing.

</br>

## Streamlit

one of the most flexible tools when it comes to visualizing data from .parquet files in a way that makes decision making easier for the consumers of the pipeline. It uses python to create robust visualizations that are relatively easy to manipulate for the end user, and if setup correctly can help simplify complicated business decisions that involves large amounts of data down to a glance.
