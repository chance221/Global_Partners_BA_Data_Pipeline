# Notes on Architecture

Because this will be a daily batch job that may includes variable amounts of data having a fully scalable pipeline where maintenance cost is one of the primary concerns. Using AWS presented a lot of options but there are a few things I wanted to call out in the architecture and why the choices were made

There is a split on the type of data format. A data lake was created for the raw layer that holds parquet files in S3 object storage manually partitioned on run date to represent the daily nature of this batch job. The current data size would not require partitioning but this pipeline assumes a much larger volume of data would come through for a national company.

The silver, and gold layers are each considered data lakehouses because they store data using an iceberg format in an s3 bucket which holds not only the parquet files but metadata that allows for iceberg to operate much more efficiently than pure parquet reads through pyspark.

Partitioning in those two layers is applied where it earns its keep rather than everywhere. The rule is to partition what grows on every run and leave entity-grain tables alone. Silver's `order_items` is partitioned by `months(creation_time_utc)` and gold's `daily_restaurant_sales` by `months(sales_date)`. Because both accumulate rows with each batch and a month partition lets a query skip whole ranges of files it does not need. The rest are deliberately unpartitioned: `date_dim` is a small static dimension, and `restaurant_revenue_rank`, `customer_lifetime_value` and `customer_rfm` are each one row per entity, so splitting them would add metadata overhead and a scatter of small files without removing any work from a query that reads most of the table anyway. Partitioning a table that fits comfortably in a handful of files costs more than it saves.

</br>

## Why Iceberg vs Strictly Parquet Updates for Silver and Gold Layers

After the database tables are ingested into flat parquet files in the raw layer Apache Iceberg is implemented through a Glue job in Silver and Gold Layers for multiple reasons. Iceberg features the reduction of object scans by using Hidden Partitioning and column level statistics by maintaining metadata that lists where warehouse files are located, and what data is relevant, stale or unecessary. This means when performing SCD 1 updates, data can be safely overwritten using standard SQL commands without worry of destroying data due to a crashed run because of its snapshot retreval capability (Unless a table is dropped, because that destroys the metadata effectively removing prior snapshots but leaves the parquet file itself).

Because of this, it is also cost efficient reducing merge operations by 20-50% for write workloads. If a file doesn't contain the data you're looking for Iceberg would reduce the need to scan unecessarily, so S3 request cost can be reduced by up to 90% due to its metadata strategy. Statements based on video located [here](https://www.youtube.com/watch?v=VpihapR6550&t=1077s).
</br>

## Secrets Manager:

No credentials are kept in the repository. This is important to make sure that the data remain secure throughout the process. Because the jobs will be dispatched and run through glue the execution role associated with that AWS operation will reference all necessary secrets and provide permissions for each environment respectively.

</br>

## AWS Bucket

Because the existing data will sit in a SQL server, bucket versioning is unecessarily duplicative for this daily batch job, unless auditing becomes a major concern. Also this would duplicate what the snapshots already achieve with iceberg handles gracefuly with metadata. The Buckets are set up with basic SSE-S3 managed key encryption while SSE-KMS is preferred in production corporate environments to maintain explicit audit capability. Because this is a basic project I wanted to eliminate the unnecessary AWS KMS API request cost for this daily batch job. Storage can also be coupled with Glue Table Optimizer to automatically compact and retain snapshots relatively easily.

</br>

## AWS RDS MSSQL Server

Connection is made by generating a connection string that is built using the secret returned from the AWS RDS Secrets object directly. The certificate for RDS is automatically accepted for dev purposes but in a live production environment should have a downloaded root certificate and placed in a java trust store.

</br>

## Medallion Architecture Using SCD1

As there is no business need to maintain old records using SCD1 is the best choice to keep storage cost down as well as the most performant choice when implementing an OLAP layer. Because we are not adding unecessary columns, aggregations are much faster, queries stay lean and scan fewer rows comparatively quickly. The only issue is that the data will only reflect the current state meaning if a client or product moved locations only the most recent record entry for that sale will be reflected, but considering we are not focusing on sales data trends that are not heavily reliant on product or customer locations, the trade-off is considered acceptable for this task. If we wanted to use machine learning to make inferences on ever changing historical data SCD2 would be the most appropriate

</br>

## AWS Glue

The can run daily on a schedule started by and maintained by AWS Glue, or a step function. The glue job is linked to a folder that updates on approved CI/CD pipeline builds and an updated immutable hash is a part of the folder path. This means the glue job automatically fetches the most recent code stored in the S3 bucket linked to the github hashed repo code, as well as orchestrates the job from start to finish. If there is a specific date that needs to be re-run or re-processed you can bypass a scheduled event altogether and pass in a parameter for the date you are seeking to run directly into the glue job and trigger all or specific stages based on those arguments.

</br>

## Quarentened Data

Quarentened data is kept in a quarantene folder in the S3 bucket and allows the job to complete, so failed records can be reviewed for reprocessing, after inspected and cleaned.

</br>

## Streamlit

Uses python to create robust visualizations that are relatively easy to manipulate for the end user, and if setup correctly can help simplify complicated business decisions that involves large amounts of data down to a glance. Natively cannot read iceberg files but can be used with PyIceberg to query glue tables in gold

</br>

## Customer Segments

Built by `python_scripts/silver_to_gold/transforms/customer_rfm.py` into `gold.customer_rfm`, one row per `user_id`. The table is a full rebuild every run (`write: replace`) rather than an incremental one, because a segment describes a customer's whole relationship with the business and an incremental run would classify people on whatever fraction of their orders happened to arrive in that batch.

### The 90-day window

Everything is measured from an **anchor date**: the dataset's own most recent order date, not `batch_date`. The sample data ends 2024-02-20 while the job runs with a batch date of 2026-10-16, 969 days later, so anchoring to the batch would mark every single customer Dormant and make the segmentation meaningless. In production, where data lands daily, the two converge. The anchor is written to the table as `anchor_date` so any number below can be traced back to the window it came from.

From the anchor the trailing 90 days are cut into three 30-day periods, each with its own order count and spend:

| period     | column suffix | what it is                                                     |
| ---------- | ------------- | -------------------------------------------------------------- |
| 0-30 days  | `_last_30d`   | the current period; recency is simply whether this is non-zero |
| 31-60 days | `_31_60d`     | the 60-90 day day period before the anchor date                |
| 61-90 days | `_61_90d`     | the oldest 30-90 day period before the anchor date             |

`orders_last90d` and `spend_last90d` are the sums of the three. Anything older than 90 days is still kept (`total_orders`, `total_revenue`, `first_order_date`, `tenure_days`) and is useful for history, but it never drives the segment: what a customer did two years ago says little about whether they are leaving now.

Recency is deliberately a boolean rather than a score, and that was measured rather than assumed. Against a point-in-time holdout -- assign the segment as of anchor minus 90 days, then check who actually ordered in the following 90 days -- plain recency beat every cadence ratio tried.

### Segment rules

The rules are evaluated **top to bottom and the first match wins**, so every rule should be read as "...and none of the rules above it matched." Dormant and New are tested first on purpose: a customer with no 90-day history has no engagement to measure, and a customer who is 30 days old has not had time to build one.

| #   | segment     | rule                                                                                                | customers | 90d spend | returns |
| --- | ----------- | --------------------------------------------------------------------------------------------------- | --------- | --------- | ------- |
| 1   | Dormant     | `orders_last90d = 0`                                                                                | 16,376    | $0        | 4.8%    |
| 2   | New         | `tenure_days <= 30` and `orders_last_30d >= 1`                                                      | 455       | $8,938    | 31.8%   |
| 3   | Re-engaging | `orders_last_30d >= 1`, nothing in either older period, and `dormancy_days >= 180`                  | 206       | $3,271    | 40.1%   |
| 4   | VIP         | `orders_last_30d >= 2`, at least one order in an older period, and `spend_last90d >= vip_spend_cut` | 401       | $53,247   | 86.7%   |
| 5   | Loyal       | `orders_last_30d >= 1` and `orders_last90d >= 4`                                                    | 193       | $12,126   | 79.1%   |
| 6   | Promising   | `orders_last_30d >= 1`                                                                              | 628       | $16,397   | 51.5%   |
| 7   | Slipping    | `orders_last90d >= 3`                                                                               | 240       | $13,195   | 57.5%   |
| 8   | Lapsing     | everything else                                                                                     | 1,675     | $29,904   | 25.8%   |

Counts are taken at the 2024-02-20 anchor and sum to 20,174 customers, every one of them landing in exactly one segment. **Returns** is the measured holdout rate -- the share of customers who held that segment as of anchor minus 90 days and then placed an order in the next 90 days. It is the evidence that the rules carry real signal: they span 4.8% to 86.7%, an 82-point spread, and they rank in a sensible order.

What each segment means in practice:

- **VIP** -- ordering repeatedly right now, present in an earlier period too, and in the top fifth of 90-day spend. 2.0% of customers producing 38.8% of trailing-90-day revenue, at $132.78 each. This is the group to protect.
- **Loyal** -- frequent and current, but under the spend cut; the natural VIP feeder at $62.83 each. Worth watching: their spend declines across all three periods ($4,873 -> $4,118 -> $3,135 oldest to newest) while VIP spend grows ($15,890 -> $15,740 -> $21,617).
- **Promising** -- current but infrequent, and with an average tenure of 422 days these are established occasional buyers, not newcomers. The name describes the opportunity, not their age.
- **Re-engaging** -- won back after at least 180 days away, 367 on average. Fragile at 40.1%, and the second order is what decides whether the win-back actually held.
- **New** -- first order inside the last 30 days (average tenure 15 days), no prior history to judge them on. They return only 31.8% of the time unaided, which is the whole argument for onboarding and habit-forming follow-up.
- **Slipping** -- had a rhythm of three or more orders in the window and then went quiet inside the last 30 days. At 57.5% they still return more often than not, and at $54.98 each they are worth roughly three Lapsing customers, which makes this the highest-leverage group in the table.
- **Lapsing** -- never built a rhythm, and now quiet. The largest at-risk pool both by headcount and by total money ($29,904) but the smallest per head ($17.85), so the honest play is cheap and broad rather than individual attention.
- **Dormant** -- no orders at all in the window. 81.2% of all customers, and 4.8% of them come back without being asked.

Two thresholds are calibrated from the data rather than hardcoded:

- **`vip_spend_cut`** is the 80th percentile of `spend_last90d` among **active** customers only, recomputed on every run ($47.93 at this anchor). 90-day spend is badly skewed -- median $20, mean $36.09, maximum $887.43 -- so a mean would drift with outliers and still qualify 27% of customers. It is computed over `Window.partitionBy("is_active")` because ranking across everyone would drag the cut toward zero, 81% of customers having no orders in the window at all. The value is written out as a column so a dashboard can display the threshold it used and a reviewer can audit it instead of taking it on trust.
- **`orders_last90d >= 4`** for Loyal is the top fifth of the 90-day order count, whose 80th percentile among active customers is 3.

### Segment groups

`segment_group` rolls the eight segments into five buckets, so a dashboard filter or a headline number can work at a level a reader can actually act on.

| segment_group | segments               | customers | 90d spend | the question it answers                     |
| ------------- | ---------------------- | --------- | --------- | ------------------------------------------- |
| Core          | VIP, Loyal             | 594       | $65,373   | who is carrying revenue right now           |
| Developing    | Promising, Re-engaging | 834       | $19,668   | who is engaged but has not become habitual  |
| New Growth    | New                    | 455       | $8,938    | who just arrived and still needs onboarding |
| Churn Risk    | Slipping, Lapsing      | 1,915     | $43,099   | who we are losing, and how much is at stake |
| Dormant       | Dormant                | 16,376    | $0        | the reactivation pool                       |

Churn Risk is kept as a parent over two segments rather than flattened into one label because the two groups leave for different reasons and need different treatment. **Slipping** customers had an established rhythm and broke it, so something changed recently and a reminder or a service recovery is the appropriate response. **Lapsing** customers never had a rhythm to break -- they were occasional from the start -- so broad, low-cost reactivation fits them and individual attention does not. Collapsing the two would bury 240 recoverable, high-value customers inside 1,675 low-value ones and invite the wrong spend.
