"""The session configuration the jobs depend on being identical everywhere.

The `spark` fixture builds the real `create_spark_session()`, so these test the
shipped configuration rather than a hand-made session.
"""

import pytest
from pyspark.sql import functions as F

from python_scripts.spark.spark_session import SESSION_TIME_ZONE

pytestmark = pytest.mark.spark

# The actual last order in silver. Chosen because it is the row that exposed
# the bug: 02:52 UTC on the 21st is 21:52 on the 20th in America/New_York.
BOUNDARY_UTC = "2024-02-21T02:52:35.366+00:00"


def test_session_timezone_is_pinned(spark):
    """Unset, this inherits the JVM default and differs between machines."""
    assert spark.conf.get("spark.sql.session.timeZone") == SESSION_TIME_ZONE
    assert SESSION_TIME_ZONE == "UTC"


def test_to_date_resolves_a_late_utc_timestamp_to_its_utc_day(spark):
    """The regression. Six gold tables derive their date grain this way.

    Under America/New_York this returns 2024-02-20, which made the
    since-removed loyalty_era_profile compute a 331-day era locally and 332 on
    Glue from identical silver.
    """
    df = spark.sql(f"SELECT CAST('{BOUNDARY_UTC}' AS timestamp) AS ts")
    got = df.select(F.to_date("ts").alias("d")).collect()[0]["d"]

    assert str(got) == "2024-02-21"


def test_the_pin_is_load_bearing(spark):
    """Shows the assertion above is not vacuous.

    If this test ever fails, `to_date` has stopped depending on the session
    zone and the pin could be dropped -- which is worth knowing. Restores the
    conf in `finally` because the fixture is session-scoped and every later
    test would otherwise inherit the wrong zone.
    """
    original = spark.conf.get("spark.sql.session.timeZone")
    try:
        spark.conf.set("spark.sql.session.timeZone", "America/New_York")
        df = spark.sql(f"SELECT CAST('{BOUNDARY_UTC}' AS timestamp) AS ts")
        shifted = df.select(F.to_date("ts").alias("d")).collect()[0]["d"]
    finally:
        spark.conf.set("spark.sql.session.timeZone", original)

    assert str(shifted) == "2024-02-20", (
        "to_date no longer shifts with the session zone; the pin may be "
        "redundant, but check before removing it"
    )
    assert spark.conf.get("spark.sql.session.timeZone") == original
