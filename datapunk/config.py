from __future__ import annotations

# NYC TLC Yellow Taxi trip data (public).
TRIP_URL_TEMPLATE = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_{year}-{month:02d}.parquet"
)
LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
LOOKUP_FILENAME = "taxi_zone_lookup.csv"

# Full benchmark window.
START = (2024, 1)
END = (2025, 12)

# --- Two-run model -------------------------------------------------------
# SMALL: a single month, NO memory cap — apples-to-apples correctness/speed.
# LARGE: the full window, WITH a physical RSS cap — surfaces out-of-core
#        behaviour and OOMs for engines that must materialise everything.
SMALL_MONTHS = 1

# Default cap for the large run, in MB. 1 GiB is a deliberate starting point:
# well above the streaming working set of these queries, but far below the
# multi-GB resident footprint pandas needs to hold full window at once — so the
# contrast is visible without starving the well-behaved engines. Run
# ``datapunk.calibrate`` on your machine to tune per suite (the global sort in
# the window suite typically needs a higher cap than the aggregations).
LARGE_CAP_MB = 1024

# Unit constants for memory accounting
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * 1024 * 1024
KIB_PER_MB = 1024

# Timing.
# SMALL run: repeated, uncapped timing for stable speed comparisons.
ITERATIONS = 5
WARMUP = 1

# LARGE run: one cold-ish capped pass. The large window is primarily a
# feasibility/stress check, so repeated full-window materializations add noise
# and memory pressure without improving the story.
LARGE_ITERATIONS = 1
LARGE_WARMUP = 0
