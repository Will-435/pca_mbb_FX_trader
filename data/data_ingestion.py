"""
data_ingestion.py pulls the raw daily series and stores them for the rest of
the pipeline. It is the only file that touches the network.

It downloads every series (LBMA fix for gold, FRED for everything else), stacks
them into 1 table, and resamples them into weekly data. Both are saved as parquet.
The parquet is used to build the SQL export.

Outputs:

    Data/Input/raw_series.parquet - every series stacked. Delete for each new run

    Data/Input/weekly_levels.parquet, This is the file pca.py reads.

    Database/raw_series.sql and Database/weekly_levels.sql, MySQL Workbench imports.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Running "python3 data/data_ingestion.py" only puts data/ on the import path,
# not the repository root, so utils/ is added here before its helper imports.
# Although this is more complicated, there is no way around it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.sql_conversion import write_parquet_to_sql

load_dotenv()

FRED_API = os.getenv("FRED_API_KEY")

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

# Official LBMA endpoint for the daily PM gold price in dollars. FRED removed
# its LBMA gold series, so gold is the one series not sourced from FRED.
GOLD_PRICE_URL = "https://prices.lbma.org.uk/json/gold_pm.json"
GOLD_SERIES_NAME = "gold"
USD_PRICE_POSITION = 0

FRED_SERIES_IDENTIFIERS = {
    "usd_krw": "DEXKOUS",
    "dollar_index": "DTWEXBGS",
    "us_2y_yield": "DGS2",
    "us_10y_yield": "DGS10",
    "oil_wti": "DCOILWTICO",
    "vix": "VIXCLS",
}

RAW_DATA_DIRECTORY = Path("Data/Input")
RAW_SERIES_PATH = RAW_DATA_DIRECTORY / "raw_series.parquet"
WEEKLY_LEVELS_PATH = RAW_DATA_DIRECTORY / "weekly_levels.parquet"

DATE_COLUMN_NAME = "week_date"

# Column names of the long raw table.
SERIES_NAME_COLUMN = "series_name"
OBSERVATION_DATE_COLUMN = "observation_date"
VALUE_COLUMN = "value"

# Weeks are stamped on Fridays. Daily gaps from market holidays are forward
# filled, but only for a few days so a dead series cannot fake a level.
WEEKLY_RESAMPLE_RULE = "W-FRI"
FORWARD_FILL_LIMIT_DAYS = 5

REQUEST_TIMEOUT_SECONDS = 60
HTTP_OK_STATUS = 200


def fetch_fred_series(series_identifier = None):
    """
    Downloads one daily series from the FRED API and returns it.

    INPUTS:
        * series_identifier, the FRED series code such as DEXKOUS

    OUTPUTS:
        * a dataframe with columns observation_date and value
    """
    request_parameters = {
        "series_id": series_identifier,
        "api_key": FRED_API,
        "file_type": "json",
    }
    response = requests.get(
        FRED_OBSERVATIONS_URL,
        params = request_parameters,
        timeout = REQUEST_TIMEOUT_SECONDS,
    )
    response_payload = response.json()

    # An error catch
    if response.status_code != HTTP_OK_STATUS:
        error_detail = response_payload.get("error_message", "no detail returned")
        raise RuntimeError(
            "FRED request for " + series_identifier + " failed: " + error_detail
            + " Check the FRED_API_KEY value in the .env file."
        )

    observation_dates = []
    observation_values = []
    for observation in response_payload["observations"]:
        # FRED marks missing days with a full stop instead of a number.
        if observation["value"] == ".":
            continue
        observation_dates.append(observation["date"])
        observation_values.append(float(observation["value"]))

    return pd.DataFrame({
        OBSERVATION_DATE_COLUMN: pd.to_datetime(observation_dates),
        VALUE_COLUMN: observation_values,
    })


def fetch_gold_series():
    """
    Downloads + engineers daily LBMA PM gold price into FRED series shape.

    INPUTS:
        * none

    OUTPUTS:
        * a dataframe with columns observation_date and value
    """
    response = requests.get(GOLD_PRICE_URL, timeout = REQUEST_TIMEOUT_SECONDS)
    if response.status_code != HTTP_OK_STATUS:
        raise RuntimeError("LBMA gold request failed with status " + str(response.status_code))

    observation_dates = []
    observation_values = []
    for daily_entry in response.json():
        usd_price = daily_entry["v"][USD_PRICE_POSITION]
        # Some early entries carry no dollar price, they are skipped.
        if not usd_price:
            continue
        observation_dates.append(daily_entry["d"])
        observation_values.append(float(usd_price))

    return pd.DataFrame({
        OBSERVATION_DATE_COLUMN: pd.to_datetime(observation_dates),
        VALUE_COLUMN: observation_values,
    })


def download_raw_series_frame():
    """
    Downloads every series and stacks them into one table.

    INPUTS:
        * none

    OUTPUTS:
        * a df with columns series_name, observation_date, value
    """
    long_frames = []
    for series_name, series_identifier in FRED_SERIES_IDENTIFIERS.items():
        print("Downloading " + series_identifier + " from FRED")
        series_frame = fetch_fred_series(series_identifier = series_identifier)
        series_frame.insert(0, SERIES_NAME_COLUMN, series_name)
        long_frames.append(series_frame)

    print("Downloading gold from LBMA")
    gold_frame = fetch_gold_series()
    gold_frame.insert(0, SERIES_NAME_COLUMN, GOLD_SERIES_NAME)
    long_frames.append(gold_frame)

    return pd.concat(long_frames, ignore_index = True)


def build_weekly_level_frame(raw_series_frame = None):
    """
    Resamplse the full df to weekly. Different markets have different holidays, so
    short daily gaps are forward filled before the Friday.

    INPUTS:
        * raw_series_frame, the long raw table from download_raw_series_frame

    OUTPUTS:
        * a dataframe of weekly levels, one column per series, indexed by date
    """
    daily_level_frame = raw_series_frame.pivot(
        index = OBSERVATION_DATE_COLUMN,
        columns = SERIES_NAME_COLUMN,
        values = VALUE_COLUMN,
    ).sort_index()

    # pivot orders the columns alphabetically, put them back in the series order
    # the rest of the pipeline expects.
    column_order = list(FRED_SERIES_IDENTIFIERS) + [GOLD_SERIES_NAME]
    daily_level_frame = daily_level_frame[column_order]

    daily_level_frame = daily_level_frame.ffill(limit = FORWARD_FILL_LIMIT_DAYS)
    weekly_level_frame = daily_level_frame.resample(WEEKLY_RESAMPLE_RULE).last()
    weekly_level_frame.index.name = DATE_COLUMN_NAME
    return weekly_level_frame


def main():
    """
    The main file. Get raw data, build weekly df, save as parquet, export as .sql.

    INPUTS:
        * none

    OUTPUTS:
        * Data/Input/raw_series.parquet and weekly_levels.parquet
        * Database/raw_series.sql and Database/weekly_levels.sql
    """
    # This directory is git ignored, so it may be absent on a fresh clone. The
    # Database directory is created by sql_conversion.py when it writes.
    RAW_DATA_DIRECTORY.mkdir(parents = True, exist_ok = True)

    # The raw parquet is the immutable raw layer. If it already exists it is
    # kept and reused, so a rerun does not hit the network again. Delete it to
    # force a refresh.
    if RAW_SERIES_PATH.exists():
        print("Keeping existing raw layer " + str(RAW_SERIES_PATH))
        raw_series_frame = pd.read_parquet(RAW_SERIES_PATH)
    else:
        raw_series_frame = download_raw_series_frame()
        raw_series_frame.to_parquet(RAW_SERIES_PATH, index = False)
        print("Written " + str(RAW_SERIES_PATH))

    weekly_level_frame = build_weekly_level_frame(raw_series_frame = raw_series_frame)
    weekly_level_frame.reset_index().to_parquet(WEEKLY_LEVELS_PATH, index = False)
    print("Written " + str(WEEKLY_LEVELS_PATH))

    write_parquet_to_sql(
        parquet_path = RAW_SERIES_PATH,
        table_name = "raw_series",
    )
    write_parquet_to_sql(
        parquet_path = WEEKLY_LEVELS_PATH,
        table_name = "weekly_levels",
    )


if __name__ == "__main__":
    main()