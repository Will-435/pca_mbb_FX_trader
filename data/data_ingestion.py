"""
data_ingestion.py pulls the raw daily series and stores them for the rest of
the pipeline. It is the only file that touches the network.

The file does two things. It downloads one raw CSV per series (FRED for
everything except gold, the LBMA fix for gold), then aligns and resamples all
of them into one weekly level frame.

Outputs:

    Data/Input/<series_name>_raw.csv, one immutable file per series, exactly
    as given by the source. Existing files are never overwritten.

    Data/Input/weekly_levels.parquet, the aligned weekly levels for every
    series. This is the file pca.py and any other consumer reads, none of
    them download or resample raw data themselves.

    Database/raw_series.sql and Database/weekly_levels.sql, MySQL Workbench
    imports of the same two layers.

Dependencies: pandas, requests, pyarrow.
Run from the repository root: python3 Data/Input/data_ingestion.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

import os
from dotenv import load_dotenv

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
RAW_FILE_SUFFIX = "_raw.csv"
WEEKLY_LEVELS_PATH = RAW_DATA_DIRECTORY / "weekly_levels.parquet"

DATABASE_DIRECTORY = Path("Database")
DATE_COLUMN_NAME = "week_date"

# Weeks are stamped on Fridays. Daily gaps from market holidays are forward
# filled, but only for a few days so a dead series cannot fake a level.
WEEKLY_RESAMPLE_RULE = "W-FRI"
FORWARD_FILL_LIMIT_DAYS = 5

REQUEST_TIMEOUT_SECONDS = 60
HTTP_OK_STATUS = 200

INSERT_BATCH_ROW_COUNT = 500


def build_raw_series_path(series_name = None):
    """
    Builds the path of the raw CSV for one series inside Data/Input.

    INPUTS:
        * series_name, the internal name such as usd_krw

    OUTPUTS:
        * the Path of the raw CSV for that series
    """
    return RAW_DATA_DIRECTORY / (series_name + RAW_FILE_SUFFIX)


def fetch_fred_series(series_identifier = None, output_path = None):
    """
    Downloads one daily series from the FRED API and saves it as a raw CSV.
    The raw layer keeps the data exactly as given, one file per series.

    INPUTS:
        * series_identifier, the FRED series code such as DEXKOUS
        * output_path, where the raw CSV is written

    OUTPUTS:
        * a CSV at output_path with columns observation_date and value
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
    if response.status_code != HTTP_OK_STATUS:
        error_detail = response_payload.get("error_message", "no detail returned")
        raise RuntimeError(
            "FRED request for " + series_identifier + " failed: " + error_detail
            + " Check the FRED_API value at the top of data_ingestion.py."
        )

    observation_dates = []
    observation_values = []
    for observation in response_payload["observations"]:
        # FRED marks missing days with a full stop instead of a number.
        if observation["value"] == ".":
            continue
        observation_dates.append(observation["date"])
        observation_values.append(float(observation["value"]))

    raw_frame = pd.DataFrame({
        "observation_date": observation_dates,
        "value": observation_values,
    })
    raw_frame.to_csv(output_path, index = False)


def fetch_gold_series(output_path = None):
    """
    Downloads the daily LBMA PM gold price in dollars and saves it as a raw CSV
    in the same shape as the FRED files, so the rest of the pipeline does not
    care where a series came from.

    INPUTS:
        * output_path, where the raw CSV is written

    OUTPUTS:
        * a CSV at output_path with columns observation_date and value
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

    raw_frame = pd.DataFrame({
        "observation_date": observation_dates,
        "value": observation_values,
    })
    raw_frame.to_csv(output_path, index = False)


def download_all_raw_series():
    """
    Downloads every raw series that is not already on disk. Existing files are
    never overwritten, which keeps the raw layer immutable and the history
    point in time. Delete a file in Data/Input to force a refresh.

    INPUTS:
        * none, the series list comes from the constants above

    OUTPUTS:
        * one raw CSV per series in Data/Input
    """
    for series_name, series_identifier in FRED_SERIES_IDENTIFIERS.items():
        output_path = build_raw_series_path(series_name = series_name)
        if output_path.exists():
            print("Keeping existing raw file " + str(output_path))
            continue
        print("Downloading " + series_identifier + " from FRED")
        fetch_fred_series(series_identifier = series_identifier, output_path = output_path)

    gold_output_path = build_raw_series_path(series_name = GOLD_SERIES_NAME)
    if gold_output_path.exists():
        print("Keeping existing raw file " + str(gold_output_path))
    else:
        print("Downloading gold from LBMA")
        fetch_gold_series(output_path = gold_output_path)


def load_raw_series(series_name = None):
    """
    Loads one raw CSV from Data/Input as a dated series.

    INPUTS:
        * series_name, the internal name such as usd_krw

    OUTPUTS:
        * a pandas Series of daily values indexed by date
    """
    raw_frame = pd.read_csv(
        build_raw_series_path(series_name = series_name),
        parse_dates = ["observation_date"],
        index_col = "observation_date",
    )
    return raw_frame["value"].rename(series_name)


def build_weekly_level_frame():
    """
    Aligns all raw daily series on one calendar and resamples them to weekly
    levels. Different markets have different holidays, so short daily gaps are
    forward filled before the Friday stamp is taken.

    INPUTS:
        * none, the series list comes from the constants above

    OUTPUTS:
        * a dataframe of weekly levels, one column per series
    """
    series_collection = {}
    for series_name in FRED_SERIES_IDENTIFIERS:
        series_collection[series_name] = load_raw_series(series_name = series_name)
    series_collection[GOLD_SERIES_NAME] = load_raw_series(series_name = GOLD_SERIES_NAME)

    daily_level_frame = pd.concat(series_collection, axis = 1).sort_index()
    daily_level_frame = daily_level_frame.ffill(limit = FORWARD_FILL_LIMIT_DAYS)
    weekly_level_frame = daily_level_frame.resample(WEEKLY_RESAMPLE_RULE).last()
    weekly_level_frame.index.name = DATE_COLUMN_NAME
    return weekly_level_frame


def build_raw_series_export_frame():
    """
    Stacks every raw series into one long table for the SQL export, so the
    untouched inputs can be inspected in MySQL Workbench.

    INPUTS:
        * none, the series list comes from the constants above

    OUTPUTS:
        * a dataframe with columns series_name, observation_date and value
    """
    long_frames = []
    all_series_names = list(FRED_SERIES_IDENTIFIERS) + [GOLD_SERIES_NAME]
    for series_name in all_series_names:
        raw_frame = pd.read_csv(
            build_raw_series_path(series_name = series_name),
            parse_dates = ["observation_date"],
        )
        raw_frame.insert(0, "series_name", series_name)
        long_frames.append(raw_frame)
    return pd.concat(long_frames, ignore_index = True)


def format_value_for_sql(single_value = None):
    """
    Formats one Python value as a MySQL literal.

    INPUTS:
        * single_value, any cell value from a dataframe

    OUTPUTS:
        * the value as a string ready to sit inside an INSERT statement
    """
    if pd.isna(single_value):
        return "NULL"
    if isinstance(single_value, pd.Timestamp):
        return "'" + single_value.strftime("%Y-%m-%d") + "'"
    if isinstance(single_value, (int, float, np.integer, np.floating)):
        return str(single_value)
    return "'" + str(single_value).replace("'", "''") + "'"


def write_dataframe_to_sql(dataframe = None, table_name = None, output_path = None):
    """
    Writes a dataframe as a MySQL script with one CREATE TABLE and batched
    INSERT statements, so every pipeline artefact can be imported into MySQL
    Workbench and inspected at any point.

    INPUTS:
        * dataframe, the table content
        * table_name, the table name used in the script
        * output_path, where the .sql file is written

    OUTPUTS:
        * a .sql file at output_path
    """
    column_definitions = []
    for column_name in dataframe.columns:
        if pd.api.types.is_datetime64_any_dtype(dataframe[column_name]):
            column_type = "DATE"
        elif pd.api.types.is_numeric_dtype(dataframe[column_name]):
            column_type = "DOUBLE"
        else:
            column_type = "VARCHAR(255)"
        column_definitions.append("    " + column_name + " " + column_type)

    script_lines = []
    script_lines.append("DROP TABLE IF EXISTS " + table_name + ";")
    script_lines.append("CREATE TABLE " + table_name + " (")
    script_lines.append(",\n".join(column_definitions))
    script_lines.append(");")

    formatted_rows = []
    for row_values in dataframe.itertuples(index = False):
        formatted_cells = []
        for single_value in row_values:
            formatted_cells.append(format_value_for_sql(single_value = single_value))
        formatted_rows.append("(" + ", ".join(formatted_cells) + ")")

    # Rows are inserted in batches, one giant statement would be unreadable and
    # one statement per row would import slowly.
    for batch_start in range(0, len(formatted_rows), INSERT_BATCH_ROW_COUNT):
        batch_rows = formatted_rows[batch_start:batch_start + INSERT_BATCH_ROW_COUNT]
        script_lines.append("INSERT INTO " + table_name + " VALUES")
        script_lines.append(",\n".join(batch_rows) + ";")

    output_path.write_text("\n".join(script_lines) + "\n")
    print("Written " + str(output_path))


def main():
    """
    Runs the ingestion stage: download every raw series, build the weekly
    level frame, and export both layers for MySQL Workbench.

    INPUTS:
        * none, configuration comes from the constants above

    OUTPUTS:
        * Data/Input raw CSVs and weekly_levels.parquet
        * Database/raw_series.sql and Database/weekly_levels.sql
    """
    if not RAW_DATA_DIRECTORY.exists():
        raise RuntimeError("Data/Input not found. Run this file from the repository root.")

    download_all_raw_series()

    weekly_level_frame = build_weekly_level_frame()
    weekly_level_frame.reset_index().to_parquet(WEEKLY_LEVELS_PATH, index = False)
    print("Written " + str(WEEKLY_LEVELS_PATH))

    write_dataframe_to_sql(
        dataframe = build_raw_series_export_frame(),
        table_name = "raw_series",
        output_path = DATABASE_DIRECTORY / "raw_series.sql",
    )
    write_dataframe_to_sql(
        dataframe = weekly_level_frame.reset_index(),
        table_name = "weekly_levels",
        output_path = DATABASE_DIRECTORY / "weekly_levels.sql",
    )


if __name__ == "__main__":
    main()