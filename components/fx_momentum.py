"""
This file does the data handling for momentum.cpp

NOTE - I have manually made all FRED spots to have be USD denominated. Some
are priced in USD/FX. Ensuring USD is the denominator ensures consistency.

I have taken logs to make the momentum calculation a linear calculation, 
log(A) - log(B). This simplifies momentum.cpp.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.sql_conversion import write_parquet_to_sql


FX_WEEKLY_LEVELS_PATH = Path("Data/Input/fx_weekly_levels.parquet")

OUTPUT_DIRECTORY = Path("components/momentum_output")
MOMENTUM_INPUT_PATH = OUTPUT_DIRECTORY / "momentum_input.parquet"
MOMENTUM_TIMESERIES_PATH = OUTPUT_DIRECTORY / "momentum_2w.parquet"
MOMENTUM_CROSS_PATH = OUTPUT_DIRECTORY / "momentum_cross_latest.parquet"

DATE_COLUMN_NAME = "week_date"

# Pairs FRED already quotes as dollars per unit of currency. Every other
# column in fx_weekly_levels.parquet is currency per dollar and is inverted.
USD_PER_CURRENCY_PAIRS = ["gbp_usd", "eur_usd", "aud_usd"]

MOMENTUM_PRINT_FORMAT = "+.4f"


def build_momentum_input_frame():
    """
    Makes every pair to dollars per unit of currency, and takes logs. 
    The result is the input matrix momentum.cpp.

    INPUTS:
        * none

    OUTPUTS:
        * a df of log dollar prices indexed by week
    """

    # Error cach
    if not FX_WEEKLY_LEVELS_PATH.exists():
        raise RuntimeError(
            "Missing " + str(FX_WEEKLY_LEVELS_PATH)
            + ". Run data/data_ingestion.py first, and run this file from the"
            + " repository root."
        )
    
    fx_level_frame = pd.read_parquet(FX_WEEKLY_LEVELS_PATH)
    fx_level_frame = fx_level_frame.set_index(DATE_COLUMN_NAME)

    for pair_name in fx_level_frame.columns:
        if pair_name not in USD_PER_CURRENCY_PAIRS:
            fx_level_frame[pair_name] = 1.0 / fx_level_frame[pair_name]

    log_level_frame = np.log(fx_level_frame)
    log_level_frame = log_level_frame.dropna()

    return log_level_frame


def export_momentum_outputs():
    """
    Exports the momentum.cpp files to Database so they can be inspected with sql.

    INPUTS:
        * none

    OUTPUTS:
        * Database/momentum_2w.sql and Database/momentum_cross_latest.sql
    """
    write_parquet_to_sql(
        parquet_path = MOMENTUM_TIMESERIES_PATH,
        table_name = "momentum_2w",
    )
    write_parquet_to_sql(
        parquet_path = MOMENTUM_CROSS_PATH,
        table_name = "momentum_cross_latest",
    )


def print_latest_momentum_ranking():
    """
    Prints the current two week momentum of every currency

    INPUTS:
        * none

    OUTPUTS:
        * a printed ranking, nothing written to disk
    """
    momentum_frame = pd.read_parquet(MOMENTUM_TIMESERIES_PATH)
    latest_row = momentum_frame.iloc[-1]
    latest_week = latest_row[DATE_COLUMN_NAME]
    latest_momentum = latest_row.drop(DATE_COLUMN_NAME).astype(float)

    print("Two week momentum against the dollar, week ending " + str(latest_week) + ":")
    for pair_name, momentum_value in latest_momentum.sort_values(ascending = False).items():
        print("    " + pair_name + "  " + format(momentum_value, MOMENTUM_PRINT_FORMAT))


def main():
    """
    The main file that runs all the above in order

    INPUTS:
        * none

    OUTPUTS:
        * momentum_output/momentum_input.parquet
        * Database exports and a printed ranking, once ./momentum has run
    """
    # The output directory is git ignored, so it may be absent on a fresh clone.
    OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)

    log_level_frame = build_momentum_input_frame()
    log_level_frame.reset_index().to_parquet(MOMENTUM_INPUT_PATH, index = False)
    print(
        "Written " + str(MOMENTUM_INPUT_PATH) + " with "
        + str(len(log_level_frame)) + " weeks of "
        + str(len(log_level_frame.columns)) + " currencies."
    )

    if MOMENTUM_TIMESERIES_PATH.exists() and MOMENTUM_CROSS_PATH.exists():
        export_momentum_outputs()
        print_latest_momentum_ranking()
    else:
        print("No momentum results yet. Run ./momentum next, then rerun this file.")


if __name__ == "__main__":
    main()