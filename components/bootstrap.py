"""
bootstrap.py builds the empirical conditional distribution of the forward four
week USD/KRW return given the current regime.

It reads the neighbourhood members written by euclidean.cpp and applies a
moving block bootstrap to their forward returns. Moving block boottstrap maintains
the autoregression in the data.

The output is the empirical distribution itself in order to protect the tails and
avoid rebuilding a gaussian.
"""

import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.sql_conversion import write_dataframe_to_sql


REGIME_MEMBERS_DATABASE_PATH = Path("components/euclidean_output/regime_members.duckdb")
CONDITIONAL_PDF_PATH = Path("components/bootstrap_output/conditional_pdf.parquet")

# The block length must be at least the four week return horizon, otherwise
# the overlap autocorrelation is broken up and the resampled spread is wrong.
# TODO Decide on a way to choose teh block length
RETURN_HORIZON_WEEKS = 4
BLOCK_LENGTH_WEEKS = 4

NUMBER_OF_REPLICATIONS = 2000
RANDOM_SEED = 42

LOWEST_PERCENTILE = 1
HIGHEST_PERCENTILE = 99
PERCENT_DIVISOR = 100

MOMENT_PRINT_FORMAT = ".6f"

def load_member_returns():
    """
    Loads the neighbourhood from the DuckDB handoff file.

    INPUTS:
        * none
    OUTPUTS:
        * a numpy array of forward four week returns in order
    """
    if not REGIME_MEMBERS_DATABASE_PATH.exists():
        raise RuntimeError(
            "Missing " + str(REGIME_MEMBERS_DATABASE_PATH)
            + ". Run euclidean first, and run this file from the repository root."
        )
    database_connection = duckdb.connect(str(REGIME_MEMBERS_DATABASE_PATH), read_only = True)
    member_frame = database_connection.execute(
        "SELECT forward_return_4w FROM regime_members ORDER BY week_date"
    ).fetchdf()
    database_connection.close()
    return member_frame["forward_return_4w"].to_numpy()


def run_moving_block_bootstrap(member_returns = None):
    """
    Resamples the member returns with a moving block bootstrap. Each
    replication rebuilds a series of the original length by drawing blocks of
    consecutive returns with replacement, then every resampled value from
    every replication is pooled into one empirical distribution.

    INPUTS:
        * member_returns - the forward returns in week order

    OUTPUTS:
        * a numpy array of all resampled returns
    """
    member_count = len(member_returns)
    possible_block_starts = member_count - BLOCK_LENGTH_WEEKS + 1
    if possible_block_starts < 1:
        raise RuntimeError(
            "The neighbourhood holds fewer members than one block. Widen the"
            " neighbourhood in euclidean.cpp or shorten BLOCK_LENGTH_WEEKS."
        )
    blocks_per_replication = math.ceil(member_count / BLOCK_LENGTH_WEEKS)

    random_generator = np.random.default_rng(RANDOM_SEED)
    resampled_values = []
    for replication_index in range(NUMBER_OF_REPLICATIONS):
        replication_values = []
        for block_index in range(blocks_per_replication):
            block_start = random_generator.integers(possible_block_starts)
            block_values = member_returns[block_start:block_start + BLOCK_LENGTH_WEEKS]
            replication_values.extend(block_values)
        # The last block can overshoot the original length, trim it back.
        resampled_values.extend(replication_values[:member_count])

    return np.array(resampled_values)


def build_output_frame(resampled_returns = None):
    """
    Packs the empirical distribution into one flat table.

    INPUTS:
        * resampled_returns - the pooled draws from the bootstrap

    OUTPUTS:
        * a dataframe ready for conditional_pdf.parquet and the SQL export
    """
    output_records = []

    draw_index = 0
    for resampled_return in resampled_returns:
        output_records.append({
            "record_type": "resampled_return",
            "draw_index": draw_index,
            "percentile": None,
            "value": float(resampled_return),
        })
        draw_index = draw_index + 1

    for percentile_level in range(LOWEST_PERCENTILE, HIGHEST_PERCENTILE + 1):
        quantile_value = np.quantile(resampled_returns, percentile_level / PERCENT_DIVISOR)
        output_records.append({
            "record_type": "empirical_quantile",
            "draw_index": None,
            "percentile": percentile_level,
            "value": float(quantile_value),
        })

    output_columns = ["record_type", "draw_index", "percentile", "value"]
    return pd.DataFrame(output_records, columns = output_columns)


def print_moment_summary(resampled_returns = None):
    """
    Prints the moments of the resampled distribution as a diagnostic-only tool.

    INPUTS:
        * resampled_returns - the pooled draws from the bootstrap

    OUTPUTS:
        * a printed summary
    """
    mean_value = float(np.mean(resampled_returns))
    deviations = resampled_returns - mean_value
    variance_value = float(np.mean(deviations ** 2))
    standard_deviation = float(np.sqrt(variance_value))
    skew_value = float(np.mean(deviations ** 3) / standard_deviation ** 3)
    excess_kurtosis = float(np.mean(deviations ** 4) / variance_value ** 2) - 3.0

    print("Console summary only, the saved artefact is the full empirical distribution.")
    print("Mean " + format(mean_value, MOMENT_PRINT_FORMAT))
    print("Standard deviation " + format(standard_deviation, MOMENT_PRINT_FORMAT))
    print("Skew " + format(skew_value, MOMENT_PRINT_FORMAT))
    print("Excess kurtosis " + format(excess_kurtosis, MOMENT_PRINT_FORMAT))


def main():
    """
    The main function for this file.

    INPUTS:
        * none

    OUTPUTS:
        * bootstrap_output/conditional_pdf.parquet
        * Database/conditional_pdf.sql
    """
    member_returns = load_member_returns()
    print("Loaded " + str(len(member_returns)) + " neighbourhood member returns.")

    resampled_returns = run_moving_block_bootstrap(member_returns = member_returns)
    print(
        "Pooled " + str(len(resampled_returns)) + " resampled returns from "
        + str(NUMBER_OF_REPLICATIONS) + " replications."
    )

    output_frame = build_output_frame(resampled_returns = resampled_returns)

    # The output directory is git ignored, so it may be absent on a fresh clone.
    CONDITIONAL_PDF_PATH.parent.mkdir(parents = True, exist_ok = True)
    output_frame.to_parquet(CONDITIONAL_PDF_PATH, index = False)
    print("Written " + str(CONDITIONAL_PDF_PATH))

    write_dataframe_to_sql(
        dataframe = output_frame,
        table_name = "conditional_pdf",
    )

    print_moment_summary(resampled_returns = resampled_returns)


if __name__ == "__main__":
    main()