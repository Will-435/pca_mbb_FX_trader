"""
mi_spearman.py is a candidate file. It measures the dependence between the dollar
and each other asset with a signed mutual information score, rolling through time.

The magnitude is the mutual information, which unlike a correlation picks up
nonlinear dependence as well as linear. Mutual information has no sign, so the
sign is taken from the Spearman rank correlation. The result is a value between
-1 and 1 that can be compared directly against the rolling Pearson and Spearman
outputs.

Nothing in the pipeline reads this yet. It sits beside pearson_correlation.cpp
and spearman_correlation.cpp so the three can be validated against each other
before any of them is wired into the regime step.

Dependency: mutual information uses scikit-learn, which also pulls in scipy. Both
are needed on top of the existing requirements.

Run from the repository root: python3 candidate/secondMoment/mi_spearman.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# Running this file directly only puts candidate/secondMoment/ on the import
# path, not the repository root. Three parents up is the root, so utils/ can be
# imported for the shared SQL export helper.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.sql_conversion import write_dataframe_to_sql


WEEKLY_LEVELS_PATH = Path("Data/Input/weekly_levels.parquet")

OUTPUT_DIRECTORY = Path("candidate/secondMoment/second_moment_output")
SIGNED_MI_OUTPUT_PATH = OUTPUT_DIRECTORY / "mi_spearman.parquet"

DATE_COLUMN_NAME = "week_date"
DOLLAR_COLUMN_NAME = "dollar_index"

# Assets measured against the dollar. Key is the column name in the parquet,
# value is the label used in the output column.
ASSET_COLUMN_LABELS = {
    "gold": "gold",
    "oil_wti": "oil",
}

# Weeks of returns in one rolling window. Matches the window used by the two
# correlation candidates so all three stay comparable.
ROLLING_WINDOW_WEEKS = 26
SINGLE_WEEK_LAG = 1

# Neighbour count for the mutual information estimator. It compares each point to
# its nearest neighbours, so a short window needs a small count.
MUTUAL_INFORMATION_NEIGHBOURS = 3

RANDOM_SEED = 42

# Mutual information (in nats) is rescaled onto a correlation scale with
# sqrt(1 - exp(-2 * mutual_information)). For a Gaussian pair this returns the
# absolute correlation, so the magnitude lines up with Pearson and Spearman.
INFORMATION_TO_CORRELATION_SCALE = 2.0

# Below this Spearman size the direction is treated as unknown and the score is
# set to zero, so the sign does not flip on noise near zero.
SIGN_DEADBAND = 0.05

SCORE_PRINT_FORMAT = "+.4f"


def load_weekly_return_frame():
    """
    Loads the weekly levels and turns them into one week log returns for the
    dollar and every asset.

    INPUTS:
        * none

    OUTPUTS:
        * a dataframe of one week log returns, one column per series
    """
    if not WEEKLY_LEVELS_PATH.exists():
        raise RuntimeError(
            "Missing " + str(WEEKLY_LEVELS_PATH)
            + ". Run data/data_ingestion.py first, and run this file from the"
            + " repository root."
        )

    weekly_level_frame = pd.read_parquet(WEEKLY_LEVELS_PATH).set_index(DATE_COLUMN_NAME)

    needed_columns = [DOLLAR_COLUMN_NAME] + list(ASSET_COLUMN_LABELS.keys())
    return_frame = pd.DataFrame(index = weekly_level_frame.index)
    for column_name in needed_columns:
        return_frame[column_name] = np.log(
            weekly_level_frame[column_name]
            / weekly_level_frame[column_name].shift(SINGLE_WEEK_LAG)
        )

    return return_frame.dropna()


def compute_signed_mutual_information(asset_window = None, dollar_window = None):
    """
    Turns one window of two return series into a signed dependence score between
    -1 and 1. The magnitude is the rescaled mutual information, the sign is the
    direction of the Spearman relationship.

    INPUTS:
        * asset_window - one window of asset returns
        * dollar_window - the matching window of dollar returns

    OUTPUTS:
        * a signed dependence score between -1 and 1
    """
    mutual_information = mutual_info_regression(
        asset_window.reshape(-1, 1),
        dollar_window,
        n_neighbors = MUTUAL_INFORMATION_NEIGHBOURS,
        random_state = RANDOM_SEED,
    )[0]

    information_correlation = math.sqrt(
        1.0 - math.exp(-INFORMATION_TO_CORRELATION_SCALE * mutual_information)
    )

    spearman_correlation = pd.Series(asset_window).corr(
        pd.Series(dollar_window), method = "spearman"
    )

    # A near zero or undefined Spearman means the direction is unreliable, so the
    # score is dropped to zero rather than allowed to flip on noise.
    if math.isnan(spearman_correlation) or abs(spearman_correlation) < SIGN_DEADBAND:
        return 0.0

    direction = 1.0 if spearman_correlation > 0.0 else -1.0
    return direction * information_correlation


def build_rolling_signed_mi(return_frame = None):
    """
    Rolls a fixed window across the return history and scores every asset against
    the dollar at each window end. The score is stamped on the last week of the
    window, so the leading weeks with no full window are dropped.

    INPUTS:
        * return_frame - weekly returns for the dollar and every asset

    OUTPUTS:
        * a dataframe of signed dependence series, one column per asset
    """
    dollar_returns = return_frame[DOLLAR_COLUMN_NAME].to_numpy()
    week_count = len(return_frame)

    signed_mi_frame = pd.DataFrame(index = return_frame.index)
    for asset_column_name, asset_label in ASSET_COLUMN_LABELS.items():
        asset_returns = return_frame[asset_column_name].to_numpy()

        signed_scores = []
        for window_end in range(week_count):
            if window_end + 1 < ROLLING_WINDOW_WEEKS:
                signed_scores.append(float("nan"))
                continue

            window_start = window_end + 1 - ROLLING_WINDOW_WEEKS
            asset_window = asset_returns[window_start:window_end + 1]
            dollar_window = dollar_returns[window_start:window_end + 1]
            signed_scores.append(
                compute_signed_mutual_information(
                    asset_window = asset_window, dollar_window = dollar_window
                )
            )

        signed_mi_frame[asset_label + "_usd_signed_mi"] = signed_scores

    signed_mi_frame = signed_mi_frame.dropna()
    signed_mi_frame.index.name = DATE_COLUMN_NAME
    return signed_mi_frame


def main():
    """
    Loads the returns, builds the rolling signed mutual information and writes it.

    INPUTS:
        * none

    OUTPUTS:
        * second_moment_output/mi_spearman.parquet
        * Database/mi_spearman.sql
    """
    OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)

    return_frame = load_weekly_return_frame()
    print("Loaded " + str(len(return_frame)) + " weeks of returns.")

    signed_mi_frame = build_rolling_signed_mi(return_frame = return_frame)
    signed_mi_frame.reset_index().to_parquet(SIGNED_MI_OUTPUT_PATH, index = False)
    print("Written " + str(SIGNED_MI_OUTPUT_PATH))

    write_dataframe_to_sql(
        dataframe = signed_mi_frame.reset_index(),
        table_name = "mi_spearman",
    )

    latest_row = signed_mi_frame.iloc[-1]
    print(
        "Latest signed mutual information, week ending "
        + str(signed_mi_frame.index[-1].date()) + ":"
    )
    for column_name, score_value in latest_row.items():
        print("    " + column_name + "  " + format(score_value, SCORE_PRINT_FORMAT))


if __name__ == "__main__":
    main()
