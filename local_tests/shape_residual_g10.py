"""
shape_residual_g10.py is a research test, kept apart from the main pipeline.
It modifies no tracked file.

It repeats shape_residual_correlation.py across the G10 pairs plus USD/KRW as a
reference, to see whether the two findings replicate cross-sectionally:
    kurtosis versus surprise magnitude (expected null),
    skew versus signed surprise (a significant contrarian signal on USD/KRW that
        needs out-of-sample support).

The macro PCA regime and neighbour selection are unchanged. Only the outcome
series (the pair's forward return, its realised forward variance, its trailing
vol and vol regime) changes per pair. Skew and kurtosis are sign-flip invariant
in the way that matters, so the FRED quote direction of each pair is irrelevant.

Run from the repository root:
    python local_tests/shape_residual_g10.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kurtosis_impact_backtest as engine
import pair_variance_comparison as pair_tools
import shape_residual_correlation as shape_tools

WEEKLY_LEVELS_PATH = Path("Data/Input/weekly_levels.parquet")
FX_WEEKLY_LEVELS_PATH = Path("Data/Input/fx_weekly_levels.parquet")
FEATURE_MATRIX_PATH = Path("Data/Input/feature_matrix.parquet")
PCA_MODEL_PATH = Path("components/pca_output/pca_model.parquet")

DATE_COLUMN_NAME = "week_date"
VIX_COLUMN_NAME = "vix"

# USD/KRW reads from the macro frame, the G10 majors from the FX basket.
PAIR_CONFIGURATIONS = [
    {"pair": "usd_krw", "source_frame": "macro"},
    {"pair": "eur_usd", "source_frame": "fx"},
    {"pair": "gbp_usd", "source_frame": "fx"},
    {"pair": "aud_usd", "source_frame": "fx"},
    {"pair": "jpy_usd", "source_frame": "fx"},
    {"pair": "chf_usd", "source_frame": "fx"},
    {"pair": "cad_usd", "source_frame": "fx"},
]

RESULTS_CSV_PATH = engine.OUTPUT_DIRECTORY / "shape_residual_g10.csv"


def realized_forward_variance_for_level(pair_level = None):
    """
    Realised variance of each forward window from the four squared weekly returns
    inside it, for one pair.

    INPUTS:
        * pair_level - the weekly level series for the pair

    OUTPUTS:
        * the realised forward variance series
    """
    weekly_return = np.log(pair_level / pair_level.shift(engine.SINGLE_WEEK_LAG))
    squared_weekly_return = weekly_return ** 2
    return squared_weekly_return.rolling(engine.HORIZON_WEEKS).sum().shift(-engine.HORIZON_WEEKS)


def pick_result(results_frame = None, regime = None, test = None, residual_option = None):
    """
    Pulls one row out of a build_results frame, or None if it is absent.

    INPUTS:
        * results_frame - the per-pair results
        * regime - the regime bucket
        * test - the test name
        * residual_option - the residual option label

    OUTPUTS:
        * the matching row as a Series, or None
    """
    matches = results_frame[
        (results_frame["regime"] == regime)
        & (results_frame["test"] == test)
        & (results_frame["residual_option"] == residual_option)
    ]
    return matches.iloc[0] if len(matches) else None


def summarise_pair(pair_name = None, results_frame = None):
    """
    Reduces one pair's results to the two headline tests, overall and in the
    high-vol regime.

    INPUTS:
        * pair_name - the pair label
        * results_frame - that pair's build_results frame

    OUTPUTS:
        * a list of compact summary rows
    """
    rows = []
    for regime in ["overall", "high"]:
        skew_row = pick_result(results_frame, regime, "skew_vs_direction", "signed_standardised_surprise")
        kurtosis_row = pick_result(results_frame, regime, "kurtosis_vs_magnitude", "b_standardised_surprise")
        if skew_row is None or kurtosis_row is None:
            continue
        rows.append({
            "pair": pair_name,
            "regime": regime,
            "weeks": int(skew_row["n"]),
            "skew_dir_corr": skew_row["correlation"],
            "skew_dir_p": skew_row["p_value"],
            "kurt_mag_corr": kurtosis_row["correlation"],
            "kurt_mag_p": kurtosis_row["p_value"],
        })
    return rows


def main():
    """
    Runs the shape-residual study for every pair and prints a cross-pair table.

    INPUTS:
        * none

    OUTPUTS:
        * a printed comparison and a saved csv under local_tests/output
    """
    engine.OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)

    frozen_model = engine.load_frozen_pca_model(pca_model_path = PCA_MODEL_PATH)
    feature_frame = pd.read_parquet(FEATURE_MATRIX_PATH).set_index(DATE_COLUMN_NAME).sort_index()
    weekly_level_frame = pd.read_parquet(WEEKLY_LEVELS_PATH).set_index(DATE_COLUMN_NAME).sort_index()
    fx_level_frame = pd.read_parquet(FX_WEEKLY_LEVELS_PATH).set_index(DATE_COLUMN_NAME).sort_index()

    scores_array = engine.project_features_onto_frozen_axes(feature_frame = feature_frame, frozen_model = frozen_model)
    vix_series = weekly_level_frame[VIX_COLUMN_NAME]
    fit_end_date = frozen_model["fit_end_date"]

    summary_rows = []
    for configuration in PAIR_CONFIGURATIONS:
        source_frame = weekly_level_frame if configuration["source_frame"] == "macro" else fx_level_frame
        pair_level = source_frame[configuration["pair"]]

        forward_return, market_variance, regime_label, _ = pair_tools.build_pair_target_variance_and_regime(
            pair_level = pair_level, vix_series = vix_series, fit_end_date = fit_end_date, variance_source = "realized"
        )
        realized_forward_variance = realized_forward_variance_for_level(pair_level = pair_level)

        feature_variant = feature_frame.copy()
        feature_variant[engine.TARGET_COLUMN_NAME] = forward_return.reindex(feature_variant.index)

        per_week_frame = shape_tools.run_shape_residual_walk_forward(
            feature_frame = feature_variant,
            scores_array = scores_array,
            market_variance = market_variance,
            regime_label = regime_label,
            realized_forward_variance = realized_forward_variance,
            fit_end_date = fit_end_date,
        )
        results_frame = shape_tools.build_results(per_week_frame = per_week_frame)
        summary_rows.extend(summarise_pair(pair_name = configuration["pair"], results_frame = results_frame))

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(RESULTS_CSV_PATH, index = False)

    print("Cross-pair replication of the two shape-residual findings.")
    print("skew_dir_corr: engine skew versus the signed standardised surprise. On USD/KRW")
    print("  this was a significant contrarian (negative) signal. Replication means the same")
    print("  sign and small p across other pairs.")
    print("kurt_mag_corr: engine kurtosis versus surprise magnitude. Expected near zero.")
    print("High-vol buckets are small for G10 pairs (2023-26 was calm by their history), so")
    print("  the overall rows carry the weight.")
    print()
    print(summary_frame.to_string(index = False, float_format = lambda value: format(value, ".4f")))
    print()
    print("Written " + str(RESULTS_CSV_PATH))


if __name__ == "__main__":
    main()
