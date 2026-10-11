"""
pair_variance_comparison.py is a research test, kept apart from the main
pipeline. It reuses the engine from kurtosis_impact_backtest.py and
answers two questions raised about the first run:

    1. Arm A used trailing realised vol. Does swapping the variance input for a
       forward-looking implied-vol proxy (VIX, rescaled) change the picture. A
       and C share the variance input, so this moves both together, but it lets
       arm B (bootstrap variance) compete against a weaker market variance.
    2. Does switching the forecast target from managed USD/KRW to a liquid,
       free-floating G10 pair (EUR/USD, GBP/USD) help, since the PCA regime is
       built from global macro factors that map better onto G10 vol.

The macro PCA regime and neighbour selection are unchanged. Only the outcome
series (the target and the neighbour forward returns) and the variance input
change per configuration.

Run from the repository root:
    python local_tests/pair_variance_comparison.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kurtosis_impact_backtest as engine

WEEKLY_LEVELS_PATH = Path("Data/Input/weekly_levels.parquet")
FX_WEEKLY_LEVELS_PATH = Path("Data/Input/fx_weekly_levels.parquet")
FEATURE_MATRIX_PATH = Path("Data/Input/feature_matrix.parquet")
PCA_MODEL_PATH = Path("components/pca_output/pca_model.parquet")

DATE_COLUMN_NAME = "week_date"
VIX_COLUMN_NAME = "vix"

# Each configuration is a target pair plus a variance input.
CONFIGURATIONS = [
    {"pair": "usd_krw", "source_frame": "macro", "variance_source": "realized"},
    {"pair": "eur_usd", "source_frame": "fx", "variance_source": "realized"},
    {"pair": "gbp_usd", "source_frame": "fx", "variance_source": "realized"},
    {"pair": "jpy_usd", "source_frame": "fx", "variance_source": "realized"},
    {"pair": "chf_usd", "source_frame": "fx", "variance_source": "realized"},
]


def build_pair_target_variance_and_regime(pair_level = None, vix_series = None, fit_end_date = None, variance_source = None):
    """
    Builds the forward return target, the 4-week market variance and the lagged
    volatility regime label for one pair. The variance is trailing realised vol
    scaled to four weeks, or a VIX proxy rescaled with a fit-window scale.

    INPUTS:
        * pair_level - the weekly level series for the pair
        * vix_series - the weekly VIX series, only used for the proxy
        * fit_end_date - the frozen fit window end
        * variance_source - "realized" or "vix_proxy"

    OUTPUTS:
        * forward_return - the forward 4-week log return target
        * market_variance - the 4-week variance series
        * regime_label - the low / mid / high label series
        * variance_source_used - the string actually used
    """
    forward_return = np.log(pair_level.shift(-engine.HORIZON_WEEKS) / pair_level)

    weekly_return = np.log(pair_level / pair_level.shift(engine.SINGLE_WEEK_LAG))
    rolling_weekly_variance = weekly_return.rolling(engine.REALIZED_VOL_WINDOW_WEEKS).var()
    realized_market_variance = rolling_weekly_variance * engine.HORIZON_WEEKS

    variance_source_used = variance_source
    if variance_source == "vix_proxy" and vix_series is not None:
        vix_four_week_std = (vix_series / 100.0) * np.sqrt(engine.HORIZON_WEEKS / engine.WEEKS_PER_YEAR)
        realized_four_week_std = np.sqrt(realized_market_variance)
        fit_mask = realized_market_variance.index <= fit_end_date
        scale = float(realized_four_week_std[fit_mask].mean() / vix_four_week_std[fit_mask].mean())
        market_variance = (scale * vix_four_week_std) ** 2
    else:
        variance_source_used = "realized"
        market_variance = realized_market_variance

    regime_measure = np.sqrt(rolling_weekly_variance).shift(engine.REGIME_LABEL_LAG_WEEKS)
    fit_measure = regime_measure[regime_measure.index <= fit_end_date].dropna()
    lower_threshold = fit_measure.quantile(engine.TERCILE_LOWER)
    upper_threshold = fit_measure.quantile(engine.TERCILE_UPPER)

    def label_regime(measure_value):
        if pd.isna(measure_value):
            return None
        if measure_value <= lower_threshold:
            return "low"
        if measure_value <= upper_threshold:
            return "mid"
        return "high"

    return forward_return, market_variance, regime_measure.apply(label_regime), variance_source_used


def summarise_configuration(configuration = None, feature_frame = None, scores_array = None, pair_level = None, vix_series = None, fit_end_date = None):
    """
    Runs the backtest for one configuration and pulls out the headline arm
    comparison overall and in the high-vol regime.

    INPUTS:
        * configuration - one entry from CONFIGURATIONS
        * feature_frame - the macro feature matrix
        * scores_array - the frozen PC scores
        * pair_level - the pair level series
        * vix_series - the weekly VIX series
        * fit_end_date - the frozen fit window end

    OUTPUTS:
        * a list of summary rows, one per reported regime
    """
    forward_return, market_variance, regime_label, variance_source_used = build_pair_target_variance_and_regime(
        pair_level = pair_level,
        vix_series = vix_series,
        fit_end_date = fit_end_date,
        variance_source = configuration["variance_source"],
    )

    feature_variant = feature_frame.copy()
    feature_variant[engine.TARGET_COLUMN_NAME] = forward_return.reindex(feature_variant.index)

    per_week_frame = engine.run_walk_forward_backtest(
        feature_frame = feature_variant,
        scores_array = scores_array,
        market_variance = market_variance,
        regime_label = regime_label,
        fit_end_date = fit_end_date,
    )
    score_summary = engine.build_score_summary(per_week_frame = per_week_frame)

    rows = []
    for regime_name in ["overall", "high"]:
        regime_summary = score_summary[score_summary["regime"] == regime_name].set_index("arm")
        if len(regime_summary) == 0 or regime_summary["weeks"].iloc[0] == 0:
            continue
        candidate_beats_baseline = regime_summary.loc["C", "mean_log_score"] > regime_summary.loc["A", "mean_log_score"]
        rows.append({
            "pair": configuration["pair"],
            "variance": variance_source_used,
            "regime": regime_name,
            "weeks": int(regime_summary.loc["A", "weeks"]),
            "logscore_A": regime_summary.loc["A", "mean_log_score"],
            "logscore_B": regime_summary.loc["B", "mean_log_score"],
            "logscore_C": regime_summary.loc["C", "mean_log_score"],
            "logscore_D": regime_summary.loc["D", "mean_log_score"],
            "twcrps_A": regime_summary.loc["A", "mean_twcrps"],
            "twcrps_C": regime_summary.loc["C", "mean_twcrps"],
            "C_beats_A_logscore": bool(candidate_beats_baseline),
            "dm_log_p_C_vs_A": regime_summary.loc["C", "dm_logscore_p_vs_A"],
            "dm_tw_p_C_vs_A": regime_summary.loc["C", "dm_twcrps_p_vs_A"],
        })
    return rows


def main():
    """
    Runs every configuration and prints one compact comparison table.

    INPUTS:
        * none

    OUTPUTS:
        * a printed comparison and a saved csv under local_tests/output
    """
    frozen_model = engine.load_frozen_pca_model(pca_model_path = PCA_MODEL_PATH)
    feature_frame = pd.read_parquet(FEATURE_MATRIX_PATH).set_index(DATE_COLUMN_NAME).sort_index()
    weekly_level_frame = pd.read_parquet(WEEKLY_LEVELS_PATH).set_index(DATE_COLUMN_NAME).sort_index()
    fx_level_frame = pd.read_parquet(FX_WEEKLY_LEVELS_PATH).set_index(DATE_COLUMN_NAME).sort_index()

    scores_array = engine.project_features_onto_frozen_axes(feature_frame = feature_frame, frozen_model = frozen_model)
    vix_series = weekly_level_frame[VIX_COLUMN_NAME]
    fit_end_date = frozen_model["fit_end_date"]

    all_rows = []
    for configuration in CONFIGURATIONS:
        if configuration["source_frame"] == "macro":
            pair_level = weekly_level_frame[configuration["pair"]]
        else:
            pair_level = fx_level_frame[configuration["pair"]]
        all_rows.extend(summarise_configuration(
            configuration = configuration,
            feature_frame = feature_frame,
            scores_array = scores_array,
            pair_level = pair_level,
            vix_series = vix_series,
            fit_end_date = fit_end_date,
        ))

    comparison_frame = pd.DataFrame(all_rows)
    output_path = engine.OUTPUT_DIRECTORY / "pair_variance_comparison.csv"
    comparison_frame.to_csv(output_path, index = False)

    print("Higher log score is better. Lower twCRPS is better. The shape adds value only")
    print("if C beats A. DM p is the C versus A test, and the sign of (logscore_C minus")
    print("logscore_A) tells you which way the significant difference points.")
    print()
    print(comparison_frame.to_string(index = False, float_format = lambda value: format(value, ".5f")))
    print()
    print("Written " + str(output_path))


if __name__ == "__main__":
    main()
