"""
shape_residual_correlation.py is a research test, kept apart from the main
pipeline. It modifies no tracked file.

Question: is the engine's distributional shape a signal for when arm A's simple
volatility forecast fails? Arm A forecasts the 4-week variance from trailing
realised vol. This test correlates the skew and excess kurtosis of the
PCA+bootstrap conditional distribution at week t (known at t) with how wrong A
turned out to be over the forward window (realised after t).

Two residual measures, both requested:
    a) volatility forecast error: the realised within-window variance (sum of the
       four squared weekly returns inside the window) divided by A's forecast
       variance. A log ratio, so a calibrated forecast sits near zero.
    b) standardised surprise: the realised 4-week return divided by A's forecast
       standard deviation. Its magnitude is how far the move blew through A's
       predicted spread, and dividing by the forecast already controls for the
       vol level.

Two hypotheses:
    kurtosis versus surprise MAGNITUDE (option a and option b): fat tails should
        predict a larger vol miss.
    skew versus SIGNED standardised surprise: asymmetry should predict which way
        the surprise lands.

Controls: a vol-level partial correlation (kurtosis versus magnitude, holding the
log forecast variance fixed), per-regime correlations, and block-bootstrap
significance to handle the 3-week overlap of consecutive 4-week windows.

Run from the repository root:
    python local_tests/shape_residual_correlation.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kurtosis_impact_backtest as engine

WEEKLY_LEVELS_PATH = Path("Data/Input/weekly_levels.parquet")
FEATURE_MATRIX_PATH = Path("Data/Input/feature_matrix.parquet")
PCA_MODEL_PATH = Path("components/pca_output/pca_model.parquet")

DATE_COLUMN_NAME = "week_date"
USD_KRW_COLUMN_NAME = "usd_krw"

BLOCK_BOOTSTRAP_WEEKS = 8
BLOCK_BOOTSTRAP_DRAWS = 2000
CORRELATION_METHOD = "spearman"
REPORTED_REGIMES = ["overall", "high", "mid"]

RESULTS_CSV_PATH = engine.OUTPUT_DIRECTORY / "shape_residual_correlation.csv"
PER_WEEK_CSV_PATH = engine.OUTPUT_DIRECTORY / "shape_residual_per_week.csv"


def compute_pool_moments(pool = None):
    """
    Skew and excess kurtosis of the pooled bootstrap draws, the same moments the
    engine prints, computed here as the shape predictor.

    INPUTS:
        * pool - the pooled bootstrap draws for one week

    OUTPUTS:
        * the skew and the excess kurtosis
    """
    mean_value = float(np.mean(pool))
    deviations = pool - mean_value
    variance_value = float(np.mean(deviations ** 2))
    standard_deviation = math.sqrt(variance_value)
    skew_value = float(np.mean(deviations ** 3) / standard_deviation ** 3)
    excess_kurtosis = float(np.mean(deviations ** 4) / variance_value ** 2) - 3.0
    return skew_value, excess_kurtosis


def build_realized_forward_variance(weekly_level_frame = None):
    """
    Realised variance of each forward window, from the four squared weekly
    returns inside it. This is the "what the volatility turned out to be" side of
    option a. It uses future weeks, which is correct because it is the outcome,
    not a predictor.

    INPUTS:
        * weekly_level_frame - weekly levels indexed by week

    OUTPUTS:
        * the realised forward variance series
    """
    weekly_return = np.log(weekly_level_frame[USD_KRW_COLUMN_NAME] / weekly_level_frame[USD_KRW_COLUMN_NAME].shift(engine.SINGLE_WEEK_LAG))
    squared_weekly_return = weekly_return ** 2
    return squared_weekly_return.rolling(engine.HORIZON_WEEKS).sum().shift(-engine.HORIZON_WEEKS)


def run_shape_residual_walk_forward(feature_frame = None, scores_array = None, market_variance = None, regime_label = None, realized_forward_variance = None, fit_end_date = None):
    """
    Walks forward over every post-fit week, reconstructs the neighbourhood, and
    records the engine shape alongside the realised outcome.

    INPUTS:
        * feature_frame - the feature matrix with the target column
        * scores_array - the frozen PC scores
        * market_variance - arm A's 4-week forecast variance series
        * regime_label - the lagged volatility regime label series
        * realized_forward_variance - the realised forward variance series
        * fit_end_date - the frozen fit window end

    OUTPUTS:
        * a per-week dataframe of shape predictors and realised outcomes
    """
    index = feature_frame.index
    realised_target = feature_frame[engine.TARGET_COLUMN_NAME].to_numpy(dtype = float)
    has_outcome = ~np.isnan(realised_target)
    positions = np.arange(len(index))
    market_variance_array = market_variance.reindex(index).to_numpy(dtype = float)
    regime_array = regime_label.reindex(index).to_numpy(dtype = object)
    forward_variance_array = realized_forward_variance.reindex(index).to_numpy(dtype = float)

    random_generator = np.random.default_rng(engine.RANDOM_SEED)
    records = []
    for test_position in positions:
        if not has_outcome[test_position] or index[test_position] <= fit_end_date:
            continue
        forecast_variance = market_variance_array[test_position]
        regime_value = regime_array[test_position]
        realised_forward_variance_value = forward_variance_array[test_position]
        if not np.isfinite(forecast_variance) or forecast_variance <= 0 or regime_value is None:
            continue
        if not np.isfinite(realised_forward_variance_value) or realised_forward_variance_value <= 0:
            continue

        eligible = has_outcome & (positions + engine.HORIZON_WEEKS <= test_position)
        if eligible.sum() < engine.NEIGHBOUR_COUNT:
            continue

        distances = np.linalg.norm(scores_array[eligible] - scores_array[test_position], axis = 1)
        eligible_positions = positions[eligible]
        nearest_positions = np.sort(eligible_positions[np.argsort(distances)[:engine.NEIGHBOUR_COUNT]])
        member_returns = realised_target[nearest_positions]

        pool = engine.moving_block_bootstrap_pool(member_returns = member_returns, random_generator = random_generator)
        skew_value, excess_kurtosis = compute_pool_moments(pool = pool)
        records.append({
            "week": index[test_position],
            "regime": regime_value,
            "skew": skew_value,
            "excess_kurtosis": excess_kurtosis,
            "forecast_variance": forecast_variance,
            "realised_return": realised_target[test_position],
            "realized_forward_variance": realised_forward_variance_value,
        })
    return pd.DataFrame(records)


def correlation_value(first_series = None, second_series = None):
    """
    Spearman or Pearson correlation of two arrays.

    INPUTS:
        * first_series - one array
        * second_series - the matching array

    OUTPUTS:
        * the correlation, or nan if undefined
    """
    if len(first_series) < 3:
        return float("nan")
    if CORRELATION_METHOD == "spearman":
        return float(spearmanr(first_series, second_series).correlation)
    return float(pearsonr(first_series, second_series)[0])


def partial_correlation_value(predictor = None, outcome = None, control = None):
    """
    Correlation of predictor and outcome holding the control fixed. Ranks first
    when the method is Spearman, then the standard partial-correlation formula.

    INPUTS:
        * predictor - the shape predictor
        * outcome - the surprise magnitude
        * control - the variable held fixed, here the log forecast variance

    OUTPUTS:
        * the partial correlation, or nan if undefined
    """
    if len(predictor) < 4:
        return float("nan")
    if CORRELATION_METHOD == "spearman":
        predictor, outcome, control = rankdata(predictor), rankdata(outcome), rankdata(control)
    predictor_outcome = pearsonr(predictor, outcome)[0]
    predictor_control = pearsonr(predictor, control)[0]
    outcome_control = pearsonr(outcome, control)[0]
    denominator = math.sqrt((1 - predictor_control ** 2) * (1 - outcome_control ** 2))
    if denominator == 0:
        return float("nan")
    return (predictor_outcome - predictor_control * outcome_control) / denominator


def block_bootstrap_interval(statistic_function = None, sample_length = None, random_generator = None):
    """
    Moving-block bootstrap confidence interval and two-sided p-value for a
    correlation statistic, resampling blocks of consecutive weeks so the
    overlapping-window autocorrelation is preserved.

    INPUTS:
        * statistic_function - maps a resampled index array to the statistic
        * sample_length - the number of weeks
        * random_generator - a numpy generator

    OUTPUTS:
        * the 2.5 and 97.5 percentile bounds and the two-sided p-value
    """
    block_count = math.ceil(sample_length / BLOCK_BOOTSTRAP_WEEKS)
    bootstrap_statistics = []
    for _ in range(BLOCK_BOOTSTRAP_DRAWS):
        resampled_index = []
        for _ in range(block_count):
            block_start = random_generator.integers(0, sample_length)
            resampled_index.extend((block_start + np.arange(BLOCK_BOOTSTRAP_WEEKS)) % sample_length)
        resampled_index = np.array(resampled_index[:sample_length])
        bootstrap_statistics.append(statistic_function(resampled_index))
    bootstrap_statistics = np.array(bootstrap_statistics)
    bootstrap_statistics = bootstrap_statistics[np.isfinite(bootstrap_statistics)]
    if len(bootstrap_statistics) == 0:
        return float("nan"), float("nan"), float("nan")
    lower_bound, upper_bound = np.percentile(bootstrap_statistics, [2.5, 97.5])
    p_value = 2.0 * min(np.mean(bootstrap_statistics <= 0), np.mean(bootstrap_statistics >= 0))
    return float(lower_bound), float(upper_bound), float(min(p_value, 1.0))


def evaluate_correlation(predictor = None, outcome = None, control = None, random_generator = None):
    """
    Point correlation (partial if a control is given) plus its block-bootstrap
    interval and p-value.

    INPUTS:
        * predictor - the shape predictor
        * outcome - the residual measure
        * control - optional, held fixed for a partial correlation
        * random_generator - a numpy generator

    OUTPUTS:
        * a dict with the correlation, interval bounds, p-value and count
    """
    if control is None:
        point_estimate = correlation_value(predictor, outcome)
        statistic_function = lambda index_array: correlation_value(predictor[index_array], outcome[index_array])
    else:
        point_estimate = partial_correlation_value(predictor, outcome, control)
        statistic_function = lambda index_array: partial_correlation_value(predictor[index_array], outcome[index_array], control[index_array])
    lower_bound, upper_bound, p_value = block_bootstrap_interval(
        statistic_function = statistic_function, sample_length = len(predictor), random_generator = random_generator
    )
    return {
        "correlation": point_estimate,
        "ci_low": lower_bound,
        "ci_high": upper_bound,
        "p_value": p_value,
        "n": len(predictor),
    }


def build_results(per_week_frame = None):
    """
    Runs every hypothesis, overall and per regime, for both residual options.

    INPUTS:
        * per_week_frame - the per-week shape and outcome table

    OUTPUTS:
        * a results dataframe
    """
    working = per_week_frame.copy()
    working["forecast_std"] = np.sqrt(working["forecast_variance"])
    working["standardised_surprise"] = working["realised_return"] / working["forecast_std"]
    working["abs_standardised_surprise"] = working["standardised_surprise"].abs()
    working["log_vol_error_ratio"] = np.log(working["realized_forward_variance"] / working["forecast_variance"])
    working["log_forecast_variance"] = np.log(working["forecast_variance"])

    random_generator = np.random.default_rng(engine.RANDOM_SEED)
    result_rows = []
    regime_groups = [("overall", working)] + [
        (label, working[working["regime"] == label]) for label in ["high", "mid"]
    ]
    for regime_name, regime_frame in regime_groups:
        if len(regime_frame) < 5:
            continue
        excess_kurtosis = regime_frame["excess_kurtosis"].to_numpy()
        skew = regime_frame["skew"].to_numpy()
        abs_surprise = regime_frame["abs_standardised_surprise"].to_numpy()
        signed_surprise = regime_frame["standardised_surprise"].to_numpy()
        log_vol_error = regime_frame["log_vol_error_ratio"].to_numpy()
        log_forecast_variance = regime_frame["log_forecast_variance"].to_numpy()

        tests = [
            ("kurtosis_vs_magnitude", "b_standardised_surprise", excess_kurtosis, abs_surprise, None),
            ("kurtosis_vs_magnitude", "a_vol_error_ratio", excess_kurtosis, log_vol_error, None),
            ("kurtosis_vs_magnitude_vol_controlled", "b_standardised_surprise", excess_kurtosis, abs_surprise, log_forecast_variance),
            ("skew_vs_direction", "signed_standardised_surprise", skew, signed_surprise, None),
        ]
        for test_name, residual_option, predictor, outcome, control in tests:
            outcome_result = evaluate_correlation(
                predictor = predictor, outcome = outcome, control = control, random_generator = random_generator
            )
            outcome_result.update({"regime": regime_name, "test": test_name, "residual_option": residual_option})
            result_rows.append(outcome_result)

    column_order = ["regime", "test", "residual_option", "n", "correlation", "ci_low", "ci_high", "p_value"]
    return pd.DataFrame(result_rows)[column_order]


def main():
    """
    Runs the shape-versus-residual correlation study and writes results locally.

    INPUTS:
        * none

    OUTPUTS:
        * printed tables and csv files under local_tests/output
    """
    engine.OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)

    frozen_model = engine.load_frozen_pca_model(pca_model_path = PCA_MODEL_PATH)
    feature_frame = pd.read_parquet(FEATURE_MATRIX_PATH).set_index(DATE_COLUMN_NAME).sort_index()
    weekly_level_frame = pd.read_parquet(WEEKLY_LEVELS_PATH).set_index(DATE_COLUMN_NAME).sort_index()

    scores_array = engine.project_features_onto_frozen_axes(feature_frame = feature_frame, frozen_model = frozen_model)
    market_variance, regime_label, _ = engine.build_market_variance_and_regime(
        weekly_level_frame = weekly_level_frame, fit_end_date = frozen_model["fit_end_date"]
    )
    realized_forward_variance = build_realized_forward_variance(weekly_level_frame = weekly_level_frame)

    per_week_frame = run_shape_residual_walk_forward(
        feature_frame = feature_frame,
        scores_array = scores_array,
        market_variance = market_variance,
        regime_label = regime_label,
        realized_forward_variance = realized_forward_variance,
        fit_end_date = frozen_model["fit_end_date"],
    )
    if len(per_week_frame) == 0:
        raise RuntimeError("No evaluable weeks. Check the frozen model and feature matrix.")

    results_frame = build_results(per_week_frame = per_week_frame)
    per_week_frame.to_csv(PER_WEEK_CSV_PATH, index = False)
    results_frame.to_csv(RESULTS_CSV_PATH, index = False)

    print("Target USD/KRW. Predictor is the engine shape at t (known at t). Outcome is")
    print("realised over the forward window. Correlation method: " + CORRELATION_METHOD + ".")
    print("Test weeks: " + str(len(per_week_frame)) + ". Regime counts: " + str(per_week_frame["regime"].value_counts().to_dict()))
    print()
    print("Hypotheses:")
    print("  kurtosis_vs_magnitude   positive means high engine kurtosis predicts a bigger vol miss.")
    print("  vol_controlled          same, holding the forecast vol level fixed, so it is not just level.")
    print("  skew_vs_direction       positive means engine skew predicts the signed standardised surprise.")
    print("Residual options: a_vol_error_ratio uses the four within-window weekly returns,")
    print("  b_standardised_surprise uses the single 4-week return over the forecast sigma.")
    print()
    print(results_frame.to_string(index = False, float_format = lambda value: format(value, ".4f")))
    print()
    print("Written " + str(RESULTS_CSV_PATH))
    print("Written " + str(PER_WEEK_CSV_PATH))


if __name__ == "__main__":
    main()
