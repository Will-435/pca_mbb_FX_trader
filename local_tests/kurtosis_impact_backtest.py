"""
kurtosis_impact_backtest.py is a research test, kept apart from the main
pipeline. It writes nothing outside local_tests/ and modifies no tracked file.

Question: does the empirical tail shape (skew and kurtosis) produced by the
PCA-regime plus moving-block-bootstrap engine improve the forecast of forward
return DISPERSION AND TAILS over a plain Gaussian, measured inside volatility
regimes. Drift is not tested: every candidate distribution is centred at zero.

The engine is mirrored in Python from the frozen artefacts, never refitted:
    project the six 4-week features onto the frozen PCA axes (pca_model.parquet),
    pick the 40 nearest historical weeks by plain unwhitened Euclidean distance,
    moving-block-bootstrap their forward 4-week returns (block 4, 2000 reps).

Four zero-centred candidate predictive distributions per test week:
    A) Gaussian, variance from the market (realised vol, or a VIX proxy switch).
    B) Gaussian, variance from the bootstrap distribution itself.
    C) Bootstrap shape standardised to unit variance then rescaled to the market
       variance. Shape from the engine, variance from the market. The candidate.
    D) The full bootstrap distribution, its own variance and shape.
The headline test is A versus C: same variance, so it isolates the added shape.
B and D attribute any gain between variance and shape.

Leakage controls:
    frozen PCA axes, never refitted on test data. The neighbour pool for test
    week t only contains weeks whose 4-week forward window closed on or before t
    (s + 4 <= t), which also purges and embargoes the 3-week horizon overlap.
    Regime labels use a lagged volatility measure known at prediction time.

Run from the repository root:
    python local_tests/kurtosis_impact_backtest.py
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, gaussian_kde, norm

FEATURE_MATRIX_PATH = Path("Data/Input/feature_matrix.parquet")
PCA_MODEL_PATH = Path("components/pca_output/pca_model.parquet")
WEEKLY_LEVELS_PATH = Path("Data/Input/weekly_levels.parquet")
OUTPUT_DIRECTORY = Path("local_tests/output")

DATE_COLUMN_NAME = "week_date"
TARGET_COLUMN_NAME = "usd_krw_forward_return_4w"
USD_KRW_COLUMN_NAME = "usd_krw"
VIX_COLUMN_NAME = "vix"

# Engine parameters, matched to the live pipeline.
HORIZON_WEEKS = 4
NEIGHBOUR_COUNT = 40
BLOCK_LENGTH_WEEKS = 4
NUMBER_OF_REPLICATIONS = 2000

SINGLE_WEEK_LAG = 1
REALIZED_VOL_WINDOW_WEEKS = 26
WEEKS_PER_YEAR = 52

# "realized" uses trailing realised vol of weekly USD/KRW returns. "vix_proxy"
# uses lagged VIX rescaled to a USD/KRW 4-week variance with a scale fitted on
# the frozen window. No genuine USD/KRW implied vol exists in the repo, so
# "realized" is the honest default and the printed report states which was used.
VARIANCE_SOURCE = "realized"

# The threshold-weighted CRPS caps values at this lower-tail quantile of the
# market-variance Gaussian, so the score weights the loss (left) tail. The same
# threshold is used for every arm so the comparison is fair.
LOSS_TAIL_QUANTILE = 0.10

VAR_LEVELS = [1, 5, 95, 99]
LOWER_TAIL_MAX_LEVEL = 50

# Overlapping 4-week horizons make consecutive weekly scores autocorrelated up
# to three lags, so the Diebold-Mariano variance is Newey-West with this lag.
NEWEY_WEST_LAG_WEEKS = HORIZON_WEEKS - 1

SCORE_SAMPLE_COUNT = 8000
DENSITY_FLOOR = 1e-300
RANDOM_SEED = 42

REGIME_LABELS = ["low", "mid", "high"]
TERCILE_LOWER = 1.0 / 3.0
TERCILE_UPPER = 2.0 / 3.0
REGIME_LABEL_LAG_WEEKS = 1

PLOT_PATH = OUTPUT_DIRECTORY / "kurtosis_impact_c_minus_a.png"
PER_WEEK_PARQUET_PATH = OUTPUT_DIRECTORY / "per_week_scores.parquet"
PER_WEEK_CSV_PATH = OUTPUT_DIRECTORY / "per_week_scores.csv"
SCORE_SUMMARY_CSV_PATH = OUTPUT_DIRECTORY / "score_summary.csv"
VAR_SUMMARY_CSV_PATH = OUTPUT_DIRECTORY / "var_summary.csv"

PLOT_WIDTH_INCHES = 12
PLOT_HEIGHT_INCHES = 6
PLOT_DPI = 150
PER_WEEK_COLOUR = "blue"
CUMULATIVE_COLOUR = "red"
ZERO_LINE_COLOUR = "green"

ARM_LABELS = ["A", "B", "C", "D"]


def load_frozen_pca_model(pca_model_path = None):
    """
    Reads the frozen PCA model back from the flat parquet the pipeline writes.

    INPUTS:
        * pca_model_path - path to pca_model.parquet

    OUTPUTS:
        * a dict with ordered feature names, the feature mean vector, the loading
          matrix (features by components), the component count and the fit dates
    """
    model_frame = pd.read_parquet(pca_model_path)

    mean_rows = model_frame[model_frame["record_type"] == "mean"].sort_values("feature_order")
    feature_names = mean_rows["feature_name"].tolist()
    feature_means = mean_rows["value"].to_numpy(dtype = float)

    loading_rows = model_frame[model_frame["record_type"] == "loading"]
    component_count = int(loading_rows["component_number"].max())
    loading_matrix = np.zeros((len(feature_names), component_count))
    for _, loading_row in loading_rows.iterrows():
        feature_position = int(loading_row["feature_order"]) - 1
        component_position = int(loading_row["component_number"]) - 1
        loading_matrix[feature_position, component_position] = float(loading_row["value"])

    fit_rows = model_frame[model_frame["record_type"] == "fit_date_range"]
    fit_dates = dict(zip(fit_rows["feature_name"], fit_rows["date_text"]))

    return {
        "feature_names": feature_names,
        "feature_means": feature_means,
        "loading_matrix": loading_matrix,
        "component_count": component_count,
        "fit_start_date": pd.Timestamp(fit_dates["fit_start_date"]),
        "fit_end_date": pd.Timestamp(fit_dates["fit_end_date"]),
    }


def project_features_onto_frozen_axes(feature_frame = None, frozen_model = None):
    """
    Projects the feature history onto the frozen axes. Centre with the frozen
    means, multiply by the frozen loadings. Never refits.

    INPUTS:
        * feature_frame - the weekly feature matrix indexed by week
        * frozen_model - from load_frozen_pca_model

    OUTPUTS:
        * a numpy array of PC scores, one row per week, one column per component
    """
    ordered_features = feature_frame[frozen_model["feature_names"]].to_numpy(dtype = float)
    centred_features = ordered_features - frozen_model["feature_means"]
    return centred_features @ frozen_model["loading_matrix"]


def moving_block_bootstrap_pool(member_returns = None, random_generator = None):
    """
    Mirrors bootstrap.py: draw blocks of consecutive members with replacement,
    rebuild a series of the original length per replication, and pool every
    resampled value across all replications into one empirical distribution.

    INPUTS:
        * member_returns - the 40 neighbour forward returns in week order
        * random_generator - a numpy generator

    OUTPUTS:
        * a numpy array of all pooled resampled returns
    """
    member_count = len(member_returns)
    possible_block_starts = member_count - BLOCK_LENGTH_WEEKS + 1
    blocks_per_replication = math.ceil(member_count / BLOCK_LENGTH_WEEKS)

    block_starts = random_generator.integers(
        0, possible_block_starts, size = (NUMBER_OF_REPLICATIONS, blocks_per_replication)
    )
    block_offsets = np.arange(BLOCK_LENGTH_WEEKS)
    draw_indices = block_starts[:, :, None] + block_offsets[None, None, :]
    draw_indices = draw_indices.reshape(NUMBER_OF_REPLICATIONS, blocks_per_replication * BLOCK_LENGTH_WEEKS)
    draw_indices = draw_indices[:, :member_count]
    return member_returns[draw_indices].reshape(-1)


def gaussian_log_density(realised_value = None, variance = None):
    """
    Log density of a zero-mean Gaussian at the realised value.

    INPUTS:
        * realised_value - the realised forward return
        * variance - the Gaussian variance

    OUTPUTS:
        * the log density
    """
    return -0.5 * math.log(2.0 * math.pi * variance) - (realised_value * realised_value) / (2.0 * variance)


def kernel_log_density(predictive_sample = None, realised_value = None):
    """
    Log density of an empirical sample at the realised value, via a Gaussian
    kernel density estimate. Floored so a deep-tail outcome cannot give minus
    infinity.

    INPUTS:
        * predictive_sample - draws from the candidate distribution
        * realised_value - the realised forward return

    OUTPUTS:
        * the log density
    """
    density_estimate = gaussian_kde(predictive_sample)
    density_value = float(density_estimate.evaluate([realised_value])[0])
    return math.log(max(density_value, DENSITY_FLOOR))


def empirical_crps_from_sorted(sorted_sample = None, observation = None):
    """
    CRPS of a sorted sample against one scalar outcome.

    INPUTS:
        * sorted_sample - the predictive sample, ascending
        * observation - the realised value

    OUTPUTS:
        * the CRPS
    """
    sample_size = len(sorted_sample)
    mean_absolute_error = np.mean(np.abs(sorted_sample - observation))
    positions = np.arange(1, sample_size + 1)
    expected_spread = (2.0 / (sample_size * sample_size)) * np.sum((2 * positions - sample_size - 1) * sorted_sample)
    return float(mean_absolute_error - 0.5 * expected_spread)


def threshold_weighted_crps(predictive_sample = None, realised_value = None, loss_tail_threshold = None):
    """
    Threshold-weighted CRPS focused on the loss (left) tail. The chaining
    function caps values at the threshold, so only the tail below it contributes.

    INPUTS:
        * predictive_sample - draws from the candidate distribution
        * realised_value - the realised forward return
        * loss_tail_threshold - the cap, set from the market variance

    OUTPUTS:
        * the threshold-weighted CRPS
    """
    transformed_sample = np.minimum(predictive_sample, loss_tail_threshold)
    transformed_observation = min(realised_value, loss_tail_threshold)
    return empirical_crps_from_sorted(np.sort(transformed_sample), transformed_observation)


def value_at_risk_breaches(gaussian_variance = None, empirical_sample = None, realised_value = None):
    """
    Breach flags at the 1st, 5th, 95th and 99th percentiles. If a Gaussian
    variance is given the quantiles are closed form, otherwise they come from the
    empirical sample. Lower levels breach below, upper levels breach above.

    INPUTS:
        * gaussian_variance - variance for a Gaussian arm, or None for empirical
        * empirical_sample - the sample for an empirical arm, or None for Gaussian
        * realised_value - the realised forward return

    OUTPUTS:
        * a dict from VaR level to breach flag (1 or 0)
    """
    breaches = {}
    for var_level in VAR_LEVELS:
        if gaussian_variance is not None:
            quantile_value = norm.ppf(var_level / 100.0) * math.sqrt(gaussian_variance)
        else:
            quantile_value = np.percentile(empirical_sample, var_level)
        if var_level <= LOWER_TAIL_MAX_LEVEL:
            breaches[var_level] = int(realised_value < quantile_value)
        else:
            breaches[var_level] = int(realised_value > quantile_value)
    return breaches


def kupiec_pof_pvalue(breach_flags = None, target_rate = None):
    """
    Kupiec proportion-of-failures test of unconditional VaR coverage.

    INPUTS:
        * breach_flags - the breach series, 1 or 0
        * target_rate - the expected breach rate

    OUTPUTS:
        * the test p-value, or nan when it is undefined
    """
    observation_count = len(breach_flags)
    breach_count = int(np.sum(breach_flags))
    if observation_count == 0 or breach_count == 0 or breach_count == observation_count:
        return float("nan")
    observed_rate = breach_count / observation_count
    log_likelihood_null = (observation_count - breach_count) * math.log(1 - target_rate) + breach_count * math.log(target_rate)
    log_likelihood_observed = (observation_count - breach_count) * math.log(1 - observed_rate) + breach_count * math.log(observed_rate)
    test_statistic = -2.0 * (log_likelihood_null - log_likelihood_observed)
    return float(chi2.sf(test_statistic, 1))


def christoffersen_cc_pvalue(breach_flags = None, target_rate = None):
    """
    Christoffersen conditional-coverage test, combining unconditional coverage
    with independence of consecutive breaches.

    INPUTS:
        * breach_flags - the breach series in time order, 1 or 0
        * target_rate - the expected breach rate

    OUTPUTS:
        * the test p-value, or nan when it is undefined
    """
    breach_flags = np.asarray(breach_flags)
    observation_count = len(breach_flags)
    breach_count = int(np.sum(breach_flags))
    if observation_count < 2 or breach_count == 0 or breach_count == observation_count:
        return float("nan")

    transition_00 = transition_01 = transition_10 = transition_11 = 0
    for previous_flag, current_flag in zip(breach_flags[:-1], breach_flags[1:]):
        if previous_flag == 0 and current_flag == 0:
            transition_00 += 1
        elif previous_flag == 0 and current_flag == 1:
            transition_01 += 1
        elif previous_flag == 1 and current_flag == 0:
            transition_10 += 1
        else:
            transition_11 += 1

    observed_rate = breach_count / observation_count
    log_likelihood_null = (observation_count - breach_count) * math.log(1 - target_rate) + breach_count * math.log(target_rate)
    log_likelihood_observed = (observation_count - breach_count) * math.log(1 - observed_rate) + breach_count * math.log(observed_rate)
    unconditional_statistic = -2.0 * (log_likelihood_null - log_likelihood_observed)

    conditional_1_rate = transition_01 / (transition_00 + transition_01) if (transition_00 + transition_01) > 0 else 0.0
    conditional_2_rate = transition_11 / (transition_10 + transition_11) if (transition_10 + transition_11) > 0 else 0.0
    pooled_rate = (transition_01 + transition_11) / (transition_00 + transition_01 + transition_10 + transition_11)
    if pooled_rate <= 0 or pooled_rate >= 1 or conditional_1_rate <= 0 or conditional_2_rate <= 0:
        independence_statistic = 0.0
    else:
        log_likelihood_pooled = (transition_00 + transition_10) * math.log(1 - pooled_rate) + (transition_01 + transition_11) * math.log(pooled_rate)
        log_likelihood_markov = (
            transition_00 * math.log(1 - conditional_1_rate)
            + transition_01 * math.log(conditional_1_rate)
            + transition_10 * math.log(1 - conditional_2_rate)
            + transition_11 * math.log(conditional_2_rate)
        )
        independence_statistic = -2.0 * (log_likelihood_pooled - log_likelihood_markov)

    return float(chi2.sf(unconditional_statistic + independence_statistic, 2))


def diebold_mariano_pvalue(loss_differential = None):
    """
    Diebold-Mariano test on a loss differential, with a Newey-West (HAC)
    variance to handle the overlapping-horizon autocorrelation. A negative mean
    differential means the candidate has the lower loss.

    INPUTS:
        * loss_differential - per-week (candidate loss minus baseline loss)

    OUTPUTS:
        * the mean differential, the test statistic and the two-sided p-value
    """
    loss_differential = np.asarray(loss_differential, dtype = float)
    observation_count = len(loss_differential)
    if observation_count < 3:
        return float("nan"), float("nan"), float("nan")
    mean_differential = float(np.mean(loss_differential))
    deviations = loss_differential - mean_differential

    long_run_variance = float(np.mean(deviations * deviations))
    for lag in range(1, NEWEY_WEST_LAG_WEEKS + 1):
        if lag >= observation_count:
            break
        bartlett_weight = 1.0 - lag / (NEWEY_WEST_LAG_WEEKS + 1)
        autocovariance = float(np.mean(deviations[lag:] * deviations[:-lag]))
        long_run_variance += 2.0 * bartlett_weight * autocovariance

    if long_run_variance <= 0:
        return mean_differential, float("nan"), float("nan")
    test_statistic = mean_differential / math.sqrt(long_run_variance / observation_count)
    p_value = 2.0 * (1.0 - norm.cdf(abs(test_statistic)))
    return mean_differential, float(test_statistic), float(p_value)


def build_market_variance_and_regime(weekly_level_frame = None, fit_end_date = None):
    """
    Builds the 4-week market variance series and the lagged volatility regime
    label. The market variance is trailing realised vol of weekly USD/KRW
    returns scaled to four weeks, or a VIX proxy rescaled with a fit-window
    scale. Regime terciles are frozen on the fit window so labels never peek at
    the test period.

    INPUTS:
        * weekly_level_frame - weekly levels indexed by week
        * fit_end_date - the frozen fit window end

    OUTPUTS:
        * market_variance - the 4-week variance series
        * regime_label - the low / mid / high label series
        * variance_source_used - the string actually used
    """
    weekly_return = np.log(weekly_level_frame[USD_KRW_COLUMN_NAME] / weekly_level_frame[USD_KRW_COLUMN_NAME].shift(SINGLE_WEEK_LAG))
    rolling_weekly_variance = weekly_return.rolling(REALIZED_VOL_WINDOW_WEEKS).var()
    realized_market_variance = rolling_weekly_variance * HORIZON_WEEKS

    variance_source_used = VARIANCE_SOURCE
    if VARIANCE_SOURCE == "vix_proxy" and VIX_COLUMN_NAME in weekly_level_frame.columns:
        # VIX is annualised equity implied vol, a proxy only. Rescale it to a
        # USD/KRW 4-week standard deviation with a ratio fitted on the fit window.
        vix_four_week_std = (weekly_level_frame[VIX_COLUMN_NAME] / 100.0) * math.sqrt(HORIZON_WEEKS / WEEKS_PER_YEAR)
        realized_four_week_std = np.sqrt(realized_market_variance)
        fit_mask = realized_market_variance.index <= fit_end_date
        scale = float(realized_four_week_std[fit_mask].mean() / vix_four_week_std[fit_mask].mean())
        market_variance = (scale * vix_four_week_std) ** 2
    else:
        variance_source_used = "realized"
        market_variance = realized_market_variance

    regime_measure = np.sqrt(rolling_weekly_variance).shift(REGIME_LABEL_LAG_WEEKS)
    fit_measure = regime_measure[regime_measure.index <= fit_end_date].dropna()
    lower_threshold = fit_measure.quantile(TERCILE_LOWER)
    upper_threshold = fit_measure.quantile(TERCILE_UPPER)

    def label_regime(measure_value):
        if pd.isna(measure_value):
            return None
        if measure_value <= lower_threshold:
            return "low"
        if measure_value <= upper_threshold:
            return "mid"
        return "high"

    regime_label = regime_measure.apply(label_regime)
    return market_variance, regime_label, variance_source_used


def score_candidates_for_week(pool = None, market_variance = None, realised_value = None, random_generator = None):
    """
    Scores the four zero-centred candidates for one test week: log score,
    threshold-weighted CRPS and the VaR breach flags.

    INPUTS:
        * pool - the pooled bootstrap draws for this week
        * market_variance - the 4-week market variance
        * realised_value - the realised forward return
        * random_generator - a numpy generator for the Gaussian score samples

    OUTPUTS:
        * a dict of per-arm log scores, CRPS values and breach flags
    """
    bootstrap_mean = float(np.mean(pool))
    bootstrap_variance = float(np.var(pool))
    bootstrap_std = math.sqrt(bootstrap_variance)

    standardised_shape = (pool - bootstrap_mean) / bootstrap_std
    candidate_c_sample = random_generator.choice(standardised_shape, SCORE_SAMPLE_COUNT, replace = False) * math.sqrt(market_variance)
    candidate_d_sample = random_generator.choice(pool - bootstrap_mean, SCORE_SAMPLE_COUNT, replace = False)

    gaussian_a_sample = random_generator.normal(0.0, math.sqrt(market_variance), SCORE_SAMPLE_COUNT)
    gaussian_b_sample = random_generator.normal(0.0, bootstrap_std, SCORE_SAMPLE_COUNT)

    loss_tail_threshold = norm.ppf(LOSS_TAIL_QUANTILE) * math.sqrt(market_variance)

    scores = {"bootstrap_variance": bootstrap_variance, "market_variance": market_variance}

    scores["log_score_A"] = gaussian_log_density(realised_value, market_variance)
    scores["log_score_B"] = gaussian_log_density(realised_value, bootstrap_variance)
    scores["log_score_C"] = kernel_log_density(candidate_c_sample, realised_value)
    scores["log_score_D"] = kernel_log_density(candidate_d_sample, realised_value)

    scores["twcrps_A"] = threshold_weighted_crps(gaussian_a_sample, realised_value, loss_tail_threshold)
    scores["twcrps_B"] = threshold_weighted_crps(gaussian_b_sample, realised_value, loss_tail_threshold)
    scores["twcrps_C"] = threshold_weighted_crps(candidate_c_sample, realised_value, loss_tail_threshold)
    scores["twcrps_D"] = threshold_weighted_crps(candidate_d_sample, realised_value, loss_tail_threshold)

    breaches_by_arm = {
        "A": value_at_risk_breaches(gaussian_variance = market_variance, empirical_sample = None, realised_value = realised_value),
        "B": value_at_risk_breaches(gaussian_variance = bootstrap_variance, empirical_sample = None, realised_value = realised_value),
        "C": value_at_risk_breaches(gaussian_variance = None, empirical_sample = candidate_c_sample, realised_value = realised_value),
        "D": value_at_risk_breaches(gaussian_variance = None, empirical_sample = candidate_d_sample, realised_value = realised_value),
    }
    for arm_label, breaches in breaches_by_arm.items():
        for var_level, breach_flag in breaches.items():
            scores["breach_" + arm_label + "_" + str(var_level)] = breach_flag
    return scores


def run_walk_forward_backtest(feature_frame = None, scores_array = None, market_variance = None, regime_label = None, fit_end_date = None):
    """
    Walks forward over every post-fit test week, reconstructs the regime
    conditional distribution and scores the four candidates.

    INPUTS:
        * feature_frame - the weekly feature matrix with the target column
        * scores_array - the frozen PC scores, one row per week
        * market_variance - the 4-week market variance series
        * regime_label - the lagged volatility regime label series
        * fit_end_date - the frozen fit window end

    OUTPUTS:
        * a dataframe of per-week scores
    """
    index = feature_frame.index
    realised_target = feature_frame[TARGET_COLUMN_NAME].to_numpy(dtype = float)
    has_outcome = ~np.isnan(realised_target)
    positions = np.arange(len(index))
    market_variance_array = market_variance.reindex(index).to_numpy(dtype = float)
    regime_array = regime_label.reindex(index).to_numpy(dtype = object)

    random_generator = np.random.default_rng(RANDOM_SEED)
    records = []
    for test_position in positions:
        if not has_outcome[test_position] or index[test_position] <= fit_end_date:
            continue
        market_variance_value = market_variance_array[test_position]
        regime_value = regime_array[test_position]
        if not np.isfinite(market_variance_value) or market_variance_value <= 0 or regime_value is None:
            continue

        # Point in time: only weeks whose forward window closed on or before this
        # week. This purges and embargoes the three-week horizon overlap.
        eligible = has_outcome & (positions + HORIZON_WEEKS <= test_position)
        if eligible.sum() < NEIGHBOUR_COUNT:
            continue

        distances = np.linalg.norm(scores_array[eligible] - scores_array[test_position], axis = 1)
        eligible_positions = positions[eligible]
        nearest_positions = np.sort(eligible_positions[np.argsort(distances)[:NEIGHBOUR_COUNT]])
        member_returns = realised_target[nearest_positions]

        pool = moving_block_bootstrap_pool(member_returns = member_returns, random_generator = random_generator)
        week_scores = score_candidates_for_week(
            pool = pool,
            market_variance = market_variance_value,
            realised_value = realised_target[test_position],
            random_generator = random_generator,
        )
        week_scores["week"] = index[test_position]
        week_scores["regime"] = regime_value
        week_scores["realised"] = realised_target[test_position]
        records.append(week_scores)

    return pd.DataFrame(records)


def build_score_summary(per_week_frame = None):
    """
    Builds the per regime, per arm summary of mean log score and mean
    threshold-weighted CRPS, with Diebold-Mariano p-values versus baseline A.

    INPUTS:
        * per_week_frame - the per-week scores

    OUTPUTS:
        * a summary dataframe
    """
    summary_rows = []
    regime_groups = [("overall", per_week_frame)] + [
        (label, per_week_frame[per_week_frame["regime"] == label]) for label in REGIME_LABELS
    ]
    for regime_name, regime_frame in regime_groups:
        for arm_label in ARM_LABELS:
            row = {
                "regime": regime_name,
                "arm": arm_label,
                "weeks": len(regime_frame),
                "mean_log_score": regime_frame["log_score_" + arm_label].mean(),
                "mean_twcrps": regime_frame["twcrps_" + arm_label].mean(),
                "dm_logscore_p_vs_A": float("nan"),
                "dm_twcrps_p_vs_A": float("nan"),
            }
            if arm_label != "A" and len(regime_frame) >= 3:
                log_loss_differential = (-regime_frame["log_score_" + arm_label].to_numpy()) - (-regime_frame["log_score_A"].to_numpy())
                _, _, log_p = diebold_mariano_pvalue(log_loss_differential)
                twcrps_differential = regime_frame["twcrps_" + arm_label].to_numpy() - regime_frame["twcrps_A"].to_numpy()
                _, _, twcrps_p = diebold_mariano_pvalue(twcrps_differential)
                row["dm_logscore_p_vs_A"] = log_p
                row["dm_twcrps_p_vs_A"] = twcrps_p
            summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def build_var_summary(per_week_frame = None):
    """
    Builds the per regime, per arm, per level VaR summary: breach rate, target
    rate, Kupiec and Christoffersen p-values.

    INPUTS:
        * per_week_frame - the per-week scores, time ordered

    OUTPUTS:
        * a summary dataframe
    """
    summary_rows = []
    regime_groups = [("overall", per_week_frame)] + [
        (label, per_week_frame[per_week_frame["regime"] == label]) for label in REGIME_LABELS
    ]
    for regime_name, regime_frame in regime_groups:
        ordered_frame = regime_frame.sort_values("week")
        for arm_label in ARM_LABELS:
            for var_level in VAR_LEVELS:
                breach_flags = ordered_frame["breach_" + arm_label + "_" + str(var_level)].to_numpy()
                target_rate = var_level / 100.0 if var_level <= LOWER_TAIL_MAX_LEVEL else (100 - var_level) / 100.0
                summary_rows.append({
                    "regime": regime_name,
                    "arm": arm_label,
                    "var_level": var_level,
                    "target_rate": target_rate,
                    "breach_rate": float(np.mean(breach_flags)) if len(breach_flags) else float("nan"),
                    "kupiec_p": kupiec_pof_pvalue(breach_flags, target_rate),
                    "christoffersen_p": christoffersen_cc_pvalue(breach_flags, target_rate),
                })
    return pd.DataFrame(summary_rows)


def save_c_minus_a_plot(per_week_frame = None):
    """
    Saves a local PNG of the candidate C minus baseline A log score difference
    over time, per week and cumulative.

    INPUTS:
        * per_week_frame - the per-week scores

    OUTPUTS:
        * a PNG at local_tests/output
    """
    ordered_frame = per_week_frame.sort_values("week")
    difference = ordered_frame["log_score_C"].to_numpy() - ordered_frame["log_score_A"].to_numpy()
    weeks = ordered_frame["week"].to_numpy()

    figure, (per_week_axis, cumulative_axis) = plt.subplots(
        nrows = 2, ncols = 1, figsize = (PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES), sharex = True
    )
    per_week_axis.bar(weeks, difference, width = 5, color = PER_WEEK_COLOUR)
    per_week_axis.axhline(0.0, color = ZERO_LINE_COLOUR, linestyle = "--")
    per_week_axis.set_ylabel("C minus A log score")

    cumulative_axis.plot(weeks, np.cumsum(difference), color = CUMULATIVE_COLOUR)
    cumulative_axis.axhline(0.0, color = ZERO_LINE_COLOUR, linestyle = "--")
    cumulative_axis.set_ylabel("cumulative")
    cumulative_axis.set_xlabel("week")

    figure.suptitle("Candidate C (added shape) minus baseline A (Gaussian) log score, above zero favours C")
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi = PLOT_DPI)
    plt.close(figure)


def print_report(score_summary = None, var_summary = None, per_week_frame = None, variance_source_used = None):
    """
    Prints the summary tables and the headline read.

    INPUTS:
        * score_summary - from build_score_summary
        * var_summary - from build_var_summary
        * per_week_frame - the per-week scores
        * variance_source_used - the market variance source string

    OUTPUTS:
        * printed tables
    """
    print("Market variance source used: " + variance_source_used)
    print("No USD/KRW implied vol exists in the repo, so realised vol is the honest default.")
    print("Test weeks scored (overlapping): " + str(len(per_week_frame)))
    print("Regime week counts: " + str(per_week_frame["regime"].value_counts().to_dict()))
    print()
    print("Arms: A Gaussian at market variance, B Gaussian at bootstrap variance,")
    print("      C bootstrap shape at market variance (the candidate), D full bootstrap.")
    print("Higher mean log score is better. Lower mean threshold-weighted CRPS is better.")
    print("DM p-value tests each arm against baseline A. Small p with the arm scoring")
    print("better means a real difference.")
    print()
    print("SCORE SUMMARY")
    print(score_summary.to_string(index = False, float_format = lambda value: format(value, ".5f")))
    print()
    print("VALUE-AT-RISK SUMMARY (breach rate versus target, with Kupiec and Christoffersen p-values)")
    print(var_summary.to_string(index = False, float_format = lambda value: format(value, ".4f")))
    print()

    headline = score_summary[(score_summary["regime"] == "overall") & (score_summary["arm"] == "C")].iloc[0]
    high_regime = score_summary[(score_summary["regime"] == "high") & (score_summary["arm"] == "C")]
    print("HEADLINE (C versus A):")
    print("    overall log score DM p-value   " + format(headline["dm_logscore_p_vs_A"], ".4f"))
    print("    overall twCRPS DM p-value      " + format(headline["dm_twcrps_p_vs_A"], ".4f"))
    if len(high_regime) > 0:
        print("    high-vol log score DM p-value  " + format(high_regime.iloc[0]["dm_logscore_p_vs_A"], ".4f"))
        print("    high-vol twCRPS DM p-value     " + format(high_regime.iloc[0]["dm_twcrps_p_vs_A"], ".4f"))


def main():
    """
    Runs the local-only backtest end to end and writes results under
    local_tests/output.

    INPUTS:
        * none

    OUTPUTS:
        * printed tables, per-week and summary files, and a PNG under local_tests/output
    """
    OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)

    frozen_model = load_frozen_pca_model(pca_model_path = PCA_MODEL_PATH)
    feature_frame = pd.read_parquet(FEATURE_MATRIX_PATH).set_index(DATE_COLUMN_NAME).sort_index()
    weekly_level_frame = pd.read_parquet(WEEKLY_LEVELS_PATH).set_index(DATE_COLUMN_NAME).sort_index()

    scores_array = project_features_onto_frozen_axes(feature_frame = feature_frame, frozen_model = frozen_model)
    market_variance, regime_label, variance_source_used = build_market_variance_and_regime(
        weekly_level_frame = weekly_level_frame, fit_end_date = frozen_model["fit_end_date"]
    )

    per_week_frame = run_walk_forward_backtest(
        feature_frame = feature_frame,
        scores_array = scores_array,
        market_variance = market_variance,
        regime_label = regime_label,
        fit_end_date = frozen_model["fit_end_date"],
    )
    if len(per_week_frame) == 0:
        raise RuntimeError("No test weeks were evaluable. Check the frozen model and feature matrix.")

    score_summary = build_score_summary(per_week_frame = per_week_frame)
    var_summary = build_var_summary(per_week_frame = per_week_frame)

    per_week_frame.to_parquet(PER_WEEK_PARQUET_PATH, index = False)
    per_week_frame.to_csv(PER_WEEK_CSV_PATH, index = False)
    score_summary.to_csv(SCORE_SUMMARY_CSV_PATH, index = False)
    var_summary.to_csv(VAR_SUMMARY_CSV_PATH, index = False)
    save_c_minus_a_plot(per_week_frame = per_week_frame)

    print_report(
        score_summary = score_summary,
        var_summary = var_summary,
        per_week_frame = per_week_frame,
        variance_source_used = variance_source_used,
    )
    print()
    print("Written " + str(PER_WEEK_PARQUET_PATH))
    print("Written " + str(SCORE_SUMMARY_CSV_PATH))
    print("Written " + str(VAR_SUMMARY_CSV_PATH))
    print("Written " + str(PLOT_PATH))


if __name__ == "__main__":
    main()
