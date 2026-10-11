"""
vol_blend_and_signal.py is a research test, kept apart from the main pipeline.
It builds on the volatility-forecast backtest and runs two
follow-ups off one enriched walk-forward:

    BLEND: does mixing the regime forecast ENG1 into the EWMA baseline beat EWMA
    alone. ENG1 adds level calibration in stress, EWMA adds week-to-week timing,
    so a blend might beat either. A fixed 0.5 blend is a legitimate, non-fitted
    forecast; the in-sample best weight is reported as an upper bound only.

    SIGNAL: does the bootstrap neighbourhood's shape (excess kurtosis, skew),
    known at the prediction week, predict the SIZE of the EWMA vol-forecast
    error over the coming 4 weeks. If so, the shape is a warning gauge of when
    even a good vol forecast is about to be unreliable, controlling for the
    vol level so the result is not just "high vol is harder".

Run from the repository root:
    python local_tests/vol_blend_and_signal.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, rankdata, skew

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kurtosis_impact_backtest as engine
import vol_forecast_backtest as volbt

BLEND_WEIGHT_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)
FIXED_BLEND_WEIGHT = 0.5
BLOCK_BOOTSTRAP_DRAWS = 1000
BLOCK_BOOTSTRAP_BLOCK_WEEKS = 8
RANDOM_SEED = 42
REPORTED_REGIMES = ["overall", "high"]

BLEND_SUMMARY_PATH = engine.OUTPUT_DIRECTORY / "vol_blend_summary.csv"
SIGNAL_SUMMARY_PATH = engine.OUTPUT_DIRECTORY / "kurtosis_signal_summary.csv"


def run_enriched_walk_forward(pair = None, pair_frame = None, scores_array = None, weekly_index = None, fit_end_date = None):
    """
    Walks forward as the vol backtest does, but also records the neighbourhood's
    forward-return skew and excess kurtosis, both known at the prediction week.

    INPUTS:
        * pair - the pair name
        * pair_frame - from volbt.build_pair_series
        * scores_array - the frozen PC scores
        * weekly_index - the weekly decision grid
        * fit_end_date - the frozen fit window end

    OUTPUTS:
        * a per-week dataframe with forecasts and neighbourhood shape
    """
    positions = np.arange(len(weekly_index))
    forward_return = pair_frame["forward_return"].to_numpy()
    forward_variance = pair_frame["forward_realised_variance"].to_numpy()
    trailing_variance = pair_frame["trailing_realised_variance"].to_numpy()
    ewma_variance = pair_frame["ewma_horizon_variance"].to_numpy()
    regime = pair_frame["regime"].to_numpy(dtype = object)
    has_forward = np.isfinite(forward_return) & np.isfinite(forward_variance) & (forward_variance > 0)

    records = []
    for test_position in positions:
        if weekly_index[test_position] <= fit_end_date:
            continue
        if not has_forward[test_position] or regime[test_position] is None:
            continue
        if not np.isfinite(trailing_variance[test_position]) or trailing_variance[test_position] <= 0:
            continue
        if not np.isfinite(ewma_variance[test_position]) or ewma_variance[test_position] <= 0:
            continue
        eligible = has_forward & (positions + engine.HORIZON_WEEKS <= test_position)
        if eligible.sum() < engine.NEIGHBOUR_COUNT:
            continue

        distances = np.linalg.norm(scores_array[eligible] - scores_array[test_position], axis = 1)
        eligible_positions = positions[eligible]
        nearest_positions = eligible_positions[np.argsort(distances)[:engine.NEIGHBOUR_COUNT]]
        neighbour_returns = forward_return[nearest_positions]

        records.append({
            "pair": pair,
            "week": weekly_index[test_position],
            "regime": regime[test_position],
            "actual_variance": forward_variance[test_position],
            "trailing_variance": trailing_variance[test_position],
            "ewma_variance": ewma_variance[test_position],
            "eng1_variance": float(np.mean(forward_variance[nearest_positions])),
            "neighbour_skew": float(skew(neighbour_returns)),
            "neighbour_excess_kurtosis": float(kurtosis(neighbour_returns, fisher = True)),
        })
    return pd.DataFrame(records)


def blend_summary_rows(per_week_frame = None):
    """
    For each regime, compares EWMA, ENG1, a fixed 0.5 blend and the in-sample
    best blend, and tests the fixed blend against EWMA with Diebold-Mariano.

    INPUTS:
        * per_week_frame - one pair's enriched per-week frame

    OUTPUTS:
        * a list of blend summary rows
    """
    rows = []
    regime_groups = [("overall", per_week_frame)] + [
        (label, per_week_frame[per_week_frame["regime"] == label]) for label in ["low", "mid", "high"]
    ]
    for regime_name, regime_frame in regime_groups:
        if regime_name not in REPORTED_REGIMES or len(regime_frame) < 3:
            continue
        actual = regime_frame["actual_variance"].to_numpy()
        ewma = regime_frame["ewma_variance"].to_numpy()
        eng1 = regime_frame["eng1_variance"].to_numpy()

        qlike_ewma = float(np.mean(volbt.qlike_loss(actual, ewma)))
        qlike_eng1 = float(np.mean(volbt.qlike_loss(actual, eng1)))

        best_weight = 0.0
        best_qlike = qlike_ewma
        for weight in BLEND_WEIGHT_GRID:
            blend = (1.0 - weight) * ewma + weight * eng1
            blend_qlike = float(np.mean(volbt.qlike_loss(actual, blend)))
            if blend_qlike < best_qlike:
                best_qlike = blend_qlike
                best_weight = weight

        fixed_blend = (1.0 - FIXED_BLEND_WEIGHT) * ewma + FIXED_BLEND_WEIGHT * eng1
        fixed_blend_loss = volbt.qlike_loss(actual, fixed_blend)
        ewma_loss = volbt.qlike_loss(actual, ewma)
        _, _, dm_p = engine.diebold_mariano_pvalue(fixed_blend_loss - ewma_loss)

        rows.append({
            "pair": regime_frame["pair"].iloc[0],
            "regime": regime_name,
            "weeks": len(regime_frame),
            "qlike_EWMA": qlike_ewma,
            "qlike_ENG1": qlike_eng1,
            "qlike_blend_0.5": float(np.mean(fixed_blend_loss)),
            "qlike_blend_best": best_qlike,
            "best_weight": best_weight,
            "dm_p_blend0.5_vs_EWMA": dm_p,
        })
    return rows


def spearman_correlation(first_values = None, second_values = None):
    """
    Spearman correlation via Pearson on the ranks.

    INPUTS:
        * first_values - one array
        * second_values - the matching array

    OUTPUTS:
        * the Spearman correlation
    """
    return float(np.corrcoef(rankdata(first_values), rankdata(second_values))[0, 1])


def partial_spearman_correlation(predictor = None, outcome = None, control = None):
    """
    Spearman correlation of predictor and outcome after removing the control,
    using the standard partial-correlation formula on rank correlations.

    INPUTS:
        * predictor - the predictor array
        * outcome - the outcome array
        * control - the variable to control for

    OUTPUTS:
        * the partial Spearman correlation
    """
    r_po = spearman_correlation(predictor, outcome)
    r_pc = spearman_correlation(predictor, control)
    r_oc = spearman_correlation(outcome, control)
    denominator = np.sqrt((1 - r_pc ** 2) * (1 - r_oc ** 2))
    if denominator <= 0:
        return float("nan")
    return float((r_po - r_pc * r_oc) / denominator)


def block_bootstrap_correlation_interval(predictor = None, outcome = None, control = None, random_generator = None):
    """
    Block-bootstrap 95% interval for the (partial) Spearman correlation, so the
    overlapping 4-week horizons do not overstate the significance.

    INPUTS:
        * predictor, outcome - the two series in week order
        * control - the control series, or None for the raw correlation
        * random_generator - a numpy generator

    OUTPUTS:
        * the 2.5th and 97.5th percentile of the resampled correlation
    """
    observation_count = len(predictor)
    block_count = int(np.ceil(observation_count / BLOCK_BOOTSTRAP_BLOCK_WEEKS))
    estimates = []
    for _ in range(BLOCK_BOOTSTRAP_DRAWS):
        picks = []
        for _ in range(block_count):
            start = random_generator.integers(0, observation_count)
            picks.extend(range(start, start + BLOCK_BOOTSTRAP_BLOCK_WEEKS))
        picks = [position % observation_count for position in picks][:observation_count]
        if control is None:
            estimates.append(spearman_correlation(predictor[picks], outcome[picks]))
        else:
            estimates.append(partial_spearman_correlation(predictor[picks], outcome[picks], control[picks]))
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def signal_summary_rows(per_week_frame = None):
    """
    Tests whether the neighbourhood's excess kurtosis and skew predict the size
    of the EWMA vol-forecast error, controlling for the vol level.

    INPUTS:
        * per_week_frame - one pair's enriched per-week frame

    OUTPUTS:
        * a list of signal summary rows
    """
    random_generator = np.random.default_rng(RANDOM_SEED)
    rows = []
    regime_groups = [("overall", per_week_frame)] + [
        (label, per_week_frame[per_week_frame["regime"] == label]) for label in ["low", "mid", "high"]
    ]
    for regime_name, regime_frame in regime_groups:
        if regime_name not in REPORTED_REGIMES or len(regime_frame) < 10:
            continue
        ordered_frame = regime_frame.sort_values("week")
        actual_vol = np.sqrt(ordered_frame["actual_variance"].to_numpy())
        ewma_vol = np.sqrt(ordered_frame["ewma_variance"].to_numpy())
        trailing_vol = np.sqrt(ordered_frame["trailing_variance"].to_numpy())
        absolute_relative_error = np.abs(actual_vol - ewma_vol) / ewma_vol
        excess_kurtosis = ordered_frame["neighbour_excess_kurtosis"].to_numpy()
        neighbour_skew = ordered_frame["neighbour_skew"].to_numpy()

        kurtosis_partial = partial_spearman_correlation(excess_kurtosis, absolute_relative_error, trailing_vol)
        kurtosis_partial_low, kurtosis_partial_high = block_bootstrap_correlation_interval(
            excess_kurtosis, absolute_relative_error, trailing_vol, random_generator
        )
        skew_partial = partial_spearman_correlation(neighbour_skew, absolute_relative_error, trailing_vol)
        skew_partial_low, skew_partial_high = block_bootstrap_correlation_interval(
            neighbour_skew, absolute_relative_error, trailing_vol, random_generator
        )
        rows.append({
            "pair": ordered_frame["pair"].iloc[0],
            "regime": regime_name,
            "weeks": len(ordered_frame),
            "kurtosis_vs_error_raw": spearman_correlation(excess_kurtosis, absolute_relative_error),
            "kurtosis_vs_error_partial": kurtosis_partial,
            "kurtosis_partial_ci_low": kurtosis_partial_low,
            "kurtosis_partial_ci_high": kurtosis_partial_high,
            "skew_vs_error_partial": skew_partial,
            "skew_partial_ci_low": skew_partial_low,
            "skew_partial_ci_high": skew_partial_high,
        })
    return rows


def main():
    """
    Runs the enriched walk-forward across pairs, then the blend and the signal
    analyses. Prints both tables and saves them locally.

    INPUTS:
        * none

    OUTPUTS:
        * printed tables and saved csv under local_tests/output
    """
    engine.OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)

    frozen_model = engine.load_frozen_pca_model(pca_model_path = volbt.PCA_MODEL_PATH)
    feature_frame = pd.read_parquet(volbt.FEATURE_MATRIX_PATH).set_index(volbt.DATE_COLUMN_NAME).sort_index()
    scores_array = engine.project_features_onto_frozen_axes(feature_frame = feature_frame, frozen_model = frozen_model)
    weekly_index = feature_frame.index
    fit_end_date = frozen_model["fit_end_date"]

    weekly_frames = {
        "macro": pd.read_parquet(volbt.WEEKLY_LEVELS_PATH).set_index(volbt.DATE_COLUMN_NAME).sort_index(),
        "fx": pd.read_parquet(volbt.FX_WEEKLY_LEVELS_PATH).set_index(volbt.DATE_COLUMN_NAME).sort_index(),
    }
    daily_frames = {
        "macro": pd.read_parquet(volbt.DAILY_RAW_PATH),
        "fx": pd.read_parquet(volbt.FX_DAILY_RAW_PATH),
    }

    blend_rows = []
    signal_rows = []
    for pair_configuration in volbt.PAIR_CONFIGURATIONS:
        pair_frame = volbt.build_pair_series(
            pair_configuration = pair_configuration,
            weekly_index = weekly_index,
            weekly_frames = weekly_frames,
            daily_frames = daily_frames,
            fit_end_date = fit_end_date,
        )
        enriched_frame = run_enriched_walk_forward(
            pair = pair_configuration["pair"],
            pair_frame = pair_frame,
            scores_array = scores_array,
            weekly_index = weekly_index,
            fit_end_date = fit_end_date,
        )
        if len(enriched_frame) == 0:
            continue
        blend_rows.extend(blend_summary_rows(per_week_frame = enriched_frame))
        signal_rows.extend(signal_summary_rows(per_week_frame = enriched_frame))

    blend_frame = pd.DataFrame(blend_rows)
    signal_frame = pd.DataFrame(signal_rows)
    blend_frame.to_csv(BLEND_SUMMARY_PATH, index = False)
    signal_frame.to_csv(SIGNAL_SUMMARY_PATH, index = False)

    print("BLEND: EWMA vs ENG1 vs a fixed 0.5 blend vs the in-sample best blend.")
    print("Lower QLIKE is better. best_weight is the in-sample optimum, an upper bound.")
    print("dm_p tests the fixed 0.5 blend against EWMA.")
    print(blend_frame.to_string(index = False, float_format = lambda value: format(value, ".4f")))
    print()
    print("SIGNAL: does neighbourhood shape predict the size of the EWMA vol error,")
    print("controlling for the vol level. Positive partial correlation with a 95%")
    print("interval clear of zero means the shape is a genuine warning gauge.")
    print(signal_frame.to_string(index = False, float_format = lambda value: format(value, ".3f")))
    print()
    print("Written " + str(BLEND_SUMMARY_PATH))
    print("Written " + str(SIGNAL_SUMMARY_PATH))


if __name__ == "__main__":
    main()
