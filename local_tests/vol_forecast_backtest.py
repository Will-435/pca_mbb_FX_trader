"""
vol_forecast_backtest.py is a research test, kept apart from the main pipeline.
It adapts the earlier local backtest toolkit from forecasting
the forward RETURN to forecasting forward VOLATILITY, because the 4-week drift is
close to unforecastable while volatility clusters and is forecastable.

Target: forward realised variance over the 4-week window, measured from daily
returns (roughly 20 daily observations, far cleaner than 4 weekly points).

Two engine forecasts, both from the same frozen PCA regime neighbourhood (macro
PC nearest 40, point in time):
    ENG1 neighbours' forward realised variance - the mean of the 40 neighbour
         weeks' own forward realised variances. Asks: does the macro regime
         predict forward vol.
    ENG2 bootstrap return spread - the cross-sectional variance of the 40
         neighbours' forward 4-week returns. This is the variance the earlier
         arms B and D used, now judged directly as a vol forecast.

Two baselines:
    RW   random walk - trailing realised variance over the previous 4 weeks.
    EWMA RiskMetrics exponentially weighted daily variance (lambda 0.94), scaled
         to the 4-week horizon.

Scored with the standard vol-forecast tools: QLIKE (robust to realised-vol
noise), log-vol RMSE, a Mincer-Zarnowitz regression (slope and R squared), and a
Diebold-Mariano test of each engine forecast against each baseline, per
volatility regime and overall, across several currency pairs.

Leakage controls carried over: frozen PCA axes, neighbours only from weeks whose
forward window closed on or before the test week, lagged regime labels.

Run from the repository root:
    python local_tests/vol_forecast_backtest.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kurtosis_impact_backtest as engine

WEEKLY_LEVELS_PATH = Path("Data/Input/weekly_levels.parquet")
FX_WEEKLY_LEVELS_PATH = Path("Data/Input/fx_weekly_levels.parquet")
DAILY_RAW_PATH = Path("Data/Input/raw_series.parquet")
FX_DAILY_RAW_PATH = Path("Data/Input/fx_raw_series.parquet")
FEATURE_MATRIX_PATH = Path("Data/Input/feature_matrix.parquet")
PCA_MODEL_PATH = Path("components/pca_output/pca_model.parquet")

DATE_COLUMN_NAME = "week_date"
OBSERVATION_DATE_COLUMN = "observation_date"
SERIES_NAME_COLUMN = "series_name"
VALUE_COLUMN = "value"

EWMA_LAMBDA = 0.94
TRADING_DAYS_PER_HORIZON = 20
MINIMUM_VARIANCE = 1e-12
SINGLE_WEEK_LAG = 1

FORECAST_LABELS = ["RW", "EWMA", "ENG1", "ENG2", "BLEND"]
BASELINE_LABELS = ["RW", "EWMA"]
BLEND_WEIGHT = 0.5
REGIME_LABELS = ["low", "mid", "high"]

# Each pair names its weekly and daily level source.
PAIR_CONFIGURATIONS = [
    {"pair": "usd_krw", "weekly_source": "macro", "daily_source": "macro"},
    {"pair": "eur_usd", "weekly_source": "fx", "daily_source": "fx"},
    {"pair": "gbp_usd", "weekly_source": "fx", "daily_source": "fx"},
    {"pair": "jpy_usd", "weekly_source": "fx", "daily_source": "fx"},
    {"pair": "chf_usd", "weekly_source": "fx", "daily_source": "fx"},
]

OUTPUT_SUMMARY_PATH = engine.OUTPUT_DIRECTORY / "vol_forecast_summary.csv"
OUTPUT_PER_WEEK_PATH = engine.OUTPUT_DIRECTORY / "vol_forecast_per_week.parquet"


def load_daily_returns(daily_frame = None, pair = None):
    """
    Pulls one pair's daily levels from a long raw table and turns them into
    daily log returns.

    INPUTS:
        * daily_frame - the long daily table with series_name, observation_date, value
        * pair - the series name to extract

    OUTPUTS:
        * a daily log return series indexed by date
    """
    pair_levels = daily_frame[daily_frame[SERIES_NAME_COLUMN] == pair]
    pair_levels = pair_levels.set_index(OBSERVATION_DATE_COLUMN)[VALUE_COLUMN].sort_index()
    return np.log(pair_levels / pair_levels.shift(SINGLE_WEEK_LAG)).dropna()


def realised_variance_between(cumulative_squared = None, start_date = None, end_date = None):
    """
    Realised variance over (start_date, end_date] from a cumulative sum of
    squared daily returns.

    INPUTS:
        * cumulative_squared - cumulative sum of squared daily returns, by date
        * start_date - the exclusive lower bound
        * end_date - the inclusive upper bound

    OUTPUTS:
        * the realised variance, or nan if either bound has no data
    """
    end_value = cumulative_squared.asof(end_date)
    start_value = cumulative_squared.asof(start_date)
    if pd.isna(end_value) or pd.isna(start_value):
        return float("nan")
    return float(end_value - start_value)


def build_pair_series(pair_configuration = None, weekly_index = None, weekly_frames = None, daily_frames = None, fit_end_date = None):
    """
    Builds every weekly series a pair needs: the forward return, the forward and
    trailing realised variance, the EWMA variance forecast and the lagged
    volatility regime label, all aligned to the weekly index.

    INPUTS:
        * pair_configuration - one entry from PAIR_CONFIGURATIONS
        * weekly_index - the weekly decision grid
        * weekly_frames - dict of the macro and fx weekly level frames
        * daily_frames - dict of the macro and fx daily raw frames
        * fit_end_date - the frozen fit window end

    OUTPUTS:
        * a dataframe indexed by weekly_index with the columns needed downstream
    """
    pair = pair_configuration["pair"]
    weekly_levels = weekly_frames[pair_configuration["weekly_source"]][pair]
    daily_returns = load_daily_returns(
        daily_frame = daily_frames[pair_configuration["daily_source"]], pair = pair
    )

    forward_return = np.log(weekly_levels.shift(-engine.HORIZON_WEEKS) / weekly_levels)

    cumulative_squared = (daily_returns ** 2).cumsum()
    ewma_daily_variance = (daily_returns ** 2).ewm(alpha = 1 - EWMA_LAMBDA, adjust = False).mean()

    horizon_offset = pd.Timedelta(weeks = engine.HORIZON_WEEKS)
    forward_realised_variance = pd.Series(index = weekly_index, dtype = float)
    trailing_realised_variance = pd.Series(index = weekly_index, dtype = float)
    ewma_horizon_variance = pd.Series(index = weekly_index, dtype = float)
    for week_date in weekly_index:
        forward_realised_variance[week_date] = realised_variance_between(
            cumulative_squared = cumulative_squared, start_date = week_date, end_date = week_date + horizon_offset
        )
        trailing_realised_variance[week_date] = realised_variance_between(
            cumulative_squared = cumulative_squared, start_date = week_date - horizon_offset, end_date = week_date
        )
        daily_variance_at_week = ewma_daily_variance.asof(week_date)
        ewma_horizon_variance[week_date] = daily_variance_at_week * TRADING_DAYS_PER_HORIZON

    regime_measure = np.sqrt(trailing_realised_variance).shift(engine.REGIME_LABEL_LAG_WEEKS)
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

    pair_frame = pd.DataFrame(index = weekly_index)
    pair_frame["forward_return"] = forward_return.reindex(weekly_index)
    pair_frame["forward_realised_variance"] = forward_realised_variance
    pair_frame["trailing_realised_variance"] = trailing_realised_variance
    pair_frame["ewma_horizon_variance"] = ewma_horizon_variance
    pair_frame["regime"] = regime_measure.reindex(weekly_index).apply(label_regime)
    return pair_frame


def run_pair_vol_backtest(pair = None, pair_frame = None, scores_array = None, weekly_index = None, fit_end_date = None):
    """
    Walks forward over the post-fit weeks, forms the two engine vol forecasts and
    the two baselines, and records the realised variance and each forecast.

    INPUTS:
        * pair - the pair name
        * pair_frame - from build_pair_series
        * scores_array - the frozen PC scores, one row per week
        * weekly_index - the weekly decision grid
        * fit_end_date - the frozen fit window end

    OUTPUTS:
        * a per-week dataframe of realised variance and forecasts
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

        engine1_forecast = float(np.mean(forward_variance[nearest_positions]))
        engine2_forecast = float(np.var(forward_return[nearest_positions]))
        blend_forecast = (1.0 - BLEND_WEIGHT) * ewma_variance[test_position] + BLEND_WEIGHT * engine1_forecast

        records.append({
            "week": weekly_index[test_position],
            "regime": regime[test_position],
            "actual_variance": forward_variance[test_position],
            "RW": trailing_variance[test_position],
            "EWMA": ewma_variance[test_position],
            "ENG1": engine1_forecast,
            "ENG2": max(engine2_forecast, MINIMUM_VARIANCE),
            "BLEND": blend_forecast,
        })

    per_week_frame = pd.DataFrame(records)
    per_week_frame.insert(0, "pair", pair)
    return per_week_frame


def qlike_loss(actual_variance = None, forecast_variance = None):
    """
    QLIKE loss, robust to noise in the realised-variance proxy. Zero when the
    forecast equals the realised variance, positive otherwise.

    INPUTS:
        * actual_variance - realised variance array
        * forecast_variance - forecast variance array

    OUTPUTS:
        * the per-observation QLIKE loss array
    """
    ratio = actual_variance / forecast_variance
    return ratio - np.log(ratio) - 1.0


def mincer_zarnowitz(actual_variance = None, forecast_variance = None):
    """
    Regresses realised variance on the forecast. Slope near one and a high R
    squared mean the forecast is well calibrated and informative.

    INPUTS:
        * actual_variance - realised variance array
        * forecast_variance - forecast variance array

    OUTPUTS:
        * the regression slope and the R squared
    """
    if len(actual_variance) < 3 or np.var(forecast_variance) <= 0:
        return float("nan"), float("nan")
    slope = float(np.cov(actual_variance, forecast_variance)[0, 1] / np.var(forecast_variance))
    r_squared = float(np.corrcoef(actual_variance, forecast_variance)[0, 1] ** 2)
    return slope, r_squared


def summarise_pair(per_week_frame = None):
    """
    Builds the per regime, per forecast summary for one pair: mean QLIKE, log-vol
    RMSE, Mincer-Zarnowitz slope and R squared, and Diebold-Mariano p-values of
    each engine forecast against each baseline.

    INPUTS:
        * per_week_frame - one pair's per-week forecasts

    OUTPUTS:
        * a list of summary rows
    """
    summary_rows = []
    regime_groups = [("overall", per_week_frame)] + [
        (label, per_week_frame[per_week_frame["regime"] == label]) for label in REGIME_LABELS
    ]
    for regime_name, regime_frame in regime_groups:
        if len(regime_frame) == 0:
            continue
        ordered_frame = regime_frame.sort_values("week")
        actual_variance = ordered_frame["actual_variance"].to_numpy()
        qlike_by_forecast = {
            forecast_label: qlike_loss(actual_variance, ordered_frame[forecast_label].to_numpy())
            for forecast_label in FORECAST_LABELS
        }
        ewma_mean_qlike = float(np.mean(qlike_by_forecast["EWMA"]))
        for forecast_label in FORECAST_LABELS:
            forecast_variance = ordered_frame[forecast_label].to_numpy()
            slope, r_squared = mincer_zarnowitz(actual_variance, forecast_variance)
            mean_qlike_value = float(np.mean(qlike_by_forecast[forecast_label]))
            row = {
                "pair": ordered_frame["pair"].iloc[0],
                "regime": regime_name,
                "forecast": forecast_label,
                "weeks": len(ordered_frame),
                "mean_qlike": mean_qlike_value,
                "qlike_pct_reduction_vs_EWMA": 100.0 * (ewma_mean_qlike - mean_qlike_value) / ewma_mean_qlike,
                "logvol_rmse": float(np.sqrt(np.mean((0.5 * np.log(forecast_variance) - 0.5 * np.log(actual_variance)) ** 2))),
                "mz_slope": slope,
                "mz_r_squared": r_squared,
                "dm_p_vs_RW": float("nan"),
                "dm_p_vs_EWMA": float("nan"),
            }
            if forecast_label not in BASELINE_LABELS:
                for baseline_label in BASELINE_LABELS:
                    loss_differential = qlike_by_forecast[forecast_label] - qlike_by_forecast[baseline_label]
                    _, _, p_value = engine.diebold_mariano_pvalue(loss_differential)
                    row["dm_p_vs_" + baseline_label] = p_value
            summary_rows.append(row)
    return summary_rows


def main():
    """
    Runs the volatility-forecast backtest across every pair and prints the
    comparison. Writes summary and per-week files under local_tests/output.

    INPUTS:
        * none

    OUTPUTS:
        * printed tables and saved csv / parquet under local_tests/output
    """
    engine.OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)

    frozen_model = engine.load_frozen_pca_model(pca_model_path = PCA_MODEL_PATH)
    feature_frame = pd.read_parquet(FEATURE_MATRIX_PATH).set_index(DATE_COLUMN_NAME).sort_index()
    scores_array = engine.project_features_onto_frozen_axes(feature_frame = feature_frame, frozen_model = frozen_model)
    weekly_index = feature_frame.index
    fit_end_date = frozen_model["fit_end_date"]

    weekly_frames = {
        "macro": pd.read_parquet(WEEKLY_LEVELS_PATH).set_index(DATE_COLUMN_NAME).sort_index(),
        "fx": pd.read_parquet(FX_WEEKLY_LEVELS_PATH).set_index(DATE_COLUMN_NAME).sort_index(),
    }
    daily_frames = {
        "macro": pd.read_parquet(DAILY_RAW_PATH),
        "fx": pd.read_parquet(FX_DAILY_RAW_PATH),
    }

    all_per_week = []
    all_summary_rows = []
    for pair_configuration in PAIR_CONFIGURATIONS:
        pair_frame = build_pair_series(
            pair_configuration = pair_configuration,
            weekly_index = weekly_index,
            weekly_frames = weekly_frames,
            daily_frames = daily_frames,
            fit_end_date = fit_end_date,
        )
        per_week_frame = run_pair_vol_backtest(
            pair = pair_configuration["pair"],
            pair_frame = pair_frame,
            scores_array = scores_array,
            weekly_index = weekly_index,
            fit_end_date = fit_end_date,
        )
        if len(per_week_frame) == 0:
            continue
        all_per_week.append(per_week_frame)
        all_summary_rows.extend(summarise_pair(per_week_frame = per_week_frame))

    summary_frame = pd.DataFrame(all_summary_rows)
    per_week_combined = pd.concat(all_per_week, ignore_index = True)
    summary_frame.to_csv(OUTPUT_SUMMARY_PATH, index = False)
    per_week_combined.to_parquet(OUTPUT_PER_WEEK_PATH, index = False)

    print("Forecasts: RW trailing realised vol, EWMA RiskMetrics, ENG1 neighbours'")
    print("forward realised vol, ENG2 bootstrap return spread. Lower QLIKE is better.")
    print("Higher Mincer-Zarnowitz R squared means the forecast explains more of the")
    print("forward vol variation. DM p tests each engine forecast against a baseline;")
    print("a small p with lower QLIKE means the engine genuinely beats that baseline.")
    print()
    reported = summary_frame[summary_frame["regime"].isin(["overall", "high"])]
    print(reported.to_string(index = False, float_format = lambda value: format(value, ".4f")))
    print()
    print("Written " + str(OUTPUT_SUMMARY_PATH))
    print("Written " + str(OUTPUT_PER_WEEK_PATH))


if __name__ == "__main__":
    main()
