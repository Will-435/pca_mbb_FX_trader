"""
garch_comparison.py tests the regime volatility forecast (ENG1) against a
GARCH(1,1) forecast of the next four weeks of volatility.

GARCH is the usual benchmark for currency volatility. It predicts tomorrow's
variance from a long run level, yesterday's squared move and yesterday's
variance, then lets the forecast drift back to the long run level. ENG1 instead
averages the volatility that followed the forty most similar past macro states.

Scoring matches the EWMA comparison: QLIKE, log vol error, a Mincer-Zarnowitz
fit and a Diebold-Mariano test, per pair and per volatility regime.

Research file, kept apart from the main pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

# This file sits one folder below local_tests, so that folder is added to the
# import path to reuse the ENG1 walk forward exactly as the EWMA test ran it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kurtosis_impact_backtest as engine
import vol_forecast_backtest as volbt


OUTPUT_DIRECTORY = Path("local_tests/garch_baseline/garch_comparison_output")
PER_WEEK_OUTPUT_PATH = OUTPUT_DIRECTORY / "garch_comparison_per_week.parquet"
SUMMARY_PARQUET_PATH = OUTPUT_DIRECTORY / "garch_comparison_summary.parquet"
SUMMARY_CSV_PATH = OUTPUT_DIRECTORY / "garch_comparison_summary.csv"

# "expanding" refits GARCH every few weeks on every daily return seen so far.
# "frozen" fits it once on the same 2006 to 2022 window as the PCA.
GARCH_FIT_SCHEME = "expanding"
REFIT_EVERY_WEEKS = 4
FIT_SCHEME_DESCRIPTIONS = {
    "expanding": "expanding, refitted every 4 weeks on daily returns since the PCA window start",
    "frozen": "frozen, fitted once on the PCA window",
}

# Returns are scaled to percent so the optimiser works on sensible sized numbers.
RETURN_SCALE = 100.0
START_ALPHA = 0.05
START_BETA = 0.90
PARAMETER_FLOOR = 1e-6
PARAMETER_CEILING = 1.0
PERSISTENCE_CEILING = 0.9999
OPTIMISER_MAX_ITERATIONS = 200
LOG_TWO_PI = np.log(2.0 * np.pi)
TRADING_DAYS_PER_HORIZON = volbt.TRADING_DAYS_PER_HORIZON

BLEND_WEIGHT = 0.5
PERCENT_SCALE = 100.0
BASELINE_LABEL = "GARCH"
FORECAST_LABELS = ["GARCH", "EWMA", "ENG1", "ENG1_GARCH_BLEND"]
REGIME_ORDER = ["low", "medium", "high", "overall"]
REGIME_RENAMES = {"mid": "medium"}
PRINT_FORMAT = ".4f"


def filter_garch_variance(scaled_returns = None, omega = None, alpha = None, beta = None, starting_variance = None):
    """
    Runs the GARCH(1,1) recursion through the return history. Each day's
    variance is a long run constant, plus a share of yesterday's squared move,
    plus a share of yesterday's variance.

    INPUTS:
        * scaled_returns - daily returns in percent
        * omega, alpha, beta - the GARCH settings
        * starting_variance - the variance assumed for the first day

    OUTPUTS:
        * each day's variance, built only from earlier days
    """
    # The recursion h[t] = omega + alpha * r[t-1]^2 + beta * h[t-1] is a simple
    # linear filter, so scipy runs it in one fast pass instead of a Python loop.
    filter_input = np.empty(len(scaled_returns))
    filter_input[0] = starting_variance
    filter_input[1:] = omega + alpha * scaled_returns[:-1] ** 2
    return lfilter([1.0], [1.0, -beta], filter_input)


def garch_negative_log_likelihood(parameters = None, scaled_returns = None, starting_variance = None):
    """
    How unlikely the returns look under a given set of GARCH settings. The fit
    searches for the settings that make this as small as possible.

    INPUTS:
        * parameters - omega, alpha and beta
        * scaled_returns - daily returns in percent
        * starting_variance - the variance assumed for the first day

    OUTPUTS:
        * the negative normal log likelihood
    """
    omega, alpha, beta = parameters
    variance = filter_garch_variance(
        scaled_returns = scaled_returns, omega = omega, alpha = alpha, beta = beta, starting_variance = starting_variance
    )
    # Normal log likelihood of each day's return, summed over days.
    return 0.5 * np.sum(LOG_TWO_PI + np.log(variance) + scaled_returns ** 2 / variance)


def persistence_headroom(parameters = None):
    """
    How far alpha plus beta sits below one. Keeping this positive means the
    forecast always drifts back to a finite long run level.

    INPUTS:
        * parameters - omega, alpha and beta

    OUTPUTS:
        * the gap below the persistence ceiling
    """
    omega, alpha, beta = parameters
    return PERSISTENCE_CEILING - alpha - beta


def fit_garch_parameters(scaled_returns = None):
    """
    Picks the GARCH settings that make the past returns most likely, treating
    each day as a normal draw with the variance the model expected that day.

    INPUTS:
        * scaled_returns - daily returns in percent

    OUTPUTS:
        * a dict of omega, alpha, beta and the starting variance
    """
    sample_variance = float(np.var(scaled_returns))
    starting_guess = [sample_variance * (1.0 - START_ALPHA - START_BETA), START_ALPHA, START_BETA]
    parameter_bounds = [
        (PARAMETER_FLOOR, None),
        (PARAMETER_FLOOR, PARAMETER_CEILING),
        (PARAMETER_FLOOR, PARAMETER_CEILING),
    ]
    fit_result = minimize(
        garch_negative_log_likelihood,
        x0 = starting_guess,
        args = (scaled_returns, sample_variance),
        method = "SLSQP",
        bounds = parameter_bounds,
        constraints = [{"type": "ineq", "fun": persistence_headroom}],
        options = {"maxiter": OPTIMISER_MAX_ITERATIONS},
    )
    fitted_values = fit_result.x if np.all(np.isfinite(fit_result.x)) else np.array(starting_guess)
    omega, alpha, beta = fitted_values
    return {"omega": omega, "alpha": alpha, "beta": beta, "starting_variance": sample_variance}


def four_week_variance_forecasts(daily_returns = None, garch_parameters = None):
    """
    For every day, the GARCH forecast of total variance over the next twenty
    trading days. Tomorrow comes from the recursion, then each later day drifts
    back towards the long run level at a pace set by alpha plus beta.

    INPUTS:
        * daily_returns - daily log returns indexed by date
        * garch_parameters - from fit_garch_parameters

    OUTPUTS:
        * the four week variance forecast for each day, in raw return units
    """
    omega = garch_parameters["omega"]
    alpha = garch_parameters["alpha"]
    beta = garch_parameters["beta"]
    scaled_returns = daily_returns.to_numpy() * RETURN_SCALE

    variance = filter_garch_variance(
        scaled_returns = scaled_returns,
        omega = omega,
        alpha = alpha,
        beta = beta,
        starting_variance = garch_parameters["starting_variance"],
    )
    next_day_variance = omega + alpha * scaled_returns ** 2 + beta * variance

    persistence = alpha + beta
    long_run_variance = omega / (1.0 - persistence)
    decay_total = (1.0 - persistence ** TRADING_DAYS_PER_HORIZON) / (1.0 - persistence)
    horizon_variance = TRADING_DAYS_PER_HORIZON * long_run_variance + (next_day_variance - long_run_variance) * decay_total
    return pd.Series(horizon_variance / RETURN_SCALE ** 2, index = daily_returns.index)


def build_garch_forecasts_for_pair(daily_returns = None, test_weeks = None, estimation_start_date = None, fit_end_date = None, garch_fit_scheme = None):
    """
    Produces the GARCH four week forecast for each test week. Every fit only
    uses returns up to the week it serves, so nothing leaks from the future.

    INPUTS:
        * daily_returns - daily log returns indexed by date
        * test_weeks - sorted test week dates
        * estimation_start_date - first day of returns the fit may use
        * fit_end_date - end of the PCA window, used by the frozen scheme
        * garch_fit_scheme - "expanding" or "frozen"

    OUTPUTS:
        * a series of GARCH variance forecasts indexed by test week
    """
    estimation_returns = daily_returns[daily_returns.index >= estimation_start_date]
    forecast_by_week = {}

    if garch_fit_scheme == "frozen":
        fit_sample = estimation_returns[estimation_returns.index <= fit_end_date]
        garch_parameters = fit_garch_parameters(scaled_returns = fit_sample.to_numpy() * RETURN_SCALE)
        horizon_variance = four_week_variance_forecasts(daily_returns = estimation_returns, garch_parameters = garch_parameters)
        for test_week in test_weeks:
            forecast_by_week[test_week] = horizon_variance.asof(test_week)
        return pd.Series(forecast_by_week)

    # Expanding: refit at the start of each block of weeks, then reuse those
    # settings for the rest of the block.
    for block_start in range(0, len(test_weeks), REFIT_EVERY_WEEKS):
        refit_week = test_weeks[block_start]
        fit_sample = estimation_returns[estimation_returns.index <= refit_week]
        garch_parameters = fit_garch_parameters(scaled_returns = fit_sample.to_numpy() * RETURN_SCALE)
        horizon_variance = four_week_variance_forecasts(daily_returns = estimation_returns, garch_parameters = garch_parameters)
        for test_week in test_weeks[block_start:block_start + REFIT_EVERY_WEEKS]:
            forecast_by_week[test_week] = horizon_variance.asof(test_week)
    return pd.Series(forecast_by_week)


def summarise_against_garch(per_week_frame = None):
    """
    Scores every forecast against GARCH for one pair, per regime and overall,
    using the same measures as the EWMA comparison.

    INPUTS:
        * per_week_frame - one pair's weekly forecasts

    OUTPUTS:
        * a list of summary rows
    """
    summary_rows = []
    for regime_name in REGIME_ORDER:
        if regime_name == "overall":
            regime_frame = per_week_frame
        else:
            regime_frame = per_week_frame[per_week_frame["regime"] == regime_name]
        if len(regime_frame) == 0:
            continue

        ordered_frame = regime_frame.sort_values("week")
        actual_variance = ordered_frame["actual_variance"].to_numpy()
        garch_loss = volbt.qlike_loss(actual_variance, ordered_frame[BASELINE_LABEL].to_numpy())
        garch_mean_loss = float(np.mean(garch_loss))

        for forecast_label in FORECAST_LABELS:
            forecast_variance = ordered_frame[forecast_label].to_numpy()
            forecast_loss = volbt.qlike_loss(actual_variance, forecast_variance)
            mean_loss = float(np.mean(forecast_loss))
            slope, r_squared = volbt.mincer_zarnowitz(actual_variance, forecast_variance)

            dm_p_value = float("nan")
            if forecast_label != BASELINE_LABEL:
                _, _, dm_p_value = engine.diebold_mariano_pvalue(forecast_loss - garch_loss)

            summary_rows.append({
                "pair": ordered_frame["pair"].iloc[0],
                "regime": regime_name,
                "forecast": forecast_label,
                "weeks": len(ordered_frame),
                "mean_qlike": mean_loss,
                "qlike_change_vs_garch_pct": PERCENT_SCALE * (garch_mean_loss - mean_loss) / garch_mean_loss,
                "logvol_rmse": float(np.sqrt(np.mean((np.log(np.sqrt(forecast_variance)) - np.log(np.sqrt(actual_variance))) ** 2))),
                "mz_slope": slope,
                "mz_r_squared": r_squared,
                "dm_p_vs_garch": dm_p_value,
            })
    return summary_rows


def build_comparison_tables(garch_fit_scheme = None):
    """
    Runs the weekly walk forward for every pair, adds the GARCH forecast beside
    ENG1 and EWMA, and scores everything against GARCH.

    INPUTS:
        * garch_fit_scheme - "expanding" or "frozen"

    OUTPUTS:
        * per_week_frame - one row per pair and week, every forecast and its loss
        * summary_frame - scores per pair, regime and forecast
    """
    frozen_model = engine.load_frozen_pca_model(pca_model_path = volbt.PCA_MODEL_PATH)
    feature_frame = pd.read_parquet(volbt.FEATURE_MATRIX_PATH).set_index(volbt.DATE_COLUMN_NAME).sort_index()
    scores_array = engine.project_features_onto_frozen_axes(feature_frame = feature_frame, frozen_model = frozen_model)
    weekly_index = feature_frame.index

    weekly_frames = {
        "macro": pd.read_parquet(volbt.WEEKLY_LEVELS_PATH).set_index(volbt.DATE_COLUMN_NAME).sort_index(),
        "fx": pd.read_parquet(volbt.FX_WEEKLY_LEVELS_PATH).set_index(volbt.DATE_COLUMN_NAME).sort_index(),
    }
    daily_frames = {
        "macro": pd.read_parquet(volbt.DAILY_RAW_PATH),
        "fx": pd.read_parquet(volbt.FX_DAILY_RAW_PATH),
    }

    pair_frames = []
    summary_rows = []
    for pair_configuration in volbt.PAIR_CONFIGURATIONS:
        pair_series_frame = volbt.build_pair_series(
            pair_configuration = pair_configuration,
            weekly_index = weekly_index,
            weekly_frames = weekly_frames,
            daily_frames = daily_frames,
            fit_end_date = frozen_model["fit_end_date"],
        )
        per_week_frame = volbt.run_pair_vol_backtest(
            pair = pair_configuration["pair"],
            pair_frame = pair_series_frame,
            scores_array = scores_array,
            weekly_index = weekly_index,
            fit_end_date = frozen_model["fit_end_date"],
        )
        if len(per_week_frame) == 0:
            continue

        daily_returns = volbt.load_daily_returns(
            daily_frame = daily_frames[pair_configuration["daily_source"]], pair = pair_configuration["pair"]
        )
        garch_forecasts = build_garch_forecasts_for_pair(
            daily_returns = daily_returns,
            test_weeks = per_week_frame["week"].sort_values().tolist(),
            estimation_start_date = frozen_model["fit_start_date"],
            fit_end_date = frozen_model["fit_end_date"],
            garch_fit_scheme = garch_fit_scheme,
        )

        per_week_frame["GARCH"] = per_week_frame["week"].map(garch_forecasts)
        per_week_frame = per_week_frame.dropna(subset = ["GARCH"]).copy()
        per_week_frame["ENG1_GARCH_BLEND"] = (1.0 - BLEND_WEIGHT) * per_week_frame["GARCH"] + BLEND_WEIGHT * per_week_frame["ENG1"]
        per_week_frame["regime"] = per_week_frame["regime"].replace(REGIME_RENAMES)
        for forecast_label in FORECAST_LABELS:
            per_week_frame["qlike_" + forecast_label] = volbt.qlike_loss(
                per_week_frame["actual_variance"].to_numpy(), per_week_frame[forecast_label].to_numpy()
            )

        kept_columns = ["pair", "week", "regime", "actual_variance"] + FORECAST_LABELS + ["qlike_" + label for label in FORECAST_LABELS]
        per_week_frame = per_week_frame[kept_columns].reset_index(drop = True)
        pair_frames.append(per_week_frame)
        summary_rows.extend(summarise_against_garch(per_week_frame = per_week_frame))

    return pd.concat(pair_frames, ignore_index = True), pd.DataFrame(summary_rows)


def print_pair_tables(summary_frame = None, garch_fit_scheme = None):
    """
    Prints one table per pair, split by regime, with GARCH as the yardstick.

    INPUTS:
        * summary_frame - from build_comparison_tables
        * garch_fit_scheme - the scheme used, for the header

    OUTPUTS:
        * printed tables
    """
    print("GARCH fit: " + FIT_SCHEME_DESCRIPTIONS[garch_fit_scheme] + ".")
    print("Lower QLIKE is better. qlike_change_vs_garch_pct above zero means closer than GARCH.")
    print("dm_p_vs_garch is the Diebold-Mariano test against GARCH. A small p with a")
    print("positive change is a real improvement, a small p with a negative change is a real loss.")
    for pair_configuration in volbt.PAIR_CONFIGURATIONS:
        pair_summary = summary_frame[summary_frame["pair"] == pair_configuration["pair"]]
        if len(pair_summary) == 0:
            continue
        print()
        print(pair_configuration["pair"])
        print(pair_summary.drop(columns = ["pair"]).to_string(
            index = False, float_format = lambda value: format(value, PRINT_FORMAT)
        ))


def main():
    """
    Runs the comparison, prints the per pair, per regime tables and saves the
    weekly results and the summary.

    INPUTS:
        * none

    OUTPUTS:
        * printed tables, plus parquet and csv files in garch_comparison_output
    """
    OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)

    per_week_frame, summary_frame = build_comparison_tables(garch_fit_scheme = GARCH_FIT_SCHEME)
    per_week_frame.to_parquet(PER_WEEK_OUTPUT_PATH, index = False)
    summary_frame.to_parquet(SUMMARY_PARQUET_PATH, index = False)
    summary_frame.to_csv(SUMMARY_CSV_PATH, index = False)

    print_pair_tables(summary_frame = summary_frame, garch_fit_scheme = GARCH_FIT_SCHEME)
    print()
    print("Written " + str(PER_WEEK_OUTPUT_PATH))
    print("Written " + str(SUMMARY_PARQUET_PATH))
    print("Written " + str(SUMMARY_CSV_PATH))


if __name__ == "__main__":
    main()
