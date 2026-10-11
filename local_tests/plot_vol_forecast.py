"""
plot_vol_forecast.py is a research plot, kept apart from the main pipeline.
It draws realised vol against the EWMA baseline and the
ENG1+EWMA blend for one pair over the out-of-sample period, so the difference is
visible on one axis. It also prints the exact forecast-error numbers (QLIKE and
log-vol RMSE, EWMA versus blend) so they can be quoted directly.

Reads the per-week forecasts written by vol_forecast_backtest.py.

Run from the repository root:
    python local_tests/plot_vol_forecast.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kurtosis_impact_backtest as engine

PER_WEEK_PATH = engine.OUTPUT_DIRECTORY / "vol_forecast_per_week.parquet"
PLOT_PAIR = "jpy_usd"
BLEND_WEIGHT = 0.5
WEEKS_PER_YEAR = 52
HORIZON_WEEKS = 4

PLOT_PATH = engine.OUTPUT_DIRECTORY / ("vol_forecast_lines_" + PLOT_PAIR + ".png")
PLOT_WIDTH_INCHES = 13
PLOT_HEIGHT_INCHES = 6
PLOT_DPI = 150
REALISED_COLOUR = "blue"
EWMA_COLOUR = "red"
BLEND_COLOUR = "green"
HIGH_VOL_SHADE_COLOUR = "grey"
HIGH_VOL_SHADE_ALPHA = 0.15


def annualised_vol_percent(variance_series = None):
    """
    Turns a 4-week variance into an annualised volatility in percent, so the plot
    reads in familiar units.

    INPUTS:
        * variance_series - a 4-week return variance

    OUTPUTS:
        * the annualised volatility in percent
    """
    return np.sqrt(variance_series * WEEKS_PER_YEAR / HORIZON_WEEKS) * 100.0


def qlike_loss(actual_variance = None, forecast_variance = None):
    """
    QLIKE loss, zero when the forecast equals the realised variance.

    INPUTS:
        * actual_variance - realised variance array
        * forecast_variance - forecast variance array

    OUTPUTS:
        * the per-observation QLIKE loss
    """
    ratio = actual_variance / forecast_variance
    return ratio - np.log(ratio) - 1.0


def log_vol_rmse(actual_variance = None, forecast_variance = None):
    """
    Root mean squared error of the log volatility, a scale-free error measure.

    INPUTS:
        * actual_variance - realised variance array
        * forecast_variance - forecast variance array

    OUTPUTS:
        * the log-vol RMSE
    """
    return float(np.sqrt(np.mean((0.5 * np.log(forecast_variance) - 0.5 * np.log(actual_variance)) ** 2)))


def print_error_metrics(pair_frame = None):
    """
    Prints EWMA versus blend forecast error for the whole period and the high-vol
    regime, with the percentage reduction and the Diebold-Mariano p-value.

    INPUTS:
        * pair_frame - the per-week frame for one pair, with a blend column

    OUTPUTS:
        * printed metrics
    """
    for regime_name, regime_frame in [("overall", pair_frame), ("high", pair_frame[pair_frame["regime"] == "high"])]:
        actual = regime_frame["actual_variance"].to_numpy()
        ewma = regime_frame["EWMA"].to_numpy()
        blend = regime_frame["blend"].to_numpy()

        ewma_qlike = float(np.mean(qlike_loss(actual, ewma)))
        blend_qlike = float(np.mean(qlike_loss(actual, blend)))
        ewma_rmse = log_vol_rmse(actual, ewma)
        blend_rmse = log_vol_rmse(actual, blend)
        _, _, dm_p = engine.diebold_mariano_pvalue(qlike_loss(actual, blend) - qlike_loss(actual, ewma))

        qlike_reduction = 100.0 * (ewma_qlike - blend_qlike) / ewma_qlike
        rmse_reduction = 100.0 * (ewma_rmse - blend_rmse) / ewma_rmse
        print(PLOT_PAIR + " " + regime_name + " (" + str(len(regime_frame)) + " weeks):")
        print("    QLIKE        EWMA " + format(ewma_qlike, ".4f") + "  blend " + format(blend_qlike, ".4f")
              + "  reduction " + format(qlike_reduction, ".1f") + "%")
        print("    log-vol RMSE EWMA " + format(ewma_rmse, ".4f") + "  blend " + format(blend_rmse, ".4f")
              + "  reduction " + format(rmse_reduction, ".1f") + "%")
        print("    Diebold-Mariano p (blend vs EWMA) " + format(dm_p, ".4f"))


def shade_high_vol_regime(plot_axis = None, pair_frame = None):
    """
    Lightly shades the weeks classified as high-vol regime, so the forecast
    differences can be read against the regime.

    INPUTS:
        * plot_axis - the axis to shade
        * pair_frame - the per-week frame, time ordered

    OUTPUTS:
        * shading drawn on the axis
    """
    is_high = (pair_frame["regime"] == "high").to_numpy()
    weeks = pair_frame["week"].to_numpy()
    span_start = None
    for position in range(len(weeks)):
        if is_high[position] and span_start is None:
            span_start = weeks[position]
        if (not is_high[position] or position == len(weeks) - 1) and span_start is not None:
            span_end = weeks[position]
            plot_axis.axvspan(span_start, span_end, color = HIGH_VOL_SHADE_COLOUR, alpha = HIGH_VOL_SHADE_ALPHA)
            span_start = None


def main():
    """
    Loads the per-week forecasts, builds the blend, plots the three lines and
    prints the error metrics.

    INPUTS:
        * none

    OUTPUTS:
        * a PNG under local_tests/output and printed metrics
    """
    if not PER_WEEK_PATH.exists():
        raise RuntimeError("Missing " + str(PER_WEEK_PATH) + ". Run vol_forecast_backtest.py first.")

    per_week_frame = pd.read_parquet(PER_WEEK_PATH)
    pair_frame = per_week_frame[per_week_frame["pair"] == PLOT_PAIR].sort_values("week").reset_index(drop = True)
    if len(pair_frame) == 0:
        raise RuntimeError("No rows for pair " + PLOT_PAIR + " in " + str(PER_WEEK_PATH))
    pair_frame["blend"] = (1.0 - BLEND_WEIGHT) * pair_frame["EWMA"] + BLEND_WEIGHT * pair_frame["ENG1"]

    figure, plot_axis = plt.subplots(figsize = (PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))
    shade_high_vol_regime(plot_axis = plot_axis, pair_frame = pair_frame)

    plot_axis.plot(pair_frame["week"], annualised_vol_percent(pair_frame["actual_variance"]),
                   color = REALISED_COLOUR, linewidth = 2.0, label = "realised vol")
    plot_axis.plot(pair_frame["week"], annualised_vol_percent(pair_frame["EWMA"]),
                   color = EWMA_COLOUR, linewidth = 1.3, label = "EWMA forecast")
    plot_axis.plot(pair_frame["week"], annualised_vol_percent(pair_frame["blend"]),
                   color = BLEND_COLOUR, linewidth = 1.3, label = "ENG1 + EWMA blend forecast")

    plot_axis.set_ylabel("annualised volatility percent")
    plot_axis.set_xlabel("week")
    plot_axis.set_title(PLOT_PAIR + " forward 4-week volatility: realised versus EWMA versus regime blend, grey shading marks high-vol regime")
    plot_axis.legend()
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi = PLOT_DPI)
    plt.close(figure)

    print_error_metrics(pair_frame = pair_frame)
    print()
    print("Written " + str(PLOT_PATH))


if __name__ == "__main__":
    main()
