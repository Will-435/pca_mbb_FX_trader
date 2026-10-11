"""
plot_garch_comparison.py draws, for each currency pair, the GARCH(1,1) and ENG1
volatility forecasts against the realised volatility that followed.

Background shading marks the low, medium and high volatility regimes. A lower
panel tracks which forecast was closer over the last quarter, so the hand over
between the two forecasts, and how quickly it happens, is easy to see.

The realised line is the volatility measured over the four weeks after each
date, plotted at the date the forecast was made, so forecast and outcome line up.

Research file, kept apart from the main pipeline. Reads the weekly
results written by garch_comparison.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


INPUT_PATH = Path("local_tests/garch_baseline/garch_comparison_output/garch_comparison_per_week.parquet")
OUTPUT_DIRECTORY = Path("local_tests/garch_baseline/plot_garch_comparison_output")

WEEKS_PER_YEAR = 52
HORIZON_WEEKS = 4
PERCENT_SCALE = 100.0
ROLLING_GAP_WINDOW_WEEKS = 13
ROLLING_GAP_MINIMUM_WEEKS = 4
ONE_WEEK = pd.Timedelta(weeks = 1)

PLOT_WIDTH_INCHES = 13
PLOT_HEIGHT_INCHES = 8
PLOT_DPI = 150
VOL_PANEL_HEIGHT = 3
GAP_PANEL_HEIGHT = 1

REALISED_COLOUR = "blue"
GARCH_COLOUR = "red"
ENG1_COLOUR = "green"
GAP_LINE_COLOUR = "black"
ZERO_LINE_COLOUR = "black"
REALISED_LINE_WIDTH = 2.0
FORECAST_LINE_WIDTH = 1.3
GAP_LINE_WIDTH = 1.0
GAP_FILL_ALPHA = 0.35

# Warmer shading means a more volatile regime.
REGIME_COLOURS = {"low": "#fff3c4", "medium": "#ffd88a", "high": "#ffb35c"}
REGIME_SHADE_ALPHA = 0.6
REGIME_LEGEND_ORDER = ["low", "medium", "high"]

VOL_AXIS_LABEL = "annualised vol, percent"
GAP_AXIS_LABEL = "QLIKE gap, " + str(ROLLING_GAP_WINDOW_WEEKS) + " week mean"
GAP_TITLE_NOTE = "Lower panel above zero means ENG1 was closer, below zero means GARCH was closer"


def annualised_vol_percent(variance_series = None):
    """
    Turns a four week variance into annualised volatility in percent, so the
    lines read in familiar units.

    INPUTS:
        * variance_series - four week return variance

    OUTPUTS:
        * annualised volatility in percent
    """
    return np.sqrt(variance_series * WEEKS_PER_YEAR / HORIZON_WEEKS) * PERCENT_SCALE


def prepare_pair_frame(per_week_frame = None, pair = None):
    """
    Pulls one pair's weeks in date order and adds the plotted columns: the
    three volatility lines and the rolling QLIKE gap between GARCH and ENG1.

    INPUTS:
        * per_week_frame - all pairs' weekly results
        * pair - the pair to plot

    OUTPUTS:
        * one pair's frame, ready to plot
    """
    pair_frame = per_week_frame[per_week_frame["pair"] == pair].sort_values("week").reset_index(drop = True)
    pair_frame["realised_vol"] = annualised_vol_percent(pair_frame["actual_variance"])
    pair_frame["garch_vol"] = annualised_vol_percent(pair_frame["GARCH"])
    pair_frame["eng1_vol"] = annualised_vol_percent(pair_frame["ENG1"])
    # Positive gap means GARCH lost more than ENG1, so ENG1 was the closer forecast.
    weekly_gap = pair_frame["qlike_GARCH"] - pair_frame["qlike_ENG1"]
    pair_frame["rolling_qlike_gap"] = weekly_gap.rolling(
        ROLLING_GAP_WINDOW_WEEKS, min_periods = ROLLING_GAP_MINIMUM_WEEKS
    ).mean()
    return pair_frame


def regime_spans(pair_frame = None):
    """
    Groups consecutive weeks that share a regime into one span, so each stretch
    of calm or stress is shaded as a single block.

    INPUTS:
        * pair_frame - one pair's weeks in date order

    OUTPUTS:
        * a list of (start, end, regime) spans
    """
    spans = []
    span_start = None
    span_regime = None
    previous_week = None
    for week, regime in zip(pair_frame["week"], pair_frame["regime"]):
        if regime != span_regime:
            if span_regime is not None:
                spans.append((span_start, previous_week + ONE_WEEK, span_regime))
            span_start = week
            span_regime = regime
        previous_week = week
    if span_regime is not None:
        spans.append((span_start, previous_week + ONE_WEEK, span_regime))
    return spans


def shade_regimes(plot_axis = None, pair_frame = None):
    """
    Paints the regime spans behind whatever is drawn on the axis.

    INPUTS:
        * plot_axis - the axis to shade
        * pair_frame - one pair's weeks in date order

    OUTPUTS:
        * shading on the axis
    """
    for span_start, span_end, regime in regime_spans(pair_frame = pair_frame):
        plot_axis.axvspan(span_start, span_end, color = REGIME_COLOURS[regime], alpha = REGIME_SHADE_ALPHA, linewidth = 0)


def regime_legend_patches():
    """
    Legend swatches for the three regime colours.

    INPUTS:
        * none

    OUTPUTS:
        * a list of legend patches
    """
    return [
        Patch(facecolor = REGIME_COLOURS[regime], alpha = REGIME_SHADE_ALPHA, label = regime + " vol regime")
        for regime in REGIME_LEGEND_ORDER
    ]


def main():
    """
    Loads the weekly results and saves one two panel figure per pair.

    INPUTS:
        * none

    OUTPUTS:
        * PNG files in plot_garch_comparison_output
    """
    if not INPUT_PATH.exists():
        raise RuntimeError("Missing " + str(INPUT_PATH) + ". Run garch_comparison.py first.")
    OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)
    per_week_frame = pd.read_parquet(INPUT_PATH)

    # USD/KRW: forecasts against realised vol, then which forecast was closer.
    krw_frame = prepare_pair_frame(per_week_frame = per_week_frame, pair = "usd_krw")
    figure, (vol_axis, gap_axis) = plt.subplots(
        nrows = 2, ncols = 1, figsize = (PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES), sharex = True,
        gridspec_kw = {"height_ratios": [VOL_PANEL_HEIGHT, GAP_PANEL_HEIGHT]},
    )
    shade_regimes(plot_axis = vol_axis, pair_frame = krw_frame)
    vol_axis.plot(krw_frame["week"], krw_frame["realised_vol"], color = REALISED_COLOUR, linewidth = REALISED_LINE_WIDTH, label = "rolling realised vol, next 4 weeks")
    vol_axis.plot(krw_frame["week"], krw_frame["garch_vol"], color = GARCH_COLOUR, linewidth = FORECAST_LINE_WIDTH, label = "GARCH(1,1) forecast")
    vol_axis.plot(krw_frame["week"], krw_frame["eng1_vol"], color = ENG1_COLOUR, linewidth = FORECAST_LINE_WIDTH, label = "ENG1 forecast")
    vol_axis.set_ylabel(VOL_AXIS_LABEL)
    vol_axis.legend(handles = vol_axis.get_legend_handles_labels()[0] + regime_legend_patches(), loc = "upper left", ncol = 2)
    shade_regimes(plot_axis = gap_axis, pair_frame = krw_frame)
    gap_axis.plot(krw_frame["week"], krw_frame["rolling_qlike_gap"], color = GAP_LINE_COLOUR, linewidth = GAP_LINE_WIDTH)
    gap_axis.fill_between(krw_frame["week"], krw_frame["rolling_qlike_gap"], 0.0, where = krw_frame["rolling_qlike_gap"] > 0.0, color = ENG1_COLOUR, alpha = GAP_FILL_ALPHA, interpolate = True)
    gap_axis.fill_between(krw_frame["week"], krw_frame["rolling_qlike_gap"], 0.0, where = krw_frame["rolling_qlike_gap"] < 0.0, color = GARCH_COLOUR, alpha = GAP_FILL_ALPHA, interpolate = True)
    gap_axis.axhline(0.0, color = ZERO_LINE_COLOUR, linestyle = "--", linewidth = GAP_LINE_WIDTH)
    gap_axis.set_ylabel(GAP_AXIS_LABEL)
    gap_axis.set_xlabel("week")
    figure.suptitle("usd_krw: GARCH(1,1) and ENG1 against realised vol. " + GAP_TITLE_NOTE)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "usd_krw_garch_vs_eng1.png", dpi = PLOT_DPI)
    plt.close(figure)

    # EUR/USD: forecasts against realised vol, then which forecast was closer.
    eur_frame = prepare_pair_frame(per_week_frame = per_week_frame, pair = "eur_usd")
    figure, (vol_axis, gap_axis) = plt.subplots(
        nrows = 2, ncols = 1, figsize = (PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES), sharex = True,
        gridspec_kw = {"height_ratios": [VOL_PANEL_HEIGHT, GAP_PANEL_HEIGHT]},
    )
    shade_regimes(plot_axis = vol_axis, pair_frame = eur_frame)
    vol_axis.plot(eur_frame["week"], eur_frame["realised_vol"], color = REALISED_COLOUR, linewidth = REALISED_LINE_WIDTH, label = "rolling realised vol, next 4 weeks")
    vol_axis.plot(eur_frame["week"], eur_frame["garch_vol"], color = GARCH_COLOUR, linewidth = FORECAST_LINE_WIDTH, label = "GARCH(1,1) forecast")
    vol_axis.plot(eur_frame["week"], eur_frame["eng1_vol"], color = ENG1_COLOUR, linewidth = FORECAST_LINE_WIDTH, label = "ENG1 forecast")
    vol_axis.set_ylabel(VOL_AXIS_LABEL)
    vol_axis.legend(handles = vol_axis.get_legend_handles_labels()[0] + regime_legend_patches(), loc = "upper left", ncol = 2)
    shade_regimes(plot_axis = gap_axis, pair_frame = eur_frame)
    gap_axis.plot(eur_frame["week"], eur_frame["rolling_qlike_gap"], color = GAP_LINE_COLOUR, linewidth = GAP_LINE_WIDTH)
    gap_axis.fill_between(eur_frame["week"], eur_frame["rolling_qlike_gap"], 0.0, where = eur_frame["rolling_qlike_gap"] > 0.0, color = ENG1_COLOUR, alpha = GAP_FILL_ALPHA, interpolate = True)
    gap_axis.fill_between(eur_frame["week"], eur_frame["rolling_qlike_gap"], 0.0, where = eur_frame["rolling_qlike_gap"] < 0.0, color = GARCH_COLOUR, alpha = GAP_FILL_ALPHA, interpolate = True)
    gap_axis.axhline(0.0, color = ZERO_LINE_COLOUR, linestyle = "--", linewidth = GAP_LINE_WIDTH)
    gap_axis.set_ylabel(GAP_AXIS_LABEL)
    gap_axis.set_xlabel("week")
    figure.suptitle("eur_usd: GARCH(1,1) and ENG1 against realised vol. " + GAP_TITLE_NOTE)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "eur_usd_garch_vs_eng1.png", dpi = PLOT_DPI)
    plt.close(figure)

    # GBP/USD: forecasts against realised vol, then which forecast was closer.
    gbp_frame = prepare_pair_frame(per_week_frame = per_week_frame, pair = "gbp_usd")
    figure, (vol_axis, gap_axis) = plt.subplots(
        nrows = 2, ncols = 1, figsize = (PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES), sharex = True,
        gridspec_kw = {"height_ratios": [VOL_PANEL_HEIGHT, GAP_PANEL_HEIGHT]},
    )
    shade_regimes(plot_axis = vol_axis, pair_frame = gbp_frame)
    vol_axis.plot(gbp_frame["week"], gbp_frame["realised_vol"], color = REALISED_COLOUR, linewidth = REALISED_LINE_WIDTH, label = "rolling realised vol, next 4 weeks")
    vol_axis.plot(gbp_frame["week"], gbp_frame["garch_vol"], color = GARCH_COLOUR, linewidth = FORECAST_LINE_WIDTH, label = "GARCH(1,1) forecast")
    vol_axis.plot(gbp_frame["week"], gbp_frame["eng1_vol"], color = ENG1_COLOUR, linewidth = FORECAST_LINE_WIDTH, label = "ENG1 forecast")
    vol_axis.set_ylabel(VOL_AXIS_LABEL)
    vol_axis.legend(handles = vol_axis.get_legend_handles_labels()[0] + regime_legend_patches(), loc = "upper left", ncol = 2)
    shade_regimes(plot_axis = gap_axis, pair_frame = gbp_frame)
    gap_axis.plot(gbp_frame["week"], gbp_frame["rolling_qlike_gap"], color = GAP_LINE_COLOUR, linewidth = GAP_LINE_WIDTH)
    gap_axis.fill_between(gbp_frame["week"], gbp_frame["rolling_qlike_gap"], 0.0, where = gbp_frame["rolling_qlike_gap"] > 0.0, color = ENG1_COLOUR, alpha = GAP_FILL_ALPHA, interpolate = True)
    gap_axis.fill_between(gbp_frame["week"], gbp_frame["rolling_qlike_gap"], 0.0, where = gbp_frame["rolling_qlike_gap"] < 0.0, color = GARCH_COLOUR, alpha = GAP_FILL_ALPHA, interpolate = True)
    gap_axis.axhline(0.0, color = ZERO_LINE_COLOUR, linestyle = "--", linewidth = GAP_LINE_WIDTH)
    gap_axis.set_ylabel(GAP_AXIS_LABEL)
    gap_axis.set_xlabel("week")
    figure.suptitle("gbp_usd: GARCH(1,1) and ENG1 against realised vol. " + GAP_TITLE_NOTE)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "gbp_usd_garch_vs_eng1.png", dpi = PLOT_DPI)
    plt.close(figure)

    # JPY/USD: forecasts against realised vol, then which forecast was closer.
    jpy_frame = prepare_pair_frame(per_week_frame = per_week_frame, pair = "jpy_usd")
    figure, (vol_axis, gap_axis) = plt.subplots(
        nrows = 2, ncols = 1, figsize = (PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES), sharex = True,
        gridspec_kw = {"height_ratios": [VOL_PANEL_HEIGHT, GAP_PANEL_HEIGHT]},
    )
    shade_regimes(plot_axis = vol_axis, pair_frame = jpy_frame)
    vol_axis.plot(jpy_frame["week"], jpy_frame["realised_vol"], color = REALISED_COLOUR, linewidth = REALISED_LINE_WIDTH, label = "rolling realised vol, next 4 weeks")
    vol_axis.plot(jpy_frame["week"], jpy_frame["garch_vol"], color = GARCH_COLOUR, linewidth = FORECAST_LINE_WIDTH, label = "GARCH(1,1) forecast")
    vol_axis.plot(jpy_frame["week"], jpy_frame["eng1_vol"], color = ENG1_COLOUR, linewidth = FORECAST_LINE_WIDTH, label = "ENG1 forecast")
    vol_axis.set_ylabel(VOL_AXIS_LABEL)
    vol_axis.legend(handles = vol_axis.get_legend_handles_labels()[0] + regime_legend_patches(), loc = "upper left", ncol = 2)
    shade_regimes(plot_axis = gap_axis, pair_frame = jpy_frame)
    gap_axis.plot(jpy_frame["week"], jpy_frame["rolling_qlike_gap"], color = GAP_LINE_COLOUR, linewidth = GAP_LINE_WIDTH)
    gap_axis.fill_between(jpy_frame["week"], jpy_frame["rolling_qlike_gap"], 0.0, where = jpy_frame["rolling_qlike_gap"] > 0.0, color = ENG1_COLOUR, alpha = GAP_FILL_ALPHA, interpolate = True)
    gap_axis.fill_between(jpy_frame["week"], jpy_frame["rolling_qlike_gap"], 0.0, where = jpy_frame["rolling_qlike_gap"] < 0.0, color = GARCH_COLOUR, alpha = GAP_FILL_ALPHA, interpolate = True)
    gap_axis.axhline(0.0, color = ZERO_LINE_COLOUR, linestyle = "--", linewidth = GAP_LINE_WIDTH)
    gap_axis.set_ylabel(GAP_AXIS_LABEL)
    gap_axis.set_xlabel("week")
    figure.suptitle("jpy_usd: GARCH(1,1) and ENG1 against realised vol. " + GAP_TITLE_NOTE)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "jpy_usd_garch_vs_eng1.png", dpi = PLOT_DPI)
    plt.close(figure)

    # CHF/USD: forecasts against realised vol, then which forecast was closer.
    chf_frame = prepare_pair_frame(per_week_frame = per_week_frame, pair = "chf_usd")
    figure, (vol_axis, gap_axis) = plt.subplots(
        nrows = 2, ncols = 1, figsize = (PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES), sharex = True,
        gridspec_kw = {"height_ratios": [VOL_PANEL_HEIGHT, GAP_PANEL_HEIGHT]},
    )
    shade_regimes(plot_axis = vol_axis, pair_frame = chf_frame)
    vol_axis.plot(chf_frame["week"], chf_frame["realised_vol"], color = REALISED_COLOUR, linewidth = REALISED_LINE_WIDTH, label = "rolling realised vol, next 4 weeks")
    vol_axis.plot(chf_frame["week"], chf_frame["garch_vol"], color = GARCH_COLOUR, linewidth = FORECAST_LINE_WIDTH, label = "GARCH(1,1) forecast")
    vol_axis.plot(chf_frame["week"], chf_frame["eng1_vol"], color = ENG1_COLOUR, linewidth = FORECAST_LINE_WIDTH, label = "ENG1 forecast")
    vol_axis.set_ylabel(VOL_AXIS_LABEL)
    vol_axis.legend(handles = vol_axis.get_legend_handles_labels()[0] + regime_legend_patches(), loc = "upper left", ncol = 2)
    shade_regimes(plot_axis = gap_axis, pair_frame = chf_frame)
    gap_axis.plot(chf_frame["week"], chf_frame["rolling_qlike_gap"], color = GAP_LINE_COLOUR, linewidth = GAP_LINE_WIDTH)
    gap_axis.fill_between(chf_frame["week"], chf_frame["rolling_qlike_gap"], 0.0, where = chf_frame["rolling_qlike_gap"] > 0.0, color = ENG1_COLOUR, alpha = GAP_FILL_ALPHA, interpolate = True)
    gap_axis.fill_between(chf_frame["week"], chf_frame["rolling_qlike_gap"], 0.0, where = chf_frame["rolling_qlike_gap"] < 0.0, color = GARCH_COLOUR, alpha = GAP_FILL_ALPHA, interpolate = True)
    gap_axis.axhline(0.0, color = ZERO_LINE_COLOUR, linestyle = "--", linewidth = GAP_LINE_WIDTH)
    gap_axis.set_ylabel(GAP_AXIS_LABEL)
    gap_axis.set_xlabel("week")
    figure.suptitle("chf_usd: GARCH(1,1) and ENG1 against realised vol. " + GAP_TITLE_NOTE)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIRECTORY / "chf_usd_garch_vs_eng1.png", dpi = PLOT_DPI)
    plt.close(figure)

    print("Written five figures to " + str(OUTPUT_DIRECTORY))


if __name__ == "__main__":
    main()
