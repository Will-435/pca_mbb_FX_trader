"""
mwu_combination.py combines GARCH(1,1) and ENG1 with multiplicative weights,
the "best expert" method from online learning, instead of a fixed 50/50 split.

Each week both experts forecast. Once a forecast's four week window has closed,
each expert's weight is shrunk by an exponential factor of how badly it did.
The combined forecast mixes the experts in log variance, where QLIKE is convex,
so it never loses more than the weighted average of the two experts.

Two versions run side by side. Hedge learns which expert is best overall.
Fixed share keeps a little weight on both, so it can move back quickly when
the better expert changes with the regime.

Research file, kept apart from the main pipeline. Reads the weekly
results written by garch_comparison.py.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This file sits one folder below local_tests, so that folder is added to the
# import path to reuse the QLIKE loss and the Diebold-Mariano test.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kurtosis_impact_backtest as engine
import vol_forecast_backtest as volbt


INPUT_PATH = Path("local_tests/garch_baseline/garch_comparison_output/garch_comparison_per_week.parquet")
OUTPUT_DIRECTORY = Path("local_tests/garch_baseline/mwu_combination_output")
PER_WEEK_OUTPUT_PATH = OUTPUT_DIRECTORY / "mwu_per_week.parquet"
SUMMARY_PARQUET_PATH = OUTPUT_DIRECTORY / "mwu_summary.parquet"
SUMMARY_CSV_PATH = OUTPUT_DIRECTORY / "mwu_summary.csv"
DIAGNOSTICS_CSV_PATH = OUTPUT_DIRECTORY / "mwu_diagnostics.csv"
SENSITIVITY_CSV_PATH = OUTPUT_DIRECTORY / "mwu_learning_rate_sensitivity.csv"
WEIGHT_PLOT_PATH = OUTPUT_DIRECTORY / "mwu_eng1_weight.png"

EXPERT_LABELS = ["GARCH", "ENG1"]
ENG1_POSITION = EXPERT_LABELS.index("ENG1")
FORECAST_LABELS = ["GARCH", "ENG1", "ENG1_GARCH_BLEND", "MWU_HEDGE", "MWU_FIXED_SHARE"]
BASELINE_LABEL = "GARCH"
FIXED_BLEND_WEIGHT = 0.5
REGIME_ORDER = ["low", "medium", "high", "overall"]
PAIR_ORDER = ["usd_krw", "eur_usd", "gbp_usd", "jpy_usd", "chf_usd"]

# Weekly QLIKE is capped then scaled into [0, 1], the range the textbook
# learning rate assumes, so one disastrous week cannot wipe out an expert.
LOSS_CAP = 2.0
HEDGE_RATE_NUMERATOR = 8.0
FIXED_SHARE_RATE = 0.05
LEARNING_RATE_MULTIPLIERS = [1.0, 4.0, 16.0]
HORIZON_OFFSET = pd.Timedelta(weeks = engine.HORIZON_WEEKS)
PERCENT_SCALE = 100.0
PRINT_FORMAT = ".4f"

PLOT_WIDTH_INCHES = 13
PLOT_HEIGHT_INCHES = 6
PLOT_DPI = 150
LINE_WIDTH = 1.6
EVEN_WEIGHT_LINE_COLOUR = "black"
KRW_COLOUR = "red"
EUR_COLOUR = "blue"
GBP_COLOUR = "green"
JPY_COLOUR = "darkorange"
CHF_COLOUR = "purple"


def textbook_learning_rate(round_count = None):
    """
    The standard Hedge learning rate for two experts over a known number of
    rounds. It balances learning fast against overreacting to noise.

    INPUTS:
        * round_count - the number of weekly forecasts

    OUTPUTS:
        * the learning rate
    """
    return float(np.sqrt(HEDGE_RATE_NUMERATOR * np.log(len(EXPERT_LABELS)) / round_count))


def run_multiplicative_weights(pair_frame = None, learning_rate = None, fixed_share_rate = None):
    """
    Walks through the weeks in date order. Before each forecast it learns from
    every earlier forecast whose window has closed, shrinking the weight of the
    expert that lost more. The forecast is then the weighted mix of the experts
    in log variance.

    INPUTS:
        * pair_frame - one pair's weeks in date order, with expert forecasts and losses
        * learning_rate - how hard each loss shrinks a weight
        * fixed_share_rate - share of weight spread back evenly after each update, zero for Hedge

    OUTPUTS:
        * combined_forecast - the mixed variance forecast for each week
        * eng1_weight - the weight on ENG1 when each forecast was made
    """
    expert_count = len(EXPERT_LABELS)
    weights = np.full(expert_count, 1.0 / expert_count)
    log_forecasts = np.log(pair_frame[EXPERT_LABELS].to_numpy())
    scaled_losses = np.minimum(pair_frame[["qlike_" + label for label in EXPERT_LABELS]].to_numpy(), LOSS_CAP) / LOSS_CAP
    weeks = pair_frame["week"].tolist()

    combined_forecast = np.empty(len(pair_frame))
    eng1_weight = np.empty(len(pair_frame))
    next_to_learn = 0
    for week_position in range(len(pair_frame)):
        # A forecast's loss is only known once its four week window has closed.
        while next_to_learn < week_position and weeks[next_to_learn] + HORIZON_OFFSET <= weeks[week_position]:
            weights = weights * np.exp(-learning_rate * scaled_losses[next_to_learn])
            weights = weights / np.sum(weights)
            weights = (1.0 - fixed_share_rate) * weights + fixed_share_rate / expert_count
            next_to_learn = next_to_learn + 1
        combined_forecast[week_position] = np.exp(np.sum(weights * log_forecasts[week_position]))
        eng1_weight[week_position] = weights[ENG1_POSITION]
    return combined_forecast, eng1_weight


def mean_eng1_weight(regime_frame = None, forecast_label = None):
    """
    The average weight a forecast placed on ENG1 over the given weeks.

    INPUTS:
        * regime_frame - the weeks being summarised
        * forecast_label - which forecast

    OUTPUTS:
        * the mean ENG1 weight, from zero (all GARCH) to one (all ENG1)
    """
    if forecast_label == "GARCH":
        return 0.0
    if forecast_label == "ENG1":
        return 1.0
    if forecast_label == "ENG1_GARCH_BLEND":
        return FIXED_BLEND_WEIGHT
    if forecast_label == "MWU_HEDGE":
        return float(regime_frame["hedge_eng1_weight"].mean())
    return float(regime_frame["fixed_share_eng1_weight"].mean())


def summarise_pair(pair_frame = None):
    """
    Scores every forecast against GARCH for one pair, per regime and overall.

    INPUTS:
        * pair_frame - one pair's weeks with every forecast

    OUTPUTS:
        * a list of summary rows
    """
    summary_rows = []
    for regime_name in REGIME_ORDER:
        if regime_name == "overall":
            regime_frame = pair_frame
        else:
            regime_frame = pair_frame[pair_frame["regime"] == regime_name]
        if len(regime_frame) == 0:
            continue

        actual_variance = regime_frame["actual_variance"].to_numpy()
        garch_loss = volbt.qlike_loss(actual_variance, regime_frame[BASELINE_LABEL].to_numpy())
        garch_mean_loss = float(np.mean(garch_loss))
        for forecast_label in FORECAST_LABELS:
            forecast_loss = volbt.qlike_loss(actual_variance, regime_frame[forecast_label].to_numpy())
            mean_loss = float(np.mean(forecast_loss))
            dm_p_value = float("nan")
            if forecast_label != BASELINE_LABEL:
                _, _, dm_p_value = engine.diebold_mariano_pvalue(forecast_loss - garch_loss)
            summary_rows.append({
                "pair": pair_frame["pair"].iloc[0],
                "regime": regime_name,
                "forecast": forecast_label,
                "weeks": len(regime_frame),
                "mean_qlike": mean_loss,
                "qlike_change_vs_garch_pct": PERCENT_SCALE * (garch_mean_loss - mean_loss) / garch_mean_loss,
                "dm_p_vs_garch": dm_p_value,
                "eng1_weight": mean_eng1_weight(regime_frame = regime_frame, forecast_label = forecast_label),
            })
    return summary_rows


def diagnose_pair(pair_frame = None):
    """
    Measures why combining helps or not for one pair. Combining pays when the
    second forecast is not much noisier than the first and its errors do not
    move with the first's. Bias shows which forecast runs too high.

    INPUTS:
        * pair_frame - one pair's weeks with every forecast

    OUTPUTS:
        * one row of diagnostics
    """
    # Log errors: above zero means the forecast was too high.
    eng1_error = np.log(pair_frame["ENG1"] / pair_frame["actual_variance"]).to_numpy()
    garch_error = np.log(pair_frame["GARCH"] / pair_frame["actual_variance"]).to_numpy()
    low_vol_weeks = (pair_frame["regime"] == "low").to_numpy()

    eng1_spread = float(np.std(eng1_error))
    garch_spread = float(np.std(garch_error))
    error_correlation = float(np.corrcoef(eng1_error, garch_error)[0, 1])
    # The weight on ENG1 that would have minimised combined error spread, in hindsight.
    shared_term = error_correlation * eng1_spread * garch_spread
    hindsight_weight = (garch_spread ** 2 - shared_term) / (eng1_spread ** 2 + garch_spread ** 2 - 2.0 * shared_term)

    return {
        "pair": pair_frame["pair"].iloc[0],
        "low_vol_share": float(np.mean(low_vol_weeks)),
        "eng1_bias": float(np.mean(eng1_error)),
        "garch_bias": float(np.mean(garch_error)),
        "eng1_bias_low_vol": float(np.mean(eng1_error[low_vol_weeks])) if low_vol_weeks.any() else float("nan"),
        "garch_bias_low_vol": float(np.mean(garch_error[low_vol_weeks])) if low_vol_weeks.any() else float("nan"),
        "eng1_error_spread": eng1_spread,
        "garch_error_spread": garch_spread,
        "error_correlation": error_correlation,
        "hindsight_eng1_weight": float(hindsight_weight),
        "final_hedge_eng1_weight": float(pair_frame["hedge_eng1_weight"].iloc[-1]),
    }


def learning_rate_sensitivity(pair_frame = None, base_learning_rate = None):
    """
    Reruns Hedge with faster learning rates, to show how much the result
    depends on how quickly weights move. Reported for context only: picking the
    best rate here would be tuning on the test period.

    INPUTS:
        * pair_frame - one pair's weeks with both experts
        * base_learning_rate - the textbook rate

    OUTPUTS:
        * a list of rows, one per learning rate multiple
    """
    sensitivity_rows = []
    actual_variance = pair_frame["actual_variance"].to_numpy()
    garch_loss = volbt.qlike_loss(actual_variance, pair_frame[BASELINE_LABEL].to_numpy())
    for multiplier in LEARNING_RATE_MULTIPLIERS:
        combined_forecast, eng1_weight = run_multiplicative_weights(
            pair_frame = pair_frame, learning_rate = base_learning_rate * multiplier, fixed_share_rate = 0.0
        )
        combined_loss = volbt.qlike_loss(actual_variance, combined_forecast)
        _, _, dm_p_value = engine.diebold_mariano_pvalue(combined_loss - garch_loss)
        sensitivity_rows.append({
            "pair": pair_frame["pair"].iloc[0],
            "learning_rate_multiple": multiplier,
            "qlike_change_vs_garch_pct": PERCENT_SCALE * (np.mean(garch_loss) - np.mean(combined_loss)) / np.mean(garch_loss),
            "dm_p_vs_garch": dm_p_value,
            "final_eng1_weight": float(eng1_weight[-1]),
        })
    return sensitivity_rows


def save_weight_plot(per_week_frame = None):
    """
    Plots the Hedge weight on ENG1 through time for every pair, so it is clear
    which expert each pair learned to trust.

    INPUTS:
        * per_week_frame - every pair's weeks with the Hedge weights

    OUTPUTS:
        * a PNG in mwu_combination_output
    """
    figure, plot_axis = plt.subplots(figsize = (PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES))

    krw_frame = per_week_frame[per_week_frame["pair"] == "usd_krw"]
    eur_frame = per_week_frame[per_week_frame["pair"] == "eur_usd"]
    gbp_frame = per_week_frame[per_week_frame["pair"] == "gbp_usd"]
    jpy_frame = per_week_frame[per_week_frame["pair"] == "jpy_usd"]
    chf_frame = per_week_frame[per_week_frame["pair"] == "chf_usd"]

    # USD/KRW weight on ENG1.
    plot_axis.plot(krw_frame["week"], krw_frame["hedge_eng1_weight"], color = KRW_COLOUR, linewidth = LINE_WIDTH, label = "usd_krw")
    # EUR/USD weight on ENG1.
    plot_axis.plot(eur_frame["week"], eur_frame["hedge_eng1_weight"], color = EUR_COLOUR, linewidth = LINE_WIDTH, label = "eur_usd")
    # GBP/USD weight on ENG1.
    plot_axis.plot(gbp_frame["week"], gbp_frame["hedge_eng1_weight"], color = GBP_COLOUR, linewidth = LINE_WIDTH, label = "gbp_usd")
    # JPY/USD weight on ENG1.
    plot_axis.plot(jpy_frame["week"], jpy_frame["hedge_eng1_weight"], color = JPY_COLOUR, linewidth = LINE_WIDTH, label = "jpy_usd")
    # CHF/USD weight on ENG1, its GARCH fit is distorted by the 2015 franc shock.
    plot_axis.plot(chf_frame["week"], chf_frame["hedge_eng1_weight"], color = CHF_COLOUR, linewidth = LINE_WIDTH, label = "chf_usd")

    plot_axis.axhline(FIXED_BLEND_WEIGHT, color = EVEN_WEIGHT_LINE_COLOUR, linestyle = "--", label = "fixed 50/50 split")
    plot_axis.set_ylabel("Hedge weight on ENG1 (0 = all GARCH, 1 = all ENG1)")
    plot_axis.set_xlabel("week")
    plot_axis.set_title("Which expert each pair learned to trust: above the dashed line leans ENG1, below leans GARCH")
    plot_axis.legend(loc = "best")
    figure.tight_layout()
    figure.savefig(WEIGHT_PLOT_PATH, dpi = PLOT_DPI)
    plt.close(figure)


def main():
    """
    Runs Hedge and fixed share for every pair, prints the per pair, per regime
    scores, the diagnostics and the learning rate check, and saves everything.

    INPUTS:
        * none

    OUTPUTS:
        * printed tables, parquet and csv files and a PNG in mwu_combination_output
    """
    if not INPUT_PATH.exists():
        raise RuntimeError("Missing " + str(INPUT_PATH) + ". Run garch_comparison.py first.")
    OUTPUT_DIRECTORY.mkdir(parents = True, exist_ok = True)
    per_week_frame = pd.read_parquet(INPUT_PATH)

    combined_frames = []
    summary_rows = []
    diagnostic_rows = []
    sensitivity_rows = []
    for pair in PAIR_ORDER:
        pair_frame = per_week_frame[per_week_frame["pair"] == pair].sort_values("week").reset_index(drop = True)
        if len(pair_frame) == 0:
            continue
        learning_rate = textbook_learning_rate(round_count = len(pair_frame))
        pair_frame["MWU_HEDGE"], pair_frame["hedge_eng1_weight"] = run_multiplicative_weights(
            pair_frame = pair_frame, learning_rate = learning_rate, fixed_share_rate = 0.0
        )
        pair_frame["MWU_FIXED_SHARE"], pair_frame["fixed_share_eng1_weight"] = run_multiplicative_weights(
            pair_frame = pair_frame, learning_rate = learning_rate, fixed_share_rate = FIXED_SHARE_RATE
        )
        combined_frames.append(pair_frame)
        summary_rows.extend(summarise_pair(pair_frame = pair_frame))
        diagnostic_rows.append(diagnose_pair(pair_frame = pair_frame))
        sensitivity_rows.extend(learning_rate_sensitivity(pair_frame = pair_frame, base_learning_rate = learning_rate))

    combined_frame = pd.concat(combined_frames, ignore_index = True)
    summary_frame = pd.DataFrame(summary_rows)
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)

    combined_frame.to_parquet(PER_WEEK_OUTPUT_PATH, index = False)
    summary_frame.to_parquet(SUMMARY_PARQUET_PATH, index = False)
    summary_frame.to_csv(SUMMARY_CSV_PATH, index = False)
    diagnostics_frame.to_csv(DIAGNOSTICS_CSV_PATH, index = False)
    sensitivity_frame.to_csv(SENSITIVITY_CSV_PATH, index = False)
    save_weight_plot(per_week_frame = combined_frame)

    float_format = lambda value: format(value, PRINT_FORMAT)
    print("Experts: GARCH(1,1) and ENG1. Textbook Hedge learning rate, fixed share rate " + str(FIXED_SHARE_RATE) + ".")
    print("qlike_change_vs_garch_pct above zero means closer than GARCH. eng1_weight is the average weight on ENG1.")
    for pair in PAIR_ORDER:
        pair_summary = summary_frame[summary_frame["pair"] == pair]
        if len(pair_summary) == 0:
            continue
        print()
        print(pair)
        print(pair_summary.drop(columns = ["pair"]).to_string(index = False, float_format = float_format))
    print()
    print("DIAGNOSTICS (log errors, above zero means the forecast ran too high; hindsight weight is in-sample)")
    print(diagnostics_frame.to_string(index = False, float_format = float_format))
    print()
    print("LEARNING RATE CHECK, Hedge over the whole period, context only")
    print(sensitivity_frame.to_string(index = False, float_format = float_format))
    print()
    print("Written " + str(OUTPUT_DIRECTORY))


if __name__ == "__main__":
    main()
