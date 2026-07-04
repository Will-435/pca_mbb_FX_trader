"""
pca.py fits and freezes the covariance PCA that defines the regime axes.

It reads the weekly data from Data/Input/data_ingestion.py, 
builds the feature matrix, fits the PCA on the fit window, 
freezes it, and projects the full history onto the frozen axes. This
file never touches the network.

Outputs (pca_output/*):

    pca_model.parquet holds the frozen loadings, feature means,
    component variances, feature order and fit date range.

    pc_timeseries.png plots the PC score history.

    rolling_correlations.parquet holds the separate correlation tracking. Every
    artefact is also exported as a .sql script in Database/ for inspection in MySQL
    Workbench.
"""

from pathlib import Path

import matplotlib

# File output only, no display window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_DATA_DIRECTORY = Path("Data/Input")
WEEKLY_LEVELS_PATH = RAW_DATA_DIRECTORY / "weekly_levels.parquet"
FEATURE_MATRIX_PATH = RAW_DATA_DIRECTORY / "feature_matrix.parquet"

OUTPUT_DIRECTORY = Path("components/pca_output")
PCA_MODEL_PATH = OUTPUT_DIRECTORY / "pca_model.parquet"
ROLLING_CORRELATIONS_PATH = OUTPUT_DIRECTORY / "rolling_correlations.parquet"
PC_TIMESERIES_PLOT_PATH = OUTPUT_DIRECTORY / "pc_timeseries.png"

DATABASE_DIRECTORY = Path("Database")

DATE_COLUMN_NAME = "week_date"
TARGET_COLUMN_NAME = "usd_krw_forward_return_4w"

#TODO - Add more features to the geature columns to feed the
# PCA model
FEATURE_COLUMN_NAMES = [
    "dollar_index_return_4w",
    "gold_return_4w",
    "oil_return_4w",
    "us_2y_yield_change_4w",
    "us_10y_yield_change_4w",
    "vix_change_4w",
]

RETURN_HORIZON_WEEKS = 4
SINGLE_WEEK_LAG = 1
ROLLING_CORRELATION_WINDOW_WEEKS = 26

#TODO Investigate the number of PCs, k. (Issue no.7 on github)
# The eigenvectors are fitted here. Rest of the data will be used for
# validating the choice of number of Primary Coefficients, k.
FIT_START_DATE = "2006-01-01"
FIT_END_DATE = "2022-12-31"

# TODO This choice needs to be validated, as per issue #7 on github
NUMBER_OF_COMPONENTS_RETAINED = 3

FIRST_POSITION = 1

INSERT_BATCH_ROW_COUNT = 500

PLOT_WIDTH_INCHES = 12
PLOT_HEIGHT_INCHES = 9
PLOT_DPI = 150
PC_1_COLOUR = "red"
PC_2_COLOUR = "green"
PC_3_COLOUR = "blue"
FREEZE_LINE_COLOUR = "black"

SCORE_PRINT_FORMAT = ".6f"
SHARE_PRINT_FORMAT = ".3f"


def load_weekly_level_frame():
    """
    Loads the weekly level data from data_ingestion.py.

    INPUTS:
        * none

    OUTPUTS:
        * a dataframe of weekly levels, one column per series, indexed by date
    """
    # Quick error catch ---------------------------------------------------
    if not WEEKLY_LEVELS_PATH.exists():
        raise RuntimeError(
            "Missing " + str(WEEKLY_LEVELS_PATH)
            + ". Run Data/Input/data_ingestion.py first, and run this file"
            + " from the repository root."
        )
    
    weekly_level_frame = pd.read_parquet(WEEKLY_LEVELS_PATH)
    weekly_level_frame = weekly_level_frame.set_index(DATE_COLUMN_NAME)
    return weekly_level_frame


def build_weekly_feature_matrix(weekly_level_frame = None):
    """
    Turns weekly levels into features and target. Price series
    become four week log returns. Yields and the VIX become four week changes
    in their own units. Units arent standardised to keep scale.

    The target is the forward four week log return of USD/KRW. It sits beside
    the features but is never a PCA input.

    INPUTS:
        * weekly_level_frame

    OUTPUTS:
        * feature_frame - the PCA input columns
        * target_series - the forward four week USD/KRW log return
    """
    feature_frame = pd.DataFrame(index = weekly_level_frame.index)

    feature_frame["dollar_index_return_4w"] = np.log(
        weekly_level_frame["dollar_index"]
        / weekly_level_frame["dollar_index"].shift(RETURN_HORIZON_WEEKS)
    )
    feature_frame["gold_return_4w"] = np.log(
        weekly_level_frame["gold"]
        / weekly_level_frame["gold"].shift(RETURN_HORIZON_WEEKS)
    )
    feature_frame["oil_return_4w"] = np.log(
        weekly_level_frame["oil_wti"]
        / weekly_level_frame["oil_wti"].shift(RETURN_HORIZON_WEEKS)
    )
    feature_frame["us_2y_yield_change_4w"] = (
        weekly_level_frame["us_2y_yield"]
        - weekly_level_frame["us_2y_yield"].shift(RETURN_HORIZON_WEEKS)
    )
    feature_frame["us_10y_yield_change_4w"] = (
        weekly_level_frame["us_10y_yield"]
        - weekly_level_frame["us_10y_yield"].shift(RETURN_HORIZON_WEEKS)
    )
    feature_frame["vix_change_4w"] = (
        weekly_level_frame["vix"]
        - weekly_level_frame["vix"].shift(RETURN_HORIZON_WEEKS)
    )

    target_series = np.log(
        weekly_level_frame["usd_krw"].shift(-RETURN_HORIZON_WEEKS)
        / weekly_level_frame["usd_krw"]
    )

    feature_frame = feature_frame[FEATURE_COLUMN_NAMES].dropna()
    target_series = target_series.reindex(feature_frame.index)
    return feature_frame, target_series


def fit_frozen_pca(feature_frame = None):
    """
    Fits the covariance PCA on the fit window and freezes it. The frozen pieces
    are the loadings, the feature means, the component variances and the fit
    date range.

    The eigendecomposition of the fit window covariance matrix is the PCA. The
    eigenvectors with the largest eigenvalues are kept as components.

    INPUTS:
        * feature_frame - the full weekly feature history

    OUTPUTS:
        * pca_model - a dictionary with the loading frame, feature mean series,
          component variance series - total variance and fit date range
    """
    fit_window_frame = feature_frame.loc[FIT_START_DATE:FIT_END_DATE]
    feature_mean_series = fit_window_frame.mean()
    centred_fit_frame = fit_window_frame - feature_mean_series
    covariance_matrix = centred_fit_frame.cov()

    eigenvalues, eigenvector_matrix = np.linalg.eigh(covariance_matrix.to_numpy())

    # eigh returns eigenvalues in ascending order, flipped here so pc_1 is the
    # direction with the largest variance.
    descending_positions = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[descending_positions]
    eigenvector_matrix = eigenvector_matrix[:, descending_positions]

    retained_loading_matrix = eigenvector_matrix[:, :NUMBER_OF_COMPONENTS_RETAINED].copy()
    for component_position in range(NUMBER_OF_COMPONENTS_RETAINED):
        component_weights = retained_loading_matrix[:, component_position]
        largest_weight_position = np.argmax(np.abs(component_weights))
        if component_weights[largest_weight_position] < 0:
            retained_loading_matrix[:, component_position] = -component_weights

    component_labels = build_component_labels()
    loading_frame = pd.DataFrame(
        retained_loading_matrix,
        index = feature_frame.columns,
        columns = component_labels,
    )
    component_variance_series = pd.Series(
        eigenvalues[:NUMBER_OF_COMPONENTS_RETAINED],
        index = component_labels,
    )

    pca_model = {
        "loading_frame": loading_frame,
        "feature_mean_series": feature_mean_series,
        "component_variance_series": component_variance_series,
        "total_variance": float(eigenvalues.sum()),
        "fit_start_date": fit_window_frame.index[0],
        "fit_end_date": fit_window_frame.index[-1],
    }
    return pca_model


def build_component_labels():
    """
    Builds the component column labels pc_1 up to the retained count.

    INPUTS:
        * none

    OUTPUTS:
        * list of labels, pc_1, pc_2, pc_3
    """
    label_list = []
    for component_number in range(FIRST_POSITION, NUMBER_OF_COMPONENTS_RETAINED + 1):
        label_list.append("pc_" + str(component_number))
    return label_list


def project_onto_frozen_axes(feature_frame = None, pca_model = None):
    """
    Projects the feature history onto the pca axes. Centre with the frozen
    means, multiply by the frozen loadings.

    INPUTS:
        * feature_frame - the full weekly feature history
        * pca_model - the frozen model from fit_frozen_pca

    OUTPUTS:
        * score_frame - the PC score history, one column per component
    """
    centred_feature_frame = feature_frame - pca_model["feature_mean_series"]
    score_array = centred_feature_frame.to_numpy() @ pca_model["loading_frame"].to_numpy()
    score_frame = pd.DataFrame(
        score_array,
        index = feature_frame.index,
        columns = pca_model["loading_frame"].columns,
    )
    return score_frame


def build_model_export_frame(pca_model = None):
    """
    Flattens the frozen model into one plain table so any language can read it.
    This is needed for the C++ Euclidean file. This function is made by Claude.

    INPUTS:
        * pca_model, the frozen model from fit_frozen_pca

    OUTPUTS:
        * a dataframe ready for pca_model.parquet and the SQL export
    """
    model_records = []

    feature_position = FIRST_POSITION
    for feature_name in pca_model["loading_frame"].index:
        component_position = FIRST_POSITION
        for component_label in pca_model["loading_frame"].columns:
            model_records.append({
                "record_type": "loading",
                "feature_name": feature_name,
                "feature_order": feature_position,
                "component_number": component_position,
                "value": float(pca_model["loading_frame"].loc[feature_name, component_label]),
                "date_text": None,
            })
            component_position = component_position + 1
        feature_position = feature_position + 1

    feature_position = FIRST_POSITION
    for feature_name in pca_model["feature_mean_series"].index:
        model_records.append({
            "record_type": "mean",
            "feature_name": feature_name,
            "feature_order": feature_position,
            "component_number": None,
            "value": float(pca_model["feature_mean_series"][feature_name]),
            "date_text": None,
        })
        feature_position = feature_position + 1

    component_position = FIRST_POSITION
    for component_label in pca_model["component_variance_series"].index:
        model_records.append({
            "record_type": "component_variance",
            "feature_name": None,
            "feature_order": None,
            "component_number": component_position,
            "value": float(pca_model["component_variance_series"][component_label]),
            "date_text": None,
        })
        component_position = component_position + 1

    model_records.append({
        "record_type": "fit_date_range",
        "feature_name": "fit_start_date",
        "feature_order": None,
        "component_number": None,
        "value": None,
        "date_text": pca_model["fit_start_date"].strftime("%Y-%m-%d"),
    })
    model_records.append({
        "record_type": "fit_date_range",
        "feature_name": "fit_end_date",
        "feature_order": None,
        "component_number": None,
        "value": None,
        "date_text": pca_model["fit_end_date"].strftime("%Y-%m-%d"),
    })

    export_columns = [
        "record_type", "feature_name", "feature_order",
        "component_number", "value", "date_text",
    ]
    return pd.DataFrame(model_records, columns = export_columns)


def compute_rolling_correlations(weekly_level_frame = None):
    """
    The rolling correlation btween gold and oil is a time-variate measure.
    They are important for regime classification, so we still concider them.

    INPUTS:
        * weekly_level_frame - weekly levels from load_weekly_level_frame

    OUTPUTS:
        * a df with the 2 rolling correlation columns
    """
    weekly_dollar_return_1w = np.log(
        weekly_level_frame["dollar_index"]
        / weekly_level_frame["dollar_index"].shift(SINGLE_WEEK_LAG)
    )
    weekly_gold_return_1w = np.log(
        weekly_level_frame["gold"]
        / weekly_level_frame["gold"].shift(SINGLE_WEEK_LAG)
    )
    weekly_oil_return_1w = np.log(
        weekly_level_frame["oil_wti"]
        / weekly_level_frame["oil_wti"].shift(SINGLE_WEEK_LAG)
    )

    gold_usd_correlation = weekly_gold_return_1w.rolling(
        window = ROLLING_CORRELATION_WINDOW_WEEKS
    ).corr(weekly_dollar_return_1w)
    oil_usd_correlation = weekly_oil_return_1w.rolling(
        window = ROLLING_CORRELATION_WINDOW_WEEKS
    ).corr(weekly_dollar_return_1w)

    rolling_correlation_frame = pd.DataFrame({
        "gold_usd_correlation": gold_usd_correlation,
        "oil_usd_correlation": oil_usd_correlation,
    }).dropna()
    rolling_correlation_frame.index.name = DATE_COLUMN_NAME

    return rolling_correlation_frame


def format_value_for_sql(single_value = None):
    """
    Formats one Python value as a MySQL literal.

    INPUTS:
        * single_value - any cell value from a dataframe

    OUTPUTS:
        * values as strings
    """
    if pd.isna(single_value):
        return "NULL"
    if isinstance(single_value, pd.Timestamp):
        return "'" + single_value.strftime("%Y-%m-%d") + "'"
    if isinstance(single_value, (int, float, np.integer, np.floating)):
        return str(single_value)
    
    return "'" + str(single_value).replace("'", "''") + "'"


def write_dataframe_to_sql(dataframe = None, table_name = None, output_path = None):
    """

    Another function written by Claude, so that I can access the data easily 
    using MySQLWorkbench.

    Writes a dataframe as a MySQL script with one CREATE TABLE and batched
    INSERT statements.

    INPUTS:
        * dataframe, the table content
        * table_name, the table name used in the script
        * output_path, where the .sql file is written

    OUTPUTS:
        * a .sql file at output_path
    """
    column_definitions = []
    for column_name in dataframe.columns:
        if pd.api.types.is_datetime64_any_dtype(dataframe[column_name]):
            column_type = "DATE"
        elif pd.api.types.is_numeric_dtype(dataframe[column_name]):
            column_type = "DOUBLE"
        else:
            column_type = "VARCHAR(255)"
        column_definitions.append("    " + column_name + " " + column_type)

    script_lines = []
    script_lines.append("DROP TABLE IF EXISTS " + table_name + ";")
    script_lines.append("CREATE TABLE " + table_name + " (")
    script_lines.append(",\n".join(column_definitions))
    script_lines.append(");")

    formatted_rows = []
    for row_values in dataframe.itertuples(index = False):
        formatted_cells = []
        for single_value in row_values:
            formatted_cells.append(format_value_for_sql(single_value = single_value))
        formatted_rows.append("(" + ", ".join(formatted_cells) + ")")

    # Rows are inserted in batches, one giant statement would be unreadable and
    # one statement per row would import slowly.
    for batch_start in range(0, len(formatted_rows), INSERT_BATCH_ROW_COUNT):
        batch_rows = formatted_rows[batch_start:batch_start + INSERT_BATCH_ROW_COUNT]
        script_lines.append("INSERT INTO " + table_name + " VALUES")
        script_lines.append(",\n".join(batch_rows) + ";")

    output_path.write_text("\n".join(script_lines) + "\n")
    print("Written " + str(output_path))


def main():
    """
    Runs the full PCA stage: load, feature build, frozen fit, projection,
    plots and exports. 

    INPUTS:
        * none

    OUTPUTS:
        * Data/Input/feature_matrix.parquet
        * pca_output/pca_model.parquet, rolling_correlations.parquet and
          pc_timeseries.png
        * Database .sql exports for every artefact
    """
    weekly_level_frame = load_weekly_level_frame()
    feature_frame, target_series = build_weekly_feature_matrix(
        weekly_level_frame = weekly_level_frame
    )

    # The feature matrix parquet is the shared input for data_relation.cpp and
    # euclidean.cpp. The target column sits last and is empty for the newest
    # weeks whose forward return has not happened yet.
    feature_matrix_frame = feature_frame.copy()
    feature_matrix_frame[TARGET_COLUMN_NAME] = target_series
    feature_matrix_frame.index.name = DATE_COLUMN_NAME
    feature_matrix_frame.reset_index().to_parquet(FEATURE_MATRIX_PATH, index = False)
    print("Written " + str(FEATURE_MATRIX_PATH))

    pca_model = fit_frozen_pca(feature_frame = feature_frame)
    score_frame = project_onto_frozen_axes(
        feature_frame = feature_frame, pca_model = pca_model
    )
    score_frame.index.name = DATE_COLUMN_NAME

    model_export_frame = build_model_export_frame(pca_model = pca_model)
    model_export_frame.to_parquet(PCA_MODEL_PATH, index = False)
    print("Written " + str(PCA_MODEL_PATH))

    rolling_correlation_frame = compute_rolling_correlations(
        weekly_level_frame = weekly_level_frame
    )
    rolling_correlation_frame.reset_index().to_parquet(
        ROLLING_CORRELATIONS_PATH, index = False
    )
    print("Written " + str(ROLLING_CORRELATIONS_PATH))

    write_dataframe_to_sql(
        dataframe = feature_matrix_frame.reset_index(),
        table_name = "feature_matrix",
        output_path = DATABASE_DIRECTORY / "feature_matrix.sql",
    )
    write_dataframe_to_sql(
        dataframe = model_export_frame,
        table_name = "pca_model",
        output_path = DATABASE_DIRECTORY / "pca_model.sql",
    )
    write_dataframe_to_sql(
        dataframe = score_frame.reset_index(),
        table_name = "pc_timeseries",
        output_path = DATABASE_DIRECTORY / "pc_timeseries.sql",
    )
    write_dataframe_to_sql(
        dataframe = rolling_correlation_frame.reset_index(),
        table_name = "rolling_correlations",
        output_path = DATABASE_DIRECTORY / "rolling_correlations.sql",
    )

    # One panel per retained component, plotted one block at a time so a single
    # panel can be adjusted without touching the others. The blocks assume
    # NUMBER_OF_COMPONENTS_RETAINED is three, add or remove blocks if that
    # changes.
    figure, (axis_pc_1, axis_pc_2, axis_pc_3) = plt.subplots(
        nrows = NUMBER_OF_COMPONENTS_RETAINED,
        ncols = 1,
        figsize = (PLOT_WIDTH_INCHES, PLOT_HEIGHT_INCHES),
        sharex = True,
    )

    # PC 1 score history with the loading freeze date marked.
    axis_pc_1.plot(score_frame.index, score_frame["pc_1"], color = PC_1_COLOUR)
    axis_pc_1.axvline(pca_model["fit_end_date"], color = FREEZE_LINE_COLOUR, linestyle = "--")
    axis_pc_1.set_ylabel("pc_1 score")

    # PC 2 score history with the loading freeze date marked.
    axis_pc_2.plot(score_frame.index, score_frame["pc_2"], color = PC_2_COLOUR)
    axis_pc_2.axvline(pca_model["fit_end_date"], color = FREEZE_LINE_COLOUR, linestyle = "--")
    axis_pc_2.set_ylabel("pc_2 score")

    # PC 3 score history with the loading freeze date marked.
    axis_pc_3.plot(score_frame.index, score_frame["pc_3"], color = PC_3_COLOUR)
    axis_pc_3.axvline(pca_model["fit_end_date"], color = FREEZE_LINE_COLOUR, linestyle = "--")
    axis_pc_3.set_ylabel("pc_3 score")
    axis_pc_3.set_xlabel("week")

    figure.suptitle("PC scores on frozen axes, dashed line marks the loading freeze date")
    figure.tight_layout()
    figure.savefig(PC_TIMESERIES_PLOT_PATH, dpi = PLOT_DPI)
    plt.close(figure)
    print("Written " + str(PC_TIMESERIES_PLOT_PATH))

    retained_variance_share = float(
        pca_model["component_variance_series"].sum() / pca_model["total_variance"]
    )
    print(
        "Retained components explain "
        + format(retained_variance_share, SHARE_PRINT_FORMAT)
        + " of fit window variance. Reference only, k is chosen by out of"
        + " sample regime stability, not by this number."
    )

    latest_score_row = score_frame.iloc[-1]
    coordinate_text = " ".join(
        format(score_value, SCORE_PRINT_FORMAT) for score_value in latest_score_row
    )
    print("Latest week in frozen PC space: " + str(score_frame.index[-1].date()))
    print("Neighbourhood step command: ./euclidean " + coordinate_text)


if __name__ == "__main__":
    main()