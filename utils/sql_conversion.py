"""
sql_conversion.py turns a dataframe into a MySQL script.

Each call writes one script with a single CREATE TABLE followed by batched
INSERT statements. The pipeline outputs can be inspected in mySQLworkbench.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Every .sql file is written here, so callers only pass a table name, never a
# full path. The directory is created if it does not exist.
DATABASE_DIRECTORY = Path("Database")
SQL_FILE_SUFFIX = ".sql"

# One INSERT statement carries this many rows. One giant statement would be
# unreadable, one statement per row would import slowly.
INSERT_BATCH_ROW_COUNT = 500

# Width of the text columns in the generated tables.
VARCHAR_COLUMN_LENGTH = 255


def format_value_for_sql(single_value = None):
    """
    makes each Python value a MySQL literal, so it can sit inside an
    INSERT statement.

    INPUTS:
        * single_value - any value from a df

    OUTPUTS:
        * the value as a string for the INSERT statement
    """
    if pd.isna(single_value):
        return "NULL"
    if isinstance(single_value, pd.Timestamp):
        return "'" + single_value.strftime("%Y-%m-%d") + "'"
    if isinstance(single_value, (int, float, np.integer, np.floating)):
        return str(single_value)

    return "'" + str(single_value).replace("'", "''") + "'"


def write_dataframe_to_sql(dataframe = None, table_name = None):
    """
    Writes a dataframe as a MySQL script with one CREATE TABLE and batched
    INSERT statements.

    The column types are read from the df and capatilised (per SQL convention).

    INPUTS:
        * df - the df content
        * table_name - the table name used in the script and the file name

    OUTPUTS:
        * a .sql file at Database/<table_name>.sql
    """
    column_definitions = []
    for column_name in dataframe.columns:
        if pd.api.types.is_datetime64_any_dtype(dataframe[column_name]):
            column_type = "DATE"
        elif pd.api.types.is_numeric_dtype(dataframe[column_name]):
            column_type = "DOUBLE"
        else:
            column_type = "VARCHAR(" + str(VARCHAR_COLUMN_LENGTH) + ")"
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

    for batch_start in range(0, len(formatted_rows), INSERT_BATCH_ROW_COUNT):
        batch_rows = formatted_rows[batch_start:batch_start + INSERT_BATCH_ROW_COUNT]
        script_lines.append("INSERT INTO " + table_name + " VALUES")
        script_lines.append(",\n".join(batch_rows) + ";")

    DATABASE_DIRECTORY.mkdir(parents = True, exist_ok = True)
    output_path = DATABASE_DIRECTORY / (table_name + SQL_FILE_SUFFIX)
    output_path.write_text("\n".join(script_lines) + "\n")
    print("Written " + str(output_path))


def write_parquet_to_sql(parquet_path = None, table_name = None):
    """
    This is the main() file that runs the above functions

    INPUTS:
        * parquet_path - the parquet input
        * table_name - the table name used in the script and the file name

    OUTPUTS:
        * a .sql file at Database/<table_name>.sql
    """
    dataframe = pd.read_parquet(parquet_path)
    write_dataframe_to_sql(
        dataframe = dataframe,
        table_name = table_name,
    )