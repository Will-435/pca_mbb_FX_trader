/*
spearman_correlation.cpp is a candidate file. It measures the co-movement between
the dollar and each other asset with a rolling Spearman rank correlation.

Spearman is Pearson computed on the ranks instead of the raw values. Because it
works on ranks it captures monotone relationships that bend, not just straight
line ones, and one outlier week cannot swing it far. Nothing in the pipeline
reads this yet. The output sits beside pearson_correlation.cpp and mi_spearman.py
so the three can be validated against each other before any of them is wired into
the regime step.

Build (from the repository root, same flags as the other C++ files):
    clang++ -std=c++17 candidate/secondMoment/spearman_correlation.cpp -o spearman_correlation \
        -I/opt/homebrew/opt/duckdb/include -I/opt/homebrew/opt/eigen/include/eigen3 \
        -L/opt/homebrew/opt/duckdb/lib -lduckdb

Outputs:
    second_moment_output/spearman.parquet - one rolling correlation column per
    asset, stamped on the last week of each window.
    Database/spearman.sql - the same table as a MySQL script for MySQL Workbench.
*/

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "duckdb.hpp"

const std::string WEEKLY_LEVELS_PATH = "Data/Input/weekly_levels.parquet";
const std::string OUTPUT_DIRECTORY = "candidate/secondMoment/second_moment_output/";
const std::string SPEARMAN_OUTPUT_PATH = OUTPUT_DIRECTORY + "spearman.parquet";
const std::string SPEARMAN_SQL_PATH = "Database/spearman.sql";

const std::string DATE_COLUMN_NAME = "week_date";
const std::string DOLLAR_COLUMN_NAME = "dollar_index";

// Assets measured against the dollar. First entry is the column name in the
// parquet, second is the label used in the output column.
const std::vector<std::pair<std::string, std::string>> ASSET_COLUMNS = {
    {"gold", "gold"},
    {"oil_wti", "oil"},
};

// Weeks of returns in one rolling window. Matches the window already used by the
// rolling correlation in pca.py so the candidates stay comparable.
const int ROLLING_WINDOW_WEEKS = 26;
const int SINGLE_WEEK_LAG = 1;

const int SQL_VALUE_PRECISION = 12;

const double NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();

// One based ranks start at this value.
const double FIRST_RANK = 1.0;


/*
Runs one query and stops with the error if it fails. Every read goes through here
so a broken query is easy to trace.

INPUTS:
    * connection - an open DuckDB connection
    * query_text - the SQL to run

OUTPUTS:
    * the materialised query result
*/
std::unique_ptr<duckdb::MaterializedQueryResult> run_query(
    duckdb::Connection& connection, const std::string& query_text)
{
    auto query_result = connection.Query(query_text);
    if (query_result->HasError()) {
        throw std::runtime_error("Query failed: " + query_text + "\n" + query_result->GetError());
    }
    return query_result;
}


// Creates the directory that will hold an output file if it is missing.
void ensure_parent_directory_exists(const std::string& file_path)
{
    std::filesystem::path parent_directory = std::filesystem::path(file_path).parent_path();
    if (!parent_directory.empty()) {
        std::filesystem::create_directories(parent_directory);
    }
}


// Reads the week_date column as plain date strings, ordered by date.
std::vector<std::string> load_week_dates(duckdb::Connection& connection)
{
    auto date_result = run_query(connection,
        "SELECT CAST(" + DATE_COLUMN_NAME + " AS DATE) FROM read_parquet('"
        + WEEKLY_LEVELS_PATH + "') ORDER BY " + DATE_COLUMN_NAME);

    std::vector<std::string> week_dates;
    for (duckdb::idx_t row_index = 0; row_index < date_result->RowCount(); row_index++) {
        week_dates.push_back(date_result->GetValue(0, row_index).ToString());
    }
    return week_dates;
}


/*
Reads one column of weekly levels from the parquet, ordered by date. A missing
value is read back as not a number so the return step can skip it.

INPUTS:
    * connection - an open DuckDB connection
    * column_name - the level column to read

OUTPUTS:
    * the column values in date order
*/
std::vector<double> load_level_column(duckdb::Connection& connection, const std::string& column_name)
{
    auto column_result = run_query(connection,
        "SELECT " + column_name + " FROM read_parquet('" + WEEKLY_LEVELS_PATH
        + "') ORDER BY " + DATE_COLUMN_NAME);

    std::vector<double> level_values;
    for (duckdb::idx_t row_index = 0; row_index < column_result->RowCount(); row_index++) {
        duckdb::Value cell_value = column_result->GetValue(0, row_index);
        if (cell_value.IsNull()) {
            level_values.push_back(NOT_A_NUMBER);
        } else {
            level_values.push_back(cell_value.GetValue<double>());
        }
    }
    return level_values;
}


/*
Turns a level column into a one week log return column. The first week has no
prior week, so it is not a number. A missing or non positive level also gives not
a number.

INPUTS:
    * level_values - one column of weekly levels

OUTPUTS:
    * the matching one week log returns, same length, first entry not a number
*/
Eigen::VectorXd compute_log_returns(const std::vector<double>& level_values)
{
    Eigen::VectorXd return_values(level_values.size());
    return_values(0) = NOT_A_NUMBER;
    for (std::size_t week_index = SINGLE_WEEK_LAG; week_index < level_values.size(); week_index++) {
        double previous_level = level_values[week_index - SINGLE_WEEK_LAG];
        double current_level = level_values[week_index];
        if (std::isnan(previous_level) || std::isnan(current_level)
            || previous_level <= 0.0 || current_level <= 0.0) {
            return_values(week_index) = NOT_A_NUMBER;
        } else {
            return_values(week_index) = std::log(current_level / previous_level);
        }
    }
    return return_values;
}


// True when the window holds no missing values, so it can be measured.
bool window_is_complete(const Eigen::VectorXd& window_values)
{
    return !window_values.array().isNaN().any();
}


/*
Turns a window of values into ranks. The smallest value gets rank one. Tied
values share the average of the ranks they span, which is the standard rule for
Spearman.

INPUTS:
    * window_values - one window of values

OUTPUTS:
    * the matching ranks, same length
*/
Eigen::VectorXd rank_values(const Eigen::VectorXd& window_values)
{
    std::size_t value_count = window_values.size();
    std::vector<std::size_t> sorted_positions(value_count);
    std::iota(sorted_positions.begin(), sorted_positions.end(), 0);
    std::sort(sorted_positions.begin(), sorted_positions.end(),
              [&window_values](std::size_t left_position, std::size_t right_position) {
                  return window_values(left_position) < window_values(right_position);
              });

    Eigen::VectorXd ranks(value_count);
    std::size_t block_start = 0;
    while (block_start < value_count) {
        // Group equal values so tied entries can share the average rank.
        std::size_t block_end = block_start;
        while (block_end + 1 < value_count
               && window_values(sorted_positions[block_end + 1]) == window_values(sorted_positions[block_start])) {
            block_end = block_end + 1;
        }
        double average_rank =
            (static_cast<double>(block_start) + static_cast<double>(block_end)) / 2.0 + FIRST_RANK;
        for (std::size_t tie_index = block_start; tie_index <= block_end; tie_index++) {
            ranks(sorted_positions[tie_index]) = average_rank;
        }
        block_start = block_end + 1;
    }
    return ranks;
}


/*
Pearson correlation of two equal length windows. Centre both, then divide the
summed product by the product of the two lengths. Returns not a number if either
window has no spread.

INPUTS:
    * first_window - one window of values
    * second_window - the matching window of values

OUTPUTS:
    * the Pearson correlation, between -1 and 1
*/
double compute_pearson_correlation(const Eigen::VectorXd& first_window,
                                   const Eigen::VectorXd& second_window)
{
    Eigen::VectorXd first_centred = first_window.array() - first_window.mean();
    Eigen::VectorXd second_centred = second_window.array() - second_window.mean();
    double normalising_product = std::sqrt(first_centred.squaredNorm() * second_centred.squaredNorm());
    if (normalising_product == 0.0) {
        return NOT_A_NUMBER;
    }
    return first_centred.dot(second_centred) / normalising_product;
}


// Spearman correlation is the Pearson correlation of the two windows' ranks.
double compute_spearman_correlation(const Eigen::VectorXd& first_window,
                                    const Eigen::VectorXd& second_window)
{
    return compute_pearson_correlation(rank_values(first_window), rank_values(second_window));
}


// Formats a correlation for the SQL insert, writing NULL for a missing value.
std::string format_correlation(double correlation_value)
{
    if (std::isnan(correlation_value)) {
        return "NULL";
    }
    std::ostringstream value_stream;
    value_stream << std::setprecision(SQL_VALUE_PRECISION) << correlation_value;
    return value_stream.str();
}


/*
Writes the rolling correlation table as a MySQL script so it can be opened in
MySQL Workbench. One INSERT per row, the same rows as the parquet output.

INPUTS:
    * table_name - the table name used in the script
    * column_list - the column definitions, shared with the parquet table
    * row_value_tuples - one VALUES tuple string per output row
    * sql_path - where the script is written

OUTPUTS:
    * a .sql script at sql_path
*/
void write_correlation_sql(const std::string& table_name,
                           const std::string& column_list,
                           const std::vector<std::string>& row_value_tuples,
                           const std::string& sql_path)
{
    ensure_parent_directory_exists(sql_path);
    std::ofstream sql_file(sql_path);
    if (!sql_file.is_open()) {
        throw std::runtime_error("Cannot open " + sql_path + " for writing.");
    }
    sql_file << "DROP TABLE IF EXISTS " << table_name << ";\n";
    sql_file << "CREATE TABLE " << table_name << " (" << column_list << ");\n";
    for (const std::string& row_value_tuple : row_value_tuples) {
        sql_file << "INSERT INTO " << table_name << " VALUES " << row_value_tuple << ";\n";
    }
}


/*
Loads the levels, builds one week returns, rolls a fixed window and writes the
rolling Spearman correlation of every asset against the dollar.

INPUTS:
    * none, paths come from the constants above

OUTPUTS:
    * second_moment_output/spearman.parquet
    * Database/spearman.sql
*/
int main()
{
    try {
        duckdb::DuckDB in_memory_database(nullptr);
        duckdb::Connection connection(in_memory_database);

        std::vector<std::string> week_dates = load_week_dates(connection);
        Eigen::VectorXd dollar_returns =
            compute_log_returns(load_level_column(connection, DOLLAR_COLUMN_NAME));

        // One return column per asset, in the asset list order.
        std::vector<Eigen::VectorXd> asset_returns;
        for (const auto& asset_entry : ASSET_COLUMNS) {
            asset_returns.push_back(compute_log_returns(load_level_column(connection, asset_entry.first)));
        }

        std::size_t week_count = week_dates.size();
        if (static_cast<int>(week_count) <= ROLLING_WINDOW_WEEKS) {
            throw std::runtime_error("Not enough weeks of data to fill one rolling window.");
        }
        std::cout << "Loaded " << week_count << " weeks of "
                  << ASSET_COLUMNS.size() << " assets against the dollar." << std::endl;

        // Output table: date column then one correlation column per asset.
        std::string column_list = DATE_COLUMN_NAME + " DATE";
        for (const auto& asset_entry : ASSET_COLUMNS) {
            column_list = column_list + ", " + asset_entry.second + "_usd_spearman DOUBLE";
        }
        run_query(connection, "CREATE OR REPLACE TABLE spearman_correlation (" + column_list + ")");

        // Every row is kept as a VALUES tuple so the same rows feed both the
        // parquet table and the MySQL script.
        std::vector<std::string> row_value_tuples;

        // Each window ends on window_end and looks back ROLLING_WINDOW_WEEKS weeks.
        for (std::size_t window_end = ROLLING_WINDOW_WEEKS - 1; window_end < week_count; window_end++) {
            std::size_t window_start = window_end + 1 - ROLLING_WINDOW_WEEKS;
            Eigen::VectorXd dollar_window = dollar_returns.segment(window_start, ROLLING_WINDOW_WEEKS);

            std::vector<double> correlation_values;
            for (const Eigen::VectorXd& single_asset_returns : asset_returns) {
                Eigen::VectorXd asset_window = single_asset_returns.segment(window_start, ROLLING_WINDOW_WEEKS);
                if (!window_is_complete(dollar_window) || !window_is_complete(asset_window)) {
                    correlation_values.push_back(NOT_A_NUMBER);
                } else {
                    correlation_values.push_back(compute_spearman_correlation(asset_window, dollar_window));
                }
            }

            // Skip the leading weeks where no asset has a full window yet.
            bool all_missing = true;
            for (double correlation_value : correlation_values) {
                if (!std::isnan(correlation_value)) {
                    all_missing = false;
                    break;
                }
            }
            if (all_missing) {
                continue;
            }

            std::ostringstream value_stream;
            value_stream << "('" << week_dates[window_end] << "'";
            for (double correlation_value : correlation_values) {
                value_stream << ", " << format_correlation(correlation_value);
            }
            value_stream << ")";
            std::string row_value_tuple = value_stream.str();
            row_value_tuples.push_back(row_value_tuple);
            run_query(connection, "INSERT INTO spearman_correlation VALUES " + row_value_tuple);
        }

        ensure_parent_directory_exists(SPEARMAN_OUTPUT_PATH);
        run_query(connection, "COPY spearman_correlation TO '" + SPEARMAN_OUTPUT_PATH + "' (FORMAT PARQUET)");
        std::cout << "Written " << SPEARMAN_OUTPUT_PATH << std::endl;

        write_correlation_sql("spearman_correlation", column_list, row_value_tuples, SPEARMAN_SQL_PATH);
        std::cout << "Written " << SPEARMAN_SQL_PATH << std::endl;
    } catch (const std::exception& thrown_error) {
        std::cerr << thrown_error.what() << std::endl;
        return 1;
    }
    return 0;
}
