/*
momentum.cpp computes the two week FX momentum with Eigen. This file is only the
linear algebra computation. Data handling is done in momentum.py.

The whole file is only two linear opperation. 

    Momentum. With a horizon of h weeks, the momentum matrix is
    M = L(h..n) - L(1..n-h), a difference of two row blocks. On log levels
    that subtraction is exactly the h week log return of each currency
    against the dollar.

    Cross rates. The log price of currency a in terms of currency b is
    log(a/USD) - log(b/USD), so just the log diff between the two momenta.
    For the newest week, the full matrix of cross momenta is built as 
    C = (m 1^T) - (1 m^T), where m is the latest momentum column vector and 1
    is a vector of all ones.

** I am currently researching for a more indsutry standard approach. **
** This works for now. **

Outputs, both read back by fx_momentum.py:
    momentum_2w.parquet - momentum series per currency
    momentum_cross_latest.parquet - latest cross matrix

*/

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "duckdb.hpp"

const std::string MOMENTUM_INPUT_PATH = "components/momentum_output/momentum_input.parquet";
const std::string MOMENTUM_TIMESERIES_PATH = "components/momentum_output/momentum_2w.parquet";
const std::string MOMENTUM_CROSS_PATH = "components/momentum_output/momentum_cross_latest.parquet";

const std::string DATE_COLUMN_NAME = "week_date";
const std::string BASE_CURRENCY_COLUMN_NAME = "base_currency";

// The momentum horizon. Two weekly rows back equals two calendar weeks on the
// Friday stamped grid.
const int MOMENTUM_HORIZON_WEEKS = 2;

const int SQL_VALUE_PRECISION = 12;


/*
Runs one query and fails loudly if DuckDB reports a problem.

INPUTS:
    * connection, an open DuckDB connection
    * query_text, the SQL to run

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


// This just makes the output file fr our outputs
void ensure_parent_directory_exists(const std::string& file_path)
{
    std::filesystem::path parent_directory = std::filesystem::path(file_path).parent_path();
    if (!parent_directory.empty()) {
        std::filesystem::create_directories(parent_directory);
    }
}


/*
Reads the log level matrix from fx_momentum.py into an Eigen matrix,
one row per week and one column per currency, with dates and names.

INPUTS:
    * connection, an open DuckDB connection

OUTPUTS:
    * week_dates - one date string per row
    * currency_names - one name per column
    * log_level_matrix - the Eigen matrix of log dollar prices
*/
void load_log_level_matrix(duckdb::Connection& connection,
                           std::vector<std::string>& week_dates,
                           std::vector<std::string>& currency_names,
                           Eigen::MatrixXd& log_level_matrix)
{
    auto input_result = run_query(connection,
        "SELECT * FROM read_parquet('" + MOMENTUM_INPUT_PATH + "') ORDER BY " + DATE_COLUMN_NAME);

    const std::vector<std::string>& column_names = input_result->names;
    if (column_names.empty() || column_names.front() != DATE_COLUMN_NAME) {
        throw std::runtime_error(
            "Unexpected column layout in " + MOMENTUM_INPUT_PATH
            + ". Run fx_momentum.py first, and run this program from the repository root.");
    }
    for (std::size_t column_position = 1; column_position < column_names.size(); column_position++) {
        currency_names.push_back(column_names[column_position]);
    }

    std::size_t week_count = input_result->RowCount();
    log_level_matrix.resize(week_count, currency_names.size());
    for (duckdb::idx_t row_index = 0; row_index < week_count; row_index++) {
        week_dates.push_back(input_result->GetValue(0, row_index).ToString());
        for (std::size_t currency_position = 0; currency_position < currency_names.size(); currency_position++) {
            log_level_matrix(row_index, currency_position) =
                input_result->GetValue(currency_position + 1, row_index).GetValue<double>();
        }
    }
}


/*
Writes the momentum series as parquet with DuckDB. The table carries the
end week of each momentum window and one column per currency.

INPUTS:
    * connection, an open DuckDB connection
    * week_dates, the full week date list from the input
    * currency_names, the column labels
    * momentum_matrix, the Eigen momentum matrix

OUTPUTS:
    * momentum_output/momentum_2w.parquet
*/
void write_momentum_timeseries(duckdb::Connection& connection,
                               const std::vector<std::string>& week_dates,
                               const std::vector<std::string>& currency_names,
                               const Eigen::MatrixXd& momentum_matrix)
{
    std::string column_list = DATE_COLUMN_NAME + " DATE";
    for (const std::string& currency_name : currency_names) {
        column_list = column_list + ", " + currency_name + " DOUBLE";
    }
    run_query(connection, "CREATE OR REPLACE TABLE momentum_timeseries (" + column_list + ")");

    for (Eigen::Index momentum_row = 0; momentum_row < momentum_matrix.rows(); momentum_row++) {
        std::ostringstream value_stream;
        value_stream << std::setprecision(SQL_VALUE_PRECISION);
        // Momentum row r spans input weeks r to r + horizon, so it is stamped
        // with the end week of that window.
        value_stream << "('" << week_dates[momentum_row + MOMENTUM_HORIZON_WEEKS] << "'";
        for (Eigen::Index currency_position = 0; currency_position < momentum_matrix.cols(); currency_position++) {
            value_stream << ", " << momentum_matrix(momentum_row, currency_position);
        }
        value_stream << ")";
        run_query(connection, "INSERT INTO momentum_timeseries VALUES " + value_stream.str());
    }

    run_query(connection,
        "COPY momentum_timeseries TO '" + MOMENTUM_TIMESERIES_PATH + "' (FORMAT PARQUET)");
}


/*
Writes the latest cross momentum matrix as parquet through DuckDB. Row a,
column b holds the two week momentum of currency a priced in currency b.

INPUTS:
    * connection, an open DuckDB connection
    * currency_names, the row and column labels
    * cross_momentum_matrix, the Eigen cross matrix for the newest week

OUTPUTS:
    * momentum_output/momentum_cross_latest.parquet
*/
void write_cross_momentum_matrix(duckdb::Connection& connection,
                                 const std::vector<std::string>& currency_names,
                                 const Eigen::MatrixXd& cross_momentum_matrix)
{
    std::string column_list = BASE_CURRENCY_COLUMN_NAME + " VARCHAR";
    for (const std::string& currency_name : currency_names) {
        column_list = column_list + ", " + currency_name + " DOUBLE";
    }
    run_query(connection, "CREATE OR REPLACE TABLE momentum_cross (" + column_list + ")");

    for (Eigen::Index base_position = 0; base_position < cross_momentum_matrix.rows(); base_position++) {
        std::ostringstream value_stream;
        value_stream << std::setprecision(SQL_VALUE_PRECISION);
        value_stream << "('" << currency_names[base_position] << "'";
        for (Eigen::Index quote_position = 0; quote_position < cross_momentum_matrix.cols(); quote_position++) {
            value_stream << ", " << cross_momentum_matrix(base_position, quote_position);
        }
        value_stream << ")";
        run_query(connection, "INSERT INTO momentum_cross VALUES " + value_stream.str());
    }

    run_query(connection,
        "COPY momentum_cross TO '" + MOMENTUM_CROSS_PATH + "' (FORMAT PARQUET)");
}


/*
Loads the log levels, runs the two linear operations and writes both outputs.

INPUTS:
    * none, paths come from the constants above

OUTPUTS:
    * momentum_output/momentum_2w.parquet
    * momentum_output/momentum_cross_latest.parquet
*/
int main()
{
    try {
        duckdb::DuckDB in_memory_database(nullptr);
        duckdb::Connection connection(in_memory_database);

        std::vector<std::string> week_dates;
        std::vector<std::string> currency_names;
        Eigen::MatrixXd log_level_matrix;
        load_log_level_matrix(connection, week_dates, currency_names, log_level_matrix);

        Eigen::Index week_count = log_level_matrix.rows();
        if (week_count <= MOMENTUM_HORIZON_WEEKS) {
            throw std::runtime_error("Not enough weeks of data to span the momentum horizon.");
        }
        std::cout << "Loaded " << week_count << " weeks of "
                  << currency_names.size() << " currencies." << std::endl;

        // Momentum as a block difference: row t of the result is the log
        // level at week t + horizon minus the log level at week t.
        Eigen::Index momentum_row_count = week_count - MOMENTUM_HORIZON_WEEKS;
        Eigen::MatrixXd momentum_matrix =
            log_level_matrix.bottomRows(momentum_row_count)
            - log_level_matrix.topRows(momentum_row_count);

        // Latest cross matrix as an outer difference: C = m 1' - 1 m'. Row a,
        // column b is momentum of a minus momentum of b, the momentum of the
        // a/b cross rate. The diagonal is zero by construction.
        Eigen::VectorXd latest_momentum_vector =
            momentum_matrix.row(momentum_row_count - 1).transpose();
        Eigen::Index currency_count = latest_momentum_vector.size();
        Eigen::MatrixXd cross_momentum_matrix =
            latest_momentum_vector * Eigen::RowVectorXd::Ones(currency_count)
            - Eigen::VectorXd::Ones(currency_count) * latest_momentum_vector.transpose();

        ensure_parent_directory_exists(MOMENTUM_TIMESERIES_PATH);
        write_momentum_timeseries(connection, week_dates, currency_names, momentum_matrix);
        std::cout << "Written " << MOMENTUM_TIMESERIES_PATH << std::endl;

        write_cross_momentum_matrix(connection, currency_names, cross_momentum_matrix);
        std::cout << "Written " << MOMENTUM_CROSS_PATH << std::endl;

        std::cout << "Rerun fx_momentum.py to export the results and print the ranking."
                  << std::endl;
    } catch (const std::exception& thrown_error) {
        std::cerr << thrown_error.what() << std::endl;
        return 1;
    }
    return 0;
}