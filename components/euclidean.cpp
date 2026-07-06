/*
euclidean.cpp selects the historic neighbourhood of the current point in
frozen PC space. The neighbourhood is the regime. There is no predefined
regime label anywhere, the input is only the current point's coordinates.

The program reads the frozen PCA model from pca_model.parquet through DuckDB,
so no pickle crosses the language boundary. It projects every historic week
onto the frozen axes, measures plain Euclidean distance to the current point,
selects the neighbourhood, and writes the members' forward four week returns
to regime_members.duckdb for bootstrap.py.

Why plain Euclidean on unwhitened scores: the PC scores keep their natural
variance weighting, so distance along a dominant macro direction counts for
more than distance along a minor one. Whitening the scores or recomputing a
fresh covariance on live data would erase exactly the drift this step is
meant to detect. Euclidean distance on frozen unwhitened scores is a frozen
Mahalanobis distance in feature space.

Working assumption, accepted: drift behaviour inside a neighbourhood is
consistent, so the members' forward returns are informative about the
current point's forward return.

Build from the repository root, DuckDB is installed via Homebrew:
    g++ -std=c++17 components/euclidean.cpp -o euclidean \
        -I/opt/homebrew/opt/duckdb/include -L/opt/homebrew/opt/duckdb/lib -lduckdb

Usage, run from the repository root after pca.py:
    ./euclidean <pc_1> <pc_2> <pc_3>
The number of coordinates must match the component count of the frozen model.
pca.py prints the newest week's coordinates in exactly this format.
*/

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "duckdb.hpp"

const std::string PCA_MODEL_PATH = "components/pca_output/pca_model.parquet";
const std::string FEATURE_MATRIX_PATH = "Data/Input/feature_matrix.parquet";
const std::string REGIME_MEMBERS_DATABASE_PATH = "components/euclidean_output/regime_members.duckdb";
const std::string REGIME_MEMBERS_SQL_PATH = "Database/regime_members.sql";

const std::string DATE_COLUMN_NAME = "week_date";
const std::string TARGET_COLUMN_NAME = "usd_krw_forward_return_4w";

// TODO: choose the neighbourhood rule, k_nearest or distance_threshold. The
// choice sets the bootstrap sample size. k_nearest fixes the sample size and
// lets the radius float, distance_threshold fixes the radius and lets the
// sample size float. Both are implemented below, this constant picks one.
const std::string NEIGHBOURHOOD_RULE = "k_nearest";

// TODO: calibrate both parameters. Overlapping four week returns carry about
// thirteen independent observations per year, so forty weekly members hold
// roughly ten independent observations. Do not grow this casually.
const int NEIGHBOUR_COUNT = 40;
const double DISTANCE_THRESHOLD = 1.0;

const int SQL_VALUE_PRECISION = 12;


// One historic week: its date, frozen PC scores, forward return and distance
// to the current point.
struct HistoricPoint {
    std::string week_date;
    std::vector<double> pc_scores;
    double forward_return;
    double distance_to_current_point;
};


// The frozen PCA model as read back from pca_model.parquet.
struct FrozenPcaModel {
    std::vector<std::string> feature_names;
    std::vector<double> feature_means;
    // loadings[feature_position][component_position]
    std::vector<std::vector<double>> loadings;
    int component_count;
};


/*
Runs one query and fails loudly if DuckDB reports a problem. Every read in
this program goes through this helper so errors carry the failing query.

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


/*
Reads the frozen PCA model from pca_model.parquet. The file is one flat table
where record_type marks each row as a loading, a feature mean or something
else. The feature_order column carries the exact feature ordering that the
projection depends on. Numeric columns are read as doubles because the flat
table leaves them blank on rows where they do not apply.

INPUTS:
    * connection, an open DuckDB connection

OUTPUTS:
    * the frozen model with feature names, means and loadings
*/
FrozenPcaModel load_frozen_pca_model(duckdb::Connection& connection)
{
    FrozenPcaModel frozen_model;

    auto mean_result = run_query(connection,
        "SELECT feature_name, value FROM read_parquet('" + PCA_MODEL_PATH + "') "
        "WHERE record_type = 'mean' ORDER BY feature_order");
    for (duckdb::idx_t row_index = 0; row_index < mean_result->RowCount(); row_index++) {
        frozen_model.feature_names.push_back(mean_result->GetValue(0, row_index).ToString());
        frozen_model.feature_means.push_back(mean_result->GetValue(1, row_index).GetValue<double>());
    }
    if (frozen_model.feature_names.empty()) {
        throw std::runtime_error("No mean records found in " + PCA_MODEL_PATH + ". Run pca.py first.");
    }

    auto component_result = run_query(connection,
        "SELECT MAX(component_number) FROM read_parquet('" + PCA_MODEL_PATH + "') "
        "WHERE record_type = 'loading'");
    frozen_model.component_count =
        static_cast<int>(component_result->GetValue(0, 0).GetValue<double>());

    // The loadings arrive ordered by feature then component, so they fill the
    // matrix row by row.
    frozen_model.loadings.assign(
        frozen_model.feature_names.size(),
        std::vector<double>(frozen_model.component_count, 0.0));
    auto loading_result = run_query(connection,
        "SELECT feature_order, component_number, value "
        "FROM read_parquet('" + PCA_MODEL_PATH + "') "
        "WHERE record_type = 'loading' ORDER BY feature_order, component_number");
    for (duckdb::idx_t row_index = 0; row_index < loading_result->RowCount(); row_index++) {
        // feature_order and component_number are one based in the file.
        int feature_position =
            static_cast<int>(loading_result->GetValue(0, row_index).GetValue<double>()) - 1;
        int component_position =
            static_cast<int>(loading_result->GetValue(1, row_index).GetValue<double>()) - 1;
        frozen_model.loadings[feature_position][component_position] =
            loading_result->GetValue(2, row_index).GetValue<double>();
    }

    return frozen_model;
}


/*
Projects one week's feature values onto the frozen axes: subtract the frozen
means, multiply by the frozen loadings. No whitening.

INPUTS:
    * feature_values, one week's features in model feature order
    * frozen_model, from load_frozen_pca_model

OUTPUTS:
    * the week's PC scores
*/
std::vector<double> project_onto_frozen_axes(const std::vector<double>& feature_values,
                                             const FrozenPcaModel& frozen_model)
{
    std::vector<double> pc_scores(frozen_model.component_count, 0.0);
    for (int component_position = 0; component_position < frozen_model.component_count; component_position++) {
        for (std::size_t feature_position = 0; feature_position < feature_values.size(); feature_position++) {
            double centred_value = feature_values[feature_position]
                                 - frozen_model.feature_means[feature_position];
            pc_scores[component_position] = pc_scores[component_position]
                + centred_value * frozen_model.loadings[feature_position][component_position];
        }
    }
    return pc_scores;
}


/*
Loads every historic week that already has a known forward return and puts it
in frozen PC space. Weeks whose forward return has not happened yet cannot be
neighbourhood members, so they are filtered out in the query. The date column
is cast to a plain DATE, parquet otherwise carries it as a timestamp. The
column order is checked against the model feature order, a silent mismatch
there would corrupt every projection.

INPUTS:
    * connection, an open DuckDB connection
    * frozen_model, from load_frozen_pca_model

OUTPUTS:
    * the list of historic points with scores and forward returns
*/
std::vector<HistoricPoint> load_historic_points(duckdb::Connection& connection,
                                                const FrozenPcaModel& frozen_model)
{
    auto matrix_result = run_query(connection,
        "SELECT CAST(" + DATE_COLUMN_NAME + " AS DATE) AS " + DATE_COLUMN_NAME
        + ", * EXCLUDE (" + DATE_COLUMN_NAME + ") FROM read_parquet('" + FEATURE_MATRIX_PATH + "') "
        "WHERE " + TARGET_COLUMN_NAME + " IS NOT NULL ORDER BY " + DATE_COLUMN_NAME);

    const std::vector<std::string>& column_names = matrix_result->names;
    std::size_t expected_column_count = frozen_model.feature_names.size() + 2;
    if (column_names.size() != expected_column_count
        || column_names.front() != DATE_COLUMN_NAME
        || column_names.back() != TARGET_COLUMN_NAME) {
        throw std::runtime_error("Unexpected column layout in " + FEATURE_MATRIX_PATH);
    }
    for (std::size_t feature_position = 0; feature_position < frozen_model.feature_names.size(); feature_position++) {
        if (column_names[feature_position + 1] != frozen_model.feature_names[feature_position]) {
            throw std::runtime_error(
                "Feature order mismatch between " + FEATURE_MATRIX_PATH + " and "
                + PCA_MODEL_PATH + " at column " + column_names[feature_position + 1]
                + ". Re-run pca.py so both files come from the same run.");
        }
    }

    std::vector<HistoricPoint> historic_points;
    for (duckdb::idx_t row_index = 0; row_index < matrix_result->RowCount(); row_index++) {
        HistoricPoint historic_point;
        historic_point.week_date = matrix_result->GetValue(0, row_index).ToString();

        std::vector<double> feature_values;
        for (std::size_t feature_position = 0; feature_position < frozen_model.feature_names.size(); feature_position++) {
            feature_values.push_back(
                matrix_result->GetValue(feature_position + 1, row_index).GetValue<double>());
        }
        historic_point.pc_scores = project_onto_frozen_axes(feature_values, frozen_model);
        historic_point.forward_return =
            matrix_result->GetValue(column_names.size() - 1, row_index).GetValue<double>();
        historic_point.distance_to_current_point = 0.0;
        historic_points.push_back(historic_point);
    }
    return historic_points;
}


// Computes plain Euclidean distance between two points in PC space.
double compute_euclidean_distance(const std::vector<double>& first_point,
                                  const std::vector<double>& second_point)
{
    double squared_distance = 0.0;
    for (std::size_t coordinate_position = 0; coordinate_position < first_point.size(); coordinate_position++) {
        double coordinate_difference = first_point[coordinate_position] - second_point[coordinate_position];
        squared_distance = squared_distance + coordinate_difference * coordinate_difference;
    }
    return std::sqrt(squared_distance);
}


/*
Selects the neighbourhood as the fixed number of nearest historic points.
Sample size is fixed, radius floats with how crowded the region is.

INPUTS:
    * sorted_points, historic points sorted by ascending distance

OUTPUTS:
    * the selected members
*/
std::vector<HistoricPoint> select_neighbours_by_count(const std::vector<HistoricPoint>& sorted_points)
{
    std::size_t member_count = std::min(
        static_cast<std::size_t>(NEIGHBOUR_COUNT), sorted_points.size());
    return std::vector<HistoricPoint>(sorted_points.begin(), sorted_points.begin() + member_count);
}


/*
Selects the neighbourhood as every historic point within the distance
threshold. Radius is fixed, sample size floats with how crowded the region
is. The threshold has not been calibrated yet.

INPUTS:
    * sorted_points, historic points sorted by ascending distance

OUTPUTS:
    * the selected members
*/
std::vector<HistoricPoint> select_neighbours_by_threshold(const std::vector<HistoricPoint>& sorted_points)
{
    std::vector<HistoricPoint> selected_members;
    for (const HistoricPoint& historic_point : sorted_points) {
        if (historic_point.distance_to_current_point <= DISTANCE_THRESHOLD) {
            selected_members.push_back(historic_point);
        }
    }
    return selected_members;
}


// Builds the shared column list of the regime_members table.
std::string build_member_column_list(int component_count)
{
    std::string column_list = "week_date DATE, distance_to_current_point DOUBLE";
    for (int component_position = 1; component_position <= component_count; component_position++) {
        column_list = column_list + ", pc_" + std::to_string(component_position) + " DOUBLE";
    }
    column_list = column_list + ", forward_return_4w DOUBLE";
    return column_list;
}


// Formats one member as a SQL values tuple, shared by both writers.
std::string build_member_value_tuple(const HistoricPoint& member)
{
    std::ostringstream value_stream;
    value_stream << std::setprecision(SQL_VALUE_PRECISION);
    value_stream << "('" << member.week_date << "', " << member.distance_to_current_point;
    for (double pc_score : member.pc_scores) {
        value_stream << ", " << pc_score;
    }
    value_stream << ", " << member.forward_return << ")";
    return value_stream.str();
}


// Creates the directory that will hold an output file if it does not exist,
// so a fresh clone with git ignored output folders still writes cleanly.
void ensure_parent_directory_exists(const std::string& file_path)
{
    std::filesystem::path parent_directory = std::filesystem::path(file_path).parent_path();
    if (!parent_directory.empty()) {
        std::filesystem::create_directories(parent_directory);
    }
}


/*
Writes the neighbourhood members into regime_members.duckdb, the handoff file
that bootstrap.py reads. The table is replaced on every run, one run means
one neighbourhood.

INPUTS:
    * members, the selected neighbourhood
    * component_count, sets the pc score columns

OUTPUTS:
    * the regime_members table in regime_members.duckdb
*/
void write_regime_members_database(const std::vector<HistoricPoint>& members, int component_count)
{
    ensure_parent_directory_exists(REGIME_MEMBERS_DATABASE_PATH);
    duckdb::DuckDB output_database(REGIME_MEMBERS_DATABASE_PATH);
    duckdb::Connection output_connection(output_database);

    run_query(output_connection,
        "CREATE OR REPLACE TABLE regime_members (" + build_member_column_list(component_count) + ")");

    for (const HistoricPoint& member : members) {
        run_query(output_connection,
            "INSERT INTO regime_members VALUES " + build_member_value_tuple(member));
    }
}


/*
Writes the same members as a MySQL script so the neighbourhood can be
inspected in MySQL Workbench like every other pipeline artefact.

INPUTS:
    * members, the selected neighbourhood
    * component_count, sets the pc score columns

OUTPUTS:
    * Database/regime_members.sql
*/
void write_regime_members_sql(const std::vector<HistoricPoint>& members, int component_count)
{
    ensure_parent_directory_exists(REGIME_MEMBERS_SQL_PATH);
    std::ofstream sql_file(REGIME_MEMBERS_SQL_PATH);
    if (!sql_file.is_open()) {
        throw std::runtime_error("Cannot open " + REGIME_MEMBERS_SQL_PATH + " for writing.");
    }
    sql_file << "DROP TABLE IF EXISTS regime_members;\n";
    sql_file << "CREATE TABLE regime_members (" << build_member_column_list(component_count) << ");\n";
    for (const HistoricPoint& member : members) {
        sql_file << "INSERT INTO regime_members VALUES " << build_member_value_tuple(member) << ";\n";
    }
}


/*
Reads the current point from the command line, measures its distance to every
historic week in frozen PC space, selects the neighbourhood under the chosen
rule and writes the members for bootstrap.py.

INPUTS:
    * one command line argument per PC coordinate of the current point

OUTPUTS:
    * regime_members.duckdb and Database/regime_members.sql
*/
int main(int argument_count, char* argument_values[])
{
    try {
        duckdb::DuckDB in_memory_database(nullptr);
        duckdb::Connection connection(in_memory_database);

        FrozenPcaModel frozen_model = load_frozen_pca_model(connection);

        int coordinate_count = argument_count - 1;
        if (coordinate_count != frozen_model.component_count) {
            std::cerr << "Usage: ./euclidean <pc_1> ... <pc_" << frozen_model.component_count
                      << ">" << std::endl;
            std::cerr << "The frozen model has " << frozen_model.component_count
                      << " components, so exactly that many coordinates are required."
                      << std::endl;
            std::cerr << "pca.py prints the newest week's coordinates in this format."
                      << std::endl;
            return 1;
        }
        std::vector<double> current_point;
        for (int argument_index = 1; argument_index < argument_count; argument_index++) {
            current_point.push_back(std::stod(argument_values[argument_index]));
        }

        std::vector<HistoricPoint> historic_points = load_historic_points(connection, frozen_model);
        for (HistoricPoint& historic_point : historic_points) {
            historic_point.distance_to_current_point =
                compute_euclidean_distance(historic_point.pc_scores, current_point);
        }
        std::sort(historic_points.begin(), historic_points.end(),
                  [](const HistoricPoint& left_point, const HistoricPoint& right_point) {
                      return left_point.distance_to_current_point
                           < right_point.distance_to_current_point;
                  });

        std::vector<HistoricPoint> members;
        if (NEIGHBOURHOOD_RULE == "k_nearest") {
            members = select_neighbours_by_count(historic_points);
        } else if (NEIGHBOURHOOD_RULE == "distance_threshold") {
            members = select_neighbours_by_threshold(historic_points);
        } else {
            throw std::runtime_error("Unknown NEIGHBOURHOOD_RULE: " + NEIGHBOURHOOD_RULE);
        }
        if (members.empty()) {
            throw std::runtime_error(
                "The neighbourhood is empty. Loosen DISTANCE_THRESHOLD or check the input point.");
        }

        write_regime_members_database(members, frozen_model.component_count);
        write_regime_members_sql(members, frozen_model.component_count);

        double return_total = 0.0;
        for (const HistoricPoint& member : members) {
            return_total = return_total + member.forward_return;
        }
        std::cout << "Rule " << NEIGHBOURHOOD_RULE << " selected " << members.size()
                  << " members from " << historic_points.size() << " historic weeks." << std::endl;
        std::cout << "Distance range " << members.front().distance_to_current_point
                  << " to " << members.back().distance_to_current_point << "." << std::endl;
        std::cout << "Mean forward four week return of members "
                  << return_total / members.size()
                  << ". Reference only, bootstrap.py builds the distribution." << std::endl;
        std::cout << "Written " << REGIME_MEMBERS_DATABASE_PATH << std::endl;
        std::cout << "Written " << REGIME_MEMBERS_SQL_PATH << std::endl;
    } catch (const std::exception& thrown_error) {
        std::cerr << thrown_error.what() << std::endl;
        return 1;
    }
    return 0;
}