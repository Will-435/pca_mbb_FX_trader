/*
data_relation.cpp is a diagnostic file. It manually runs PCA to see the Cov and Cor
between features. If the Corr of Cov matricies are near diagonal, then it will have
the same number of eigenevctors as features, making PCA redundant.

The outputs will tell us if PCA is worth doing. If Corr and Cov matricies have strong
diaginals, then PCA won't be able to compress features well.
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "duckdb.hpp"

const std::string FEATURE_MATRIX_PATH = "Data/Input/feature_matrix.parquet";
const std::string OUTPUT_DIRECTORY = "components/data_relation_output/";
const std::string CORRELATION_IMAGE_PATH = OUTPUT_DIRECTORY + "correlation_matrix.png";
const std::string COVARIANCE_IMAGE_PATH = OUTPUT_DIRECTORY + "covariance_matrix.png";
const std::string EIGEN_REPORT_PATH = OUTPUT_DIRECTORY + "eigen.txt";

const std::string DATE_COLUMN_NAME = "week_date";
const std::string TARGET_COLUMN_NAME = "usd_krw_forward_return_4w";

// Sample covariance divides by row count minus one.
const int SAMPLE_COVARIANCE_DEGREES_OF_FREEDOM = 1;

const int REPORT_DECIMAL_PLACES = 6;

// Heatmap rendering. Each matrix cell becomes a square of this many pixels.
const int CELL_PIXEL_SIZE = 60;
const int CHANNELS_PER_PIXEL = 3;
const int FULL_COLOUR_INTENSITY = 255;
const double MINIMUM_DISPLAY_VALUE = -1.0;
const double MAXIMUM_DISPLAY_VALUE = 1.0;

// PNG and zlib format numbers. The writer emits uncompressed PNG files, which
// keeps this diagnostic free of image libraries.
const int BITS_PER_BYTE = 8;
const unsigned char PNG_BIT_DEPTH = 8;
const unsigned char PNG_COLOUR_TYPE_RGB = 2;
const unsigned char ZLIB_HEADER_FIRST_BYTE = 0x78;
const unsigned char ZLIB_HEADER_SECOND_BYTE = 0x01;
const std::size_t MAXIMUM_STORED_BLOCK_SIZE = 65535;
const uint32_t ADLER_MODULUS = 65521;
const uint32_t CRC_POLYNOMIAL = 0xEDB88320u;
const uint32_t CRC_INITIAL_VALUE = 0xFFFFFFFFu;


/*
Runs one query and fails DuckDB reports a problem.

INPUTS:
    * connection - an open DuckDB connection
    * query_text - the SQL to run

OUTPUTS:
    * the query output
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
Reads the weekly feature matrix parquet and extracts the feature columns.

INPUTS:
    * file_path - the parquet

OUTPUTS:
    * feature_names - the feature column names in file order
    * feature_rows - one vector of feature values per week
*/
void read_feature_matrix(const std::string& file_path,
                         std::vector<std::string>& feature_names,
                         std::vector<std::vector<double>>& feature_rows)
{
    duckdb::DuckDB in_memory_database(nullptr);
    duckdb::Connection connection(in_memory_database);

    std::unique_ptr<duckdb::MaterializedQueryResult> matrix_result;
    try {
        matrix_result = run_query(connection,
            "SELECT * FROM read_parquet('" + file_path + "') ORDER BY " + DATE_COLUMN_NAME);
    } catch (const std::exception& thrown_error) {
        throw std::runtime_error(
            "Cannot read " + file_path + ". Run pca.py first and run this program"
            + " from the repository root.\n" + thrown_error.what());
    }

    const std::vector<std::string>& column_names = matrix_result->names;
    std::vector<std::size_t> feature_column_positions;
    for (std::size_t column_position = 0; column_position < column_names.size(); column_position++) {
        if (column_names[column_position] == DATE_COLUMN_NAME
            || column_names[column_position] == TARGET_COLUMN_NAME) {
            continue;
        }
        feature_column_positions.push_back(column_position);
        feature_names.push_back(column_names[column_position]);
    }

    for (duckdb::idx_t row_index = 0; row_index < matrix_result->RowCount(); row_index++) {
        std::vector<double> feature_values;
        for (std::size_t column_position : feature_column_positions) {
            feature_values.push_back(
                matrix_result->GetValue(column_position, row_index).GetValue<double>());
        }
        feature_rows.push_back(feature_values);
    }
}


/*
Just computes the mean of each feature column

INPUTS:
    * feature_rows, one vector of feature values per week

OUTPUTS:
    * the vector of column means
*/
std::vector<double> compute_column_means(const std::vector<std::vector<double>>& feature_rows)
{
    std::size_t feature_count = feature_rows.front().size();
    std::vector<double> column_means(feature_count, 0.0);
    for (const std::vector<double>& feature_values : feature_rows) {
        for (std::size_t feature_index = 0; feature_index < feature_count; feature_index++) {
            column_means[feature_index] = column_means[feature_index] + feature_values[feature_index];
        }
    }
    for (std::size_t feature_index = 0; feature_index < feature_count; feature_index++) {
        column_means[feature_index] = column_means[feature_index] / feature_rows.size();
    }
    return column_means;
}


/*
Computes the covariance matrix of the columns.

INPUTS:
    * feature_rows - one vector of feature values per week

OUTPUTS:
    * the covariance matrix as nested vectors
*/
std::vector<std::vector<double>> compute_covariance_matrix(
    const std::vector<std::vector<double>>& feature_rows)
{
    std::size_t feature_count = feature_rows.front().size();
    std::vector<double> column_means = compute_column_means(feature_rows);
    std::vector<std::vector<double>> covariance_matrix(
        feature_count, std::vector<double>(feature_count, 0.0));

    for (const std::vector<double>& feature_values : feature_rows) {
        for (std::size_t row_index = 0; row_index < feature_count; row_index++) {
            for (std::size_t column_index = 0; column_index < feature_count; column_index++) {
                double row_deviation = feature_values[row_index] - column_means[row_index];
                double column_deviation = feature_values[column_index] - column_means[column_index];
                covariance_matrix[row_index][column_index] =
                    covariance_matrix[row_index][column_index] + row_deviation * column_deviation;
            }
        }
    }

    double sample_denominator =
        static_cast<double>(feature_rows.size()) - SAMPLE_COVARIANCE_DEGREES_OF_FREEDOM;
    for (std::size_t row_index = 0; row_index < feature_count; row_index++) {
        for (std::size_t column_index = 0; column_index < feature_count; column_index++) {
            covariance_matrix[row_index][column_index] =
                covariance_matrix[row_index][column_index] / sample_denominator;
        }
    }
    return covariance_matrix;
}


/*
Derives the correlation matrix from the covariance matrix by dividing each
entry by the 2 sigma.

INPUTS:
    * covariance_matri

OUTPUTS:
    * correlation matrix as vectors
*/
std::vector<std::vector<double>> compute_correlation_matrix(
    const std::vector<std::vector<double>>& covariance_matrix)
{
    std::size_t feature_count = covariance_matrix.size();
    std::vector<std::vector<double>> correlation_matrix(
        feature_count, std::vector<double>(feature_count, 0.0));
    for (std::size_t row_index = 0; row_index < feature_count; row_index++) {
        for (std::size_t column_index = 0; column_index < feature_count; column_index++) {
            double scale_product = std::sqrt(
                covariance_matrix[row_index][row_index]
                * covariance_matrix[column_index][column_index]);
            correlation_matrix[row_index][column_index] =
                covariance_matrix[row_index][column_index] / scale_product;
        }
    }
    return correlation_matrix;
}


/*
Decomposes a symmetric matrix with Eigen's self adjoint solver for symmetric matrices.
Results are sorted by descending eigenvalue and each eigenvector has its sign fixed so 
its largest weight is positive (same as in pca.py).

INPUTS:
    * symmetric_matrix

OUTPUTS:
    * eigenvalues
    * eigenvectors
*/
void compute_eigen_decomposition(const std::vector<std::vector<double>>& symmetric_matrix,
                                 std::vector<double>& eigenvalues,
                                 std::vector<std::vector<double>>& eigenvectors)
{
    std::size_t dimension = symmetric_matrix.size();

    Eigen::MatrixXd eigen_input(dimension, dimension);
    for (std::size_t row_index = 0; row_index < dimension; row_index++) {
        for (std::size_t column_index = 0; column_index < dimension; column_index++) {
            eigen_input(row_index, column_index) = symmetric_matrix[row_index][column_index];
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(eigen_input);
    if (eigen_solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed on the diagnostic matrix.");
    }

    // Eigen returns eigenvalues in ascending order, so the pairs are read
    // back to front to get the descending order the report expects.
    eigenvalues.clear();
    eigenvectors.clear();
    for (std::size_t pair_index = 0; pair_index < dimension; pair_index++) {
        std::size_t ascending_position = dimension - 1 - pair_index;
        eigenvalues.push_back(eigen_solver.eigenvalues()(ascending_position));

        Eigen::VectorXd eigenvector_column = eigen_solver.eigenvectors().col(ascending_position);

        // Sign convention: the largest weight in each eigenvector is positive.
        Eigen::Index largest_weight_position = 0;
        eigenvector_column.cwiseAbs().maxCoeff(&largest_weight_position);
        if (eigenvector_column(largest_weight_position) < 0.0) {
            eigenvector_column = -eigenvector_column;
        }

        eigenvectors.push_back(std::vector<double>(
            eigenvector_column.data(), eigenvector_column.data() + dimension));
    }
}


/*
Appends one 32 bit value to a byte buffer

INPUTS:
    * byte_buffer, the buffer being built
    * value_to_append, the 32 bit value

OUTPUTS:
    * byte_buffer grows by four bytes
*/
void append_big_endian_value(std::vector<unsigned char>& byte_buffer, uint32_t value_to_append)
{
    byte_buffer.push_back(static_cast<unsigned char>((value_to_append >> 24) & 0xFFu));
    byte_buffer.push_back(static_cast<unsigned char>((value_to_append >> 16) & 0xFFu));
    byte_buffer.push_back(static_cast<unsigned char>((value_to_append >> 8) & 0xFFu));
    byte_buffer.push_back(static_cast<unsigned char>(value_to_append & 0xFFu));
}


// Computes the CRC32 checksum PNG uses to guard every chunk.
uint32_t compute_crc32(const std::vector<unsigned char>& bytes)
{
    uint32_t crc_value = CRC_INITIAL_VALUE;
    for (unsigned char current_byte : bytes) {
        crc_value = crc_value ^ current_byte;
        for (int bit_index = 0; bit_index < BITS_PER_BYTE; bit_index++) {
            if (crc_value & 1u) {
                crc_value = (crc_value >> 1) ^ CRC_POLYNOMIAL;
            } else {
                crc_value = crc_value >> 1;
            }
        }
    }
    return crc_value ^ CRC_INITIAL_VALUE;
}


// Computes the Adler32 checksum that closes a zlib stream.
uint32_t compute_adler32(const std::vector<unsigned char>& bytes)
{
    uint32_t low_sum = 1;
    uint32_t high_sum = 0;
    for (unsigned char current_byte : bytes) {
        low_sum = (low_sum + current_byte) % ADLER_MODULUS;
        high_sum = (high_sum + low_sum) % ADLER_MODULUS;
    }
    return (high_sum << 16) | low_sum;
}


/*
Writes one PNG chunk: length, type, data, then a CRC over type and data

INPUTS:
    * output_file, the open PNG file
    * chunk_type, the four character chunk name
    * chunk_data, the chunk payload

OUTPUTS:
    * the chunk appended to output_file
*/
void write_png_chunk(std::ofstream& output_file,
                     const std::string& chunk_type,
                     const std::vector<unsigned char>& chunk_data)
{
    std::vector<unsigned char> length_bytes;
    append_big_endian_value(length_bytes, static_cast<uint32_t>(chunk_data.size()));
    output_file.write(reinterpret_cast<const char*>(length_bytes.data()), length_bytes.size());

    std::vector<unsigned char> type_and_data(chunk_type.begin(), chunk_type.end());
    type_and_data.insert(type_and_data.end(), chunk_data.begin(), chunk_data.end());
    output_file.write(reinterpret_cast<const char*>(type_and_data.data()), type_and_data.size());

    std::vector<unsigned char> crc_bytes;
    append_big_endian_value(crc_bytes, compute_crc32(type_and_data));
    output_file.write(reinterpret_cast<const char*>(crc_bytes.data()), crc_bytes.size());
}


/*
Writes an RGB pixel buffer as a PNG file. The pixel data is wrapped in a
zlib stream made of uncompressed deflate blocks. - Partially rebuilt with Gemini.

INPUTS
    * file_path, where the PNG is written
    * image_width and image_height in pixels
    * rgb_pixel_bytes, three bytes per pixel, rows top to bottom

OUTPUTS:
    * a PNG file at file_path
*/
void write_png_image(const std::string& file_path,
                     int image_width,
                     int image_height,
                     const std::vector<unsigned char>& rgb_pixel_bytes)
{
    // Every scanline is prefixed with a zero byte meaning no PNG filter.
    std::vector<unsigned char> filtered_data;
    std::size_t bytes_per_row = static_cast<std::size_t>(image_width) * CHANNELS_PER_PIXEL;
    for (int row_index = 0; row_index < image_height; row_index++) {
        filtered_data.push_back(0);
        std::size_t row_start = static_cast<std::size_t>(row_index) * bytes_per_row;
        filtered_data.insert(filtered_data.end(),
                             rgb_pixel_bytes.begin() + row_start,
                             rgb_pixel_bytes.begin() + row_start + bytes_per_row);
    }

    // Wrap the filtered data in a zlib stream of stored deflate blocks.
    std::vector<unsigned char> zlib_stream;
    zlib_stream.push_back(ZLIB_HEADER_FIRST_BYTE);
    zlib_stream.push_back(ZLIB_HEADER_SECOND_BYTE);
    std::size_t remaining_bytes = filtered_data.size();
    std::size_t data_position = 0;
    while (remaining_bytes > 0) {
        std::size_t block_size = std::min(remaining_bytes, MAXIMUM_STORED_BLOCK_SIZE);
        bool is_final_block = (block_size == remaining_bytes);
        zlib_stream.push_back(is_final_block ? 1 : 0);
        zlib_stream.push_back(static_cast<unsigned char>(block_size & 0xFFu));
        zlib_stream.push_back(static_cast<unsigned char>((block_size >> 8) & 0xFFu));
        zlib_stream.push_back(static_cast<unsigned char>(~block_size & 0xFFu));
        zlib_stream.push_back(static_cast<unsigned char>((~block_size >> 8) & 0xFFu));
        zlib_stream.insert(zlib_stream.end(),
                           filtered_data.begin() + data_position,
                           filtered_data.begin() + data_position + block_size);
        data_position = data_position + block_size;
        remaining_bytes = remaining_bytes - block_size;
    }
    append_big_endian_value(zlib_stream, compute_adler32(filtered_data));

    std::ofstream output_file(file_path, std::ios::binary);
    if (!output_file.is_open()) {
        throw std::runtime_error("Cannot open " + file_path + " for writing.");
    }

    const unsigned char png_signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    output_file.write(reinterpret_cast<const char*>(png_signature), sizeof(png_signature));

    std::vector<unsigned char> header_data;
    append_big_endian_value(header_data, static_cast<uint32_t>(image_width));
    append_big_endian_value(header_data, static_cast<uint32_t>(image_height));
    header_data.push_back(PNG_BIT_DEPTH);
    header_data.push_back(PNG_COLOUR_TYPE_RGB);
    header_data.push_back(0);
    header_data.push_back(0);
    header_data.push_back(0);
    write_png_chunk(output_file, "IHDR", header_data);

    write_png_chunk(output_file, "IDAT", zlib_stream);
    write_png_chunk(output_file, "IEND", std::vector<unsigned char>());
}


/*
Maps heatmap colours to the [-1, +1] range.

INPUTS:
    * display_value, the scaled matrix entry

OUTPUTS:
    * red_channel, green_channel and blue_channel bytes
*/
void colour_for_value(double display_value,
                      unsigned char& red_channel,
                      unsigned char& green_channel,
                      unsigned char& blue_channel)
{
    double clamped_value = std::max(MINIMUM_DISPLAY_VALUE,
                                    std::min(MAXIMUM_DISPLAY_VALUE, display_value));
    if (clamped_value < 0.0) {
        red_channel = 0;
        green_channel = static_cast<unsigned char>(FULL_COLOUR_INTENSITY * (1.0 + clamped_value));
        blue_channel = static_cast<unsigned char>(FULL_COLOUR_INTENSITY * (-clamped_value));
    } else {
        red_channel = static_cast<unsigned char>(FULL_COLOUR_INTENSITY * clamped_value);
        green_channel = static_cast<unsigned char>(FULL_COLOUR_INTENSITY * (1.0 - clamped_value));
        blue_channel = 0;
    }
}


/*
Renders a square matrix as an RGB heatmap pixel buffer.

INPUTS:
    * matrix_values, the square matrix to render
    * value_scale, the divisor applied before colouring

OUTPUTS:
    * the RGB byte buffer
*/
std::vector<unsigned char> build_heatmap_pixel_bytes(
    const std::vector<std::vector<double>>& matrix_values, double value_scale)
{
    std::size_t dimension = matrix_values.size();
    std::size_t image_size = dimension * CELL_PIXEL_SIZE;
    std::vector<unsigned char> pixel_bytes(image_size * image_size * CHANNELS_PER_PIXEL);

    for (std::size_t pixel_row = 0; pixel_row < image_size; pixel_row++) {
        for (std::size_t pixel_column = 0; pixel_column < image_size; pixel_column++) {
            std::size_t cell_row = pixel_row / CELL_PIXEL_SIZE;
            std::size_t cell_column = pixel_column / CELL_PIXEL_SIZE;
            double display_value = matrix_values[cell_row][cell_column] / value_scale;

            unsigned char red_channel = 0;
            unsigned char green_channel = 0;
            unsigned char blue_channel = 0;
            colour_for_value(display_value, red_channel, green_channel, blue_channel);

            std::size_t byte_position = (pixel_row * image_size + pixel_column) * CHANNELS_PER_PIXEL;
            pixel_bytes[byte_position] = red_channel;
            pixel_bytes[byte_position + 1] = green_channel;
            pixel_bytes[byte_position + 2] = blue_channel;
        }
    }
    return pixel_bytes;
}


// Finds the largest absolute entry of a matrix, used to scale the covariance
// heatmap onto the shared colour range.
double find_largest_absolute_entry(const std::vector<std::vector<double>>& matrix_values)
{
    double largest_entry = 0.0;
    for (const std::vector<double>& matrix_row : matrix_values) {
        for (double matrix_entry : matrix_row) {
            largest_entry = std::max(largest_entry, std::fabs(matrix_entry));
        }
    }
    return largest_entry;
}


/*
Prints one labelled matrix into the .txt output file. Columns follow the 
same feature order as the rows

INPUTS:
    * report_file, the open report
    * section_heading, the matrix name
    * matrix_values, the square matrix
    * feature_names, the row labels

OUTPUTS:
    * the section appended to report_file
*/
void write_matrix_section(std::ofstream& report_file,
                          const std::string& section_heading,
                          const std::vector<std::vector<double>>& matrix_values,
                          const std::vector<std::string>& feature_names)
{
    report_file << section_heading << "\n\n";
    for (std::size_t row_index = 0; row_index < matrix_values.size(); row_index++) {
        report_file << std::left << std::setw(26) << feature_names[row_index] << std::right;
        for (double matrix_entry : matrix_values[row_index]) {
            report_file << std::setw(14) << matrix_entry;
        }
        report_file << "\n";
    }
    report_file << "\n";
}


/*
Prints the eigenvalues and eigenvectors of one matrix into the same txt file, with
each eigenvalue's share of the total so we can se the concentration of the varoance.

INPUTS:
    * report_file, the open report
    * section_heading, the matrix name
    * eigenvalues and eigenvectors, from compute_eigen_decomposition
    * feature_names, labels for the eigenvector weights

OUTPUTS:
    * the section appended to report_file
*/
void write_eigen_section(std::ofstream& report_file,
                         const std::string& section_heading,
                         const std::vector<double>& eigenvalues,
                         const std::vector<std::vector<double>>& eigenvectors,
                         const std::vector<std::string>& feature_names)
{
    double eigenvalue_total = 0.0;
    for (double eigenvalue : eigenvalues) {
        eigenvalue_total = eigenvalue_total + eigenvalue;
    }

    report_file << section_heading << "\n\n";
    for (std::size_t pair_index = 0; pair_index < eigenvalues.size(); pair_index++) {
        report_file << "Eigenvalue " << pair_index + 1 << ": " << eigenvalues[pair_index]
                    << "  share of total " << eigenvalues[pair_index] / eigenvalue_total << "\n";
        for (std::size_t feature_index = 0; feature_index < feature_names.size(); feature_index++) {
            report_file << "    " << std::left << std::setw(26) << feature_names[feature_index]
                        << std::right << std::setw(14)
                        << eigenvectors[pair_index][feature_index] << "\n";
        }
        report_file << "\n";
    }
}


/*
The main function

INPUTS:
    * none, paths come from the constants above

OUTPUTS:
    * correlation_matrix.png, covariance_matrix.png and eigen.txt in
      components/data_relation_output
*/
int main()
{
    try {
        // The output directory is git ignored, so it may be absent on a fresh
        // clone. Created here before anything is written into it.
        std::filesystem::create_directories(OUTPUT_DIRECTORY);

        std::vector<std::string> feature_names;
        std::vector<std::vector<double>> feature_rows;
        read_feature_matrix(FEATURE_MATRIX_PATH, feature_names, feature_rows);
        std::cout << "Loaded " << feature_rows.size() << " weeks of "
                  << feature_names.size() << " features." << std::endl;

        std::vector<std::vector<double>> covariance_matrix = compute_covariance_matrix(feature_rows);
        std::vector<std::vector<double>> correlation_matrix = compute_correlation_matrix(covariance_matrix);

        std::vector<double> covariance_eigenvalues;
        std::vector<std::vector<double>> covariance_eigenvectors;
        compute_eigen_decomposition(covariance_matrix, covariance_eigenvalues, covariance_eigenvectors);

        std::vector<double> correlation_eigenvalues;
        std::vector<std::vector<double>> correlation_eigenvectors;
        compute_eigen_decomposition(correlation_matrix, correlation_eigenvalues, correlation_eigenvectors);

        // Correlation entries already live in the shared colour range. The
        // covariance heatmap is scaled by its largest absolute entry.
        std::vector<unsigned char> correlation_pixels =
            build_heatmap_pixel_bytes(correlation_matrix, 1.0);
        std::size_t image_size = feature_names.size() * CELL_PIXEL_SIZE;
        write_png_image(CORRELATION_IMAGE_PATH,
                        static_cast<int>(image_size), static_cast<int>(image_size),
                        correlation_pixels);
        std::cout << "Written " << CORRELATION_IMAGE_PATH << std::endl;

        double covariance_scale = find_largest_absolute_entry(covariance_matrix);
        std::vector<unsigned char> covariance_pixels =
            build_heatmap_pixel_bytes(covariance_matrix, covariance_scale);
        write_png_image(COVARIANCE_IMAGE_PATH,
                        static_cast<int>(image_size), static_cast<int>(image_size),
                        covariance_pixels);
        std::cout << "Written " << COVARIANCE_IMAGE_PATH << std::endl;

        std::ofstream report_file(EIGEN_REPORT_PATH);
        report_file << std::fixed << std::setprecision(REPORT_DECIMAL_PLACES);
        report_file << "Eigenstructure diagnostic of the weekly feature matrix.\n";
        report_file << "Sample: all " << feature_rows.size() << " weeks in "
                    << FEATURE_MATRIX_PATH << ".\n\n";
        report_file << "The correlation eigenstructure is diagnostic only. Production PCA\n";
        report_file << "uses the covariance matrix and lives in pca.py. Strong off diagonal\n";
        report_file << "structure below means PCA compresses shared directions well. A near\n";
        report_file << "diagonal matrix means PCA buys little.\n\n";
        report_file << "Feature order for all matrices and eigenvectors below, and for the\n";
        report_file << "heatmap rows and columns:\n";
        for (std::size_t feature_index = 0; feature_index < feature_names.size(); feature_index++) {
            report_file << "    " << feature_index + 1 << "  " << feature_names[feature_index] << "\n";
        }
        report_file << "\n";

        write_matrix_section(report_file, "Correlation matrix", correlation_matrix, feature_names);
        write_eigen_section(report_file, "Correlation matrix eigenstructure",
                            correlation_eigenvalues, correlation_eigenvectors, feature_names);
        write_matrix_section(report_file, "Covariance matrix", covariance_matrix, feature_names);
        write_eigen_section(report_file, "Covariance matrix eigenstructure",
                            covariance_eigenvalues, covariance_eigenvectors, feature_names);
        std::cout << "Written " << EIGEN_REPORT_PATH << std::endl;
    } catch (const std::exception& thrown_error) {
        std::cerr << thrown_error.what() << std::endl;
        return 1;
    }
    return 0;
}