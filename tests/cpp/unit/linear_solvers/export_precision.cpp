#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "version.hpp"
#include "csr_matrix.hpp"
#include "linear_solvers_data_types.hpp"

#include "test_common.hpp"

int test_csr_roundtrip_preserves_values();

int main()
{
  /*
    export_precision
    Guards the numeric precision of csr_matrix::export_matrix_to_file /
    import_matrix_from_file in csr format.

    The csr dump is what a Jacobian is inspected through (conditioning
    analysis, the GPU engine's print_linear_system), so it has to be lossless:
    exporting a matrix and importing it back must return bitwise identical
    values -- including the very small entries a reservoir Jacobian carries,
    where a mineral balance's pressure derivative runs down to 1e-20.

    This deliberately does NOT compare against a reference file. The golden
    file tests (write_mat_csr, read_mat_csr) pin the exact text layout, and
    they pass for any self-consistent format, lossy or not: the previous
    std::fixed / 16-decimals format wrote every entry below ~1e-16 as
    0.0000000000000000 and those tests were perfectly happy. This test pins
    the property instead of the layout, so it survives a format change and
    fails a precision regression.
  */
  return test_csr_roundtrip_preserves_values();
}

/** Exports a block csr matrix carrying values across the full double range and
    imports it back, checking that nothing was lost in the text representation.

    @return 0 if every value, column index and row pointer survived the round
            trip unchanged, the number of mismatching entries otherwise.
*/
int test_csr_roundtrip_preserves_values()
{
  const opendarts::config::index_t n_block_rows = 3; // rows, in blocks
  const opendarts::config::index_t n_block_cols = 3; // columns, in blocks
  const opendarts::config::index_t n_non_zeros = 4;  // nonzero blocks
  const opendarts::config::index_t b_sqr = 3 * 3;    // values per block, csr_matrix<3>

  // The output file to save the matrix to
  std::string output_filename("export_precision__csr_matrix.txt");

  // One block worth of probe values, spanning the range a Jacobian entry can
  // take. The three smallest ones are the interesting cases: they are exactly
  // what a fixed-point format with a bounded number of decimals flushes to zero.
  const std::vector<opendarts::config::mat_float> probe_block = {0.0, 1.0, -2.0, 0.1,
      1.0 / 3.0, 1.0e-20, -3.7e-18, 1.0e+12,
      std::numeric_limits<opendarts::config::mat_float>::denorm_min()};

  // Tile the probe block over the nonzero blocks, flipping sign on odd blocks so
  // that the blocks are not all identical
  std::vector<opendarts::config::mat_float> values(n_non_zeros * b_sqr);
  for (opendarts::config::index_t block_idx = 0; block_idx < n_non_zeros; block_idx++)
  {
    const opendarts::config::mat_float sign = (block_idx % 2 == 0) ? 1.0 : -1.0;
    for (opendarts::config::index_t value_idx = 0; value_idx < b_sqr; value_idx++)
      values[block_idx * b_sqr + value_idx] = sign * probe_block[value_idx];
  }

  const std::vector<opendarts::config::index_t> rows_ptr = {0, 2, 3, 4};
  const std::vector<opendarts::config::index_t> cols_ind = {0, 2, 1, 2};

  // Build the matrix and write it out
  opendarts::linear_solvers::csr_matrix<3> A;
  A.init(n_block_rows, n_block_cols, n_non_zeros, values, rows_ptr, cols_ind);

  int error_output = A.export_matrix_to_file(output_filename,
      opendarts::linear_solvers::sparse_matrix_export_format::csr);
  if (error_output != 0)
    return error_output;

  // Read it back into a fresh matrix
  opendarts::linear_solvers::csr_matrix<3> B;
  error_output = B.import_matrix_from_file(output_filename,
      opendarts::linear_solvers::sparse_matrix_import_format::csr);
  if (error_output != 0)
    return error_output;

  // Compare the structure
  int n_mismatches = 0;
  if (B.n_rows != n_block_rows || B.n_cols != n_block_cols || B.n_non_zeros != n_non_zeros)
  {
    std::printf("Dimensions changed over the round trip: %d x %d with %d nonzero blocks, "
                "expected %d x %d with %d\n",
        B.n_rows, B.n_cols, B.n_non_zeros, n_block_rows, n_block_cols, n_non_zeros);
    return 1; // the value comparison below would be meaningless
  }

  for (opendarts::config::index_t row_idx = 0; row_idx <= n_block_rows; row_idx++)
  {
    if (B.rows_ptr[row_idx] != rows_ptr[row_idx])
    {
      std::printf("rows_ptr[%d]: got %d, expected %d\n", row_idx, B.rows_ptr[row_idx], rows_ptr[row_idx]);
      n_mismatches++;
    }
  }

  for (opendarts::config::index_t block_idx = 0; block_idx < n_non_zeros; block_idx++)
  {
    if (B.cols_ind[block_idx] != cols_ind[block_idx])
    {
      std::printf("cols_ind[%d]: got %d, expected %d\n", block_idx, B.cols_ind[block_idx], cols_ind[block_idx]);
      n_mismatches++;
    }
  }

  // Compare the values bitwise: anything short of equality means the text
  // representation is not round-trip exact
  for (opendarts::config::index_t value_idx = 0; value_idx < n_non_zeros * b_sqr; value_idx++)
  {
    if (B.values[value_idx] != values[value_idx])
    {
      std::printf("values[%d]: got %.17e, expected %.17e -- the csr dump is lossy\n",
          value_idx, B.values[value_idx], values[value_idx]);
      n_mismatches++;
    }
  }

  return n_mismatches;
}
