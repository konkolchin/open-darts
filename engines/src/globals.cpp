#include "globals.h"
#include <iomanip>

#ifdef WITH_GPU
int device_num = 0;
#endif

void write_vector_to_file(std::string file_name, std::vector<value_t> &v)
{
  std::ofstream outFile(file_name);
  // setprecision(5) kept 6 significant digits, which is not enough to read a
  // residual or a Newton update back for analysis (and not round-trippable).
  for (const auto &e : v) outFile << std::scientific << std::setprecision(17) << e << "\n";
}
