// Hyperloom harness. One isothermal 2c/2p assembly launch, plus a CPR
// pressure-restrict launch. Neither starts a timestep.

#include "port_bench_time.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "engine_super_gpu.hpp"

extern "C" void od_hyperloom_cpr(int n, double *B, double *P_B);

constexpr double kTol = 1e-12;
constexpr int kWarmup = 10;
constexpr int kSamples = 100;

using Eng = engine_super_gpu<2, 2, false>;

struct LaunchState {
  unsigned n = 0;
  value_t *X = nullptr;
  value_t *RHS = nullptr;
  value_t *Jac = nullptr;
  index_t *rows = nullptr;
  index_t *cols = nullptr;
  index_t *diag = nullptr;
  value_t *op = nullptr;
  value_t *op_n = nullptr;
  value_t *der = nullptr;
  value_t *tran = nullptr;
  value_t *tranD = nullptr;
  value_t *hcap = nullptr;
  value_t *rcond = nullptr;
  value_t *poro = nullptr;
  value_t *PV = nullptr;
  value_t *RV = nullptr;
  value_t *grav = nullptr;
  value_t *kin = nullptr;
  value_t *spe = nullptr;
  int *flags = nullptr;
};

LaunchState g_asm;

// The block-row thread that owns a matrix position is the only writer of that
// Jacobian entry, and the v==0 thread of an equation is the only writer of
// that residual. A second launch is therefore comparable at 1e-12.
extern "C" void od_hyperloom_assemble()
{
  assemble_jacobian_array_kernel<2, 2, Eng::NE, Eng::N_VARS, Eng::P_VAR, Eng::T_VAR, Eng::N_OPS,
                                 Eng::ACC_OP, Eng::FLUX_OP, Eng::DENS_OP, Eng::UPSAT_OP, Eng::GRAD_OP,
                                 Eng::KIN_OP, Eng::GRAV_OP, Eng::PC_OP, Eng::MULT_OP, Eng::LAMBDA_OP,
                                 Eng::SAT_OP, Eng::ENTH_OP, Eng::TEMP_OP, Eng::PRES_OP, false>
      KERNEL_1D(g_asm.n, Eng::N_VARS * Eng::N_VARS, 64)(
          g_asm.n, g_asm.n, true, 1e-8, 1.0, g_asm.X, g_asm.RHS, g_asm.rows, g_asm.cols, g_asm.Jac,
          g_asm.diag, g_asm.op, g_asm.op_n, g_asm.der, g_asm.tran, g_asm.tranD, g_asm.hcap, g_asm.rcond,
          g_asm.poro, g_asm.PV, g_asm.RV, g_asm.grav, g_asm.kin, g_asm.spe, g_asm.flags);
}

double mix(uint32_t x)
{
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return (x & 0xffffff) * (1.0 / 16777216.0);
}

struct Mesh {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int n = 0;
  int nnz = 0;
  int n_conn = 0;
  std::vector<index_t> rows;
  std::vector<index_t> cols;
  std::vector<index_t> diag;
  std::vector<value_t> tran;
  std::vector<value_t> tranD;
  std::vector<value_t> grav;
  std::vector<value_t> X;
  std::vector<value_t> op;
  std::vector<value_t> op_n;
  std::vector<value_t> der;
  std::vector<value_t> pv;
  std::vector<value_t> rv;
  std::vector<value_t> poro;
  std::vector<value_t> hcap;
  std::vector<value_t> rcond;
  std::vector<value_t> kin;
  std::vector<value_t> spe;
};

int cell_id(const Mesh &m, int x, int y, int z)
{
  return (z * m.ny + y) * m.nx + x;
}

Mesh make_mesh(int nx, int ny, int nz)
{
  Mesh m;
  m.nx = nx;
  m.ny = ny;
  m.nz = nz;
  m.n = nx * ny * nz;
  m.rows.resize((size_t)m.n + 1);
  m.diag.resize((size_t)m.n);
  m.X.resize((size_t)m.n * Eng::N_VARS);
  m.op.resize((size_t)m.n * Eng::N_OPS);
  m.op_n.resize(m.op.size());
  m.der.resize(m.op.size() * Eng::N_VARS);
  m.pv.resize((size_t)m.n);
  m.rv.resize((size_t)m.n);
  m.poro.resize((size_t)m.n);
  m.hcap.resize((size_t)m.n);
  m.rcond.resize((size_t)m.n);
  m.kin.resize((size_t)m.n);
  m.spe.resize((size_t)m.n);

  const int dirs[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
  m.rows[0] = 0;
  for (int z = 0; z < nz; ++z)
  {
    for (int y = 0; y < ny; ++y)
    {
      for (int x = 0; x < nx; ++x)
      {
        const int i = cell_id(m, x, y, z);
        for (int d = 0; d < 6; ++d)
        {
          const int x2 = x + dirs[d][0];
          const int y2 = y + dirs[d][1];
          const int z2 = z + dirs[d][2];
          if (x2 < 0 || y2 < 0 || z2 < 0 || x2 >= nx || y2 >= ny || z2 >= nz)
            continue;
          m.cols.push_back(cell_id(m, x2, y2, z2));
          m.tran.push_back(0.05 + 0.02 * mix((uint32_t)i * 6u + (uint32_t)d));
          m.tranD.push_back(0.01);
          m.grav.push_back(dirs[d][2] == 0 ? 0.0 : 0.002);
        }
        m.diag[(size_t)i] = (index_t)m.cols.size();
        m.cols.push_back(i);
        m.rows[(size_t)i + 1] = (index_t)m.cols.size();

        m.X[(size_t)i * 2] = 150.0 + 0.15 * x - 0.25 * z;
        m.X[(size_t)i * 2 + 1] = 0.3;
        m.pv[(size_t)i] = 50.0;
        m.rv[(size_t)i] = 20.0;
        m.poro[(size_t)i] = 0.2;
        m.hcap[(size_t)i] = 1.0;
        m.rcond[(size_t)i] = 1.0;
        m.kin[(size_t)i] = 0.05;
        m.spe[(size_t)i] = 0.01;
        for (int k = 0; k < Eng::N_OPS; ++k)
        {
          const double a = 0.2 + 0.05 * mix((uint32_t)i * 32u + (uint32_t)k);
          const size_t opi = (size_t)i * Eng::N_OPS + (size_t)k;
          m.op[opi] = a;
          m.op_n[opi] = 0.5 * a;
          m.der[opi * Eng::N_VARS] = 0.01 * a;
          m.der[opi * Eng::N_VARS + 1] = -0.02 * a;
        }
        m.op[(size_t)i * Eng::N_OPS + Eng::MULT_OP] = 1.0 + 0.1 * mix((uint32_t)i);
        m.op[(size_t)i * Eng::N_OPS + Eng::LAMBDA_OP] = 0.4;
        m.op[(size_t)i * Eng::N_OPS + Eng::LAMBDA_OP + 1] = 0.6;
        m.op[(size_t)i * Eng::N_OPS + Eng::UPSAT_OP] = 1.0;
        m.op[(size_t)i * Eng::N_OPS + Eng::UPSAT_OP + 1] = 1.0;
        m.op[(size_t)i * Eng::N_OPS + Eng::ACC_OP] = 0.35;
        m.op[(size_t)i * Eng::N_OPS + Eng::ACC_OP + 1] = 0.15;
        m.op_n[(size_t)i * Eng::N_OPS + Eng::ACC_OP] = 0.20;
        m.op_n[(size_t)i * Eng::N_OPS + Eng::ACC_OP + 1] = 0.10;
      }
    }
  }
  m.nnz = (int)m.cols.size();
  m.n_conn = (int)m.tran.size();
  return m;
}

int upload(void **dst, const void *src, size_t bytes)
{
  if (hipMalloc(dst, bytes) != hipSuccess)
    return 1;
  if (bytes == 0)
    return 0;
  if (hipMemcpy(*dst, src, bytes, hipMemcpyHostToDevice) != hipSuccess)
    return 1;
  return 0;
}

template <typename T>
int upload_vec(T **dst, const std::vector<T> &src)
{
  return upload((void **)dst, src.data(), src.size() * sizeof(T));
}

int download(void *dst, const void *src, size_t bytes)
{
  if (hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  return 0;
}

double max_abs(const std::vector<value_t> &a, const std::vector<value_t> &b)
{
  double worst = 0;
  for (size_t i = 0; i < a.size(); ++i)
  {
    const double diff = std::fabs((double)a[i] - (double)b[i]);
    if (diff > worst)
      worst = diff;
  }
  return worst;
}

struct GoldenHead {
  char magic[8];
  int32_t nx;
  int32_t ny;
  int32_t nz;
  int64_t n_jac;
  int64_t n_out;
};

int write_golden(const char *path, const char *magic, int nx, int ny, int nz,
                 const std::vector<value_t> &jac, const std::vector<value_t> &out)
{
  FILE *f = std::fopen(path, "wb");
  if (!f)
    return 1;
  GoldenHead head{};
  std::memcpy(head.magic, magic, 8);
  head.nx = nx;
  head.ny = ny;
  head.nz = nz;
  head.n_jac = (int64_t)jac.size();
  head.n_out = (int64_t)out.size();
  const int rc = std::fwrite(&head, sizeof(head), 1, f) != 1 ||
                 (jac.size() && std::fwrite(jac.data(), sizeof(value_t), jac.size(), f) != jac.size()) ||
                 (out.size() && std::fwrite(out.data(), sizeof(value_t), out.size(), f) != out.size());
  std::fclose(f);
  return rc ? 1 : 0;
}

int read_golden(const char *path, const char *magic, int nx, int ny, int nz,
                std::vector<value_t> *jac, std::vector<value_t> *out)
{
  FILE *f = std::fopen(path, "rb");
  if (!f)
    return 1;
  GoldenHead head{};
  if (std::fread(&head, sizeof(head), 1, f) != 1 || std::memcmp(head.magic, magic, 8) != 0 ||
      head.nx != nx || head.ny != ny || head.nz != nz || head.n_jac != (int64_t)jac->size() ||
      head.n_out != (int64_t)out->size())
  {
    std::fclose(f);
    return 1;
  }
  const int rc = (jac->size() && std::fread(jac->data(), sizeof(value_t), jac->size(), f) != jac->size()) ||
                 (out->size() && std::fread(out->data(), sizeof(value_t), out->size(), f) != out->size());
  std::fclose(f);
  return rc ? 1 : 0;
}

struct Args {
  bool assemble = false;
  bool cpr = false;
  int nx = 100;
  int ny = 100;
  int nz = 100;
  const char *write_golden = nullptr;
  const char *check = nullptr;
};

void usage()
{
  std::fprintf(stderr,
               "usage: assembly_bench assemble|cpr [--ten-m] [--nx N --ny N --nz N]\n"
               "         [--write-golden FILE] [--check FILE]\n");
}

int parse_args(int argc, char **argv, Args *a)
{
  if (argc < 2)
    return 1;
  if (std::strcmp(argv[1], "assemble") == 0)
    a->assemble = true;
  else if (std::strcmp(argv[1], "cpr") == 0)
    a->cpr = true;
  else
    return 1;
  for (int i = 2; i < argc; ++i)
  {
    if (std::strcmp(argv[i], "--ten-m") == 0)
    {
      a->nx = 200;
      a->ny = 200;
      a->nz = 250;
    }
    else if (std::strcmp(argv[i], "--nx") == 0 && i + 1 < argc)
      a->nx = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--ny") == 0 && i + 1 < argc)
      a->ny = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--nz") == 0 && i + 1 < argc)
      a->nz = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--write-golden") == 0 && i + 1 < argc)
      a->write_golden = argv[++i];
    else if (std::strcmp(argv[i], "--check") == 0 && i + 1 < argc)
      a->check = argv[++i];
    else
      return 1;
  }
  return (a->nx > 0 && a->ny > 0 && a->nz > 0) ? 0 : 1;
}

int time_launch(void (*launch)(), double *median_ms)
{
  return od_port_time(kWarmup, kSamples, launch, median_ms);
}

int run_assemble(const Args &args)
{
  const Mesh mesh = make_mesh(args.nx, args.ny, args.nz);
  g_asm.n = (unsigned)mesh.n;
  const int flags_h[2] = {1, 0};
  if (upload_vec(&g_asm.X, mesh.X) || upload_vec(&g_asm.rows, mesh.rows) || upload_vec(&g_asm.cols, mesh.cols) ||
      upload_vec(&g_asm.diag, mesh.diag) || upload_vec(&g_asm.op, mesh.op) || upload_vec(&g_asm.op_n, mesh.op_n) ||
      upload_vec(&g_asm.der, mesh.der) || upload_vec(&g_asm.tran, mesh.tran) || upload_vec(&g_asm.tranD, mesh.tranD) ||
      upload_vec(&g_asm.hcap, mesh.hcap) || upload_vec(&g_asm.rcond, mesh.rcond) ||
      upload_vec(&g_asm.poro, mesh.poro) || upload_vec(&g_asm.PV, mesh.pv) || upload_vec(&g_asm.RV, mesh.rv) ||
      upload_vec(&g_asm.grav, mesh.grav) || upload_vec(&g_asm.kin, mesh.kin) || upload_vec(&g_asm.spe, mesh.spe) ||
      upload((void **)&g_asm.flags, flags_h, sizeof(flags_h)))
    return 1;
  const size_t n_jac = (size_t)mesh.nnz * Eng::N_VARS * Eng::N_VARS;
  const size_t n_rhs = (size_t)mesh.n * Eng::N_VARS;
  if (hipMalloc(&g_asm.Jac, n_jac * sizeof(value_t)) != hipSuccess ||
      hipMalloc(&g_asm.RHS, n_rhs * sizeof(value_t)) != hipSuccess)
    return 1;

  std::vector<value_t> jac_a(n_jac), jac_b(n_jac), rhs_a(n_rhs), rhs_b(n_rhs);
  od_hyperloom_assemble();
  if (hipDeviceSynchronize() != hipSuccess || hipGetLastError() != hipSuccess)
    return 1;
  if (download(jac_a.data(), g_asm.Jac, n_jac * sizeof(value_t)) ||
      download(rhs_a.data(), g_asm.RHS, n_rhs * sizeof(value_t)))
    return 1;
  od_hyperloom_assemble();
  if (hipDeviceSynchronize() != hipSuccess || hipGetLastError() != hipSuccess)
    return 1;
  if (download(jac_b.data(), g_asm.Jac, n_jac * sizeof(value_t)) ||
      download(rhs_b.data(), g_asm.RHS, n_rhs * sizeof(value_t)))
    return 1;

  const double self_jac = max_abs(jac_a, jac_b);
  const double self_rhs = max_abs(rhs_a, rhs_b);
  std::printf("SELF_MAX_ABS_JAC %.17g\n", self_jac);
  std::printf("SELF_MAX_ABS_RHS %.17g\n", self_rhs);
  if (self_jac > kTol || self_rhs > kTol)
  {
    std::printf("SELF_CHECK_FAIL\n");
    return 2;
  }

  if (args.check)
  {
    std::vector<value_t> jac_g(n_jac), rhs_g(n_rhs);
    if (read_golden(args.check, "ODASM1\0", args.nx, args.ny, args.nz, &jac_g, &rhs_g))
    {
      std::printf("GOLDEN_READ_FAIL\n");
      return 1;
    }
    const double file_jac = max_abs(jac_a, jac_g);
    const double file_rhs = max_abs(rhs_a, rhs_g);
    std::printf("FILE_MAX_ABS_JAC %.17g\n", file_jac);
    std::printf("FILE_MAX_ABS_RHS %.17g\n", file_rhs);
    if (file_jac > kTol || file_rhs > kTol)
    {
      std::printf("GOLDEN_FAIL\n");
      return 1;
    }
  }
  if (args.write_golden &&
      write_golden(args.write_golden, "ODASM1\0", args.nx, args.ny, args.nz, jac_a, rhs_a))
  {
    std::printf("GOLDEN_WRITE_FAIL\n");
    return 1;
  }

  double median_ms = 0;
  if (time_launch(od_hyperloom_assemble, &median_ms))
    return 1;
  std::printf("CELLS %d\n", mesh.n);
  std::printf("NNZ %d\n", mesh.nnz);
  std::printf("MEDIAN_MS %.6f\n", median_ms);
  std::printf("CHECK_OK\n");
  return 0;
}

struct CprState {
  int n = 0;
  double *B = nullptr;
  double *P = nullptr;
};

CprState g_cpr;

void cpr_launch()
{
  od_hyperloom_cpr(g_cpr.n, g_cpr.B, g_cpr.P);
}

int run_cpr(const Args &args)
{
  const int n = args.nx * args.ny * args.nz;
  g_cpr.n = n;
  std::vector<double> host((size_t)n * 2);
  for (int i = 0; i < n; ++i)
  {
    host[(size_t)i * 2] = 1.0 + mix((uint32_t)i);
    host[(size_t)i * 2 + 1] = 2.0 + mix((uint32_t)i + 17u);
  }
  if (hipMalloc(&g_cpr.B, host.size() * sizeof(double)) != hipSuccess ||
      hipMalloc(&g_cpr.P, (size_t)n * sizeof(double)) != hipSuccess ||
      hipMemcpy(g_cpr.B, host.data(), host.size() * sizeof(double), hipMemcpyHostToDevice) != hipSuccess)
    return 1;

  std::vector<value_t> a((size_t)n), b((size_t)n);
  cpr_launch();
  if (hipDeviceSynchronize() != hipSuccess || hipGetLastError() != hipSuccess)
    return 1;
  if (download(a.data(), g_cpr.P, (size_t)n * sizeof(double)))
    return 1;
  cpr_launch();
  if (hipDeviceSynchronize() != hipSuccess || hipGetLastError() != hipSuccess)
    return 1;
  if (download(b.data(), g_cpr.P, (size_t)n * sizeof(double)))
    return 1;

  const double self_p = max_abs(a, b);
  std::printf("SELF_MAX_ABS_P %.17g\n", self_p);
  if (self_p > kTol)
  {
    std::printf("SELF_CHECK_FAIL\n");
    return 2;
  }

  std::vector<value_t> no_jac;
  if (args.check)
  {
    std::vector<value_t> g((size_t)n);
    if (read_golden(args.check, "ODCPR1\0", args.nx, args.ny, args.nz, &no_jac, &g))
    {
      std::printf("GOLDEN_READ_FAIL\n");
      return 1;
    }
    const double file_p = max_abs(a, g);
    std::printf("FILE_MAX_ABS_P %.17g\n", file_p);
    if (file_p > kTol)
    {
      std::printf("GOLDEN_FAIL\n");
      return 1;
    }
  }
  if (args.write_golden && write_golden(args.write_golden, "ODCPR1\0", args.nx, args.ny, args.nz, no_jac, a))
  {
    std::printf("GOLDEN_WRITE_FAIL\n");
    return 1;
  }

  double median_ms = 0;
  if (time_launch(cpr_launch, &median_ms))
    return 1;
  std::printf("CELLS %d\n", n);
  std::printf("MEDIAN_MS %.6f\n", median_ms);
  std::printf("CHECK_OK\n");
  return 0;
}

int main(int argc, char **argv)
{
  Args args;
  if (parse_args(argc, argv, &args))
  {
    usage();
    return 1;
  }
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess)
  {
    std::printf("FAIL device\n");
    return 1;
  }
  std::printf("MODE %s\n", args.assemble ? "assemble" : "cpr");
  std::printf("DEVICE %s\n", prop.name);
  std::printf("ARCH %s\n", prop.gcnArchName);
  std::printf("WARMUP %d\n", kWarmup);
  std::printf("SAMPLES %d\n", kSamples);
  const int rc = args.assemble ? run_assemble(args) : run_cpr(args);
  std::fflush(stdout);
  return rc;
}
