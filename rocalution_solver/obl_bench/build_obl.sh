#!/bin/bash
# Verify on the machine that produced the snapshot (RX 7900 XTX, gfx1100):
#   hipcc -O3 --offload-arch=gfx1100 bench_obl.hip obl_kernel.hip -o bench_obl
# Hyperloom target, Instinct MI300 CDNA3. Do not read a gfx1100 median as an MI300X speedup:
#   hipcc -O3 --offload-arch=gfx942 bench_obl.hip obl_kernel.hip -o bench_obl
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${1:-gfx1100}"
hipcc -O3 --offload-arch="${ARCH}" bench_obl.hip obl_kernel.hip -o bench_obl
