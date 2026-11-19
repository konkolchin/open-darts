# Standalone shortcut for the two rocALUTION regression suites.
# The suites are also targets of the open-DARTS CMake build:
#   cmake -S . -B build -D WITH_ROCALUTION=ON -D OPENDARTS_CONFIG=MT
#   cmake --build build --target test_linsolv_rocalution test_linsolv_rocalution_spd
# This Makefile does not compile linsolv_rocalution into linear_solvers.
# On the AMD box, from this directory:
#   make
#   ./test_linsolv_rocalution
#   ./test_linsolv_rocalution_spd

ROCM ?= /opt/rocm
CXX ?= $(ROCM)/bin/hipcc
CXXFLAGS ?= -O3 -std=c++17 -I$(ROCM)/include
LDFLAGS ?= -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lrocalution

all: test_linsolv_rocalution test_linsolv_rocalution_spd

test_linsolv_rocalution: test_linsolv_rocalution.cpp linsolv_rocalution.cpp linsolv_rocalution.hpp
	$(CXX) $(CXXFLAGS) -o $@ test_linsolv_rocalution.cpp linsolv_rocalution.cpp $(LDFLAGS)

test_linsolv_rocalution_spd: test_linsolv_rocalution_spd.cpp
	$(CXX) $(CXXFLAGS) -o $@ test_linsolv_rocalution_spd.cpp $(LDFLAGS)

clean:
	rm -f test_linsolv_rocalution test_linsolv_rocalution_spd
