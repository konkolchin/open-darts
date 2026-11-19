# Standalone rocALUTION solves of two DARTS test systems.
# On the AMD box:
#   cd rocalution_solver
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
