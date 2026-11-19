# Standalone rocALUTION solve of the 12-unknown DARTS test system.
# On the AMD box:
#   cd rocalution_solver
#   make
#   ./test_linsolv_rocalution

ROCM ?= /opt/rocm
CXX ?= $(ROCM)/bin/hipcc
CXXFLAGS ?= -O3 -std=c++17 -I$(ROCM)/include
LDFLAGS ?= -L$(ROCM)/lib -Wl,-rpath,$(ROCM)/lib -lrocalution

test_linsolv_rocalution: test_linsolv_rocalution.cpp linsolv_rocalution.cpp linsolv_rocalution.hpp
	$(CXX) $(CXXFLAGS) -o $@ test_linsolv_rocalution.cpp linsolv_rocalution.cpp $(LDFLAGS)

clean:
	rm -f test_linsolv_rocalution
