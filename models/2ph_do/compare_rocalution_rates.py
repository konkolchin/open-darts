"""Compare 2ph_do well rates: CPU solver, rocALUTION GMRES+Jacobi, rocALUTION CPR.

The gate is store_well_time_data, not iteration counts. Run from models/2ph_do
after a WITH_ROCALUTION MT build has installed the Python extensions into darts/.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model import Model


def _numbers(value, prefix, out):
    if isinstance(value, dict):
        for key in sorted(value):
            _numbers(value[key], prefix + key + ".", out)
        return
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=float)
        if arr.dtype.kind in "fc" and arr.size:
            out.append((prefix[:-1], arr.ravel()))
        return
    arr = np.asarray(value)
    if arr.dtype.kind in "fc" and arr.size:
        out.append((prefix[:-1], arr.ravel()))


def run_case(label, solver_name):
    case = Model()
    case.init()
    case.params.max_i_linear = 500
    case.params.tolerance_linear = 1e-6
    handle = None
    if solver_name is not None:
        import darts.linear_solvers as linear_solvers

        config = linear_solvers.SolverConfig()
        config.max_iterations = 500
        config.tolerance = 1e-6
        handle = linear_solvers.create_linear_solver(
            solver_name, config, case.physics.n_vars
        )
        case.physics.engine.set_linear_solver(handle, solver_name)
    case.set_output(output_folder="output_" + label, verbose=0)
    case.run(300)
    data = case.output.store_well_time_data(save_output_files=False)
    flat = []
    _numbers(data, "", flat)
    # Keep the solver alive until the run has finished.
    case._solver_handle = handle
    return flat


def compare(reference, other, label):
    ref = dict(reference)
    worst = 0.0
    worst_name = ""
    for name, values in other:
        if name not in ref:
            print("missing from CPU rates:", name)
            return False
        base = ref[name]
        if base.shape != values.shape:
            print(label, name, "shape", values.shape, "vs", base.shape)
            return False
        scale = np.maximum(np.max(np.abs(base)), 1.0)
        diff = np.max(np.abs(base - values)) / scale
        if diff > worst:
            worst = diff
            worst_name = name
    print(label, "max relative well-rate difference", worst, "on", worst_name)
    if worst > 1e-2:
        print(label, "FAIL")
        return False
    print(label, "OK")
    return True


def main():
    print("---- CPU solver")
    cpu = run_case("cpu", None)
    print("---- rocALUTION GMRES + Jacobi")
    gmres = run_case("rocalution", "rocalution")
    print("---- rocALUTION CPR (AMG pressure, ILU full system)")
    cpr = run_case("rocalution_cpr", "rocalution_cpr")
    ok = compare(cpu, gmres, "rocalution") and compare(cpu, cpr, "rocalution_cpr")
    if not ok:
        sys.exit(1)
    print("WELL RATES OK")


if __name__ == "__main__":
    main()
