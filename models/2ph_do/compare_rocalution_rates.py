"""Compare 2ph_do well rates: CPU engine against the AMD GPU rocALUTION CPR run.

The gate is store_well_time_data, not iteration counts. Run from models/2ph_do
after an OPENDARTS_CONFIG=GPU WITH_ROCALUTION build has installed the extensions.
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


def run_case(label, platform):
    case = Model()
    case.init(platform=platform)
    case.params.max_i_linear = 500
    case.params.tolerance_linear = 1e-6
    case.set_output(output_folder="output_" + label, verbose=0)
    case.run(300)
    data = case.output.store_well_time_data(save_output_files=False)
    flat = []
    _numbers(data, "", flat)
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
    print("---- CPU engine")
    cpu = run_case("cpu", "cpu")
    print("---- GPU engine, rocALUTION CPR")
    gpu = run_case("gpu", "gpu")
    if not compare(cpu, gpu, "gpu"):
        sys.exit(1)
    print("WELL RATES OK")


if __name__ == "__main__":
    main()
