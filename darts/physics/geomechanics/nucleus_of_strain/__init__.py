import os
import sys

if sys.platform == "win32":
    _cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if _cuda_path:
        _cuda_bin = os.path.join(_cuda_path, "bin")
        if os.path.isdir(_cuda_bin):
            os.add_dll_directory(_cuda_bin)
