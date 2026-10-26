import os
import sys

import numpy as np

original_stdout = os.dup(1)


def redirect_all_output(log_file, append=True):
    sys.stdout.flush()
    if append:
        log_stream = open(log_file, "a+")
    else:
        log_stream = open(log_file, "w")
    # this way truly all messages from both Python and C++, printf or std::cout, will be redirected
    os.dup2(log_stream.fileno(), sys.stdout.fileno())
    return log_stream


def abort_redirection(log_stream):
    sys.stdout.flush()
    log_stream.flush()
    os.dup2(original_stdout, sys.stdout.fileno())
    log_stream.close()


def fmt_e(x: float) -> str:
    """Formats a scalar float using exponential (scientific) notation with 3 decimal places.

    If input is non-scalar (e.g., array or list), returns its string representation.
    """
    # np.isscalar handles numbers, booleans, etc., but excludes arrays and lists
    if np.isscalar(x):
        return f"{x:.3e}"
    return str(x)


def fmt(x: float) -> str:
    """Formats a scalar float using standard notation with 3 significant figures.

    If input is non-scalar (e.g., array or list), returns its string representation.
    """
    if np.isscalar(x):
        return f"{x:.3g}"  # Changed from .3f/.3 to .3g to reliably format to 3 significant figures
    return str(x)


def print_array_range(data: dict, name: str, units: str = "") -> None:
    """Prints the min and max values of an array stored within a dictionary or structured container.

    Args:
        data: A dictionary or mapping containing arrays or array-like objects.
        name: The key corresponding to the desired array in `data`.
        units: Optional units string to append to the printed range (e.g., "m/s").
    """
    # 1. Check if key exists to prevent KeyError
    if name not in data:
        print(f"Key '{name}' not found in data.")
        return

    val = data[name]

    # 2. Check if the entry is None
    if val is None:
        return

    # 3. Convert to numpy array to handle lists or tuples safely
    arr = np.asarray(val)

    # 4. Handle empty arrays to avoid errors when calling min()/max()
    if arr.size == 0:
        print(f"{name}: [Empty Array]")
        return

    # 5. Format unit string with leading space if provided
    unit_str = f" {units.strip()}" if units.strip() else ""

    # 6. Safely retrieve min and max
    min_val = arr.min()
    max_val = arr.max()

    print(f"{name} range: {fmt(min_val)} - {fmt(max_val)}{unit_str}")


####################################################################


# Logging usage example
if __name__ == "__main__":
    from darts.engines import logging

    logging.log("screen only")
    logging.duplicate_output_to_file("log.log")
    logging.log("screen and log file")
    print("screen only")

    # Logging verbosity example
    # Default logging verbosity level is : INFO
    logging.debug("debug that should not be shown")
    logging.info("info")
    logging.set_logging_level(logging.LoggingLevel.ERROR)
    logging.info("info that should not be shown")
    logging.error("error")
    logging.critical("critical")

    """
    # Results

    ## Screen

        screen only
        screen and log file
        screen only
        info
        error
        critical

    ## Log file 'log.log'

        screen and log file
        info
        error
        critical

    """
