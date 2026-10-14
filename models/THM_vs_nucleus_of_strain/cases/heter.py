import numpy as np
from cases.homog import *
from cases.base import recompute_rock_derived


def heter():
    # Inherits `homog` (same mesh, flow/thermal props) but with
    # HETEROGENEOUS geomechanical properties per tag: a softer, more
    # compressible sand reservoir sandwiched between stiffer shale burdens.
    #
    # Per-tag array order follows homog / matrix_tags
    # (99991, 99992, 99993) = [reservoir(sand), overburden(shale), underburden(shale)].
    idata = homog()

    # Young's modulus
    E_sand  = 20.0 * 1e4   # [bars]  (1e4: [GPa] -> [bar])
    E_shale = 10.0 * 1e4  # [bars]
    idata.rock.E = np.array([E_sand, E_shale, E_shale])

    # Poisson ratio
    nu_sand  = 0.20
    nu_shale = 0.25
    idata.rock.nu = np.array([nu_sand, nu_shale, nu_shale])

    th_exp_sand = 1e-5  # [1/K] linear thermal expansion coefficient
    th_exp_shale = 1e-5  # [1/K] linear thermal expansion coefficient
    idata.rock.th_expn_orig = np.array([th_exp_sand, th_exp_shale, th_exp_shale])

    # recompute the per-tag derived quantities from the E, nu and th_expn_orig above
    recompute_rock_derived(idata)

    # the geometry is the homog one, but the mesh folder follows the case name so a later
    # geometry change here cannot silently reuse the homog mesh

    return idata


def heter_offset0():
    # Same heterogeneous properties as heter, but with the fault's
    # vertical juxtaposition of the reservoir removed (fault_offset=0): the reservoir
    # sits at the same depth on both sides of the fault, only the reservoir_thickness
    # (kept the same) sets its thickness.
    idata = heter()
    idata.other.fault_offset = 0.0
    return idata
