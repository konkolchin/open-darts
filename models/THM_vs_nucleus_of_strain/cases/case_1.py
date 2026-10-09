import numpy as np
from cases.base import (
    base_THM,
    recompute_rock_derived,
    _set_reservoir_bounds,
    _set_wells,
    _set_mesh_tags,
)

# examples which use custom meshes (without using local gmsh-based mesh generators)

def case_1():
    idata = base_THM()

    # override permeability (default is 10 mD)
    idata.rock.permx = idata.rock.permy = idata.rock.permz = 1000.0  # [mD]

    # reservoir geometry (overrides defaults: rsv_top=2000, rsv_bottom=2400, rsv_xy=1000)
    idata.other.rsv_top = 1800.0    # [m]
    idata.other.rsv_bottom = 1900.0  # [m]
    idata.other.rsv_xy = 1500.0      # [m]

    # well placement (overrides defaults: doublet_shift=500, cell_shift=0)
    idata.other.doublet_shift = 100.0  # [m]
    idata.other.cell_shift = 500.0     # [m]

    # mesh: three Gmsh physical tags (1=overburden, 2=reservoir, 3=underburden)
    idata.other.matrix_tags = (1, 2, 3)

    # case_2 and case_3 inherit this, so all three cases share one mesh. It is a committed
    # tetrahedral mesh with no generator in the repo, hence meshes/ rather than results/
    idata.other.mesh_dir = 'case_1'
    idata.other.mesh_root = 'meshes'

    # recompute derived values that depend on the overridden parameters above
    _set_reservoir_bounds(idata)
    _set_wells(idata)
    _set_mesh_tags(idata)

    idata.other.points_xy = []  # no black reference line
    idata.other.use_mesh_bounds_in_plot = True  # plot axes span the whole mesh
    return idata


def case_2():
    # same geometry as case_1, but rock properties are assigned per Gmsh tag rather than
    # by coordinate interpolation; tags: 1=overburden, 2=reservoir, 3=underburden
    idata = case_1()
    idata.other.set_props_by_tags = True

    rsv_poro = 0.2
    rsv_perm = 1000.0    # [mD]
    non_rsv_poro = 0.001
    non_rsv_perm = 0.001  # [mD]

    idata.rock.porosity = np.array([non_rsv_poro, rsv_poro, non_rsv_poro])

    perm = np.array([non_rsv_perm, rsv_perm, non_rsv_perm])  # isotropic perm per tag
    idata.rock.permx = idata.rock.permy = perm
    idata.rock.permz = perm * 0.1  # vertical perm is 10x lower

    # the per-tag porosity replaces the scalar the rock compressibility was computed from
    recompute_rock_derived(idata)

    return idata


def case_3():
    # same as case_2 but with heterogeneous rock thermal (flow) properties per tag (sand vs shale)
    idata = case_2()

    hcap_sand  = 2450.0  # [kJ/m3/K]
    hcap_shale = 2300.0  # [kJ/m3/K]
    idata.rock.heat_capacity = np.array([hcap_shale, hcap_sand, hcap_shale])

    rcond_sand  = 3.0 * 86.4  # [kJ/m/day/K]
    rcond_shale = 2.2 * 86.4  # [kJ/m/day/K]
    idata.rock.thermal_conductivity = np.array([rcond_shale, rcond_sand, rcond_shale])

    return idata


def case_4():
    # same as case_3 but with heterogeneous geomechanical properties per tag (sand vs shale):
    # a softer, more compressible reservoir sand sandwiched between stiffer shale burdens.
    idata = case_3()

    E_sand  = 8.0 * 1e4   # [bars]  (1e4: [GPa] -> [bar])
    E_shale = 20.0 * 1e4  # [bars]
    idata.rock.E = np.array([E_shale, E_sand, E_shale])

    nu_sand  = 0.20  # Poisson ratio
    # 0.25, not the original 0.30: a nu contrast of 0.10 stalls the FS-CPR
    # preconditioner outright (GMRES reaches its iteration cap with no residual
    # reduction at all), while 0.05 keeps a real sand/shale contrast and stays
    # solvable -- 3197 linear iterations in 418 s against 24280 in 2358 s, for
    # the same mesh and the same 17 Newton iterations. Contrast in E and biot is
    # cheap by comparison; nu is what drives the cost here.
    nu_shale = 0.25
    idata.rock.nu = np.array([nu_shale, nu_sand, nu_shale])

    biot_sand  = 0.8
    biot_shale = 0.6
    idata.rock.biot = np.array([biot_shale, biot_sand, biot_shale])

    th_expn_sand  = 1.2e-5  # [1/K] linear thermal expansion coefficient
    th_expn_shale = 0.8e-5
    idata.rock.th_expn_orig = np.array([th_expn_shale, th_expn_sand, th_expn_shale])  # preserve original for proxy

    # recompute the per-tag derived quantities from the E, nu, biot and th_expn_orig above
    recompute_rock_derived(idata)

    return idata
