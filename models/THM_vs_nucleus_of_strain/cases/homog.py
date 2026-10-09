import numpy as np
from cases.base import (
    base_THM,
    set_zero_rate,
    set_depth_shift,
    recompute_rock_derived,
    set_single_y_layer,
    _set_reservoir_bounds,
    _set_wells,
    _set_mesh_tags,
)
from darts.reservoirs.unstruct_reservoir_mech import (
    get_rock_compressibility,
)


def homog():
    # Homogeneous geomechanical properties (E, nu, Biot, thermal expansion) only: porosity,
    # permeability and the thermal properties differ between the reservoir and the over- and
    # underburden (per mesh tag, below). heter adds heterogeneous geomechanical properties.
    idata = base_THM()

    # override permeability (default is 10 mD)
    #idata.rock.permx = idata.rock.permy = idata.rock.permz = 1000.0  # [mD]

    # reservoir geometry (overrides defaults: rsv_top=2000, rsv_bottom=2400, rsv_xy=1000)
    idata.other.rsv_top = 2550.0    # [m]
    idata.other.rsv_bottom = 2600.0  # [m]
    idata.other.rsv_xy = 10000. # [m]

    # well placement (overrides defaults: doublet_shift=500, cell_shift=0)
    idata.other.doublet_shift = 500.0  # [m]

    # doublet rate (default 2000 m3/day); the injector and the producer share it (balanced doublet)
    idata.other.well_rate_m3_day = 8000.0  # [m3/day]
    #idata.other.cell_shift = 500.0     # [m]

    # mesh: Gmsh physical tags: rsv, overburden, underburden
    idata.other.matrix_tags = (99991, 99992, 99993)

    #idata.other.fault = (9991)


    # the fault dip comes from base.py (80 degrees); the variants below override it

    # domain size [m] and extrusion layer length [m] along y, passed to
    # gen_fault_msh_no_damage_zone by main.generate_unstructured_mesh
    idata.other.domain_w = 10000.0  # x extent; the fault crosses mid-depth at x = domain_w / 2
    idata.other.domain_h = 10000.0  # y extent (extrusion length)
    idata.other.layer_dy = 500.0    # extrusion layer length along y
    idata.other.reservoir_thickness = 300.0  # [m] same on both sides of the fault
    idata.other.fault_offset = 200.0         # [m] vertical juxtaposition/throw; 0 = no offset

    # recompute derived values that depend on the overridden parameters above
    _set_reservoir_bounds(idata)
    _set_wells(idata)
    _set_mesh_tags(idata)

    # wells on different sides of the fault (the fault crosses z = 2500 m at x = domain_w / 2, dipping 60 deg):
    # injector well_offset west and producer well_offset east of it, both at the middle of the domain in y.
    # The fault offsets the reservoir layer (tag 99991): it spans z ~ 2450..2750 m west of the fault and
    # z ~ 2250..2550 m east of it. Both wells request the depth range 2000..3000 m; since perforations are
    # restricted to the reservoir tag, each well is perforated over the whole local reservoir layer.
    x_fault, y_mid = idata.other.domain_w / 2., idata.other.domain_h / 2.
    well_offset = 500.  # [m] horizontal distance of each well from the fault at mid-depth (wells 1 km apart)
    idata.other.inj_well_coords = [x_fault - well_offset, y_mid, 2000., 3000.]  # [X, Y, Z1, Z2]
    idata.other.prod_well_coords = [x_fault + well_offset, y_mid, 2000., 3000.]  # [X, Y, Z1, Z2]
    idata.other.well_perforation_tags = (99991,)  # perforate the reservoir tag only

    idata.other.points_xy = []  # no black reference line
    idata.other.use_mesh_bounds_in_plot = True  # plot axes span the whole mesh
    # 2D slices through the well row at the injector depth: the XZ slice shows both wells and the fault.
    # y_mid lies on a boundary of the layer_dy-long mesh layers; the perforations take the layer below it.
    idata.other.plot_slice_origin = [x_fault - well_offset, y_mid - idata.other.layer_dy / 2., 2575.]

    idata.other.set_props_by_tags = True  # rock props below are per matrix tag, in matrix_tags order

    rsv_poro = 0.2
    rsv_perm = 10.0     # [mD]

    non_rsv_poro = 0.001
    non_rsv_perm = 0.001  # [mD]

    idata.rock.porosity = np.array([rsv_poro, non_rsv_poro, non_rsv_poro])

    perm = np.array([rsv_perm, non_rsv_perm,  non_rsv_perm])  # isotropic perm per tag
    idata.rock.permx = idata.rock.permy = perm
    idata.rock.permz = perm * 0.1  # vertical perm is 10x lower

    hcap_sand  = 2450.0  # [kJ/m3/K]
    hcap_shale = 2300.0  # [kJ/m3/K]
    idata.rock.heat_capacity = np.array([hcap_sand, hcap_shale, hcap_shale])

    rcond_sand  = 3.0 * 86.4  # [kJ/m/day/K]
    rcond_shale = 2.2 * 86.4  # [kJ/m/day/K]
    idata.rock.thermal_conductivity = np.array([rcond_sand, rcond_shale, rcond_shale])
    
    idata.rock.E = 20.0 * 1e4   # [bars]
    # the Poisson ratio comes from base.py (0.25), and with it K0 = nu/(1-nu) = 1/3

    # E and the per-tag porosity above replace the base values the derived quantities
    # (kd, compressibility, stiffness, th_expn) were computed from
    recompute_rock_derived(idata)

    idata.other.check_equil = True  # K0 check of the stress state after the equilibration step
    # column for the K0 equil-vs-analytic check: 10% in from the west boundary, clear
    # of the fault trace (at x = domain_w / 2) and the wells (at x_fault +- well_offset)
    idata.other.equil_column_xy = (0.1 * idata.other.domain_w, 0.5 * idata.other.domain_h)

    # the fault offsets the reservoir layer, so a flat rsv_top/rsv_bottom line is wrong here
    idata.other.plot_rsv_lines = False

    return idata

def homog_zero_rate():
    # homog with the wells shut in (rate 0): checks the initial state, nothing should change
    return set_zero_rate(homog())


def homog_deeper1km():
    # homog with the reservoir, the fault and the wells 1 km deeper; the domain grows downwards
    # (gen_fault_msh_no_damage_zone) so the free surface stays at z = 0
    return set_depth_shift(homog(), 1000.0)


def homog_coupled():
    # homog with the two-way coupling on: the mechanics feeds back into the
    # flow through the porosity update, instead of only being driven by it
    idata = homog()
    idata.other.decouple_geomech = False
    return idata


def homog_dip80():
    idata = homog()
    idata.other.fault_dip = 80.0
    return idata

def homog_offset0():
    idata = homog()
    idata.other.fault_offset = 0.0
    return idata

def homog_offset0_1layer():
    # homog_offset0 on a single y layer, at the usual lc. ~4.4 k cells
    # against 88 k, so it runs in minutes - the coarse reference for the lc50 case below.
    return set_single_y_layer(homog_offset0())


def homog_offset0_1layer_lc50():
    # as _1layer but with the cross-section refined, lc 400 -> 50 m. Cells cost ~lc^-2, so
    # this is 5 M on the full 10 km domain (more memory than a workstation has) and ~250 k
    # on the single layer, which is what makes the refinement affordable.
    idata = homog_offset0_1layer()
    idata.other.mesh_lc = 50.0
    return idata


def homog_offset0_dip80():
    idata = homog_offset0()
    idata.other.fault_dip = 80.
    return idata

def homog_offset0_dip80_biot07():
    idata = homog_offset0_dip80()
    idata.rock.biot = 0.7
    recompute_rock_derived(idata)
    return idata

def homog_offset0_dip89():
    idata = homog_offset0()
    idata.other.fault_dip = 89.  # almost vertical fault
    return idata