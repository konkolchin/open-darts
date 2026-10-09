import numpy as np
from cases.base import (
    base_THM,
    recompute_rock_derived,
    _set_reservoir_bounds,
    _set_wells,
    _set_mesh_tags,
)


def damage_zone(mesh_dir='damage_zone',
                      bulk_mesh_size=500.0, fault_dip_degrees=-45.0,
                      injector_fault_distance=300.0,
                      well_plane_strike=0, reservoir_block_offset=50,
                      damage_zone_width=50):
    """
    Build case 5 inputs.

    :param mesh_dir: Mesh folder name.
    :param bulk_mesh_size: Far-field Gmsh mesh size in metres.
    :param fault_dip_degrees: Fault dip used by the mesh and well placement.
    :param injector_fault_distance: Horizontal distance from the injector to
        the fault-core centre at the reservoir midpoint [m].
    :param well_plane_strike: Horizontal well-plane angle in degrees measured
        from the fault strike. Zero keeps the wells parallel to the fault
        strike.
    :param reservoir_block_offset: Rigid vertical offset applied to the
        reservoir block opposite the wells [m].
    :param damage_zone_width: Width of the damage zone on each side of the
        fault core [m].
    :return: Case 5 input data.
    """
    idata = base_THM()

    # override permeability (default is 10 mD)
    #idata.rock.permx = idata.rock.permy = idata.rock.permz = 1000.0  # [mD]

    # reservoir geometry (overrides defaults: rsv_top=2000, rsv_bottom=2400, rsv_xy=1000)
    idata.other.rsv_top = 2830.0    # [m]
    idata.other.rsv_bottom = 3030.0  # [m]
    idata.other.rsv_xy = 10000. # 4500.0      # [m]
    idata.other.well_rate_m3_day = 8000  # [m3/day]
    idata.other.fault_dip_degrees = float(fault_dip_degrees)  # [degrees], used by the mesh and well placement
    idata.other.injector_fault_distance = float(injector_fault_distance)  # [m]
    idata.other.fault_offset = 500.0  # [m], at z_old = 0
    idata.other.fault_thickness = 15.0  # [m], true thickness normal to fault
    idata.other.well_plane_strike = float(well_plane_strike)  # [degrees] angle of the well pair from the fault strike
    idata.other.reservoir_block_offset = float(reservoir_block_offset)  # [m] vertical shift of the block opposite the wells
    # well placement (overrides defaults: doublet_shift=500, cell_shift=0)
    idata.other.doublet_shift = 500.0  # [m]
    #idata.other.cell_shift = 500.0     # [m]

    # mesh: Gmsh physical tags of the volumes, in the order of gen_fault_msh: reservoir, overburden, underburden,
    # damage zone left, damage zone right, fault
    idata.other.matrix_tags = (1, 2, 3, 4, 5, 99991)
    # Wells must connect to reservoir rock only, never to a confining layer
    # when their requested endpoint lies exactly on a material interface.
    idata.other.well_perforation_tags = (1,)
    # Well mesh size retains the 1:20 ratio of the reference mesh.
    idata.other.bulk_mesh_size = float(bulk_mesh_size)  # [m] far-field gmsh size
    idata.other.damage_zone_width = float(damage_zone_width)  # [m] on each side of the fault core
    # mesh_dir = mesh_dir + '_' + str(int(idata.other.bulk_mesh_size))
    # # case_2 and case_3 inherit this, so all three cases share one mesh
    # idata.other.mesh_dir = mesh_dir

    # recompute derived values that depend on the overridden parameters above
    _set_reservoir_bounds(idata)
    _set_wells(idata)
    _set_mesh_tags(idata)

    # reference point at the reservoir midpoint so that the injector-to-fault distance is
    # independent of fault dip.  The mesh generator uses the same convention.
    middle_depth = 0.5 * (idata.other.rsv_top + idata.other.rsv_bottom)
    middle_z_old = 2800.0 - middle_depth
    fault_middle_y_old = (
        idata.other.fault_offset
        + middle_z_old / np.tan(np.deg2rad(idata.other.fault_dip_degrees))
        + 0.5 * idata.other.fault_thickness / np.sin(
            np.deg2rad(idata.other.fault_dip_degrees)
        )
    )
    well_y = idata.other.rsv_xy / 2.0 + fault_middle_y_old - idata.other.injector_fault_distance

    # damage_zone owns its well locations, derived from the reservoir geometry above.
    # Carried on idata to both the model and the fault-mesh generator. [X, Y, Z1, Z2].
    xc = idata.other.rsv_xy / 2.0
    inj_x = xc - idata.other.doublet_shift
    well_distance = 2.0 * idata.other.doublet_shift
    well_plane_angle = np.deg2rad(idata.other.well_plane_strike)
    idata.other.inj_well_coords = [inj_x, well_y,
                                   idata.other.rsv_top, idata.other.rsv_bottom]
    idata.other.prod_well_coords = [
        inj_x + well_distance * np.cos(well_plane_angle),
        well_y + well_distance * np.sin(well_plane_angle),
        idata.other.rsv_top,
        idata.other.rsv_bottom,
    ]

    idata.other.points_xy = []  # no black reference line
    idata.other.use_mesh_bounds_in_plot = True  # plot axes span the whole mesh
    # Plot proxy/THM slices over the complete physical mesh domain.  Other
    # cases retain the historical reservoir-focused slice bounds.
    idata.other.proxy_plot_use_mesh_bounds = True

    idata.other.set_props_by_tags = True

    rsv_poro = 0.2
    rsv_perm = 500.0    # [mD]
    damage_zone_perm = 600 # [mD]
    damage_zone_poro = 0.1
    fault_perm = 0.1 # [mD]
    fault_poro = 0.01
    non_rsv_poro = 0.001
    non_rsv_perm = 0.001  # [mD]

    idata.rock.porosity = np.array([rsv_poro, non_rsv_poro,  non_rsv_poro, damage_zone_poro,damage_zone_poro,fault_poro])
    perm = np.array([rsv_perm,non_rsv_perm,  non_rsv_perm, damage_zone_perm, damage_zone_perm, fault_perm])  # isotropic perm per tag
    idata.rock.permx = idata.rock.permy = perm
    idata.rock.permz = perm * 0.1  # vertical perm is 10x lower

    hcap_sand  = 2450.0  # [kJ/m3/K]
    hcap_shale = 2300.0  # [kJ/m3/K]
    idata.rock.heat_capacity = np.array([hcap_sand, hcap_shale,  hcap_shale, hcap_sand, hcap_sand, hcap_sand])

    rcond_sand  = 3.0 * 86.4  # [kJ/m/day/K]
    rcond_shale = 2.2 * 86.4  # [kJ/m/day/K]
    idata.rock.thermal_conductivity = np.array([rcond_sand, rcond_shale, rcond_shale, rcond_sand,rcond_sand, rcond_sand])
    # Region order: sandstone, shale, shale, damage zone, damage zone, fault core.
    # Values are regional midpoints rounded to two decimal places.
    # idata.rock.E = np.array([16.50, 21.00, 21.00, 27.50, 27.50, 5.03]) * 1e4  # [bar] (GPa -> bar)
    idata.rock.E = np.array([8, 10, 10, 5, 5, 1]) * 1e4  # [bar] (GPa -> bar)
    # idata.rock.nu = np.array([0.18, 0.28, 0.28, 0.25, 0.25, 0.26])
    idata.rock.nu = np.array([0.10, 0.20, 0.20, 0.20, 0.20, 0.20])
    # idata.rock.biot = np.array([0.74, 0.73, 0.73, 0.90, 0.90, 0.95])
    idata.rock.biot = np.array([0.5, 0.55, 0.55, 0.80, 0.80, 0.90])
    # Preserve original linear thermal expansion coefficients for the proxy [1/K].
    # idata.rock.th_expn_orig = np.array([30.00, 25.00, 25.00, 17.50, 17.50, 15.50]) * 1e-6
    idata.rock.th_expn_orig = np.array([10, 10, 10, 8, 8, 1]) * 1e-6

    # recompute the per-tag derived quantities from the E, nu, biot and th_expn_orig above
    recompute_rock_derived(idata)
    return idata
