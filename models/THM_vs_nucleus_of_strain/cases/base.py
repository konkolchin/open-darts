import os
import re
import numpy as np
from darts.engines import well_control_iface
from darts.input.input_data import InputData
from darts.reservoirs.unstruct_reservoir_mech import (
    get_bulk_modulus,
    get_isotropic_stiffness,
    get_rock_compressibility,
)


def print_idata(idata):
    """Print the rock properties of a case.

    Called by main.run() once the case is fully built (set_case.set_input_data), so the values
    shown are the ones the run uses. Printing them where they are computed instead reports values a
    later override replaces - biot in input_data_base_biot_07, or E/nu per tag in the
    heterogeneous cases.
    """
    def fmt(value, limit=8):
        a = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
        if a.size == 1:
            return '%g' % a[0]
        shown = ', '.join('%g' % x for x in a[:limit])
        return '[%s%s]' % (shown, ', ...' if a.size > limit else '')

    rock = idata.rock
    nu = np.asarray(rock.nu, dtype=float)
    print('--- rock properties ---')
    if getattr(idata.other, 'set_props_by_tags', False):
        print('  assigned per mesh tag %s' % (idata.other.matrix_tags,))
    print('  E               = %-28s [bar]' % fmt(rock.E))
    print('  nu              = %-28s [-]' % fmt(nu))
    print('  K0 = nu/(1-nu)  = %-28s [-]    uniaxial-strain stress ratio' % fmt(nu / (1. - nu)))
    print('  kd bulk modulus = %-28s [bar]' % fmt(rock.kd))
    print('  biot            = %-28s [-]' % fmt(rock.biot))
    print('  porosity        = %-28s [-]' % fmt(rock.porosity))
    print('  compressibility = %-28s [1/bar] (biot - poro)(1 - biot)/kd' % fmt(rock.compressibility))
    print('  permx/y/z       = %s / %s / %s [mD]'
          % (fmt(rock.permx), fmt(rock.permy), fmt(rock.permz)))
    print('  density         = %-28s [kg/m3]' % fmt(rock.density))
    print('  friction        = %-28s [-]    fault Mohr-Coulomb' % fmt(getattr(rock, 'friction', np.nan)))
    print('  cohesion        = %-28s [bar]  fault Mohr-Coulomb' % fmt(getattr(rock, 'cohesion', np.nan)))
    print('  th_expn         = %-28s [1/K]  linear, before kd x 3 scaling'
          % fmt(getattr(rock, 'th_expn_orig', np.nan)))


def check_rock_props(idata):
    """Check the poroelastic rock properties against the bounds the formulation assumes.

    Raises ValueError so a value outside them is reported where it is set, rather than
    surfacing later as a negative rock compressibility, a non-positive bulk modulus or a
    uniaxial-strain ratio K0 >= 1.

    Each property may be a scalar or a per-tag array; biot may also be a stack of 3x3
    tensors, which is reduced to its mean diagonal as in get_rock_compressibility.
    """
    def as_array(value):
        return np.atleast_1d(np.asarray(value, dtype=float))

    E, nu = as_array(idata.rock.E), as_array(idata.rock.nu)
    poro = as_array(idata.rock.porosity)
    biot = np.asarray(idata.rock.biot, dtype=float)
    if biot.ndim >= 2 and biot.shape[-2:] == (3, 3):
        biot = np.trace(biot, axis1=biot.ndim - 2, axis2=biot.ndim - 1) / 3.
    biot = np.atleast_1d(biot)

    errors = []
    # isotropic elasticity: kd = E/(3(1-2nu)) and G = E/(2(1+nu)) must stay positive, and
    # K0 = nu/(1-nu) reaches 1 at nu = 0.5, the incompressible limit
    if np.any(E <= 0.):
        errors.append('E must be positive, got %s' % E)
    if np.any(nu <= 0.) or np.any(nu >= 0.5):
        errors.append('nu must be in (0, 0.5), got %s' % nu)
    if np.any(poro <= 0.) or np.any(poro > 1.):
        errors.append('porosity must be in (0, 1], got %s' % poro)
    # Biot-Willis: biot = 1 - kd/ks with ks >= kd >= 0, and the storage coefficient is
    # non-negative only for biot >= porosity. Both are what keeps the rock
    # compressibility (biot - poro)(1 - biot)/kd non-negative; the ends (biot = 1,
    # incompressible grains, and biot = porosity, stiff frame) are allowed and give 0.
    biot_b, poro_b = np.broadcast_arrays(biot, poro)  # scalar and per-tag mix freely
    if np.any(biot_b > 1.) or np.any(biot_b < poro_b):
        errors.append('biot must be in [porosity, 1], got %s for porosity %s' % (biot, poro))
    if errors:
        raise ValueError('invalid rock properties:\n  ' + '\n  '.join(errors))


def parse_structured_dims(model_folder):
    # 'rect_17x17x15' -> (17, 17, 15)
    match = re.search(r'(\d+)x(\d+)x(\d+)$', os.path.basename(model_folder))
    if match is None:
        raise ValueError("structured mesh case %r is not named rect_NXxNYxNZ" % model_folder)
    return tuple(int(n) for n in match.groups())


def input_data_base(thermal=True):
    # True = single_phase_thermal (rate control); False = single_phase (BHP control)
    idata = InputData(
        type_hydr="thermal" if thermal else "isothermal",
        type_mech="thermoporoelasticity" if thermal else "poroelasticity",
        init_type="gradient",
    )

    idata.other.thermal = thermal  # True: rate control with temperature; False: isothermal, BHP control

    # --- reservoir geometry ---
    idata.other.rsv_top = 2000.0     # [m] reservoir top depth
    idata.other.rsv_bottom = 2400.0  # [m] reservoir bottom depth
    idata.other.rsv_xy = 1000.0      # [m] reservoir half-width; sets rsv_x1/x2/y1/y2 in _set_reservoir_bounds

    # --- well parameters ---
    idata.other.wells_type = "doublet"     # "doublet", "inj", or "prod"
    idata.other.doublet_shift = 500.0      # [m] half-distance between injection and production well
    idata.other.cell_shift = 0.0           # [m] lateral offset to align well with a cell centre
    idata.other.delta_temp_inj = 40.0     # [K] injected fluid temperature delta vs. initial reservoir T
    idata.other.well_rate_m3_day = 2000.0  # [m3/day] volumetric rate (thermal / rate-control mode)
    idata.other.well_perforation_tags = None  # matrix tags the wells may perforate; None: all matrix cells
    idata.other.bhp_delta_p = 10.0         # [bar] BHP offset from initial pressure (isothermal / BHP-control mode)

    # --- mesh tags ---
    idata.other.matrix_tags = (99991,)   # single tag for structured-like meshes; override for named cases
    idata.other.set_props_by_tags = False  # if True, rock props assigned per Gmsh physical tag

    _set_rock_data(idata)
    _set_reservoir_bounds(idata)
    _set_rock_mechanics(idata)
    _set_fluid_data(idata)
    _set_initial_conditions(idata)
    _set_wells(idata)
    _set_mesh_tags(idata)

    # a folder name with the mesh file, None if the mesh is generated on the fly
    idata.other.mesh_dir = None
    # directory holding <mesh_dir>/mesh.msh, relative to the model folder. Generated meshes
    # are output, so they go next to the results; a case shipping a committed mesh (case_1)
    # overrides this with 'meshes', which is where its tracked mesh.msh lives.
    idata.other.mesh_root = 'results'

    # fault dip [degrees from the horizontal], read by main.generate_unstructured_mesh and
    # by the fault post-processing. Only the cases with a fault generator use it.
    idata.other.fault_dip = 80.0

    # reference points [x, y, label] drawn as a black line in plot_vtk (empty = no line)
    idata.other.points_xy = [[250., 250., '(250,250)']]
    idata.other.use_mesh_bounds_in_plot = False  # True: plot axes span the mesh; False: manual window around the reservoir
    idata.other.plot_slice_origin = None  # [x, y, z] the 2D slice plots pass through; None: the mesh center
    # if False, plot_vtk skips the rsv_top/rsv_bottom reference lines: they assume a flat
    # reservoir, which is wrong to draw where a fault offsets it (see homog)
    idata.other.plot_rsv_lines = True
    # which of plot_vtk_pyvista's 2D slices ('xz', 'yz', 'xy') to plot; XY is left out
    # to avoid tripling the plotting cost/output
    idata.other.plot_slices = ['xz', 'yz']
    # if False, plot_vtk skips the well marker lines (e.g. for zero-rate/shut-in cases,
    # where the wells do nothing and drawing them is misleading)
    idata.other.plot_wells = True
    # if True, run() post-processes the fault vtu files (FSP, Coulomb) and writes the
    # fault_plots; cases without a fault in the mesh have nothing to post-process
    idata.other.plot_fault = True

    # if True, the mechanics does not feed back into the flow: the porosity update is
    # switched off (reservoir.decouple_geomech), so pressure is unaffected by deformation.
    # It is the cheaper and better converging direction and the field-scale error is small;
    # a case that wants the two-way coupling sets this False. run()'s argument overrides it.
    idata.other.decouple_geomech = True

    # if True, run() generates the mesh of a case that has a generator (rect_* box mesh, gmsh
    # scripts), unless the mesh on disk was made from the same input; False loads the mesh of
    # a previous run. run()'s argument overrides it.
    idata.other.generate_mesh = True

    # if True, reuse a cached discretization (meshes/<mesh_dir>/.cache/discretization_*.npz)
    # keyed on mesh content, domain tags, boundary conditions and rock data - see
    # UnstructReservoirCustom.discretization_cache_key. Off by default: a stale cache
    # silently reusing an outdated discretization is a worse failure mode than a slower
    # run. run()'s cache_discretization arg overrides this when set.
    idata.other.cache_discretization = False

    # if True, run() checks the state right after the equilibration step: it writes
    # solution0.vtu and plots the equilibrium stress state and the computed-vs-analytic
    # (uniaxial-strain K0) stress profile (check_equil.py), then continues into the
    # transient timesteps as usual
    idata.other.check_equil = False
    idata.other.equil_column_xy = None  # (x, y) of the K0 check column; None: mesh center
    # [m] depth the equilibrium stress state is reported at (run(), see check_equil):
    # inside the reservoir interval, so it follows a case that moves the reservoir
    idata.other.check_stress_depth = 2200.0

    _set_obl(idata)
    return idata


def set_zero_rate(idata):
    """Shut the wells in: rate control with a zero target.

    Isothermal cases default to BHP control in _set_wells (target = p_cell +/- delta_p),
    which ignores well_rate, so the control type has to change for the well to be genuinely
    shut in. Thermal already uses rate control, which makes this consistent for both.

    Returns idata so it can wrap a case builder.
    """
    idata.other.wctrl_type = well_control_iface.MASS_RATE
    idata.other.delta_p = None
    idata.other.well_rate = 0.
    # the wells do nothing, so drawing their marker lines is misleading
    idata.other.plot_wells = False
    return idata


def set_depth_shift(idata, depth_shift):
    """Move the reservoir, fault and wells `depth_shift` metres deeper.

    The mesh generator grows the domain downwards by the same amount, keeping the free
    surface at z = 0 so the extra overburden is actually carried - see
    gen_fault_msh_no_damage_zone. Everything here is what the generator does not know
    about: the reservoir bounds, the well intervals, the depths the plots and the
    equilibrium check are taken at, and the mesh folder.

    Returns idata so it can wrap a case builder.
    """
    idata.other.depth_shift = depth_shift
    idata.other.rsv_top += depth_shift
    idata.other.rsv_bottom += depth_shift
    for coords in (idata.other.inj_well_coords, idata.other.prod_well_coords):
        coords[2] += depth_shift  # Z1 of the requested perforation interval
        coords[3] += depth_shift  # Z2
    if idata.other.plot_slice_origin is not None:
        idata.other.plot_slice_origin[2] += depth_shift
    idata.other.check_stress_depth += depth_shift
    # the initial pressure/temperature gradients are referenced to z = 0, so they already
    # give the deeper reservoir its higher pressure; the x/y reservoir bounds do not move.
    # The mesh folder is left to the caller, which names it after the case (set_case).
    return idata


def set_single_y_layer(idata):
    """Collapse the model to one cell along y: domain_h = layer_dy, so n_extrude is 1.

    The result is quasi-2D and close to plane strain - the y faces carry roller BCs, so
    u_y is clamped a single cell apart. It cuts the cell count by the number of layers the
    full domain had, which is what makes a fine cross-section affordable.

    The wells, the slice plot and the K0 column were placed at the middle of the original
    domain, so they are moved to the middle of the remaining layer; leaving them would put
    them outside it. Returns idata so it can wrap a case builder.
    """
    idata.other.domain_h = idata.other.layer_dy
    y_mid = idata.other.domain_h / 2.
    idata.other.inj_well_coords[1] = y_mid
    idata.other.prod_well_coords[1] = y_mid
    if idata.other.plot_slice_origin is not None:
        idata.other.plot_slice_origin[1] = y_mid
    if idata.other.equil_column_xy is not None:
        idata.other.equil_column_xy = (idata.other.equil_column_xy[0], y_mid)
    return idata


def _set_run_settings(idata, physics_type, wells_type, runtime_years):
    # what a case runs with, read by run() and main_nucleus_of_strain; a case can override any of them
    idata.other.physics_type = physics_type  # 'single_phase' or 'single_phase_thermal', in the results folder name
    idata.other.wells_type = wells_type      # 'inj', 'prod' or 'doublet', in the results folder name
    idata.other.runtime_years = runtime_years            # [years] simulated time after the equilibration step
    idata.other.report_step = 365.25 / 2     # [days] interval of the vtk output
    _set_wells(idata)


def base_HM():
    """Hydro-mechanical (isothermal) layer of the cases: one injector under BHP control, 10 years."""
    idata = input_data_base(thermal=False)
    _set_run_settings(idata, physics_type='single_phase', wells_type='inj', runtime_years=10)
    return idata


def base_THM():
    """Thermo-hydro-mechanical layer of the cases: a doublet under rate control, 60 years."""
    idata = input_data_base(thermal=True)
    _set_run_settings(idata, physics_type='single_phase_thermal', wells_type='doublet', runtime_years=60)
    return idata


def input_data_base_biot_07(thermal=True):
    """Base input data (HM or THM layer) with the Biot coefficient pinned to 0.7.
    """
    idata = base_THM() if thermal else base_HM()
    idata.rock.biot = 0.7
    recompute_rock_derived(idata)
    return idata


def input_data_base_biot_1(thermal=True):
    """Base input data (HM or THM layer) with the Biot coefficient pinned to 1.0
    """
    idata = base_THM() if thermal else base_HM()
    idata.rock.biot = 1.0
    recompute_rock_derived(idata)
    return idata

def _set_rock_data(idata):
    idata.rock.density = 2650.0            # [kg/m3]
    idata.rock.porosity = 0.1              # [-]
    idata.rock.permx = idata.rock.permy = idata.rock.permz = 10.0  # [mD]
    idata.rock.thermal_conductivity = 260  # [kJ/m/day/K]
    idata.rock.heat_capacity = 2300        # [kJ/m3/K]
    # rock properties outside the reservoir coordinate bounds
    idata.rock.poro_non_rsv = 0.001
    idata.rock.perm_non_rsv = 0.01        # [mD]

    # Mohr-Coulomb fault strength, read by BOTH the THM fault post-processing
    # (fault.py via main.postprocess_fault) and the geomechanical proxy
    # (geomechanics.geomech via main_nucleus_of_strain.run_geomech_nucleus_of_strain), so the two always
    # evaluate the slip criterion with the same parameters.
    idata.rock.friction = 0.6   # fault friction coefficient [-]
    idata.rock.cohesion = 0.0   # fault cohesion [bar]; 0 assumes healing is absent


def _set_reservoir_bounds(idata):
    # permeable reservoir vertical boundaries
    idata.other.rsv_x1 = -idata.other.rsv_xy
    idata.other.rsv_x2 =  idata.other.rsv_xy
    idata.other.rsv_y1 = -idata.other.rsv_xy
    idata.other.rsv_y2 =  idata.other.rsv_xy


def recompute_rock_derived(idata):
    """Recompute the rock properties derived from E, nu, biot, porosity and th_expn_orig.

    Call this after overriding any of those five, otherwise the case keeps the derived
    values of whatever set them last - usually input_data_base, whose E, nu and scalar
    porosity a named case has just replaced. The stale ones reach the run:
    reservoir.py reads compressibility as self.cs and kd through get_biot_modulus.

    Idempotent: th_expn is rebuilt from th_expn_orig rather than scaled in place, so
    calling it twice gives the same result.
    """
    idata.rock.kd = get_bulk_modulus(E=idata.rock.E, nu=idata.rock.nu)
    idata.rock.compressibility = get_rock_compressibility(
        kd=idata.rock.kd,
        biot=idata.rock.biot,
        poro0=idata.rock.porosity,
    )
    idata.rock.stiffness = get_isotropic_stiffness(idata.rock.E, idata.rock.nu)
    # Cauchy 4.19a/4.21a, then linear -> volumetric (4.22)
    idata.rock.th_expn = idata.rock.th_expn_orig * idata.rock.kd * 3.0


def _set_rock_mechanics(idata):
    idata.rock.biot = 0.9  # Biot coefficient [-]
    idata.rock.E = 12.0 * 1e4  # [GPa]  # 1e4: [GPa] -> [bar]
    idata.rock.nu = 0.25  # Poisson ratio
    idata.rock.E_non_rsv = idata.rock.E  # E outside the reservoir bounds; equal to E = homogeneous mechanics

    idata.rock.th_expn_orig = 1e-5  # [1/K] linear thermal expansion coefficient
    recompute_rock_derived(idata)

    idata.rock.th_expn_poro = 0.0  # mechanical term in porosity update


def _set_fluid_data(idata):
    # single-phase water properties
    idata.fluid.Mw = 18.015         # molar weight [g/mol]
    idata.fluid.compressibility = 4.4e-5  # [1/bar]
    idata.fluid.viscosity = 1.0     # [cP]
    idata.fluid.density = 1000.0    # [kg/m3]
    idata.fluid.thermal_conductivity = 0.0  # not used by the engine [kJ/m/day/K]
    idata.fluid.heat_capacity = 75.0        # [kJ/kmol/K] (water: 75.37 kJ/kmol/K)


def _set_initial_conditions(idata):
    # non-zero initial temperature does not converge properly (check t_ref implementation)
    idata.initial.reference_depth_for_temperature = 0.0  # [m]
    idata.initial.temperature_gradient = 0.0             # [K/m]
    idata.initial.temperature_at_ref_depth = 0.0         # [K]
    # pressure_gradient used only for reservoir.p_init; actual initial pressure is from equilibrium
    idata.initial.pressure_gradient = 0.1   # [bar/m]
    idata.initial.pressure_at_ref_depth = 1.0  # [bar] at z = 0


def _set_wells(idata):
    """
    Well placement and the requested perforation interval [Z1, Z2].

    The wells are open to the whole reservoir thickness. model.set_wells walks
    this interval in steps of step_z_perf and perforates the nearest eligible
    cell of each step, so the interval is what decides how many layers the well
    sees. Cases whose reservoir is not a single flat layer (damage_zone,
    no_damage_zone_*) overwrite the coordinates after calling this.
    """
    # place a single well at centre; for doublet, shift both wells symmetrically
    shift = 0.0
    if idata.other.wells_type == "doublet":
        shift = idata.other.doublet_shift  # [m] half-distance between wells

    # The 1 m insets keep the scan away from the layer boundaries, where the
    # nearest cell centre can belong to the over- or underburden.
    perf_depth = idata.other.rsv_top + 1.0         # [m]
    perf_depth_end = idata.other.rsv_bottom - 1.0  # [m]

    # cell_shift aligns the well with a cell centre when the mesh is centred at (0,0)
    if idata.other.wells_type in ("doublet", "prod"):
        idata.other.prod_well_coords = [
            idata.other.cell_shift - shift,
            idata.other.cell_shift,
            perf_depth,
            perf_depth_end,
        ]  # [X, Y, Z1, Z2]
    if idata.other.wells_type in ("doublet", "inj"):
        idata.other.inj_well_coords = [
            idata.other.cell_shift + shift,
            idata.other.cell_shift,
            perf_depth,
            perf_depth_end,
        ]  # [X, Y, Z1, Z2]

    # well control: wctrl_type, target well_rate [kg/day] (rate control) or delta_p [bar] (BHP offset from the cell pressure)
    if idata.other.thermal:  # rate control
        idata.other.delta_p = None
        idata.other.wctrl_type = well_control_iface.MASS_RATE
        idata.other.well_rate = idata.other.well_rate_m3_day * idata.fluid.density  # [m3/day] -> [kg/day]
    else:  # BHP control
        idata.other.delta_p = idata.other.bhp_delta_p  # [bar]
        idata.other.wctrl_type = well_control_iface.BHP
        idata.other.well_rate = None

def _set_mesh_tags(idata):
    # Physical-group tags: the BCs and rock properties look cells up by them, and the mesh generators
    # (generate_box_3d, gen_fault_msh*) create the groups with them. An external mesh must carry the same numbers.
    # matrix_tags (volumes) are positional, the order is fixed by the generator: gen_fault_msh_no_damage_zone
    # (reservoir, overburden, underburden), gen_fault_msh (reservoir, overburden, underburden, damage zone left/right, fault).
    idata.mesh.bnd_tags = {}
    tags = idata.mesh.bnd_tags
    tags["BND_X-"] = 991
    tags["BND_X+"] = 992
    tags["BND_Y-"] = 993
    tags["BND_Y+"] = 994
    tags["BND_Z-"] = 995   # TOP
    tags["BND_Z+"] = 996   # BOTTOM

    idata.mesh.fault_tag = 9991  # fault surfaces in mesh_fault.msh of gen_fault_msh_no_damage_zone, read by fault.py

    # z is depth in the generated meshes; the depth-based initial state and the BND_Z-/BND_Z+ (top/bottom)
    # boundary conditions assume it as well, so an external mesh with z upwards needs more than this flag
    idata.mesh.z_axis_direction = 'downwards'

    idata.mesh.matrix_tags = list(idata.other.matrix_tags)
    idata.mesh.tags = idata.mesh.bnd_tags.copy()
    for i, mat_tag in enumerate(idata.mesh.matrix_tags, start=1):
        idata.mesh.tags[f"MATRIX_{i}"] = mat_tag


def _set_obl(idata):
    n_points = 400                # OBL points per axis
    idata.obl.zero = 1e-9         # keeps the composition axis off z = 0 and z = 1
    idata.obl.n_points = n_points
    idata.obl.epsilon_z = 1e-10   # composition-axis offset passed to the physics

    p_min, p_max = 0.0, 1000.0    # [bar]
    t_min, t_max = -50.0, 50.0    # [K]
    z_min, z_max = idata.obl.zero, 1 - idata.obl.zero

    # OBL grid: origin + step per axis, step = (axis_max - axis_min) / (n_points - 1)
    idata.obl.p_origin = p_min
    idata.obl.p_step = (p_max - p_min) / (n_points - 1)
    idata.obl.t_origin = t_min
    idata.obl.t_step = (t_max - t_min) / (n_points - 1)
    idata.obl.z_origin = z_min
    idata.obl.z_step = (z_max - z_min) / (n_points - 1)


def _xc_from_nx(nx):
    if nx == 1:
        xc_left = np.array([-100])
    elif nx == 7:
        xc_left = np.array([-4000, -2000, -1000, -100])
    elif nx == 17:
        xc_left = np.array([-4000, -2000, -1000, -500, -400, -300, -200, -100, -50])
    elif nx == 97:
        xc_left = np.array(
            [
                -8000,
                -6000,
                -5000,
                -4000,
                -3000,
                -2500,
                -2000,
                -1600,
                -1500,
                -1450,
                -1400,
                -1350,
                -1300,
                -1250,
                -1200,
                -1150,
            ]
            + [-1100, -1050, -1030, -1010, -1000, -990, -980, -950, -900]
            + np.arange(-800, -555, 50).tolist()
            + np.arange(-555, -455, 10).tolist()
            + np.arange(-455, -5, 50).tolist()
        )
    elif nx == 83:
        xc_left = np.array(
            [
                -8000,
                -6000,
                -5000,
                -4000,
                -3000,
                -2500,
                -2000,
                -1600,
                -1500,
                -1450,
                -1400,
                -1350,
                -1300,
                -1250,
                -1200,
                -1150,
            ]
            + [-1100, -1050, -1030, -1010, -1000, -990, -980, -950, -900]
            + np.arange(-800, -100, 100).tolist()
            + np.arange(-100, 0, 10).tolist()
        )
    elif nx == 71:
        xc_left = np.array(
            [
                -8000,
                -6000,
                -5000,
                -4000,
                -3000,
                -2500,
                -2000,
                -1600,
                -1400,
                -1200,
            ]
            + [-1100, -1050, -1030, -1010, -1000, -990, -980, -950, -900]
            + np.arange(-800, -100, 100).tolist()
            + np.arange(-100, 0, 10).tolist()
        )
    elif nx == 41:
        xc_left = np.array(
            [
                -8000,
                -6000,
                -5000,
                -4000,
                -3000,
                -2500,
                -2000,
                -1600,
                -1400,
                -1200,
            ]
            + [-1100, -1000, -900]
            + np.arange(-800, -100, 100).tolist()
            + [-50]
        )
    else:
        raise ValueError(f"not found an option to mesh with nx = {nx}")

    return np.hstack([xc_left, -xc_left[::-1]])


def get_structured_mesh_coordinates(idata):
    nx, ny, nz = idata.other.nx, idata.other.ny, idata.other.nz
    xc = _xc_from_nx(nx)
    yc = _xc_from_nx(ny)

    rsv_top = idata.other.rsv_top
    rsv_bottom = idata.other.rsv_bottom
    if nz == 5:
        zc = np.array([0, 1000, 2000, 2100, 2200, 3000])
    elif nz == 15:
        zc = np.array([0, 1000, 1500, 2000, 2100, 2120, 2140, 2160, 2180, 2200, 2300, 2500, 3000, 4000, 5000, 6000])
    elif nz == 29:
        zc = np.hstack([np.arange(0, rsv_top, 200), np.arange(rsv_top, rsv_bottom, 20), np.arange(rsv_bottom, 5000, 200)])
    elif nz == 37:
        zc = np.hstack([np.arange(0, rsv_top, 150), np.arange(rsv_top, rsv_bottom, 20), np.arange(rsv_bottom, 5000, 150)])
    elif nz == 53:
        zc = np.hstack([np.arange(0, rsv_top, 100), np.arange(rsv_top, rsv_bottom, 20), np.arange(rsv_bottom, 5000, 100)])
    elif nz == 57:
        zc = np.hstack(
            [
                np.arange(0, rsv_top - 100 + 1, 100),
                np.arange(rsv_top - 50, rsv_bottom + 50 + 1, 25),
                rsv_bottom + 100,
                np.arange(rsv_bottom + 200, 5000 + 1, 100),
            ]
        )
    elif nz == 66:
        zc = np.hstack(
            [
                np.arange(0, rsv_top - 100 + 1, 100),
                np.arange(rsv_top - 50, rsv_bottom + 50 + 1, 25),
                rsv_bottom + 100,
                np.arange(rsv_bottom + 200, 5000 + 1, 100),
            ]
        )
    elif nz == 90:
        zc = np.hstack(
            [
                np.arange(0, rsv_top - 500 + 1, 100),
                np.arange(rsv_top - 450, rsv_bottom + 450 + 1, 25),
                rsv_bottom + 500,
                np.arange(rsv_bottom + 600, 5000 + 1, 100),
            ]
        )
    else:
        raise ValueError(f"not found an option to mesh with nz = {nz}")

    return xc, yc, zc


def input_data_struct_like(model_folder, thermal):
    # unstructured mesh is generated on the fly with rectangular hexahedral cells;
    # HM: one injector, THM: a doublet (base_HM / base_THM)

    idata = input_data_base_biot_07(thermal=thermal)
    #idata = input_data_base_biot_1(thermal=thermal)

    # structured mesh dimensions parsed from the folder name (NXxNYxNZ)
    idata.other.nx, idata.other.ny, idata.other.nz = parse_structured_dims(model_folder)
    # [m] node coordinates of the generated box mesh along x, y, z (sizes nx+1, ny+1, nz+1)
    idata.other.Xc, idata.other.Yc, idata.other.Zc = get_structured_mesh_coordinates(idata)

    return idata
