from model import Model
from reservoir import file_sha256
from darts.tools.memory import print_allocated_memory
from darts.tools.vtk_io import compare_vtk_with_ref, save_vtk_ref
import json
import numpy as np
import os
import shutil
import time
from datetime import datetime
from darts.engines import redirect_darts_output, timer_node
from plot_vtk import plot_vtk_pyvista
from set_case import get_output_directory, get_case_output_directory, set_input_data
from cases.base import print_idata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
str_splitter = '-' * 100 + '\n'


def strip_case_prefixes(base):
    # "thm_" (THM physics of a rect_ mesh, see set_case.py) is not part of the mesh name, so it
    # is stripped before matching case names
    return base[len('thm_'):] if base.startswith('thm_') else base


def is_struct_like_case(case):
    # struct-like cases are named rect_NXxNYxNZ (e.g. 'rect_17x17x15', 'thm_rect_17x17x15')
    # and their box mesh is generated on the fly in the reservoir (reads idata.other.nx/ny/nz)
    return strip_case_prefixes(os.path.basename(case)).startswith('rect_')


def is_damage_zone_case(case):
    return strip_case_prefixes(os.path.basename(case)).startswith('damage')


def is_unstructured_generated_case(case):
    # Cases whose mesh is built by a dedicated gmsh script rather than by the structured box
    # path in reservoir.py. Besides rect_* and the committed-mesh case_* these are the
    # damage* cases (gen_fault_msh) and every other named case (gen_fault_msh_no_damage_zone),
    # so a new no-damage-zone case needs no registration.
    base = strip_case_prefixes(os.path.basename(case))
    return not (is_struct_like_case(base) or base.startswith('case_'))


def supports_mesh_generation(case):
    # True when the mesh of `case` can be (re)generated from the scripts in this
    # folder. case_1..case_4 are not included: they share meshes/case_1/mesh.msh,
    # a committed tetrahedral mesh with no generator in the repo (generate_box_3d
    # produces hexahedra), so they keep loading the committed file.
    return is_struct_like_case(case) or is_unstructured_generated_case(case)


def mesh_key_file(mesh_filename):
    # .cache/ is git-ignored
    return os.path.join(os.path.dirname(mesh_filename), '.cache', 'mesh_key.json')


def mesh_up_to_date(mesh_files, key):
    """True when mesh_files were generated from `key` and have not changed since."""
    try:
        with open(mesh_key_file(mesh_files[0])) as f:
            stored = json.load(f)
        return stored['key'] == key and all(
            stored['files'][os.path.basename(fn)] == file_sha256(fn) for fn in mesh_files)
    except (OSError, ValueError, KeyError):
        return False


def store_mesh_key(mesh_files, key):
    key_file = mesh_key_file(mesh_files[0])
    os.makedirs(os.path.dirname(key_file), exist_ok=True)
    with open(key_file, 'w') as f:
        json.dump({'key': key, 'files': {os.path.basename(fn): file_sha256(fn) for fn in mesh_files}},
                  f, indent=1)


def generate_unstructured_mesh(case, idata=None, bulk_mesh_size=None, force=False):
    """
    (Re)generate the mesh of a named unstructured case. The mesh on disk is kept
    when it was generated from the same input: generator arguments, well
    coordinates, generator source and gmsh version (stored in .cache/mesh_key.json).

    Structured rect_NXxNYxNZ cases are meshed inside the reservoir instead - see
    UnstructReservoirCustom.field_reservoir.

    :param case: case name, e.g. 'damage_zone' or 'homog'.
    :param idata: input data of the case; read via set_case.set_input_data when None.
    :param bulk_mesh_size: far-field Gmsh size [m]; the well size keeps the 1:20 ratio.
        None uses a size-encoded case name or idata.other.bulk_mesh_size.
    :param force: regenerate even when the mesh is up to date.
    :return: the mesh file, or None when the case has no generator.
    """
    base = os.path.basename(case)
    if not is_unstructured_generated_case(base):
        return None

    import gmsh
    if idata is None:
        idata = set_input_data(base)
    # several cases share one mesh (heter reuses
    # homog), so the folder comes from idata rather than the case name
    mesh_dir = getattr(idata.other, 'mesh_dir', None) or base
    mesh_root = getattr(idata.other, 'mesh_root', 'results')
    mesh_filename = os.path.join(BASE_DIR, mesh_root, mesh_dir, 'mesh.msh')
    os.makedirs(os.path.dirname(mesh_filename), exist_ok=True)

    if not is_damage_zone_case(base):
        import gen_fault_msh_no_damage_zone as generator
        # the fault-tagged twin is what fault.py reads when post-processing
        mesh_files = [mesh_filename, os.path.join(os.path.dirname(mesh_filename), 'mesh_fault.msh')]
        # the dip, domain size and reservoir geometry are inputs, so they belong in the
        # cache key: without them a changed value would reuse the mesh of the previous one
        dip_degrees = getattr(idata.other, 'fault_dip', generator.DIP_DEGREES)  # [degrees]
        domain_w = getattr(idata.other, 'domain_w', generator.DOMAIN_W)  # [m]
        domain_h = getattr(idata.other, 'domain_h', generator.DOMAIN_H)  # [m]
        layer_dy = getattr(idata.other, 'layer_dy', generator.LAYER_DY)  # [m]
        reservoir_thickness = getattr(idata.other, 'reservoir_thickness', generator.RESERVOIR_THICKNESS)  # [m]
        fault_offset = getattr(idata.other, 'fault_offset', generator.FAULT_OFFSET)  # [m]
        depth_shift = getattr(idata.other, 'depth_shift', generator.DEPTH_SHIFT)  # [m]
        mesh_lc = getattr(idata.other, 'mesh_lc', generator.MESH_LC)  # [m]
        key = {'generator_sha256': file_sha256(generator.__file__), 'gmsh': gmsh.__version__,
               'fault_dip': float(dip_degrees), 'domain_w': float(domain_w),
               'domain_h': float(domain_h), 'layer_dy': float(layer_dy),
               'reservoir_thickness': float(reservoir_thickness), 'fault_offset': float(fault_offset),
               'depth_shift': float(depth_shift), 'mesh_lc': float(mesh_lc),
               'bnd_tags': dict(idata.mesh.bnd_tags), 'matrix_tags': [int(t) for t in idata.mesh.matrix_tags],
               'fault_tag': int(idata.mesh.fault_tag)}
        if not force and mesh_up_to_date(mesh_files, key):
            print(f'Mesh of {mesh_dir} is up to date, not regenerated')
            return mesh_filename
        print(f'Generating mesh for {mesh_dir} (fault dip {dip_degrees:g} deg, '
              f'domain {domain_w:g}x{domain_h:g} m, fault offset {fault_offset:g} m, '
              f'depth shift {depth_shift:g} m, lc {mesh_lc:g} m)')
        for fault_mesh in (False, True):
            generator.gen_fault_msh_no_damage_zone(
                fault=fault_mesh, mesh_dir=mesh_dir, mesh_root=mesh_root, dip_degrees=dip_degrees,
                domain_w=domain_w, domain_h=domain_h, layer_dy=layer_dy,
                reservoir_thickness=reservoir_thickness, fault_offset=fault_offset,
                depth_shift=depth_shift, mesh_lc=mesh_lc, bnd_tags=idata.mesh.bnd_tags,
                matrix_tags=idata.mesh.matrix_tags, fault_tag=idata.mesh.fault_tag)
        store_mesh_key(mesh_files, key)
        return mesh_filename

    import gen_fault_msh as generator
    if bulk_mesh_size is None:
        bulk_mesh_size = damage_zone_mesh_size(
            base, default=getattr(idata.other, 'bulk_mesh_size', 200.0))
    well_mesh_size = bulk_mesh_size / 20.0
    # generate_3d_fault_mesh reads only the well coordinates from idata
    key = {'generator_sha256': file_sha256(generator.__file__), 'gmsh': gmsh.__version__,
           'bulk_mesh_size': float(bulk_mesh_size), 'well_mesh_size': float(well_mesh_size),
           'prod_well_coords': [float(c) for c in idata.other.prod_well_coords],
           'inj_well_coords': [float(c) for c in idata.other.inj_well_coords],
           'bnd_tags': dict(idata.mesh.bnd_tags), 'matrix_tags': [int(t) for t in idata.mesh.matrix_tags]}
    if not force and mesh_up_to_date([mesh_filename], key):
        print(f'Mesh of {mesh_dir} is up to date (bulk={bulk_mesh_size:g} m, well={well_mesh_size:g} m), '
              f'not regenerated')
        return mesh_filename
    print(f'Generating {mesh_dir}: bulk={bulk_mesh_size:g} m, well={well_mesh_size:g} m')
    generator.generate_3d_fault_mesh(idata, msh_filename=mesh_filename,
                                     well_mesh_size=well_mesh_size)
    store_mesh_key([mesh_filename], key)
    return mesh_filename


def damage_zone_mesh_name(mesh_size):
    """
    Return the case/output folder name for a damage_zone mesh resolution.

    :param mesh_size: Far-field characteristic Gmsh size in metres.
    :type mesh_size: float
    :return: Case name unique to the requested mesh resolution.
    :rtype: str
    """
    size = float(mesh_size)
    if size <= 0.0:
        raise ValueError('damage_zone mesh sizes must be positive')
    size_label = f'{size:g}'
    if float(size_label) != size:  # keep the name lossless: damage_zone_mesh_size() parses it back
        size_label = repr(size)
    size_label = size_label.replace('.', 'p')
    return f'damage_zone_mesh_{size_label}m'


def damage_zone_mesh_size(case, default=200.0):
    """
    Far-field Gmsh size [m] encoded in a damage_zone_mesh_name() case name, e.g. 300.0
    for 'damage_zone_mesh_300m'; `default` for any other case name.
    """
    base = os.path.basename(case)
    if base.startswith('damage_zone_mesh_'):
        return float(base[len('damage_zone_mesh_'):-1].replace('p', '.'))
    return default


def plot_results(model, time_data_dict, out_dir):
    """Plot well data for thermal/isothermal and single/doublet cases."""
    import matplotlib.pyplot as plt
    import pandas as pd

    time_data = pd.DataFrame.from_dict(time_data_dict)
    if time_data.empty or 'time' not in time_data:
        print('Skipping well plots: no well time data are available')
        return []

    # The first stored row belongs to the long-time geomechanical
    # equilibrium solve (typically time=1e8 days).
    time_data = time_data.iloc[1:].reset_index(drop=True)
    if time_data.empty:
        print('Skipping well plots: no transient well time data are available')
        return []
    time_data['Time (years)'] = time_data['time'] / 365.25

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d')
    saved_plots = []

    def save_plot(plot_name):
        filename = f'well_time_data_{plot_name}_{timestamp}.png'
        filepath = os.path.join(out_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=200)
        plt.close()
        saved_plots.append(filepath)

    plt.rc('font', size=12)
    well_names = [well.name for well in model.reservoir.wells]
    production_wells = [name for name in well_names if name.upper().startswith('PRD')]

    bhp_keys = [
        (well_name, f'well_{well_name}_BHP') for well_name in well_names
        if f'well_{well_name}_BHP' in time_data
    ]
    if bhp_keys:
        _, ax = plt.subplots()
        for well_name, key in bhp_keys:
            ax.plot(time_data['Time (years)'], time_data[key], label=well_name)
        ax.set(xlabel='Years', ylabel='BHP [bar]', title='Well BHP')
        ax.set_xlim(left=1.0)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        save_plot('well_bhp')

    # Temperature is available only for thermal physics.
    bht_keys = [
        (well_name, f'well_{well_name}_BHT') for well_name in well_names
        if f'well_{well_name}_BHT' in time_data
    ]
    if model.thermal and bht_keys:
        _, ax = plt.subplots()
        for well_name, key in bht_keys:
            ax.plot(time_data['Time (years)'], time_data[key], label=well_name)
        ax.set(xlabel='Years', ylabel='BHT [K]', title='Well Temperature')
        ax.set_xlim(left=1.0)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        save_plot('well_temperature')

    # Plot wellhead water-rate magnitude for all available wells together.
    rate_suffix = '_volumetric_rate_wat_at_wh'
    rate_keys = [
        (well_name, f'well_{well_name}{rate_suffix}') for well_name in well_names
        if f'well_{well_name}{rate_suffix}' in time_data
    ]
    if rate_keys:
        _, ax = plt.subplots()
        for well_name, key in rate_keys:
            ax.plot(
                time_data['Time (years)'], time_data[key].abs(), label=well_name
            )
        ax.set(xlabel='Years', ylabel='Water rate magnitude [m3/day]',
               title='Well Water Rate')
        ax.set_xlim(left=1.0)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        save_plot('well_water_rate')

    # Cumulative extracted energy is meaningful only for a thermal case with
    # at least one production well. Energy rate is stored in kJ/day;
    # integration over days and 1e-12 convert it to PJ.
    if model.thermal and production_wells:
        heat_suffix = '_energy_rate_relative_to_ref_state_wat_at_wh'
        heat_keys = [
            f'well_{name}{heat_suffix}' for name in production_wells
            if f'well_{name}{heat_suffix}' in time_data
        ]
        if heat_keys:
            energy_data = pd.DataFrame({
                'Time (years)': time_data['Time (years)']
            })
            heat_rate = time_data[heat_keys].sum(axis=1).abs().to_numpy()
            times = time_data['time'].to_numpy()
            dt = np.diff(times, prepend=times[0])
            energy_data['energy'] = np.cumsum(heat_rate * dt) * 1.e-12
            ax = energy_data.plot(x='Time (years)', y='energy', legend=False)
            ax.set(xlabel='Years', ylabel='Extracted energy [PJ]',
                   title='Cumulative Extracted Energy')
            ax.set_xlim(left=1.0)
            ax.grid(True, linestyle='--', alpha=0.3)
            save_plot('extracted_energy')

    print('Well time-data plots:', *saved_plots, sep='\n  ')
    return saved_plots


def run_python(m, days=0, restart_dt=0, init_step = False,
               save_well_data_after_run=True, max_init_cuts=1):
    if days:
        runtime = days
    else:
        runtime = m.ts_control.runtime

    mult_dt = m.ts_control.dt_mult
    max_dt = m.ts_control.dt_max
    m.e = m.physics.engine

    # get current engine time
    t = m.e.t

    # same logic as in engine.run
    if np.fabs(t) < 1e-15:
        dt = m.ts_control.dt_first
    elif restart_dt > 0:
        dt = restart_dt
    else:
        dt = m.ts_control.dt_max

    # evaluate end time
    runtime += t
    ts = 0
    n_init_cuts = 0

    while t < runtime:
        if init_step:   new_time = t
        else:           new_time = t + dt

        if not init_step:
            m.timer.node["update"].start()
            # store boundaries taken at previous time step
            m.reservoir.update(dt=dt, time=new_time)
            # evaluate and assign transient boundaries or sources / sinks
            # m.reservoir.update_boundary(time=new_time, idata=m.idata)
            # update transient boundaries or sources / sinks
            m.reservoir.update_trans(dt, m.physics.engine.X)
            m.timer.node["update"].stop()

        converged = m.nonlinear_solver.run_timestep(dt, t)
        if converged:
            t += dt
            ts = ts + 1
            print("# %d \tT = %f (days), %f (years)\tDT = %f\tNI = %d\tLI=%d"
                  % (ts, t, t / 365.25, dt, m.nonlinear_solver.status.n_newton, m.nonlinear_solver.status.n_linear))

            # Use the configured multiplier for successful growth as well as
            # failed-step reduction instead of a separate hard-coded factor.
            dt *= mult_dt
            if dt > max_dt:
                dt = max_dt

            if t + dt > runtime:
                dt = runtime - t
        else:
            if init_step:
                # The equilibrium step is static (find_equilibrium): a smaller dt leaves the
                # mechanical system and the state unchanged, so a retry repeats the same failed
                # solve and cutting dt would loop forever.
                n_init_cuts += 1
                if n_init_cuts >= max_init_cuts:
                    raise RuntimeError(
                        'Geomechanical equilibrium did not converge after %d attempt(s); cutting dt does '
                        'not change this static problem. Check the linear solver output above (e.g. '
                        '"displacement stage diverged" from FS-CPR).' % n_init_cuts)
            new_time -= dt
            dt /= mult_dt
            print("Cut timestep to %.5e" % dt)

    # update current engine time
    m.e.t = runtime
    if save_well_data_after_run:
        # save well data at every converged time step
        m.output.save_data_to_h5(kind="well")

    stats = m.nonlinear_solver.stats
    print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (stats.n_timesteps_total, stats.n_timesteps_wasted,
                                                     stats.n_newton_total, stats.n_newton_wasted,
                                                     stats.n_linear_total, stats.n_linear_wasted))
def postprocess_fault(m, friction=None, cohesion=None):
    """
    Fault stability post-processing at the end of a run (see fault.py): FSP and the
    Mohr-Coulomb criterion appended to fault<N>.vtu, slip time series, 2D FSP and
    delta_FSP maps per report step and 1D dip profiles through every well.
    Skipped for cases without a fault mesh (no fault<N>.vtu written).
    """
    import fault
    if not fault._fault_files(m.output_directory):
        return
    # same source as the proxy (main_nucleus_of_strain.run_geomech_nucleus_of_strain), so THM and proxy
    # evaluate the slip criterion with identical fault strength
    if friction is None:
        friction = getattr(m.idata.rock, 'friction', 0.6)
    if cohesion is None:
        cohesion = getattr(m.idata.rock, 'cohesion', 0.0)
    print(f'fault post-processing: friction={friction:g}, cohesion={cohesion:g}')
    fault.postprocess_case(m.output_directory, friction=friction, cohesion=cohesion)
    fault.plot_fault_case(m.output_directory, fields=('FSP', 'delta_FSP'))
    centroids = m.reservoir.discr_mesh.centroids
    for well, cell_ids in zip(m.reservoir.wells, m.well_cell_ids):
        point = np.mean([centroids[int(i)].values for i in cell_ids], axis=0)
        print('fault dip profile through well', well.name, 'at', point.round(0))
        fault.plot_fault_dip_profiles(m.output_directory, point=point, friction=friction, cohesion=cohesion)


def state_dependent_viscosity(m, n_ops, visc_idx):
    """Viscosity [cP] per reservoir block, evaluated at the live engine state.

    Does what output.output_properties(engine=True) does, but reads engine.X with the
    engine's own variable count instead of physics.n_vars: the poromechanics engine keeps
    the displacements in X alongside pressure and temperature, so the two differ (5 vs 2)
    and the generic path pairs pressure with a displacement.

    :return: array of length n_res_blocks, in the block order write_to_vtk indexes with.
    """
    from darts.engines import value_vector, index_vector

    nb = m.reservoir.mesh.n_res_blocks
    n_engine_vars = m.physics.engine.get_n_vars()
    x = np.array(m.physics.engine.X, copy=False)[:n_engine_vars * nb]
    # cell_property gives the order: pressure, temperature, then the displacements
    state = value_vector(np.stack([x[0::n_engine_vars],
                                   x[1::n_engine_vars]]).T.flatten())

    values = value_vector(np.zeros(n_ops * nb))
    values_numpy = np.array(values, copy=False)
    dvalues = value_vector(np.zeros(n_ops * nb * m.physics.n_vars))
    itors = (getattr(m.physics, 'output_property_itor', None)
             or m.physics.property_itor)
    op_num = np.array(m.reservoir.mesh.op_num, copy=False)[:nb]

    out = np.zeros(nb)
    for region, itor in itors.items():
        blocks = np.where(op_num == region)[0].astype(np.int32)
        itor.evaluate_with_derivatives(state, index_vector(blocks), values, dvalues)
        out[blocks] = values_numpy[visc_idx::n_ops][blocks]
    return out


def run(model_folder, decouple_geomech=None, generate_mesh=None, report_step=None, sim_time=None,
        plot_vtk_timesteps=[], clear_output_dir=False, solver_type='fs_cpr', save_well_time_data=True,
        cache_discretization=None):
    '''
    The physics and the wells are those of the case (idata.other.physics_type, wells_type,
    set by base_HM / base_THM in cases/base.py); the other settings default to the case too.
    :param model_folder: case name, see set_case.set_input_data
    :param decouple_geomech: turn off mechanics->porosity (so pressure and flow) influence;
        None: idata.other.decouple_geomech (True by default, see cases/base.py)
    :param generate_mesh: if True, the mesh is generated (unless it is up to date), otherwise the mesh
        of a previous run is loaded; None: idata.other.generate_mesh. Cases without a generator
        (case_1..case_4) always load their committed mesh.
    :param report_step: [days] vtk output interval; None: idata.other.report_step
    :param sim_time: [days] simulated time after the equilibration; None: idata.other.runtime_years
    :param solver_type: 'superlu', 'fs_cpr', or 'by_env_var' (see Model.set_solver_params)
    :param save_well_time_data: if True, write the well time-series files (pkl/xlsx).
    :param cache_discretization: reuse the discretization of an earlier run with the same input;
        None: idata.other.cache_discretization (off by default, see cases/base.py)
    :return:
    '''

    t_wall_start = time.time()

    try:
        # if compiled with OpenMP, set to run with 1 thread, as mech tests are not working in the multithread version yet
        from darts.engines import set_num_threads
        set_num_threads(1)
    except:
        pass

    idata = set_input_data(model_folder)
    print_idata(idata)
    physics_type, wells_type = idata.other.physics_type, idata.other.wells_type
    if generate_mesh is None:
        generate_mesh = idata.other.generate_mesh
    generate_mesh = generate_mesh and supports_mesh_generation(model_folder)
    if report_step is None:
        report_step = idata.other.report_step
    if sim_time is None:
        sim_time = idata.other.runtime_years * 365.25

    # The mesh is written into this run's own output folder, next to its solution*.vtu, so
    # it has to be resolved and cleared before anything meshes into it.
    output_directory = get_output_directory(model_folder, physics_type, wells_type)
    if clear_output_dir and os.path.exists(output_directory):
        try:
            shutil.rmtree(output_directory)
        except OSError:
            pass
    if idata.other.mesh_dir is not None:
        # set_input_data created the mesh folder, which may have been the folder cleared above
        os.makedirs(os.path.join(BASE_DIR, idata.other.mesh_root, idata.other.mesh_dir), exist_ok=True)

    # Structured rect_NXxNYxNZ cases are meshed inside the reservoir; the named
    # unstructured ones need their gmsh script run first, so one generate_mesh
    # flag covers both.
    if generate_mesh:
        generate_unstructured_mesh(model_folder, idata=idata)

    m = Model(model_folder=model_folder, idata=idata,
              decouple_geomech=decouple_geomech, generate_mesh=generate_mesh, solver_type=solver_type,
              cache_discretization=cache_discretization)

    plot_fault = getattr(m.idata.other, 'plot_fault', True)
    if decouple_geomech is None:
        m.decouple_geomech = getattr(m.idata.other, 'decouple_geomech', True)

    m.timer.node["model.init()"] = timer_node()
    m.timer.node["model.init()"].start()
    m.init()
    m.timer.node["model.init()"].stop()

    n_vars = m.physics.engine.get_n_vars()
    n_cells = m.reservoir.mesh.n_blocks
    est_mem_gb = 16 * n_vars * n_cells / 1024**2  # 16 KB per cell per variable (for THM)
    print(f"Estimated memory requirement: 16 KB * {n_vars} vars * {n_cells} cells = {est_mem_gb:.2f} GB")

    #m.restart = False
    #m.set_output()

    #redirect_darts_output('log.txt')
    m.timer.node["update"] = timer_node()
    # Properties for writing to vtk format: named after the case (resolved and cleared
    # above, before the mesh was written into it)
    m.output_directory = output_directory


    # Set up darts output to evaluate secondary properties (e.g. viscosity) from the
    # primary variables via the property interpolator. all_phase_props=True registers the
    # phase properties (incl. 'mu_<phase>') in output.properties and builds property_itor.
    # THMCModel.init() does not set some attributes that DartsModel.init() sets but which
    # set_output() reads (self.restart, self.has_dfm_well), so set them explicitly here.
    m.restart = False
    from darts.engines import ms_well
    m.has_dfm_well = any(well.ms_type == ms_well.MS_Type.DFM for well in m.reservoir.wells)
    m.set_output(output_folder=m.output_directory, all_phase_props=True, save_initial=False)

    # Preserve the transient first timestep while equilibrium initialization
    # temporarily replaces it with its intentionally very large timestep.
    transient_first_ts = m.ts_control.dt_first

    # For geomechanics equilibrium intialization, we initially run the simulation for a long time
    # to get the equilibrium, then store that initial displacements internally.
    # Further-timestep displacements will be relative to the initial ones.
    print(str_splitter + 'compute initialization ...\n' + str_splitter)
    m.reservoir.set_equilibrium(zero_conduction=True)
    m.physics.engine.find_equilibrium = True
    dt_init = 1.e+8 # days
    m.ts_control.dt_first = dt_init
    run_python(m, dt_init, init_step=True)
    m.reinit(zero_conduction=True)
    m.physics.engine.find_equilibrium = False
    print(str_splitter + 'initialization completed\n' + str_splitter)

    if getattr(m.idata.other, 'check_equil', False):
        # right after equilibration: write solution0.vtu and plot the equilibrium
        # stress state, then fall through into the transient timesteps below as usual
        m.reservoir.write_to_vtk(m.output_directory, 0, m.physics.engine)
        solution0 = os.path.join(m.output_directory, 'solution0.vtu')
        print('equilibrium state written to', solution0)
        import check_equil
        slice_origin = getattr(m.idata.other, 'plot_slice_origin', None)
        plot_vtk_pyvista(m.output_directory, tstep_to_plot=0, idata=m.idata,
                         slice_origin=slice_origin)

        # computed-vs-analytic (uniaxial-strain K0) stress profile at one column
        column_xy = getattr(m.idata.other, 'equil_column_xy', None)
        if column_xy is None:
            import meshio
            pts = meshio.read(solution0).points
            column_xy = (0.5 * (pts[:, 0].min() + pts[:, 0].max()),
                        0.5 * (pts[:, 1].min() + pts[:, 1].max()))
        col = check_equil.load_column(solution0, column_xy[0], column_xy[1])
        biot = m.idata.rock.biot
        biot = float(biot) if np.isscalar(biot) else float(np.mean(biot))
        ana = check_equil.analytic(col, rho_rock=m.idata.rock.density,
                                   rho_fluid=m.idata.fluid.density, biot=biot)
        check_equil.report(col, ana)
        # plot_vtk_pyvista (above) normally creates this, but not if pyvista is missing
        plots_dir = os.path.join(m.output_directory, 'thm_plots_timestep_0')
        os.makedirs(plots_dir, exist_ok=True)
        check_equil.plot_vs_analytic(col, ana, os.path.join(plots_dir, 'equil_vs_analytic.png'),
                                     'equilibrium stress at x=%.0f, y=%.0f vs uniaxial-strain solution'
                                     % (column_xy[0], column_xy[1]))
        if plot_fault:
            # preliminary fault post-processing from the equilibrium state alone
            # (fault0.vtu only); the run continues below, so postprocess_fault runs
            # again at the end of run() over the full set of fault<N>.vtu and
            # supersedes the fault_plots/dip-profile output written here
            postprocess_fault(m)

            # stresses at exactly the requested depth: the model's own (numerical, not
            # analytic-K0) vertical/horizontal stress, interpolated at that single z, and,
            # on the fault, the values postprocess_fault appended to fault0.vtu (normal
            # total/effective stress, shear, Coulomb mcc, FSP) at the nearest fault face
            target_z = getattr(m.idata.other, 'check_stress_depth', 2200.0)
            col_z = check_equil.load_column(solution0, column_xy[0], column_xy[1],
                                            n_points=2, z_range=(target_z, target_z))
            Sv, Sh_xx, Sh_yy = col_z['tot_stress'][0, 2], col_z['tot_stress'][0, 0], col_z['tot_stress'][0, 1]
            Sv_eff, Sh_xx_eff, Sh_yy_eff = (col_z['eff_stress'][0, 2], col_z['eff_stress'][0, 0],
                                            col_z['eff_stress'][0, 1])
            P = col_z['pressure'][0]
            print(f"\nnumerical stresses at z={target_z:g} m:")
            print(f"  Sv={Sv:.3f}  Sh_xx={Sh_xx:.3f}  Sh_yy={Sh_yy:.3f}  P={P:.3f}  "
                  f"Sv_eff={Sv_eff:.3f}  Sh_eff_xx={Sh_xx_eff:.3f}  Sh_eff_yy={Sh_yy_eff:.3f}  [bar]")
            friction = getattr(m.idata.rock, 'friction', 0.6)
            fault_dip = getattr(m.idata.other, 'fault_dip', float('nan'))
            print(f"  biot={biot:.3f}  friction={friction:g}  fault_dip={fault_dip:g} deg")

            fault0 = os.path.join(m.output_directory, 'fault0.vtu')
            if os.path.exists(fault0):
                import meshio
                fm = meshio.read(fault0)
                fdata = {name: np.concatenate(arrays) for name, arrays in fm.cell_data.items()}
                ffaces = [face for block in fm.cells for face in block.data]
                fdepth = np.array([fm.points[face, 2].mean() for face in ffaces])
                j = int(np.argmin(np.abs(fdepth - target_z)))
                print(f"  fault at z={fdepth[j]:.1f} m: sigma_n={fdata['sigma_n'][j]:.3f}  "
                      f"sigma_n_eff={fdata['sigma_n_eff'][j]:.3f}  tau={fdata['tau'][j]:.3f}  "
                      f"mcc={fdata['mcc'][j]:.3f} bars FSP={fdata['FSP'][j]:.4f}")

    max_dt = report_step
    m.max_dt = max_dt
    m.ts_control.dt_max = max_dt

    first_ts = min(transient_first_ts, report_step)
    m.ts_control.dt_first = first_ts
    m.set_boundary_conditions_after_initialization()

    if m.decouple_geomech:
        m.reservoir.decouple_geomech()

    m.reservoir.create_vtk_wells(output_directory=m.output_directory)


    visc_key = f'mu_{m.physics.phases[0]}'  # state-dependent viscosity property key, e.g. 'mu_wat'
    n_ops = m.output.n_ops
    visc_idx = list(next(iter(m.physics.property_containers.values()))
                    .output_props.keys()).index(visc_key)

    m.timer.node["run_python"] = timer_node()
    m.timer.node["run_python"].start()

    m.time_steps = []
    data = []
    # a case can shut the wells in during the run (cases/check_analytical.py); the report
    # steps end exactly at shut_in_time, so the switch happens at that time
    shut_in_time = getattr(m.idata.other, 'shut_in_time', None)
    # Run over all reporting time-steps:
    ith_step = 0
    t_wall_tsteps_start = time.time()
    while m.physics.engine.t < sim_time:
        days = report_step
        if shut_in_time is not None and m.physics.engine.t < shut_in_time:
            days = min(days, shut_in_time - m.physics.engine.t)
        run_python(m=m, days=days)
        if shut_in_time is not None and abs(m.physics.engine.t - shut_in_time) < 1e-6:
            print('t = %g days: wells shut in (rate 0)' % m.physics.engine.t)
            m.shut_in_wells()
        # state-dependent viscosity [cP] per reservoir block, from the live engine state.
        #
        # output.output_properties(engine=True) cannot be used here. It slices engine.X with
        # physics.n_vars, which is 2 (pressure, temperature), while this poromechanics engine
        # stores 5 values per block (pressure, temperature, ux, uy, uz). It therefore reads
        # only the first 2/5 of the array and then strides through it by 2, pairing pressure
        # with a displacement - the property comes back evaluated at meaningless states,
        # outside any physical range and uncorrelated with p and T.
        viscosity = state_dependent_viscosity(m, n_ops, visc_idx)
        m.reservoir.write_to_vtk(m.output_directory, ith_step + 1, m.physics.engine, viscosity=viscosity)
        ith_step += 1
        m.time_steps.append(m.physics.engine.t)
        data.append(m.get_performance_data(is_last_ts=(m.physics.engine.t >= sim_time)))

        # estimation should be based on elapsed for only the timesteps time, without time spent on initilization
        elapsed = time.time() - t_wall_start
        elapsed_tsteps = time.time() - t_wall_tsteps_start
        progress = m.physics.engine.t / sim_time
        estimated = elapsed_tsteps / progress + (t_wall_tsteps_start - t_wall_start) if progress > 0 else 0
        remaining = estimated - elapsed
        print(f"Wall time: elapsed={elapsed:.1f}s, estimated={estimated:.1f}s, remaining={remaining:.1f}s")

    m.timer.node["run_python"].stop()

    total_elapsed = time.time() - t_wall_start
    print(f"Total elapsed: {total_elapsed:.1f}s")

    print('Timers:')
    m.print_timers()
    #m.print_stat()
    if save_well_time_data:
        time_data_dict = m.output.store_well_time_data(
            save_output_files=True, energy_rate_relative_to_ref_state=True
        )
        plot_results(
            model=m,
            time_data_dict=time_data_dict,
            out_dir=os.path.join(m.output_directory, 'well_time_data_plots'),
        )
    print('Output folder:', m.output_directory, 'Timesteps:', ith_step, 't=', m.physics.engine.t, 'days')
    print_allocated_memory()

    #m.output.plot_well_time_data(phase_volumetric_rate=True)

    for tstep_to_plot in plot_vtk_timesteps:
        plot_vtk_pyvista(m.output_directory, tstep_to_plot=tstep_to_plot, idata=m.idata,
                         slice_origin=getattr(m.idata.other, 'plot_slice_origin', None))

    if plot_fault:
        postprocess_fault(m)

    return m, data


# ---------------------------------------------------------------------------
# Reference solution (vtk) for the test suite
# ---------------------------------------------------------------------------
# The well time-series pkl files used by most models carry only well rates and
# BHP, so they say nothing about the mechanical response (displacements and
# stresses), which is what this model is about. As in
# models/displaced_fault_reactivation, the reference is therefore the vtk
# solution of the last reported timestep, compared field by field.

# Properties compared with the reference. Displacements and effective-stress
# change carry the mechanical response; pressure/temperature carry the flow part
# it is coupled to. The remaining vtk arrays are either constant model input
# (perm, E, poisson, poro) or algebraic combinations of the compared ones
# (tot_stress, eff_stress, strain), so they are left out to keep the reference
# files small.
REF_PROPS = ['pressure', 'temperature', 'ux', 'uy', 'uz', 'delta_eff_stress']
REF_TIMESTEP = 1  # reported timestep to compare (the last one in the test configuration)



def run_test(args: list = [], platform='cpu'):
    import time
    # args: [case, ...]; the case fixes the physics and the wells (the suite appends an overwrite
    # flag, which is not used: references are (re)generated by running with UPLOAD_PKL=1)
    if len(args) < 1:
        print('Not enough arguments for run_test:', args)
        return 1, 0.0
    case = args[0]
    overwrite = os.getenv('UPLOAD_PKL') == '1'
    t0 = time.time()
    try:
        run(
            model_folder=case,
            decouple_geomech=True,
            generate_mesh=True,  # cases without a generator (case_1..case_4) load their committed mesh
            sim_time=30.0,
            report_step=30.0,
            clear_output_dir=True,
            solver_type='by_env_var',
            save_well_time_data=True,
            cache_discretization=False,  # tests exercise the discretizer, and a cache would only cost disk
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1, time.time() - t0

    vtk_fname = 'solution' + str(REF_TIMESTEP) + '.vtu'
    vtk_cur_fname = os.path.join(get_case_output_directory(case), vtk_fname)
    # ref/<case folder>/<solution file>: the leading 'results' of the output folder is
    # dropped - references are committed input, not output
    vtk_ref_fname = os.path.join('ref', os.path.basename(os.path.dirname(vtk_cur_fname)), vtk_fname)
    if overwrite:
        # the old reference is compared before it is replaced: the new file is saved either way,
        # but a difference from the old one still fails the test
        differs = 0
        if os.path.exists(vtk_ref_fname):
            differs = compare_vtk_with_ref(vtk_cur_fname, vtk_ref_fname, props=REF_PROPS, abs_tolerance=1e-7)
        save_vtk_ref(vtk_cur_fname, vtk_ref_fname, props=REF_PROPS)
        return differs, 0.0
    return compare_vtk_with_ref(vtk_cur_fname, vtk_ref_fname, props=REF_PROPS, abs_tolerance=1e-7), time.time() - t0


# The cases the CI suite runs for this model, copied from run_test_suite2.run_testing
# (test_dirs_mech / test_args_mech). Keep in sync with it.
REF_CHECK_CASES = [
    ['rect_17x17x15'],      # HM
    ['thm_rect_17x17x15'],  # THM
]


def run_ref_checks(cases=REF_CHECK_CASES):
    """Repeat the CI reference check of this model locally, without run_test_suite2.

    Each case is checked twice, as in the suite: run_test compares the vtk solution with
    ref/<case>/solution<REF_TIMESTEP>.vtu, and the well time-series pkl is compared with
    ref/well_time_data_<os><solver>_<case>.pkl. The pkl suffix follows the ODLS env var,
    so the comparison picks the reference of the solver build in use.
    Returns the number of failed cases.
    """
    import sys
    models_dir = os.path.dirname(BASE_DIR)
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    from compare_well_time_series import (create_well_time_series_snapshot,
                                          compare_generated_well_time_series,
                                          get_pkl_suffix)

    overwrite = os.getenv('UPLOAD_PKL') == '1'
    results = []
    for args in cases:
        print(str_splitter + 'reference check: ' + ' '.join(args) + '\n' + str_splitter, flush=True)
        # only the files this case writes are compared, so snapshot before it runs
        snapshot = create_well_time_series_snapshot(BASE_DIR)
        failed, t = run_test(list(args))
        if not failed:
            failed_pkl, _, _ = compare_generated_well_time_series(
                BASE_DIR, snapshot, overwrite=overwrite, pkl_suffix=get_pkl_suffix())
            failed += failed_pkl
        results.append((args, failed, t))

    print('\n' + str_splitter + 'reference check summary (pkl suffix: %s)\n' % get_pkl_suffix() + str_splitter)
    for args, failed, t in results:
        print('%-45s %s  (%.1f s)' % (' '.join(args), 'SAVED' if overwrite else
                                      ('FAIL' if failed else 'OK'), t))
    n_failed = sum(1 for _, failed, _ in results if failed)
    print('%d of %d passed' % (len(results) - n_failed, len(results)))
    return n_failed


if __name__ == '__main__':
    try:
        # if compiled with OpenMP, set to run with 1 thread, as mech tests are not working in the multithread version yet
        from darts.engines import set_num_threads
        set_num_threads(1)
    except:
        pass

    # if True, ignore the case list below and instead run the CI reference check of this
    # model (run_ref_checks): the suite's cases with its own short settings, each compared
    # with ref/. Set UPLOAD_PKL=1 to regenerate the references instead of comparing them.
    #check_ref = True
    check_ref = False

    if check_ref:
        exit(run_ref_checks())

    # A case fixes its physics, wells, duration (runtime_years), report step and mesh generation
    # (cases/base.py: base_HM, base_THM); rect_NXxNYxNZ is HM, thm_rect_NXxNYxNZ THM.
    cases = []

    # nx ny nz
    #cases += ['rect_17x17x15']  # for debugging
    #cases += ['thm_rect_17x17x15']
    #cases += ['rect_41x41x66']
    #cases += ['rect_71x71x66']
    #cases += ['rect_83x83x90']  # for isothermal (single well)
    #cases += ['thm_rect_97x97x90'] # for thermal (doublet)
    #cases += ['rect_71x1x66']  # 1 layer by Y; it is not correct to use this as it corresponds to plane-strain solution

    # compare against the analytic formulas for stress changes

    # HM: one injector in the reservoir centre, 10 years, for the poroelastic formula
    #cases += ['rect_17x17x15_check_analytical']
    cases += ['rect_41x41x66_check_analytical']
    # THM: doublet at (0, +-2 km), 100 years of injection then 5 years shut in (cases/check_analytical.py)
    #cases += ['thm_rect_17x17x15_check_analytical']
    #cases += ['thm_rect_41x41x66_check_analytical']

    # thermal cases with unstructured meshes

    #cases += ['damage_zone']

    #cases += ['homog_offset0']
    #cases += ['heter_offset0']
    #cases += ['homog']
    #cases += ['heter']
    # same as homog but with the mechanics->flow coupling on
    #cases += ['homog_coupled']

    # a single Y layer (2D)
    #cases += ['homog_offset0_1layer']
    # finer background mesh, lc = 50 m for the shear stress check
    #cases += ['homog_offset0_1layer_lc50']

    # check initialization (no production/injection)
    #cases += ['homog_zero_rate']
    # the reservoir, the fault and the wells 1 km deeper
    #cases += ['homog_deeper1km']

    for case in cases:
        #os.system("title thm_proxy: " + case + " PID=" + str(os.getpid())) # set the window title
        # a short run of any case: run(..., sim_time=30, report_step=30)
        run(model_folder=case, plot_vtk_timesteps=[0, -1]) # plot initial and last timesteps
