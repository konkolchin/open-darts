import numpy as np
import os
import meshio
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.interpolate import griddata as gd
from functools import reduce

from darts.tools.logging import fmt
from main import run
from plot_vtk import write_plot_time_file
from set_case import get_output_directory, set_input_data

# unit conversion factors
m2mm = 1e3
bars2mpa = 0.1
Pa2bars = 1e-5
g_grav = 9.81  # m/s^2

# Short file names of the contour plots. The plot titles are long ('Horizontal effective stress
# change (XX), MPa - Nucleus of strain'), and the case folder plus such a title exceeds the
# 260-character path limit of Windows.
CONTOUR_FILE_NAMES = {
    'Pressure change, MPa': 'dp',
    'Temperature change, K': 'dt',
    'Horizontal displacements (X), mm.': 'ux',
    'Vertical displacements, mm.': 'uz',
    'Horizontal effective stress change (XX), MPa': 'dsxx_eff',
    'Vertical effective stress change, MPa': 'dszz_eff',
    'Horizontal total stress change (XX), MPa': 'dsxx_tot',
    'Vertical total stress change, MPa': 'dszz_tot',
    'delta_pressure, MPa': 'dp',
}
CONTOUR_SOURCE_NAMES = {'Nucleus of strain': 'nos', 'THM': 'thm', 'Difference': 'diff'}


def contour_file_name(arr_name):
    # '<quantity>[ - <source>]' -> '<short quantity>[_<short source>]_contour.png'
    quantity, _, source = arr_name.partition(' - ')
    name = CONTOUR_FILE_NAMES.get(quantity, quantity)
    if source:
        name += '_' + CONTOUR_SOURCE_NAMES.get(source, source)
    for ch in ' ,()':
        name = name.replace(ch, '_' if ch == ' ' else '')
    return name + '_contour.png'

def read_vtk_darts_solution(folder, timestep : int):
    # read <folder>/solution<timestep>.vtu written by the THM run
    filename = os.path.join(folder, 'solution'+str(timestep)+'.vtu')
    if not os.path.isfile(filename):
        print('THM vtk file not found:', filename)
        print('Run THM first to generate vtk files with p,T changes (by setting run_thm = True)')
        raise FileNotFoundError(filename)
    msh = meshio.read(filename)
    print('Reading', filename)
    #print("\tCells:", msh.cells_dict.keys())
    #print("\tCell Data:", msh.cell_data.keys())
    return msh

# volume cells the nucleus-of-strain solution cannot represent: a bounding box misrepresents their
# volume and shape. Surface and line cells are skipped.
UNSUPPORTED_VOLUME_CELL_TYPES = {'wedge', 'tetra', 'pyramid'}


def check_volume_cell_type(cell_block):
    """
    :return: True for a hexahedral block, False for a surface or line block
    :raises ValueError: for wedge, tetra or pyramid cells
    """
    if cell_block.type in UNSUPPORTED_VOLUME_CELL_TYPES:
        raise ValueError(
            f'The nucleus-of-strain solution supports only hexahedral cells, '
            f'the mesh has {len(cell_block.data)} {cell_block.type} cells'
        )
    return cell_block.type == 'hexahedron'


def geomech_init_geometry(mesh_data):
    """
    Convert the hexahedral cells to the rectangular prisms used by the nucleus-of-strain solution.

    Each hexahedron is represented by its axis-aligned bounding box, which is exact for the
    rectangular cells of the structured meshes. Other volume cells (wedge, tetra, pyramid)
    raise ValueError: a bounding box misrepresents their volume and shape.

    :return: prisms, centroids (Y, X, Z), quadrature prisms (always None: only the
             tetrahedral meshes had them), cell sizes
    """
    prism_blocks = []
    centroid_blocks = []
    cell_size_blocks = []

    def make_prisms(centres, extents):
        lower = centres - 0.5 * extents
        upper = centres + 0.5 * extents
        return np.column_stack([
            lower[:, 1], upper[:, 1],
            lower[:, 0], upper[:, 0],
            upper[:, 2], lower[:, 2],
        ])

    for cell_block in mesh_data.cells:
        if not check_volume_cell_type(cell_block):
            continue

        vertices = mesh_data.points[cell_block.data]
        centroids = vertices.mean(axis=1)
        extents = vertices.max(axis=1) - vertices.min(axis=1)
        if np.any(np.prod(extents, axis=1) <= 0.0):
            raise ValueError('Degenerate hexahedron cell encountered while initializing nucleus-of-strain geometry')

        prism_blocks.append(make_prisms(centroids, extents))
        centroid_blocks.append(centroids[:, [1, 0, 2]])
        cell_size_blocks.append(extents.max(axis=1))

    if not prism_blocks:
        raise ValueError('No hexahedral cells found in the VTK mesh')

    prisms = np.concatenate(prism_blocks)
    centroids = np.concatenate(centroid_blocks)
    cell_sizes = np.concatenate(cell_size_blocks)
    return prisms, centroids, None, cell_sizes

class thm_solution:
    # container for the THM fields read from vtk (read_thm_solution_from_vtk), attributes are set there
    pass


def get_volume_cell_data(mesh_data, name):
    """
    Return the cell data of the hexahedral cells, concatenated in the order of the prisms
    built by geomech_init_geometry.
    """
    arrays = [
        np.asarray(values)
        for cell_block, values in zip(
            mesh_data.cells, mesh_data.cell_data[name], strict=True
        )
        if check_volume_cell_type(cell_block)
    ]
    if not arrays:
        raise ValueError(f'No hexahedral cell data found for {name!r}')
    return np.concatenate(arrays, axis=0)


def read_thm_solution_from_vtk(m, folder : str, timestep: int):
    # THM solution at `timestep` and its change from timestep 0 (p, T, displacements, strains, stresses),
    # plus the prisms of the reservoir cells that are the sources of the nucleus-of-strain solution
    thm_sol = thm_solution()

    msh_initial = read_vtk_darts_solution(folder=folder, timestep=0)
    poro = get_volume_cell_data(msh_initial, 'poro').flatten()
    thm_sol.p_init = get_volume_cell_data(msh_initial, 'pressure').flatten()
    thm_sol.Szz_init = get_volume_cell_data(msh_initial, 'tot_stress')[:, 2] * bars2mpa # ZZ

    msh_last = read_vtk_darts_solution(folder=folder, timestep=timestep)
    p_last = get_volume_cell_data(msh_last, 'pressure').flatten()
    thm_sol.ux_last = get_volume_cell_data(msh_last, 'ux').flatten()
    thm_sol.uy_last = get_volume_cell_data(msh_last, 'uy').flatten()
    thm_sol.uz_last = get_volume_cell_data(msh_last, 'uz').flatten()
    delta_eff_stress = get_volume_cell_data(msh_last, 'delta_eff_stress')
    delta_tot_stress = get_volume_cell_data(msh_last, 'delta_tot_stress')
    strain = get_volume_cell_data(msh_last, 'strain')
    thm_sol.delta_Sxx_last = delta_eff_stress[:, 0] * bars2mpa #  XX
    thm_sol.delta_Syy_last = delta_eff_stress[:, 1] * bars2mpa #  YY
    thm_sol.delta_Szz_last = delta_eff_stress[:, 2] * bars2mpa # ZZ
    thm_sol.delta_total_Sxx_last = delta_tot_stress[:, 0] * bars2mpa #  XX
    thm_sol.delta_total_Syy_last = delta_tot_stress[:, 1] * bars2mpa #  YY
    thm_sol.delta_total_Szz_last = delta_tot_stress[:, 2] * bars2mpa # ZZ
    thm_sol.qx_last = strain[:, 0]  #  XX
    thm_sol.qy_last = strain[:, 1]  #  YY
    thm_sol.qz_last = strain[:, 2]  # ZZ

    thm_sol.folder = folder

    if 'delta_pressure' in msh_last.cell_data.keys():
        thm_sol.delta_pressure = get_volume_cell_data(msh_last, 'delta_pressure').flatten()
    else:
        p_initial = get_volume_cell_data(msh_initial, 'pressure').flatten()
        thm_sol.delta_pressure = p_last - p_initial
    thm_sol.delta_pressure *= 0.1 # bars to MPa

    # delta_temperature is zero in isothermal case
    thm_sol.delta_temperature = np.zeros_like(thm_sol.delta_pressure)
    if 'delta_temperature' in msh_last.cell_data.keys():
        thm_sol.delta_temperature = get_volume_cell_data(msh_last, 'delta_temperature').flatten()

    prisms, centroids, quadrature_prisms, cell_sizes = geomech_init_geometry(msh_initial)
    print('\tprisms all', prisms.shape[0])

    #perm_threshold_for_prisms = 1e-6
    perm_threshold_for_prisms = 0 # do not use [rsv] filtering

    if m.idata.rock.perm_non_rsv <= perm_threshold_for_prisms: #use only the permeable part, assuming there is no p,T change in the impermeable part
        rsv = poro > m.idata.rock.poro_non_rsv  # reservoir (permeable) cells only will be used as input for the nucleus-of-strain solution
    else: # there will be pressure diffusion, so need to use all cells
        rsv = poro > 0 # use all cells in the nucleus-of-strain solution

    thm_sol.delta_pressure_rsv = thm_sol.delta_pressure[rsv]
    thm_sol.delta_temperature_rsv = thm_sol.delta_temperature[rsv]
    thm_sol.prisms_rsv = prisms[rsv, :]
    if quadrature_prisms is not None:
        thm_sol.quadrature_prisms_rsv = quadrature_prisms.reshape(-1, 4, 6)[rsv].reshape(-1, 6)
        thm_sol.cell_sizes_rsv = cell_sizes[rsv]
    else:
        thm_sol.quadrature_prisms_rsv = None
        thm_sol.cell_sizes_rsv = None
    print('\tprisms rsv', thm_sol.prisms_rsv.shape[0])

    # centroids are only used for THM data plotting, they are not used in the nucleus-of-strain solution
    thm_sol.centroids = centroids
    # Keep the physical mesh bounds as Y, X, Z, matching the coordinate order of the nucleus-of-strain solution
    # convention.  Cell-centre bounds omit half a boundary cell and should not
    # be used as the plotting domain for meshes such as case_5 that start at 0.
    thm_sol.mesh_bounds = [
        (msh_initial.points[:, axis].min(), msh_initial.points[:, axis].max())
        for axis in (1, 0, 2)
    ]

    n_dim = 3  # X,Y,Z
    thm_sol.n_dim = n_dim
    thm_sol.bounds = [0]*n_dim
    for k in range(n_dim):
        thm_sol.bounds[k] = centroids[:, k].min(), centroids[:, k].max()

    thm_sol.rsv_centroids = centroids[rsv, :]
    return thm_sol

def vtk_step_time_days(folder, timestep: int):
    # simulation time [days] of <folder>/solution<timestep>.vtu, from its entry in solution.pvd
    import xml.etree.ElementTree as ET
    root = ET.parse(os.path.join(folder, 'solution.pvd')).getroot()
    for dataset in root.iter('DataSet'):
        if dataset.get('file') == f'solution{timestep}.vtu':
            return float(dataset.get('timestep'))
    raise ValueError(f'solution{timestep}.vtu is not listed in {folder}/solution.pvd')

# field of plot_delta_along_x -> (thm_solution attribute, axis label)
ALONG_X_FIELDS = {
    'pressure': ('delta_pressure', 'Pressure change, MPa'),
    'temperature': ('delta_temperature', 'Temperature change, K'),
}

def plot_delta_along_x(case, timesteps, field='pressure',
                       y=0.0, z=2200.0, x_range=None, n_points=200, output_folder=None):
    """Plot the THM pressure or temperature change along the X axis at fixed Y, Z, overlaying several timesteps.

    Reads the THM solution vtk for each timestep, interpolates the change from the cell
    centers onto a line of points running along X (at the given Y and Z), and draws all
    timesteps on a single combined figure.

    :param case: case name, e.g. 'rect_41x41x66'; the results folder follows its physics and wells
    :param timesteps: list of vtk timestep indices to overlay
    :param field: 'pressure' [MPa] or 'temperature' [K]
    :param y, z: fixed coordinates [m] of the line; it runs along X (defaults Y=0, Z=2200)
    :param x_range: optional (x_min, x_max) [m]; defaults to the THM centroid X-extent
    :param n_points: number of sample points along X
    :param output_folder: where to save the png; defaults to <results folder>/plots_delta_<field>_x
    """
    attr, axis_label = ALONG_X_FIELDS[field]
    # lightweight model just to provide idata (rsv filtering etc.) to the vtk reader
    from model import Model
    m = Model(model_folder=case, decouple_geomech=True, dummy='yes')
    m.set_input_data()
    folder = get_output_directory(case, m.physics_type, m.wells_type)

    if output_folder is None:
        output_folder = os.path.join(folder, f'plots_delta_{field}_x')
    os.makedirs(output_folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for ts in timesteps:
        thm_sol = read_thm_solution_from_vtk(m, folder=folder, timestep=ts)
        # sample points along X at fixed Y, Z
        if x_range is None:
            x_min, x_max = thm_sol.centroids[:, 1].min(), thm_sol.centroids[:, 1].max()
        else:
            x_min, x_max = x_range
        points_x = np.linspace(x_min, x_max, n_points)
        points_y = np.full_like(points_x, y)
        points_z = np.full_like(points_x, z)
        # interpolate the change from cell centers (X,Y,Z) onto the line points
        line = gd((thm_sol.centroids[:, 1], thm_sol.centroids[:, 0], thm_sol.centroids[:, 2]),
                  getattr(thm_sol, attr), (points_x, points_y, points_z), method='linear')
        ax.plot(points_x, line, label=f't = {vtk_step_time_days(folder, ts) / 365.25:.4g} years')

    ax.set_xlabel('X, m.')
    ax.set_ylabel(axis_label + '.')
    ax.set_title(f'{axis_label.split(",")[0]} along X (Y={y:g}, Z={z:g})')
    ax.legend(fontsize=8)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth=0.8)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)
    fig.tight_layout()
    out = os.path.join(output_folder, f'delta_{field}_along_x.png')
    fig.savefig(out)
    plt.close(fig)
    print('Saved', out)

def plot_plan_view(thm_sol, idata, wells_type, thermal, output_folder):
    # XY maps of the pressure and temperature changes in the cell layer nearest to the reservoir
    # mid-depth, over the whole mesh: wells away from Y = 0 (e.g. the doublet at Y = +-2 km of the thermoelastic check)
    # are outside the XZ slices at Y = 0
    z_mid = 0.5 * (idata.other.rsv_top + idata.other.rsv_bottom)
    cz = thm_sol.centroids[:, 2]
    z_layer = cz[np.argmin(np.abs(cz - z_mid))]
    layer = np.isclose(cz, z_layer, atol=1.0)
    x, y = thm_sol.centroids[layer, 1], thm_sol.centroids[layer, 0]
    fields = [('Pressure change, MPa', thm_sol.delta_pressure, 'dp_xy_contour.png')]
    if thermal:
        fields.append(('Temperature change, K', thm_sol.delta_temperature, 'dt_xy_contour.png'))
    for title, values, fname in fields:
        v = values[layer]
        vmin, vmax = v.min(), v.max()
        if np.isclose(vmin, vmax):
            vmin, vmax = vmin - 0.01, vmax + 0.01
        levels = np.linspace(vmin, vmax, 12)
        if vmin < 0 < vmax:
            levels = np.sort(np.unique(np.append(levels, 0.)))
        fig, ax = plt.subplots(figsize=(7, 7))
        cs = ax.tricontourf(x, y, v, levels=levels)
        fig.colorbar(cs, ax=ax, orientation='horizontal', pad=0.1)
        rsv_xy = idata.other.rsv_xy
        if rsv_xy < x.max():
            ax.plot([-rsv_xy, rsv_xy, rsv_xy, -rsv_xy, -rsv_xy], [-rsv_xy, -rsv_xy, rsv_xy, rsv_xy, -rsv_xy],
                    color='black', linewidth=0.8, linestyle='--', label='reservoir sides')
        if wells_type in ('prod', 'doublet'):
            ax.plot(*idata.other.prod_well_coords[0:2], 'o', color='red', label='production well')
        if wells_type in ('inj', 'doublet'):
            ax.plot(*idata.other.inj_well_coords[0:2], 'o', color='cyan', markeredgecolor='black',
                    label='injection well')
        ax.set_aspect('equal')
        ax.minorticks_on()
        ax.set_xlabel('X, m.')
        ax.set_ylabel('Y, m.')
        ax.set_title(f'{title}\nXY slice at depth {z_layer:g} m')
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        fig.savefig(os.path.join(output_folder, fname), bbox_inches='tight')
        plt.close(fig)
        print(f'Saving {fname}, min={v.min():.4g}, max={v.max():.4g}')

def run_geomech_nucleus_of_strain(case, timestep=1, modes=[], n_threads=1, use_gpu=False, read_from_cache=False):
    # Compute the nucleus-of-strain solution from the THM p, T changes at `timestep` and compare it with
    # the THM displacements and stresses. `modes` selects the outputs: 'plot_2d_slices', 'plot_vertic_line',
    # 'print_at_point', 'check_initial', '2d_slices_41_71', 'plot_2d_thm_41_vs_71'.
    # init the nucleus-of-strain solution
    from darts.physics.geomechanics.nucleus_of_strain.nucleus_of_strain import geomech
    g = geomech()
    # just to set input data
    from model import Model
    m = Model(model_folder=case, decouple_geomech=True, dummy='yes')
    m.set_input_data()
    physics_type, wells_type = m.physics_type, m.wells_type  # those of the case
    thermal = physics_type == 'single_phase_thermal'
    folder = get_output_directory(case, physics_type, wells_type)  # where vtk files are located

    def scalar_or_mean(val):
        # the analytic nucleus-of-strain solution assumes a homogeneous elastic medium; if idata carries
        # per-tag heterogeneous geomech properties (e.g. case_4), fall back to their mean
        if np.isscalar(val):
            return val
        print(f'warning: the nucleus-of-strain solution assumes homogeneous properties, using mean of {val} = {val.mean()}')
        return val.mean()

    # elastic constants
    g.poisson = scalar_or_mean(m.idata.rock.nu)
    g.young = scalar_or_mean(m.idata.rock.E)
    g.young *= bars2mpa
    g.thermal_expansion = scalar_or_mean(m.idata.rock.th_expn_orig)
    g.biot = scalar_or_mean(m.idata.rock.biot)

    # Mohr-Coulomb parameters from idata, so the nucleus-of-strain solution uses the same fault strength
    # as the THM post-processing instead of the class defaults.
    g.friction = scalar_or_mean(m.idata.rock.friction)
    g.cohesion = scalar_or_mean(m.idata.rock.cohesion)
    print(f'Mohr-Coulomb from idata: friction={g.friction:g}, cohesion={g.cohesion:g}')

    g.set_num_threads(n_threads)
    print('N_THREADS =', n_threads)
    if use_gpu:
        from darts.physics.geomechanics.nucleus_of_strain.nucleus_of_strain import HAS_GPU
        if not HAS_GPU:
            raise RuntimeError('use_gpu=True but open-darts was built without GPU support')
        g.set_platform('gpu')
        print('PLATFORM = gpu')
    else:
        print('PLATFORM = cpu')

    thm_sol = read_thm_solution_from_vtk(m, folder=folder, timestep=timestep)
    g.centroids = thm_sol.rsv_centroids
    use_adaptive_proxy = thm_sol.quadrature_prisms_rsv is not None
    if use_adaptive_proxy:
        g.set_adaptive_sources(
            thm_sol.prisms_rsv, thm_sol.quadrature_prisms_rsv,
            thm_sol.cell_sizes_rsv, thm_sol.delta_pressure_rsv,
            thm_sol.delta_temperature_rsv, near_factor=2.5,
        )
        print('UNSTRUCTURED NUCLEUS OF STRAIN = adaptive four-point tetrahedron, near factor 2.5')

    output_folder = os.path.join(thm_sol.folder, 'thm_vs_nucleus_plots_timestep_' + str(timestep))
    os.makedirs(output_folder, exist_ok=True)
    write_plot_time_file(output_folder, vtk_step_time_days(folder, timestep), timestep)

    def find_cell_by_point(point):
        # find an index of the cell, closest to the desired point
        cell = ((thm_sol.centroids[:, 0] - point[0]) ** 2 + (thm_sol.centroids[:, 1] - point[1]) ** 2 + (
                    thm_sol.centroids[:, 2] - point[2]) ** 2).argmin()
        #print('get_thm_solution', 'closest cell is', thm_sol.centroids[cell, :], 'point', point)
        return cell

    def get_cell_center(point):
        # centroid of the cell closest to the point
        cell = find_cell_by_point(point)
        return thm_sol.centroids[cell, 0], thm_sol.centroids[cell, 1], thm_sol.centroids[cell, 2]

    def get_thm_by_interp(array_dict, points_x, points_y, points_z, method='linear'):
        # THM cell arrays interpolated to the given points
        array_dict_interp = dict()
        for arr_name, arr in array_dict.items():
            # interpolate the solution (arr) from cell centers to given set of points
            array_dict_interp[arr_name] = gd((thm_sol.centroids[:, 1], thm_sol.centroids[:, 0], thm_sol.centroids[:, 2]), \
                arr, (points_x, points_y, points_z), method=method)
        return array_dict_interp

    def get_thm_dp_dt(point, verbose=False): # pressure (in MPa) and temperature (in K) changes
        cell = find_cell_by_point(point)
        dp = thm_sol.delta_pressure[cell]
        dt = thm_sol.delta_temperature[cell]
        return dp, dt

    def get_thm_displs(point, verbose=False):
        # THM displacements [m] of the cell closest to the point
        cell = find_cell_by_point(point)
        ux_thm = thm_sol.ux_last[cell]
        uy_thm = thm_sol.uy_last[cell]
        uz_thm = thm_sol.uz_last[cell]
        return ux_thm, uy_thm, uz_thm

    def get_thm_strain(point, verbose=False):
        # THM strain of the cell closest to the point
        cell = find_cell_by_point(point)
        qx_thm = thm_sol.qx_last[cell]
        qy_thm = thm_sol.qy_last[cell]
        qz_thm = thm_sol.qz_last[cell]
        return qx_thm, qy_thm, qz_thm

    def get_thm_stress(point, verbose=False): # effective stress in MPa
        cell = find_cell_by_point(point)
        return thm_sol.delta_Sxx_last[cell],  thm_sol.delta_Syy_last[cell],  thm_sol.delta_Szz_last[cell]

    def get_thm_total_stress(point, verbose=False): # total stress in MPa
        cell = find_cell_by_point(point)
        return thm_sol.delta_total_Sxx_last[cell],  thm_sol.delta_total_Syy_last[cell],  thm_sol.delta_total_Szz_last[cell]

    def get_proxy_displs(eval_points):  # Y,X,Z
        # nucleus-of-strain displacements [m] at eval_points (rows Y, X, Z), poroelastic + thermoelastic
        eps = 0  # [m], to avoid r=0 for the 1/r integral of the nucleus-of-strain solution
        eval_points_eps = eval_points + eps
        #eval_points_eps = eval_points_eps.transpose()
        if use_adaptive_proxy:
            upx1, upy1, upz1, utx1, uty1, utz1 = \
                g.calc_displacements_adaptive_cpp(eval_points_eps)
        else:
            upx1, upy1, upz1, utx1, uty1, utz1 = \
                g.calc_displacements_cpp(
                    eval_points_eps, thm_sol.prisms_rsv,
                    thm_sol.delta_pressure_rsv, thm_sol.delta_temperature_rsv
                )
        ux = upx1 + utx1
        uy = upy1 + uty1
        uz = upz1 + utz1
        return uy, ux, uz # thermoporoelastic displacements [m]

    def get_eval_points(mode, shift_x=0, shift_y=0):
        # evaluation points (rows Y, X, Z): the cell centers, or a vertical line through the domain center
        if mode == 'centers':
            # X<->Y
            eval_points = np.zeros_like(thm_sol.centroids)
            eval_points[:, 1] = thm_sol.centroids[:, 0] # X
            eval_points[:, 0] = thm_sol.centroids[:, 1]
            eval_points[:, 2] = thm_sol.centroids[:, 2]
        elif mode == 'vertical':
            # test with a vertical line
            z = np.unique(thm_sol.centroids[:, 2])
            eval_points = np.zeros((z.size, 3))
            eval_points[:, 1] = thm_sol.centroids[:, 0].mean() + shift_x
            eval_points[:, 0] = thm_sol.centroids[:, 1].mean() + shift_y
            eval_points[:, 2] = z
        return eval_points.transpose()

    def get_proxy_strain_stress(eval_points_):
        # nucleus-of-strain strain and stress [MPa] at the points, one point (1D array) or several (rows Y, X, Z)
        if len(eval_points_.shape) == 1:  # just one point (1d array)
            eval_points = np.zeros((3, 1))
            eval_points[0, :] = eval_points_[0]  # X<->Y
            eval_points[1, :] = eval_points_[1]
            eval_points[2, :] = eval_points_[2]
        else: # multiple points, 2d array
            eval_points = eval_points_
        # returns thermoporoelastic strain and  stress in MPa, (6, n_points), 6 - Voight notation
        eps = 0  # [m], to avoid r=0 for the 1/r integral of the nucleus-of-strain solution
        eval_points_eps = eval_points + eps
        if use_adaptive_proxy:
            res = g.calc_strain_stress_adaptive_cpp(
                eval_points_eps, thm_sol.delta_pressure_rsv,
                thm_sol.delta_temperature_rsv,
            )
        else:
            res = g.calc_strain_stress_cpp(
                eval_points_eps, thm_sol.prisms_rsv,
                thm_sol.delta_pressure_rsv, thm_sol.delta_temperature_rsv
            )
        stress_p, strain_p, stress_total_p, stress_t, strain_t, stress_total_t, stress, strain, stress_total = res
        #[Sp_xx, Sp_yy, Sp_zz, Sp_yz, Sp_xz, Sp_xy] = stress_p
        #[St_xx, St_yy, St_zz, St_yz, St_xz, St_xy] = stress_t
        return strain[1,:], strain[0,:], strain[2,:], \
            stress[1,:], stress[0,:], stress[2,:], stress_total[1,:], stress_total[0,:], stress_total[2,:]  # xx yy zz


    def compare_vert_line(points, suffix='', loc='', output_folder='.', array_names={}):
        # plot THM vs nucleus-of-strain profiles along a vertical line, one figure per array in array_names
        z_range = points[2,:]

        array_dict = {'dp': thm_sol.delta_pressure, 'dt': thm_sol.delta_temperature}
        array_dict_interp = get_thm_by_interp(array_dict, points[1,:], points[0,:], points[2,:], method='linear') # obtain thm solutiona at points using interpolation
        dp = array_dict_interp['dp']
        dt = array_dict_interp['dt']

        ux_prx, uy_prx, uz_prx = get_proxy_displs(points)
        qx_prx, qy_prx, qz_prx, sx_prx, sy_prx, sz_prx, sx_total_prx, sy_total_prx, sz_total_prx =  get_proxy_strain_stress(points)

        # get total from effective stress
        if False: # to avoid smoothing at the rsv boundaries
            dp_z = np.zeros_like(z_range)
            z_range_rsv = reduce(np.logical_and, [z_range > 2100, z_range < 2200])
            dp_z[z_range_rsv] = dp.max() #m.idata.other.delta_p #MPa
        sx_total_prx = sx_prx + g.biot * dp
        sy_total_prx = sy_prx + g.biot * dp
        sz_total_prx = sz_prx + g.biot * dp

        n_points = points.shape[1]
        ux_thm = np.zeros(n_points); uy_thm = np.zeros(n_points); uz_thm = np.zeros(n_points);
        qx_thm = np.zeros(n_points); qy_thm = np.zeros(n_points); qz_thm = np.zeros(n_points);
        sx_thm = np.zeros(n_points); sy_thm = np.zeros(n_points); sz_thm = np.zeros(n_points);
        sx_total_thm = np.zeros(n_points); sy_total_thm = np.zeros(n_points); sz_total_thm = np.zeros(n_points);

        for i in range(n_points):  # use XY from point and different Z
            point = np.array([points[0, i], points[1, i], points[2, i]])  # YXZ - > XYZ
            ux_thm[i], uy_thm[i], uz_thm[i] = get_thm_displs(point)
            if 'strain' in array_names:
                qx_thm[i], qy_thm[i], qz_thm[i] = get_thm_strain(point)
            if 'delta_eff_stress_z' in array_names:
                sx_thm[i], sy_thm[i], sz_thm[i] = get_thm_stress(point)
            if 'delta_total_stress_z' in array_names:
                sx_total_thm[i], sy_total_thm[i], sz_total_thm[i] = get_thm_total_stress(point)

        for array_name in array_names:
            match array_name:
                case 'displ_z': prx = uz_prx * m2mm; thm = uz_thm * m2mm; s = 'Vertical displacement, mm.' + ' at ' + loc
                case 'displ_y': prx = uy_prx * m2mm; thm = uy_thm * m2mm; s = 'Horizontal displacement (Y), mm.' + ' at ' + loc
                case 'displ_x': prx = ux_prx * m2mm; thm = ux_thm * m2mm; s = 'Horizontal displacement (X), mm.' + ' at ' + loc

                case 'strain_z': prx = qz_prx; thm = qz_thm; s = 'Vertical strain' + ' at ' + loc
                case 'strain_y': prx = qy_prx; thm = qy_thm; s = 'Horizontal strain (YY)' + ' at ' + loc
                case 'strain_x': prx = qx_prx; thm = qx_thm; s = 'Horizontal strain (XX)' + ' at ' + loc

                case 'delta_eff_stress_z': prx = sz_prx; thm = sz_thm; s = 'Vertical effective stress change, MPa.' + ' at ' + loc
                case 'delta_eff_stress_y': prx = sy_prx; thm = sy_thm; s = 'Horizontal effective stress change (YY), MPa.' + ' at ' + loc
                case 'delta_eff_stress_x': prx = sx_prx; thm = sx_thm; s = 'Horizontal effective stress change (XX), MPa.' + ' at ' + loc

                case 'delta_total_stress_z': prx = sz_total_prx; thm = sz_total_thm; s = 'Vertical total stress change, MPa.' + ' at ' + loc
                case 'delta_total_stress_y': prx = sy_total_prx; thm = sy_total_thm; s = 'Horizontal total stress change (YY), MPa.' + ' at ' + loc
                case 'delta_total_stress_x': prx = sx_total_prx; thm = sx_total_thm; s = 'Horizontal total stress change (XX), MPa.' + ' at ' + loc

                case 'delta_pressure': prx = None; thm = dp; s = 'Pressure change, MPa.' + ' at ' + loc
                case 'delta_temperature': prx = None; thm = dt; s = 'Temperature change, K.' + ' at ' + loc
                #delta_temperature

            plot_mesh_layers = False
            if plot_mesh_layers:
                for zi in m.reservoir.Zc:
                    if z_range.min() <= zi <= z_range.max():
                        plt.axhline(y=zi, color='gray', linestyle='dotted')

            plt.axhline(y=m.idata.other.rsv_top, color='black', linestyle='dotted')#, xmin=0.95, xmax=1.0)
            plt.axhline(y=m.idata.other.rsv_bottom, color='black', linestyle='dotted', label=r'reservoir top/bottom')#, xmin=0.95, xmax=1.0)
            plt.plot(thm, z_range, label='THM', c='blue')#, marker='.')
            if prx is not None:
                plt.plot(prx, z_range, label='Nucleus of strain', linestyle='--', c='red')#, marker='.')

            # for comparizon with analytical solution laterally infinite rsv
            #if mode == 'delta_total_stress_z' and np.fabs(prx.max()) < 0.05 and np.fabs(thm_sol.max()) < 0.05: # vertical stress is almost zero
            #    plt.xlim(-0.25, 0.25)

            plt.gca().invert_yaxis()
            plt.xlabel(s)
            plt.title(s)
            plt.ylabel('Depth, m.')
            plt.legend()
            #plt.grid()
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth=0.8)
            plt.grid(which='minor', linestyle=':', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, array_name + '_' + loc + '_' + suffix + '.png'))
            plt.close()

            # plot the difference THM solution vs nucleus-of-strain solution
            if prx is not None:
                diff = thm - prx
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_diff = np.where(np.abs(thm) > 0, (diff / np.abs(thm)) * 100.0, np.nan)
                fig, ax1 = plt.subplots()
                ax1.axhline(y=m.idata.other.rsv_top, color='black', linestyle='dotted')
                ax1.axhline(y=m.idata.other.rsv_bottom, color='black', linestyle='dotted', label=r'reservoir top/bottom')
                ax1.plot(diff, z_range, c='green', label='abs. diff')
                ax1.invert_yaxis()
                ax1.set_xlabel('Absolute diff. ' + s)
                ax1.set_title(s)
                ax1.set_ylabel('Depth, m.')
                ax1.minorticks_on()
                ax1.grid(which='major', linestyle='-', linewidth=0.8)
                ax1.grid(which='minor', linestyle=':', linewidth=0.5)
                if True:  # plot only abs. diff
                    ax1.legend()
                else:  # plot also rel.diff
                    ax2 = ax1.twiny()
                    ax2.plot(rel_diff, z_range, c='orange', linestyle='--', label='rel. diff')
                    ax2.set_xlabel('Relative diff., %')
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2)
                fig.tight_layout()
                fig.savefig(os.path.join(output_folder, array_name + '_' + loc + '_' + suffix + '_diff'+ '.png'))
                plt.close(fig)


    def plot_contour(array_dict, points_x, points_y, output_folder, layer=0, slice='XY', vlims=None, n_levels=12, idata=None):
        # plot contours XY plane, 1 layer by z
        # vlims: optional dict {arr_name: (vmin, vmax)} to fix colorbar range
        _default_w, _default_h = plt.rcParams['figure.figsize']
        for arr_name, arr in array_dict.items():
            if slice == 'XZ':
                plt.figure(figsize=(_default_w, _default_h * 2))
            if len(arr.shape) == 3:
                arr_layer = arr[:, :, layer]
            else:
                arr_layer = arr
            if vlims is not None and arr_name in vlims:
                vmin, vmax = vlims[arr_name]
            else:
                vmin, vmax = arr_layer.min(), arr_layer.max()
            if np.isclose(vmin, vmax):
                vmin, vmax = vmin - 0.01, vmax + 0.01
            levels = np.linspace(vmin, vmax, n_levels)
            if vmin < 0 < vmax:
                levels = np.sort(np.unique(np.append(levels, 0.)))
            cs = plt.contourf(points_x, points_y, np.ma.masked_invalid(arr_layer), levels=levels, vmin=vmin, vmax=vmax)
            plt.colorbar(cs, orientation='horizontal', pad=0.12)
            #plt.colorbar(cs, extend='neither').set_ticks([vmin, vmax])
            plt.gca().set_aspect(2) # 2X vertical scale
            #plt.gca().set_aspect('equal')
            plt.minorticks_on()
            plt.xlabel('X, m.')
            if slice == 'XY':
                plt.ylabel('Y')
            elif slice == 'XZ':
                plt.ylabel('Depth, m.')
                plt.gca().invert_yaxis()
                if idata is not None:
                    plt.axhline(y=idata.other.rsv_top,    color='black', linewidth=0.8, linestyle='--')
                    plt.axhline(y=idata.other.rsv_bottom, color='black', linewidth=0.8, linestyle='--', label='reservoir top/bottom')
                    reservoir_sides = [
                        x for x in (idata.other.rsv_x1, idata.other.rsv_x2)
                        if points_x.min() <= x <= points_x.max()
                    ]
                    for side_index, x in enumerate(reservoir_sides):
                        label = 'reservoir sides' if side_index == 0 else None
                        plt.axvline(x=x, color='black', linewidth=0.5,
                                    linestyle=':', label=label)
                    z_well_top = points_y.min()
                    z_well_bot = idata.other.rsv_bottom
                    if wells_type in ('prod', 'doublet'):
                        plt.plot([idata.other.prod_well_coords[0]] * 2, [z_well_top, z_well_bot],
                                 color='red',  linewidth=1.5, label='production well')
                    if wells_type in ('inj', 'doublet'):
                        plt.plot([idata.other.inj_well_coords[0]] * 2,  [z_well_top, z_well_bot],
                                 color='cyan', linewidth=1.5, label='injection well')
                    plt.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 0.15), bbox_transform=plt.gcf().transFigure, ncol=2)
            # Annotation artists must not expand the physical slice domain.
            plt.xlim(points_x.min(), points_x.max())
            parts = arr_name.split(' - ', 1)
            plt.title('\n'.join(parts) if len(parts) > 1 else arr_name)
            fig_fname = contour_file_name(arr_name)
            fig_path = os.path.join(output_folder, fig_fname)
            print(f'Saving {fig_fname}, min={arr_layer.min():.4g}, max={arr_layer.max():.4g}')
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()


    def plot_imshow(array_dict, points_x, points_y, output_folder, layer=0, slice='XY'):
        # same fields as plot_contour, drawn as images (imshow) instead of contours, 1 layer by z
        for arr_name, arr in array_dict.items():
            if len(arr.shape) == 3:
                arr_layer = arr[:, :, layer]
            else:
                arr_layer = arr
            cs = plt.imshow(arr_layer)
            plt.colorbar(cs)
            plt.gca().set_aspect('equal')
            plt.xlabel('I')
            if slice == 'XY':
                plt.ylabel('J')
            elif slice == 'XZ':
                plt.ylabel('K')
                plt.gca().invert_yaxis()
            plt.title(arr_name)
            plt.savefig(os.path.join(output_folder, arr_name + '_imshow.png'))
            plt.close()

    def plot_mesh_skeleton(output_folder):
        # XZ view of the mesh cell boundaries with the reservoir outline and the wells (mesh_skeleton.png)
        rsv_top = m.idata.other.rsv_top
        rsv_bottom = m.idata.other.rsv_bottom
        rsv_xy = m.idata.other.rsv_xy

        fig, ax = plt.subplots(figsize=(8, 6))
        if hasattr(m.idata.other, 'Xc') and hasattr(m.idata.other, 'Zc'):
            Xc = m.idata.other.Xc
            Zc = m.idata.other.Zc
            Xc_plot = Xc[(Xc >= -2100) & (Xc <= 2100)]
            Zc_plot = Zc[(Zc >= rsv_top - 500) & (Zc <= rsv_bottom + 500)]
            for xi in Xc_plot:
                ax.axvline(x=xi, color='steelblue', linewidth=0.7, zorder=1)
            for zi in Zc_plot:
                ax.axhline(y=zi, color='coral', linewidth=0.7, zorder=1)
        else:
            # An arbitrary unstructured mesh has no global grid lines. Plot a
            # bounded projection of its cell centres instead.
            cells_x = thm_sol.centroids[:, 1]
            cells_z = thm_sol.centroids[:, 2]
            plot_mask = (cells_z >= rsv_top - 500) & (cells_z <= rsv_bottom + 500)
            if not np.any(plot_mask):
                plot_mask = np.ones(cells_z.shape, dtype=bool)
            cells_x = cells_x[plot_mask]
            cells_z = cells_z[plot_mask]
            max_plot_points = 50000
            stride = max(1, int(np.ceil(cells_x.size / max_plot_points)))
            ax.scatter(cells_x[::stride], cells_z[::stride], s=0.2,
                       color='steelblue', alpha=0.35, rasterized=True, zorder=1,
                       label='unstructured cell centres')
            Xc_plot = np.array([cells_x.min(), cells_x.max()])
            Zc_plot = np.array([cells_z.min(), cells_z.max()])
        # reservoir boundary rectangle (dashed red, lines clipped to their crossing points)
        from matplotlib.patches import Rectangle
        rect = Rectangle((-rsv_xy, rsv_top), 2 * rsv_xy, rsv_bottom - rsv_top,
                         linewidth=2., edgecolor='red', linestyle='--', facecolor='none', zorder=3,
                         label='reservoir boundary')
        ax.add_patch(rect)
        # wells (vertical lines), same as in plot_contour XZ slice
        z_well_top = Zc_plot.min()
        z_well_bot = rsv_bottom
        if wells_type in ('prod', 'doublet'):
            ax.plot([m.idata.other.prod_well_coords[0]] * 2, [z_well_top, z_well_bot],
                    color='red', linewidth=1.5, label='production well', zorder=4)
        if wells_type in ('inj', 'doublet'):
            ax.plot([m.idata.other.inj_well_coords[0]] * 2, [z_well_top, z_well_bot],
                    color='darkblue', linewidth=1.5, label='injection well', zorder=4)
        # legend (reservoir boundary is always drawn; wells added when present)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(Xc_plot.min(), Xc_plot.max())
        ax.set_ylim(Zc_plot.min(), Zc_plot.max())
        ax.invert_yaxis()
        ax.set_xlabel('X, m.')
        ax.set_ylabel('Depth, m.')
        ax.set_title('Mesh skeleton')
        ax.set_aspect('auto')
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, 'mesh_skeleton.png'))
        plt.close(fig)
        print('Saved mesh_skeleton.png')

    def save_html_prx_vs_thm(base_names, output_folder, filename='proxy_vs_thm_2d.html',
                              locs=None, array_names=None, suffix_1d='all'):
        # HTML report: dp/dt contours, then nucleus-of-strain / THM / difference 2D slices and the 1D profiles
        # dp + dt contours at the top
        contour_cells = []
        for fname, title in [('dp_contour.png', '&#916;P contour (XZ)'), ('dt_contour.png', '&#916;T contour (XZ)'),
                             ('dp_xy_contour.png', '&#916;P contour (XY)'), ('dt_xy_contour.png', '&#916;T contour (XY)')]:
            if os.path.exists(os.path.join(output_folder, fname)):
                contour_cells.append(f'<td style="text-align:center"><b>{title}</b><br>'
                                     f'<img src="{fname}" style="max-width:100%"></td>')
        contour_html = ''
        if contour_cells:
            contour_html = (f'<h2>&#916;P / &#916;T contours (XZ and XY slices)</h2>'
                            f'<table><tr>{"".join(contour_cells)}</tr></table>')

        # 2D nucleus-of-strain solution vs THM table
        html_rows = []
        for b in base_names:
            prx_file  = contour_file_name(f'{b} - Nucleus of strain')
            thm_file  = contour_file_name(f'{b} - THM')
            diff_file = contour_file_name(f'{b} - Difference')
            html_rows.append(f'''
              <tr>
                <td style="text-align:center"><img src="{prx_file}" style="max-width:100%"></td>
                <td style="text-align:center"><img src="{thm_file}" style="max-width:100%"></td>
                <td style="text-align:center"><img src="{diff_file}" style="max-width:100%"></td>
              </tr>''')

        # 1D plots: rows=array_names, columns=locations in fixed order
        col_order = ['(250,250)', 'inj_well', 'prod_well']
        active_locs = [l for l in col_order if locs and l in locs]
        plots_1d_html = ''
        if active_locs and array_names:
            th_locs = ''.join(f'<th>{l}</th>' for l in active_locs)
            rows_1d = []
            for array_name in array_names:
                cells = []
                for loc in active_locs:
                    plot_file = f'{array_name}_{loc}_{suffix_1d}.png'
                    if os.path.exists(os.path.join(output_folder, plot_file)):
                        cells.append(f'<td style="text-align:center">'
                                     f'<img src="{plot_file}" style="max-width:100%"></td>')
                    else:
                        cells.append('<td style="text-align:center;color:gray">N/A</td>')
                rows_1d.append(f'<tr><td style="white-space:nowrap"><b>{array_name}</b></td>{"".join(cells)}</tr>')
            plots_1d_html = f'''
        <h2>1D vertical profiles</h2>
        <table border="1" cellspacing="4" cellpadding="4" style="width:100%">
          <tr><th>Mode</th>{th_locs}</tr>
          {''.join(rows_1d)}
        </table>'''

        html = f'''<!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>Nucleus of strain vs THM Report</title></head>
        <body>
        {contour_html}
        <h2>Nucleus of strain vs THM 2D slices (XZ)</h2>
        <table border="1" cellspacing="4" cellpadding="4">
          <tr><th>Nucleus of strain</th><th>THM</th><th>Difference (THM - Nucleus of strain)</th></tr>
          {''.join(html_rows)}
        </table>
        {plots_1d_html}
        </body>
        </html>'''
        html_path = os.path.join(output_folder, filename)
        with open(html_path, 'w') as f:
            f.write(html)
        print(f'Saved HTML report: {html_path}')

    def save_pdf_report(base_names, output_folder, filename='proxy_vs_thm_report.pdf',
                        locs=None, array_names=None, suffix_1d='all', read_from_cache=False):
        # PDF version of the HTML report (save_html_prx_vs_thm)
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.image as mpimg

        def load_img(img_name):
            if not img_name:
                return None
            img_path = os.path.join(output_folder, img_name)
            return mpimg.imread(img_path) if os.path.exists(img_path) else None

        def add_ncol_page(pdf, names_labels, row_title='', col_fontsize=9, title_fontsize=10):
            # names_labels: list of (img_name, col_title); missing files are skipped
            loaded = [(load_img(n), lbl) for n, lbl in names_labels]
            loaded = [(im, lbl) for im, lbl in loaded if im is not None]
            if not loaded:
                return
            n = len(loaded)
            fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
            if n == 1:
                axes = [axes]
            for ax, (im, lbl) in zip(axes, loaded):
                ax.imshow(im)
                ax.axis('off')
                ax.set_title(lbl, fontsize=col_fontsize)
            if row_title:
                fig.suptitle(row_title, fontsize=title_fontsize, fontweight='bold')
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        col_order = ['(250,250)', 'inj_well', 'prod_well']
        active_locs = [l for l in col_order if locs and l in locs]

        pdf_path = os.path.join(output_folder, filename)
        with PdfPages(pdf_path) as pdf:
            # dp + dt contours
            add_ncol_page(pdf,
                [('dp_contour.png', 'ΔP contour (XZ)'), ('dt_contour.png', 'ΔT contour (XZ)')],
                row_title='ΔP / ΔT contours (XZ slice)', col_fontsize=7, title_fontsize=8)
            add_ncol_page(pdf,
                [('dp_xy_contour.png', 'ΔP contour (XY)'), ('dt_xy_contour.png', 'ΔT contour (XY)')],
                row_title='ΔP / ΔT contours (XY slice at the reservoir mid-depth)', col_fontsize=7, title_fontsize=8)
            # 2D nucleus-of-strain solution vs THM: 3 columns per page
            for b in base_names:
                add_ncol_page(pdf,
                    [(contour_file_name(f'{b} - Nucleus of strain'), 'Nucleus of strain'),
                     (contour_file_name(f'{b} - THM'), 'THM'),
                     (contour_file_name(f'{b} - Difference'), 'Difference (THM - Nucleus of strain)')],
                    row_title=b, col_fontsize=7, title_fontsize=8)
            # 1D profiles: one page per array_name, columns = locations
            if active_locs and array_names:
                for array_name in array_names:
                    add_ncol_page(pdf,
                        [(f'{array_name}_{loc}_{suffix_1d}.png', loc) for loc in active_locs],
                        row_title=array_name)
        print(f'Saved PDF report: {pdf_path}')


    # mesh skeleton with wells (independent of modes / the nucleus-of-strain computation)
    plot_mesh_skeleton(output_folder)

    # reference points for the 1D vertical profiles; taken from idata.other.points_xy
    # (set in cases/base.py) as a list of [x, y, label]
    points_xy = dict()
    for x_pt, y_pt, label in m.idata.other.points_xy:  # the order is actually Y,X
        points_xy[label] = [y_pt, x_pt]

    if True: # evaluate along the wells; the well coords are [X, Y, Z1, Z2], points_xy holds [Y, X]
        if wells_type in ['prod', 'doublet']:
            points_xy['prod_well'] = m.idata.other.prod_well_coords[1::-1]
        if wells_type in ['inj', 'doublet']:
            points_xy['inj_well'] = m.idata.other.inj_well_coords[1::-1]

    base_names = [] # for html/pdf reports

    # compute with the nucleus-of-strain solution for 2D slice
    if 'plot_2d_slices' in modes:
        if use_adaptive_proxy:
            plot_nx = getattr(m.idata.other, 'proxy_plot_nx', 41)
            plot_nz = getattr(m.idata.other, 'proxy_plot_nz', 31)
            use_mesh_plot_bounds = getattr(
                m.idata.other, 'proxy_plot_use_mesh_bounds', False
            )
            if use_mesh_plot_bounds:
                x_min, x_max = thm_sol.mesh_bounds[1]
                y_min, y_max = thm_sol.mesh_bounds[0]
                z_min, z_max = thm_sol.mesh_bounds[2]
            else:
                # Historical plotting domain for existing nucleus-of-strain cases.
                x_min, x_max = (
                    thm_sol.centroids[:, 1].min(),
                    thm_sol.centroids[:, 1].max(),
                )
                y_min, y_max = (
                    thm_sol.centroids[:, 0].min(),
                    thm_sol.centroids[:, 0].max(),
                )
                z_min = max(
                    thm_sol.centroids[:, 2].min(),
                    m.idata.other.rsv_top - 500.,
                )
                z_max = min(
                    thm_sol.centroids[:, 2].max(),
                    m.idata.other.rsv_bottom + 500.,
                )
            points_x = np.linspace(x_min, x_max, plot_nx)
            points_y = np.array([0.5 * (y_min + y_max)])
            points_z = np.linspace(z_min, z_max, plot_nz)
        else:
            points_x = np.unique(g.centroids[:, 1])
            points_y = np.array([0.])
            points_z = np.unique(g.centroids[:, 2])

        if '2d_slices_41_71' in modes: # while reading a coarser grid as sources, evaluate at finer grid centers
            # replace points_x by ones from the finer grid
            # Xc from nx == 71 case in model.py: # rsv corners and near-well (middle) are refined
            Xc_left = np.array([-8000,-6000,-5000,-4000,-3000,-2500,-2000,-1600,-1400,-1200] +
                          [-1100, -1050, -1030, -1010, -1000,  -990,  -980, -950, -900] +
                          np.arange(-800, -100, 100).tolist() +
                          np.arange(-100, 0, 10).tolist())
            Xc = np.hstack([Xc_left, -Xc_left[::-1]]) # add the right part symmetrically
            points_x = (Xc[1:] + Xc[:-1]) * 0.5 # centers

        if not use_adaptive_proxy:
            # cut points far from the reservoir for better zoom in plots
            points_x = points_x[
                reduce(np.logical_and, [points_x > -5000., points_x < 5000.])
            ]
            points_z = points_z[
                reduce(np.logical_and, [points_z > 1400., points_z < 3400.])
            ]

        points_x_3d, points_y_3d, points_z_3d = np.meshgrid(
            points_x, points_y, points_z, indexing='ij'
        )

        print('plotting 2D slices, n_eval_points =', points_x.size * points_y.size * points_z.size, 'n_src_points = ', g.centroids[:, 0].size)
        p_nx, p_ny, p_nz = points_x.size, points_y.size, points_z.size
        points = np.zeros((3, points_x_3d.size))
        # Nucleus of strain coordinates use Y, X, Z ordering.
        points[0, :] = points_y_3d.flatten()
        points[1, :] = points_x_3d.flatten()
        points[2, :] = points_z_3d.flatten()

        dp = gd((g.centroids[:, 1], g.centroids[:, 0], g.centroids[:, 2]), \
            thm_sol.delta_pressure, (points[1, :], points[0, :], points[2, :]), method='nearest', fill_value=0.).reshape((p_nx, p_ny, p_nz))
        array_dict = {'Pressure change, MPa':dp[:,0,:].transpose()}
        if thermal:
            dt = gd((g.centroids[:, 1], g.centroids[:, 0], g.centroids[:, 2]), \
                thm_sol.delta_temperature, (points[1, :], points[0, :], points[2, :]), method='nearest', fill_value=0.).reshape((p_nx, p_ny, p_nz))
            array_dict.update({'Temperature change, K':dt[:,0,:].transpose()})
        plot_contour(array_dict, points_x, points_z, output_folder=output_folder, slice='XZ', idata=m.idata)
        plot_plan_view(thm_sol, m.idata, wells_type, thermal, output_folder)
        #plot_imshow(array_dict, points_x, points_z, output_folder=output_folder, slice = 'XZ')

        # if caching is requested but the pkl is missing, fall back to recomputing the nucleus-of-strain solution
        pkl_path = os.path.join(output_folder, "displs_stresses_prx.pkl")
        cached_data = None
        if read_from_cache and not os.path.exists(pkl_path):
            print(f'read_from_cache=True but "{pkl_path}" not found; '
                  f'forcing read_from_cache=False (recomputing the nucleus-of-strain solution)')
            read_from_cache = False
        elif read_from_cache:
            import pickle
            with open(pkl_path, "rb") as f:
                cached_data = pickle.load(f)
            expected_method = (
                'adaptive_tetra_v3_mesh_bounds'
                if use_adaptive_proxy and use_mesh_plot_bounds
                else 'adaptive_tetra_v1'
                if use_adaptive_proxy
                else 'legacy_prism'
            )
            expected_shape = (p_nx, p_ny, p_nz)
            if (cached_data.get('proxy_method') != expected_method
                    or cached_data.get('grid_shape') != expected_shape):
                print('Cached nucleus-of-strain method/grid does not match; recomputing')
                read_from_cache = False

        if not read_from_cache: # run the nucleus-of-strain solution and save PKLs
            print('computing the nucleus-of-strain solution...')
            ux_prx, uy_prx, uz_prx = get_proxy_displs(points)
            ux_prx = ux_prx.reshape((p_nx, p_ny, p_nz))
            uy_prx = uy_prx.reshape((p_nx, p_ny, p_nz))
            uz_prx = uz_prx.reshape((p_nx, p_ny, p_nz))

            _, _, _, sxx_prx, syy_prx, szz_prx, sxx_total_prx, syy_total_prx, szz_total_prx = get_proxy_strain_stress(points)  # need to X<->Y if non-symmetric
            sxx_prx = sxx_prx.reshape((p_nx, p_ny, p_nz))
            syy_prx = syy_prx.reshape((p_nx, p_ny, p_nz))
            szz_prx = szz_prx.reshape((p_nx, p_ny, p_nz))
            sxx_total_prx = sxx_total_prx.reshape((p_nx, p_ny, p_nz))
            syy_total_prx = syy_total_prx.reshape((p_nx, p_ny, p_nz))
            szz_total_prx = szz_total_prx.reshape((p_nx, p_ny, p_nz))

            # save to pkl
            import pickle
            data = {'ux_prx': ux_prx, 'uy_prx': uy_prx, 'uz_prx': uz_prx,
                    'sxx_prx': sxx_prx, 'syy_prx': syy_prx, 'szz_prx': szz_prx,
                    'sxx_total_prx': sxx_total_prx,
                    'syy_total_prx': syy_total_prx,
                    'szz_total_prx': szz_total_prx,
                    'proxy_method': (
                        'adaptive_tetra_v3_mesh_bounds'
                        if use_adaptive_proxy and use_mesh_plot_bounds
                        else 'adaptive_tetra_v1'
                        if use_adaptive_proxy
                        else 'legacy_prism'
                    ),
                    'grid_shape': (p_nx, p_ny, p_nz)}
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)
        else: # do not rerun the nucleus-of-strain solution, read from PKl files (if only plotting is changed)
            data = cached_data
            ux_prx = data['ux_prx']
            uy_prx = data['uy_prx']
            uz_prx = data['uz_prx']
            sxx_prx = data['sxx_prx']
            syy_prx = data['syy_prx']
            szz_prx = data['szz_prx']
            sxx_total_prx = data['sxx_total_prx']
            syy_total_prx = data['syy_total_prx']
            szz_total_prx = data['szz_total_prx']

        array_dict = {'Horizontal displacements (X), mm. - Nucleus of strain' : ux_prx[:,0,:].transpose() * m2mm,
                      'Vertical displacements, mm. - Nucleus of strain' : uz_prx[:,0,:].transpose() * m2mm,
                      'Horizontal effective stress change (XX), MPa - Nucleus of strain' : sxx_prx[:,0,:].transpose(),
                      'Vertical effective stress change, MPa - Nucleus of strain' : szz_prx[:,0,:].transpose(),
                      'Horizontal total stress change (XX), MPa - Nucleus of strain' : sxx_total_prx[:,0,:].transpose(),
                      'Vertical total stress change, MPa - Nucleus of strain' : szz_total_prx[:,0,:].transpose()
                      }
        shared_cbar = True  # use same colorbar min/max for prx and thm plots of the same variable

        # compare the nucleus-of-strain solution vs thm on finer mesh although coarser mesh data used as the input for the nucleus-of-strain solution
        if '2d_slices_41_71' in modes:
            folder_71 = folder.replace('41x41x66', '71x71x66')
            thm_sol = read_thm_solution_from_vtk(m, folder=folder_71, timestep=timestep)

        # THM 2D plots on the same grid
        thm_raw = {'Horizontal displacements (X), mm. - THM': thm_sol.ux_last * m2mm,
                   'Vertical displacements, mm. - THM': thm_sol.uz_last * m2mm,
                   'Horizontal effective stress change (XX), MPa - THM': thm_sol.delta_Sxx_last,
                   'Vertical effective stress change, MPa - THM': thm_sol.delta_Szz_last,
                   'Horizontal total stress change (XX), MPa - THM': thm_sol.delta_total_Sxx_last,
                   'Vertical total stress change, MPa - THM': thm_sol.delta_total_Szz_last}
        thm_interp_method = 'nearest' # nearest is better here as eval points are centroids
        #thm_interp_method = 'linear' #
        thm_interp = get_thm_by_interp(thm_raw, points[1, :], points[0, :], points[2, :], method=thm_interp_method)
        array_dict_thm = {k: v.reshape((p_nx, p_ny, p_nz))[:, 0, :].transpose()
                          for k, v in thm_interp.items()}

        base_names = ['Horizontal displacements (X), mm.']
        base_names += ['Vertical displacements, mm.']
        base_names += ['Horizontal effective stress change (XX), MPa']
        base_names += ['Vertical effective stress change, MPa']
        base_names += ['Horizontal total stress change (XX), MPa']
        base_names += ['Vertical total stress change, MPa']
        if shared_cbar:
            vlims_prx = {f'{b} - Nucleus of strain': (min(array_dict[f'{b} - Nucleus of strain'].min(),  array_dict_thm[f'{b} - THM'].min()),
                                      max(array_dict[f'{b} - Nucleus of strain'].max(), array_dict_thm[f'{b} - THM'].max()))
                         for b in base_names}
            vlims_thm = {f'{b} - THM': vlims_prx[f'{b} - Nucleus of strain'] for b in base_names}
        else:
            vlims_prx = vlims_thm = None

        plot_contour(array_dict, points_x, points_z, output_folder=output_folder, slice='XZ', vlims=vlims_prx, idata=m.idata)
        #plot_imshow(array_dict, points_x, points_z, output_folder=output_folder, slice = 'XZ')

        plot_contour(array_dict_thm, points_x, points_z, output_folder=output_folder, slice='XZ', vlims=vlims_thm, idata=m.idata)

        print('Array ranges:')
        for b in base_names:
            prx = array_dict[f'{b} - Nucleus of strain']
            thm = array_dict_thm[f'{b} - THM']
            print(f'  {b}  Nucleus of strain [{prx.min():.4g}, {prx.max():.4g}]  THM [{thm.min():.4g}, {thm.max():.4g}]')

        # THM - Nucleus of strain difference 2D plots
        diff_clip = None
        #diff_clip = 0.1  # nullify stress differences larger than this value as they affect the axis range but located in very small vicinity
        array_dict_diff = {}
        for b in base_names:
            d = array_dict_thm[f'{b} - THM'] - array_dict[f'{b} - Nucleus of strain']
            if diff_clip is not None and 'stress' in b:
                d = np.where(np.abs(d) <= diff_clip, d, 0.)
            #rd = np.where(np.abs(array_dict_thm[f'{b}_thm']) > 0, (d / np.abs(array_dict_thm[f'{b}_thm'])) * 100., 0.)
            array_dict_diff[f'{b} - Difference'] = d
            #array_dict_diff[f'{b} - Relative Difference'] = rd
        plot_contour(array_dict_diff, points_x, points_z, output_folder=output_folder, slice='XZ', idata=m.idata)

        if False:
            print('Relative difference THM vs Nucleus of strain (% of |THM|):')
            for b in base_names:
                thm_arr = array_dict_thm[f'{b} - THM'].ravel()
                diff_arr = array_dict_diff[f'{b} - Difference'].ravel()
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = np.where(np.abs(thm_arr) > 0, np.abs(diff_arr / thm_arr) * 100., np.nan)
                n_total = np.sum(~np.isnan(rel))
                for thr in [1., 5., 10.]:
                    n_over = np.sum(rel > thr)
                    print(f'  {b}: >{thr:.0f}%: {n_over}/{n_total} ({100.*n_over/n_total if n_total>0 else 0:.1f}%)')

        pass  # HTML/PDF reports are saved after 1D plots are generated (see below)

    if 'plot_2d_thm_41_vs_71' in modes:
        from model import Model
        prefix = 'thm_' if physics_type == 'single_phase_thermal' else ''  # same physics as `case`
        m_41 = Model(model_folder=prefix + 'rect_41x41x66', decouple_geomech=True, dummy='yes')
        m_41.set_input_data()

        folder_41 = get_output_directory('rect_41x41x66', physics_type, wells_type)
        folder_71 = get_output_directory('rect_71x71x66', physics_type, wells_type)

        thm_sol_41 = read_thm_solution_from_vtk(m_41, folder=folder_41, timestep=timestep)
        thm_sol_71 = read_thm_solution_from_vtk(m, folder=folder_71, timestep=timestep)

        # eval grid from 71 centroids (finer), same x/z filter as plot_2d_slices
        pts_x = np.unique(thm_sol_71.centroids[:, 0])  # follows existing naming convention
        pts_y = np.array([0.])
        pts_z = np.unique(thm_sol_71.centroids[:, 2])
        pts_x = pts_x[reduce(np.logical_and, [pts_x > -5000., pts_x < 5000.])]
        pts_z = pts_z[reduce(np.logical_and, [pts_z > 1400., pts_z < 3400.])]

        p_nx, p_ny, p_nz = pts_x.size, pts_y.size, pts_z.size
        px3, py3, pz3 = np.meshgrid(pts_x, pts_y, pts_z)
        pts = np.zeros((3, px3.size))
        pts[0, :], pts[1, :], pts[2, :] = px3.flatten(), py3.flatten(), pz3.flatten()

        def interp_thm_dp(thm_s):
            return gd(
                (thm_s.centroids[:, 1], thm_s.centroids[:, 0], thm_s.centroids[:, 2]),
                thm_s.delta_pressure,
                (pts[1, :], pts[0, :], pts[2, :]),
                method='linear', fill_value=0.
            ).reshape((p_nx, p_ny, p_nz))

        dp_41 = interp_thm_dp(thm_sol_41)
        dp_71 = interp_thm_dp(thm_sol_71)

        vmin = min(dp_41.min(), dp_71.min())
        vmax = max(dp_41.max(), dp_71.max())
        vlims_cmp = {'delta_pressure, MPa - THM 41': (vmin, vmax),
                     'delta_pressure, MPa - THM 71': (vmin, vmax)}
        plot_contour({'delta_pressure, MPa - THM 41': dp_41[:, 0, :].transpose()},
                     pts_x, pts_z, output_folder=output_folder, slice='XZ', vlims=vlims_cmp, idata=m.idata)
        plot_contour({'delta_pressure, MPa - THM 71': dp_71[:, 0, :].transpose()},
                     pts_x, pts_z, output_folder=output_folder, slice='XZ', vlims=vlims_cmp, idata=m.idata)
        plot_contour({'delta_pressure, MPa - Difference (71-41)': (dp_71 - dp_41)[:, 0, :].transpose()},
                     pts_x, pts_z, output_folder=output_folder, slice='XZ', idata=m.idata)

    array_names = []
    if 'plot_vertic_line' in modes: # 1D plots (along vertical lines at points_xy)
        for k in points_xy.keys():
            point_xy = points_xy[k]
            print('1D plots for point', k, 'YX=', point_xy)
            array_names += ['delta_pressure']
            array_names += ['delta_temperature']
            array_names += ['displ_z', 'displ_y', 'displ_x']
            array_names += ['delta_eff_stress_z', 'delta_eff_stress_y', 'delta_eff_stress_x']
            array_names += ['delta_total_stress_z', 'delta_total_stress_y', 'delta_total_stress_x']
            #array_names += ['strain_z', 'strain_y', 'strain_x']
            #array_names = ['strain_x']  # debug
            #z_min = 0.
            #z_max = thm_sol.centroids[:, 2].max() #+ 1000.
            z_step = 5.  # m.
            #z_range_all = np.arange(z_min, z_max+1., z_step)# more points
            if hasattr(m.idata.other, 'Zc'):
                z_range_all = m.idata.other.Zc
            else:
                z_min, z_max = thm_sol.bounds[2]
                z_range_all = np.arange(z_min, z_max + 0.5 * z_step, z_step)

            z_interp_eps = 5. # m # remove points close to rsv boundary, as they create descrepancies even with 'nearest' interpolation
            z_range_all_filter = reduce(np.logical_and, [np.fabs(z_range_all - m.idata.other.rsv_top) > z_interp_eps, np.fabs(z_range_all - m.idata.other.rsv_bottom) > z_interp_eps])
            z_range_all = z_range_all[z_range_all_filter]
            n_points = z_range_all.size
            points_all = np.zeros((3, n_points))
            points_all[0, :] = point_xy[0]  # X<->Y
            points_all[1, :] = point_xy[1]
            points_all[2, :] = z_range_all

            compare_vert_line(points_all, suffix='all', loc=k, output_folder=output_folder, array_names=array_names)

            if False: # zoomed in plot for the rsv part, if its thickness is quite small
                z_range_rsv = np.arange(m.idata.other.rsv_top-100., m.idata.other.rsv_bottom+100., z_step)
                # remove points close to rsv boundary, as they create descrepancies even with 'nearest' interpolation
                z_range_rsv_filter = reduce(np.logical_and, [np.fabs(z_range_rsv - m.idata.other.rsv_top) > z_interp_eps, np.fabs(z_range_rsv - m.idata.other.rsv_bottom) > z_interp_eps])
                z_range_rsv = z_range_rsv[z_range_rsv_filter]
                n_points = z_range_rsv.size
                points_rsv = np.zeros((3, n_points))
                points_rsv[0, :] = point_xy[1] # X<->Y
                points_rsv[1, :] = point_xy[0]
                points_rsv[2, :] = z_range_rsv
                compare_vert_line(points_rsv, suffix='rsv', loc=k, output_folder=output_folder, array_names=array_names)

    save_html_prx_vs_thm(base_names, output_folder=output_folder,
                         locs=list(points_xy.keys()), array_names=array_names, suffix_1d='all')
    save_pdf_report(base_names, output_folder=output_folder,
                    locs=list(points_xy.keys()), array_names=array_names, suffix_1d='all', read_from_cache=read_from_cache)

    # print vert displs and stresses change at a point
    if 'print_at_point' in modes:
        point = np.array([0., 0., 0.5*(m.idata.other.rsv_bottom + m.idata.other.rsv_top)]) # middle rsv depth

        ux_thm, uy_thm, uz_thm = get_thm_displs(point)
        ux_prx, uy_prx, uz_prx = get_proxy_displs(point)# need to X<->Y
        print('Compare at the point=', point)
        print('\tTHM   ', 'ux=', fmt(ux_thm*m2mm), 'uy=', fmt(uy_thm*m2mm), 'uz=', fmt(uz_thm*m2mm), 'mm.')
        print('\tNucleus of strain ', 'ux=', fmt(ux_prx[0]*m2mm), 'uy=', fmt(uy_prx[0]*m2mm), 'uz=', fmt(uz_prx[0]*m2mm), 'mm.')

        dp = get_thm_dp_dt(point)[0]
        #dsxx_thm = get_thm_stress(point)[0]
        dsxx_total_thm = get_thm_total_stress(point)[0]
        dsxx_total_prx = get_proxy_strain_stress(point)[3][0]  # need to X<->Y if non-symmetric
        dsxx_total_prx += g.biot * dp # get total from effective stress

        print('Compare at the point=', point)
        print('\tTHM   ', 'delta_P=', fmt(dp), 'MPa')
        print('\tTHM   ', 'delta_total_Sxx=', fmt(dsxx_total_thm), 'MPa')
        print('\tNucleus of strain ', 'delta_total_Sxx=', fmt(dsxx_total_prx), 'MPa')

        # for uniform pressure change in the reservoir with Biot=1 and poisson ratio=0.25 should be 2/3
        print('THM delta_total_Sxx_thm_max =', fmt(np.fabs(thm_sol.delta_total_Sxx_last).max()))
        print('THM delta_total_Szz_thm_max =', fmt(np.fabs(thm_sol.delta_total_Szz_last).max()))
        print('THM delta_pressure_max=', np.fabs(thm_sol.delta_pressure).max())
        print('THM delta_total_Sxx_thm_max / delta_pressure_max=', fmt(np.fabs(thm_sol.delta_total_Sxx_last).max() / np.fabs(thm_sol.delta_pressure).max()))  # MAX
        print('THM delta_total_Sxx_thm_point / delta_pressure_point =', fmt(dsxx_total_thm / dp)) # at point
        print('Analytical delta_total_Sxx/dp =', g.biot * (1 - 2 * g.poisson)/(1 - g.poisson))

    if 'check_initial' in modes: # check initial pressure and stress for THM
        max_depth = thm_sol.bounds[2][1]  # max z m
        rsv_thickness = m.idata.other.rsv_bottom - m.idata.other.rsv_top
        poro_rsv = m.idata.rock.porosity
        poro_non_rsv = m.idata.rock.poro_non_rsv
        rock_dens = m.idata.rock.density
        fluid_dens = m.idata.fluid.density

        p_init_by_density = g_grav * fluid_dens * max_depth * Pa2bars
        szz_init_by_density = g_grav * (rock_dens*(1-poro_non_rsv) + fluid_dens*poro_non_rsv) * (max_depth - rsv_thickness) * Pa2bars
        szz_init_by_density += g_grav * (rock_dens*(1-poro_rsv) + fluid_dens*poro_rsv) * rsv_thickness * Pa2bars
        # Note, that THM depth is at cell center
        print('Pressure at depth ', max_depth, 'by gradient=', fmt(p_init_by_density),
              'THM=', fmt(thm_sol.p_init.max()), 'bars')
        print('StressZZ at depth ', max_depth, 'by gradient=', fmt(szz_init_by_density),
              'THM=', fmt(thm_sol.Szz_init.max()), 'bars')

if __name__ == '__main__':
    # A case fixes its physics, wells, duration (runtime_years) and report step (cases/base.py: base_HM,
    # base_THM); rect_NXxNYxNZ is HM, thm_rect_NXxNYxNZ THM.
    # nx ny nz
    cases = []
    modes = [] # where to evaluate geoemchanical solution using the nucleus-of-strain model

    # a coarse mesh for quick testing
    #cases += ['rect_17x17x15']
    #cases += ['thm_rect_17x17x15']

    # compare against the analytic formulas for stress changes
    # HM: one injector in the reservoir centre, 10 years, for the poroelastic formula
    #cases += ['rect_17x17x15_check_analytical']
    #cases += ['rect_41x41x66_check_analytical']
    #cases += ['rect_71x71x66_check_analytical']
    # THM: doublet at (0, +-2 km), 100 years of injection then 5 years shut in (cases/check_analytical.py)
    #cases += ['thm_rect_17x17x15_check_analytical']
    cases += ['thm_rect_41x41x66_check_analytical']
    modes += ['print_at_point'] # compare both THM and nucleus-of-strain solution versus analytical solution

    # isothermal
    #cases += ['rect_41x41x66'] # without mesh refinement
    #cases += ['rect_71x71x66'] # refined middle and tips
    #cases += ['rect_83x83x90'] # the mesh is also horizontally refined at the inj well location

    # thermal
    #cases += ['thm_rect_41x41x66'] # without mesh refinement
    #cases += ['thm_rect_71x71x66'] # refined middle and tips
    #cases += ['thm_rect_97x97x90'] # the mesh is also horizontally refined at the doublet locations

    run_thm = True  # runs THM simulation first, then the nucleus-of-strain solution
    #run_thm = False # don't recompute THM (use vtk files from its previous run)

    #modes += ['check_initial'] # check initial pressure and stress for THM
    modes += ['plot_horiz_line']
    modes += ['plot_vertic_line']
    modes += ['plot_2d_slices']
    #modes += ['2d_slices_41_71'] # coarse mesh THM (nx=41) => finer eval points in the nucleus-of-strain solution (nx=71) and compare it against finer THM (nx=71); only if case ='rect_41x41x66'
    #modes += ['plot_2d_thm_41_vs_71']  # compare delta_pressure: THM rect_41x41x66 vs THM rect_71x71x66. This doesn't run the nucleus-of-strain solution

    # for the nucleus-of-strain solution:
    n_threads = 24  # CPU cores
    use_gpu = False  # CUDA
    #read_from_cache = True # read stresses computed from the previous nucleus-of-strain run from .pkl file (useful when only ploting was changed)
    read_from_cache = False

    for case in cases:
        idata = set_input_data(case)
        runtime_years, report_step = idata.other.runtime_years, idata.other.report_step
        thermal = idata.other.physics_type == 'single_phase_thermal'

        def get_timestep_index(t_years):
            # vtk timestep index of the report step at t_years
            return int((t_years * 365.25) / report_step)

        # which timestep to read from vtk (delta p,T for the nucleus-of-strain solution and u,stress for comparison):
        # the last one, and for THM also intermediate 1 and 10 years
        last = get_timestep_index(runtime_years)
        timestep_list = [last]
        if thermal:  # only the intermediate times the run reaches, each once
            timestep_list += [t for t in (get_timestep_index(y) for y in [1, 10]) if t < last]
        print('timestep_list for the nucleus-of-strain solution :', timestep_list)

        print('\n\n' + '='*30)
        print(case, idata.other.physics_type, idata.other.wells_type)

        # run THM with no mechanics->flow impact
        t1 = datetime.now()
        if run_thm:
            run(model_folder=case, decouple_geomech=True, plot_vtk_timesteps=[0, -1])  # THM plots of the initial and last timesteps, as main.py
        t2 = datetime.now()
        thm_time = t2 - t1

        # run the nucleus-of-strain solution
        if 'plot_horiz_line' in modes:
            # combined pressure change along X (Y=0, Z=2200) for up to 4 report steps between the
            # first and the last one the case writes; for THM also the temperature change along X
            # through the injector at the reservoir mid-depth, where the cooled zone is
            last = get_timestep_index(runtime_years)
            steps = sorted(set(np.linspace(1, last, min(4, last)).astype(int).tolist()))
            plot_delta_along_x(case=case, timesteps=steps, field='pressure', y=0.0, z=2200.0)
            if thermal:
                plot_delta_along_x(case=case, timesteps=steps, field='temperature',
                                   y=idata.other.inj_well_coords[1],
                                   z=0.5 * (idata.other.rsv_top + idata.other.rsv_bottom))
        for timestep in timestep_list:
            print('The timestep for plots and the nucleus-of-strain solution:', timestep)
            t1 = datetime.now()
            run_geomech_nucleus_of_strain(case=case, modes=modes,
                              timestep=timestep, n_threads=n_threads, use_gpu=use_gpu,
                              read_from_cache=read_from_cache)
            t2 = datetime.now()
            proxy_time = t2 - t1

        print('case', case, 'done')
        print('THM   time', thm_time)
        print('nucleus-of-strain time', proxy_time) # counts only the last timestep
