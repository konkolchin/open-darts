"""
Plotting/analysis for the geomechanical state right after the equilibration step
(before any injection). Library only - no CLI, no self-run: main.run() imports and
calls these when idata.other.check_equil is True.

This module covers the stress profile at one column against the uniaxial-strain
K0 = nu/(1-nu) analytic solution: load_column()/analytic()/plot_vs_analytic()/report().
"""

import os

import numpy as np


def _build_interpolator(vtu_filename):
    """
    Read a solution vtu and return (centres, interp, has): interp(name, query)
    linearly interpolates cell-centred field `name` (from cell centroids) onto
    arbitrary (n, 3) query points, with a nearest-neighbour fallback outside the
    convex hull; has(name) reports whether the field is present.

    Same approach as main_nucleus_of_strain.get_thm_by_interp (there via scipy.griddata, called
    once per field), but the Delaunay triangulation of the ~1e5 centroids is built
    once here and reused for every field and every query point (load_column calls
    this for its line), since griddata would otherwise re-triangulate the same
    points on every call.
    """
    import meshio
    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    mesh = meshio.read(vtu_filename)
    cells = np.concatenate([cb.data for cb in mesh.cells])
    centres = mesh.points[cells].mean(axis=1)

    # A mesh one cell thick (domain_h == layer_dy) puts every centroid on one plane, and a
    # 3D Delaunay of coplanar points has no volume - qhull rejects it as a flat simplex.
    # Triangulate in the two dimensions that do vary and ignore the collapsed one; queries
    # are projected the same way.
    spans = centres.max(axis=0) - centres.min(axis=0)
    flat = spans <= 1e-9 * max(spans.max(), 1.0)
    keep = ~flat
    if flat.any():
        print('interpolating in %s only: %s collapsed (mesh is one cell thick there)'
              % (''.join('xyz'[i] for i in np.flatnonzero(keep)),
                 ''.join('xyz'[i] for i in np.flatnonzero(flat))))

    tri = Delaunay(centres[:, keep])
    nearest = NearestNDInterpolator(centres[:, keep], np.arange(len(centres)))

    def cell_array(name):
        return np.concatenate(
            [v.reshape(len(v), -1) for v in mesh.cell_data_dict[name].values()]
        )

    def interp(name, query):
        values = cell_array(name)
        if values.shape[1] == 1:
            values = values[:, 0]
        q = np.asarray(query)[:, keep]  # same projection the triangulation used
        out = LinearNDInterpolator(tri, values)(q)
        missing = np.isnan(out) if values.ndim == 1 else np.isnan(out).any(axis=-1)
        if missing.any():
            out[missing] = values[nearest(q[missing]).astype(int)]
        return out

    return centres, interp, (lambda name: name in mesh.cell_data_dict)


# ---------------------------------------------------------------------------
# Computed-vs-analytic (uniaxial-strain K0) stress profile at one column
# ---------------------------------------------------------------------------
# component order of tot_stress / eff_stress as written by write_to_vtk
STRESS_LABELS = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
SIGMA_EFF = 'σ′'  # sigma + PRIME, for plot labels; see plot_analytic_fsp()
FSP_LABEL = r'FSP = $\tau$ / ' + SIGMA_EFF + r'$_{n}$ [-]'
G_BAR = 9.80665e-5  # [bar/(kg/m3)/m]: rho[kg/m3] * G_BAR * z[m] -> bar


def load_column(vtu_filename, x, y, n_points=200, z_range=None):
    """
    Interpolate cell-centred fields onto a vertical line of points at (x, y),
    spanning z_range (the mesh's full depth extent when None). Same interpolation
    (via _build_interpolator) as load_slice_grid's plane, generalized here to a
    line - both replace picking the nearest cells: on an unstructured/tetrahedral
    mesh several cells can share a lateral position at nearly the same depth, and
    stacking them by z (as a naive nearest-cell selection would) double-counts
    layers when integrating the overburden weight.

    :return: dict with 'z' (n_points,), 'dz' (uniform spacing), and the
        interpolated fields ('tot_stress', 'eff_stress', 'pressure', 'poro', 'E',
        'poisson'; the last three are None if absent from the vtu).
    """
    centres, interp, has = _build_interpolator(vtu_filename)

    if z_range is None:
        z_range = (centres[:, 2].min(), centres[:, 2].max())
    z = np.linspace(z_range[0], z_range[1], n_points)
    dz = np.full_like(z, (z_range[1] - z_range[0]) / (n_points - 1))
    query = np.stack([np.full_like(z, x), np.full_like(z, y), z], axis=1)

    return {
        'centres': query,
        'z': z,
        'dz': dz,
        'tot_stress': interp('tot_stress', query),
        'eff_stress': interp('eff_stress', query),
        'pressure': interp('pressure', query),
        'poro': interp('poro', query) if has('poro') else None,
        'E': interp('E', query) if has('E') else None,
        'poisson': interp('poisson', query) if has('poisson') else None,
    }


def analytic(col, rho_rock, rho_fluid, biot):
    """
    Sv, sigma'_v, sigma'_h, Sh at the column points of load_column(), from the
    uniaxial-strain (roller sides, free top) solution:

        Sv(z)      = integral of rho_bulk * g dz          (total vertical stress)
        sigma'_v   = Sv - biot * p                        (Terzaghi)
        sigma'_h   = K0 * sigma'_v,  K0 = nu / (1 - nu)   (no lateral movement)
        Sh         = sigma'_h + biot * p

    rho_bulk (not rho_rock) is used because the engine's momentum balance weighs
    both the rock and the pore fluid; K0 applies to EFFECTIVE stress (comparing
    sigma_xx to K0 * Sv directly is a common slip).
    """
    poro = col['poro'] if col['poro'] is not None else np.zeros_like(col['z'])
    rho_bulk = (1.0 - poro) * rho_rock + poro * rho_fluid

    # integrate the bulk weight down the column: full cells above, half of this one
    weight = rho_bulk * col['dz'] * G_BAR
    sv = np.cumsum(weight) - 0.5 * weight

    p = col['pressure']
    sv_eff = sv - biot * p
    nu = col['poisson']
    k0 = nu / (1.0 - nu)
    sh_eff = k0 * sv_eff
    return {'Sv': sv, 'sv_eff': sv_eff, 'sh_eff': sh_eff, 'Sh': sh_eff + biot * p,
            'K0': k0, 'rho_bulk': rho_bulk}


def plot_vs_analytic(col, ana, png_filename, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z = col['z']
    fig, axes = plt.subplots(1, 4, figsize=(19, 7), sharey=True)

    ax = axes[0]
    for j, lab in enumerate(STRESS_LABELS):
        ax.plot(col['tot_stress'][:, j], z, lw=1.2, label=f'sigma_{lab}')
    ax.plot(ana['Sv'], z, 'k--', lw=1.6, label='Sv = int rho_bulk g dz')
    ax.plot(ana['Sh'], z, '--', color='0.45', lw=1.6, label="Sh = K0 sigma'_v + biot p")
    ax.set_title('total stress', fontsize=10)
    ax.set_xlabel('stress [bar]')
    ax.set_ylabel('depth [m]')

    ax = axes[1]
    for j, lab in enumerate(STRESS_LABELS):
        ax.plot(col['eff_stress'][:, j], z, lw=1.2, label=f"sigma'_{lab}")
    ax.plot(ana['sv_eff'], z, 'k--', lw=1.6, label="sigma'_v = Sv - biot p")
    ax.plot(ana['sh_eff'], z, '--', color='0.45', lw=1.6, label="sigma'_h = K0 sigma'_v")
    ax.set_title('effective stress', fontsize=10)
    ax.set_xlabel('stress [bar]')

    ax = axes[2]
    ax.plot(col['tot_stress'][:, 2] - ana['Sv'], z, 'k-', lw=1.4, label='sigma_zz - Sv')
    ax.plot(col['eff_stress'][:, 0] - ana['sh_eff'], z, '-', color='C0', lw=1.4,
            label="sigma'_xx - K0 sigma'_v")
    ax.plot(col['eff_stress'][:, 1] - ana['sh_eff'], z, '-', color='C1', lw=1.4,
            label="sigma'_yy - K0 sigma'_v")
    ax.axvline(0.0, color='0.7', lw=0.8)
    ax.set_title('model - analytic', fontsize=10)
    ax.set_xlabel('difference [bar]')

    ax = axes[3]
    ax.plot(col['pressure'], z, 'C0-', lw=1.4, label='p')
    ax.set_title('pore pressure', fontsize=10)
    ax.set_xlabel('pressure [bar]')

    # sharey=True links all axes' ylim; invert once here rather than once per axis
    # (an even number of invert_yaxis() calls on shared axes cancels out to no-op)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc='best')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_filename, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', png_filename)


def report(col, ana):
    print('\ncolumn of %d cells, depth %.1f .. %.1f m' % (len(col['z']), col['z'].min(), col['z'].max()))
    print('  porosity %.4f .. %.4f   bulk density %.1f .. %.1f kg/m3   K0 %.4f .. %.4f'
          % (col['poro'].min(), col['poro'].max(), ana['rho_bulk'].min(),
             ana['rho_bulk'].max(), ana['K0'].min(), ana['K0'].max()))
    rows = [
        ('sigma_zz  vs Sv', col['tot_stress'][:, 2], ana['Sv']),
        ("sigma'_zz vs sigma'_v", col['eff_stress'][:, 2], ana['sv_eff']),
        ("sigma'_xx vs K0 sigma'_v", col['eff_stress'][:, 0], ana['sh_eff']),
        ("sigma'_yy vs K0 sigma'_v", col['eff_stress'][:, 1], ana['sh_eff']),
    ]
    print('  %-26s %10s %10s %10s' % ('', 'max|diff|', 'rel', 'at depth'))
    for name, model, ref in rows:
        d = np.abs(model - ref)
        i = int(d.argmax())
        scale = max(np.abs(ref).max(), 1e-30)
        print('  %-26s %10.4g %9.3g%% %9.0f m' % (name, d.max(), 100 * d.max() / scale, col['z'][i]))
    for j in (3, 4, 5):
        s = col['tot_stress'][:, j]
        print('  %-26s %10.4g' % ('shear sigma_%s (max|.|)' % STRESS_LABELS[j], np.abs(s).max()))


def analytic_fsp(depth, biot=0.9, nu=0.2, dip_degrees=70.0, rho_bulk=2650.0,
                 p_gradient=0.1, friction=0.6, cohesion=0.0):
    """
    Initial (pre-injection) fault stability from the uniaxial-strain K0 state, without
    running the model. `depth` and `biot` are broadcast against each other, so either can
    be an array.

        Sv         = rho_bulk * g * z                     (total vertical stress)
        p          = p_gradient * z                       (hydrostatic)
        sigma'_v   = Sv - biot * p                        (matrix, Biot)
        sigma'_h   = K0 * sigma'_v,  K0 = nu / (1 - nu)   (no lateral movement)
        Sh         = sigma'_h + biot * p

    Resolved on a fault of dip delta from the horizontal, unit normal
    n = [sin(delta), 0, cos(delta)] under sigma = diag(Sh, Sh, Sv):

        sigma_n    = Sh sin^2(delta) + Sv cos^2(delta)
        tau        = |Sv - Sh| sin(delta) cos(delta)
        sigma'_n   = sigma_n - p        Terzaghi with the FULL pore pressure: the matrix
                                        biot does not apply on the fault surface
        mcc        = tau - (cohesion + friction * sigma'_n)   > 0 means slipping
        FSP        = tau / sigma'_n                           NaN where sigma'_n <= 0

    :param depth: depth(s) below the free surface [m].
    :param biot: Biot coefficient(s) of the matrix [-].
    :param nu: Poisson ratio [-]; sets K0 and so the stress ratio.
    :param dip_degrees: fault dip from the horizontal [degrees].
    :param rho_bulk: bulk density of the overburden [kg/m3].
    :param p_gradient: pore pressure gradient [bar/m] (idata.initial.pressure_gradient).
    :param friction: fault friction coefficient [-].
    :param cohesion: fault cohesion [bar]. With cohesion = 0 the criterion is homogeneous
        of degree 1, which makes FSP independent of depth - see notes/biot_FSP.md.
    :return: dict of arrays: Sv, p, sv_eff, sh_eff, Sh, sigma_n, sigma_n_eff, tau, mcc, FSP.
    """
    z, b = np.broadcast_arrays(np.asarray(depth, float), np.asarray(biot, float))
    sv = rho_bulk * G_BAR * z
    p = p_gradient * z
    sv_eff = sv - b * p
    k0 = nu / (1.0 - nu)
    sh_eff = k0 * sv_eff
    sh = sh_eff + b * p

    delta = np.radians(dip_degrees)
    sigma_n = sh * np.sin(delta) ** 2 + sv * np.cos(delta) ** 2
    tau = np.abs(sv - sh) * np.sin(delta) * np.cos(delta)
    sigma_n_eff = sigma_n - p
    mcc = tau - (cohesion + friction * sigma_n_eff)
    with np.errstate(divide='ignore', invalid='ignore'):
        fsp = np.where(sigma_n_eff > 0.0, tau / sigma_n_eff, np.nan)
    return {'Sv': sv, 'p': p, 'sv_eff': sv_eff, 'sh_eff': sh_eff, 'Sh': sh,
            'sigma_n': sigma_n, 'sigma_n_eff': sigma_n_eff, 'tau': tau,
            'mcc': mcc, 'FSP': fsp, 'K0': k0}


def plot_analytic_fsp(png_filename, depths=None, biots=None, depth=2200.0, biot=0.9,
                      dips=(70.0, 80.0, 89.0), friction=0.6, fsp_max=3.0,
                      stress_dip=None, nus=None, nu_dips=None, **kwargs):
    """
    Initial fault stability against depth (at `biot`) and against the Biot coefficient (at
    `depth`), in two rows: the effective stresses on top, FSP below with the friction
    coefficient drawn as the slip threshold (FSP > friction is a slipping fault). FSP gets
    one line per dip; the stresses are drawn for `stress_dip` only, since four curves per
    dip would be unreadable. Initial state only - see analytic_fsp().

    :param depths: depths for the left column [m]; default 500 m to 5 km.
    :param biots: Biot coefficients for the right column [-]; default 0.2 to 1.
    :param depth: depth the right column is evaluated at [m].
    :param biot: Biot coefficient the left column is evaluated at [-].
    :param dips: fault dips to draw in the FSP row [degrees].
    :param fsp_max: upper limit of the FSP axis; FSP diverges as sigma'_n -> 0.
    :param stress_dip: dip used for the stress row [degrees]; the first of `dips` if None.
        sigma'_xx and sigma'_zz do not depend on it, only sigma'_n and tau do.
    :param nus: Poisson ratios for the bottom panels; nu -/+ 0.05 if None. Values outside
        (0, 0.5) are dropped: nu = 0.5 is the incompressible limit where K0 = 1.
    :param nu_dips: one bottom panel per dip [degrees]; (stress_dip, 80) if None.
    :param kwargs: passed to analytic_fsp (nu, rho_bulk, p_gradient, cohesion).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    depths = np.linspace(500.0, 5000.0, 200) if depths is None else np.asarray(depths, float)
    biots = np.linspace(0.2, 1.0, 200) if biots is None else np.asarray(biots, float)
    stress_dip = dips[0] if stress_dip is None else stress_dip

    # (key, label) of the stress row; sigma'_zz and sigma'_xx are the K0 state itself,
    # sigma'_n and tau are that state resolved on the fault
    # The prime marks an effective stress. It is the Unicode PRIME (U+2032) outside
    # mathtext, not $\sigma'$ or $\sigma^{\prime}$: both of those render as a superscript
    # circle that reads as a zero in the DejaVu font mathtext falls back to here.
    fields = [('sv_eff', SIGMA_EFF + r'$_{zz}$ (vertical)'),
              ('sh_eff', SIGMA_EFF + r'$_{xx}$ (horizontal)'),
              ('sigma_n_eff', SIGMA_EFF + r'$_{n}$ (fault normal)'),
              ('tau', r'$\tau$ (fault shear)')]

    # the nu panel spans the bottom; nu enters only through K0 = nu/(1-nu), so it sets how
    # much of the vertical load is carried horizontally and therefore the whole deviator
    nu = kwargs.pop('nu', 0.2)
    if nus is None:
        nus = [nu - 0.05, nu, nu + 0.05]
    nus = [v for v in nus if 0.0 < v < 0.5]  # nu >= 0.5 is incompressible, K0 >= 1

    # one nu panel per dip; a single one spans the bottom, otherwise two per row
    nu_dips = (stress_dip, 80.0) if nu_dips is None else tuple(nu_dips)
    nu_dips = tuple(dict.fromkeys(nu_dips))  # drop repeats, keep the order given
    n_rows = 2 + (len(nu_dips) + 1) // 2

    fig = plt.figure(figsize=(11, 4.4 * n_rows))
    grid = fig.add_gridspec(n_rows, 2)
    axes = [[fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]]
    if len(nu_dips) == 1:
        nu_axes = [fig.add_subplot(grid[2, :])]
    else:
        nu_axes = [fig.add_subplot(grid[2 + k // 2, k % 2]) for k in range(len(nu_dips))]

    a_depth = analytic_fsp(depths, biot=biot, nu=nu, dip_degrees=stress_dip,
                           friction=friction, **kwargs)
    a_biot = analytic_fsp(depth, biot=biots, nu=nu, dip_degrees=stress_dip,
                          friction=friction, **kwargs)
    for key, label in fields:
        axes[0][0].plot(a_depth[key], depths, label=label)
        axes[0][1].plot(biots, a_biot[key], label=label)

    for dip in dips:
        a = analytic_fsp(depths, biot=biot, nu=nu, dip_degrees=dip, friction=friction, **kwargs)
        axes[1][0].plot(a['FSP'], depths, label=r'dip %g$\degree$' % dip)
        a = analytic_fsp(depth, biot=biots, nu=nu, dip_degrees=dip, friction=friction, **kwargs)
        axes[1][1].plot(biots, a['FSP'], label=r'dip %g$\degree$' % dip)

    for ax_nu, nu_dip in zip(nu_axes, nu_dips):
        for v in nus:
            a = analytic_fsp(depth, biot=biots, nu=v, dip_degrees=nu_dip,
                             friction=friction, **kwargs)
            ax_nu.plot(biots, a['FSP'], ls='-' if v == nu else '--',
                       label=r'$\nu$ = %.2f (K0 = %.2f)' % (v, v / (1.0 - v)))
        ax_nu.axhline(friction, color='k', ls=':', lw=1, label='friction = %g' % friction)
        ax_nu.set_xlabel('Biot coefficient [-]')
        ax_nu.set_ylabel(FSP_LABEL)
        ax_nu.set_ylim(0.0, fsp_max)
        ax_nu.set_title(r'FSP vs biot for a range of $\nu$ (z = %g m, dip %g$\degree$)'
                        % (depth, nu_dip))
        ax_nu.grid(alpha=0.3)
        ax_nu.legend(fontsize=8)

    stress_label = 'effective stress [bar]'
    axes[0][0].set_xlabel(stress_label)
    axes[0][0].set_title('stresses vs depth (biot = %g, dip %g$\\degree$)' % (biot, stress_dip))
    axes[0][1].set_ylabel(stress_label)
    axes[0][1].set_title('stresses vs biot (z = %g m, dip %g$\\degree$)' % (depth, stress_dip))
    # sigma'_n reaches zero at low biot, which is where FSP blows up
    axes[0][1].axhline(0.0, color='k', lw=0.8)

    # FSP diverges as sigma'_n -> 0 (low biot), so cap the axis on the interesting range
    fsp_label = FSP_LABEL
    axes[1][0].set_xlabel(fsp_label)
    axes[1][0].set_title('FSP vs depth (biot = %g)' % biot)
    axes[1][0].set_xlim(0.0, fsp_max)
    axes[1][1].set_ylabel(fsp_label)
    axes[1][1].set_title('FSP vs biot (z = %g m)' % depth)
    axes[1][1].set_ylim(0.0, fsp_max)
    # the slip threshold: FSP above the friction coefficient means the fault is slipping.
    # FSP is the x axis in the depth column and the y axis in the biot one
    label = 'friction = %g' % friction
    axes[1][0].axvline(friction, color='k', ls='--', lw=1, label=label)
    axes[1][1].axhline(friction, color='k', ls='--', lw=1, label=label)

    for row in axes:  # depth is the shared y axis of the left column, growing downwards
        row[0].set_ylabel('depth [m]')
        row[0].invert_yaxis()
        row[1].set_xlabel('Biot coefficient [-]')
        for ax in row:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    fig.suptitle('initial (pre-injection) fault stability, uniaxial-strain K0 state')
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(png_filename)), exist_ok=True)
    fig.savefig(png_filename, dpi=150)
    plt.close(fig)
    print('analytic FSP ->', png_filename)
