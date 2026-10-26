"""
Consistency checks of the mech_discretizer interface and boundary approximations.

The checks need no reference data: each one compares the discrete fluxes with an exact
property of the continuous problem on the two-layer column of the poroelastic analytics
model (layers at x < 75 and x > 75 with different stiffness, Biot coefficient and
thermal expansion):

* the total traction (Hooke + Biot + thermal) is conservative across interior faces;
* a piecewise-linear state (uniform pressure and temperature, uniform total stress,
  strain jumping at the layer interface) is reproduced exactly: every cell is in
  equilibrium and Biot's volumetric-strain term equals V * b * eps_xx;
* the interface pressure is exact for a hydrostatic state with a permeability contrast;
* on roller faces the volumetric-strain term does not depend on cell unknowns, also
  for an anisotropic stiffness;
* on loaded no-flow faces of a non-orthogonal mesh the traction equals the load;
* non-default gradient-reconstruction options keep the flux stencils consistent.

Run:  PYTHONPATH=<repo> python -m pytest tests/test_mech_discretizer_consistency.py -q
"""

import pathlib

import numpy as np
import pytest

dis = pytest.importorskip("darts.discretizer")

MESH_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "models"
    / "1ph_1comp_poroelastic_analytics"
    / "meshes"
)
TOP, BOTTOM = 99991, 99992  # x > 75 and x < 75
XM, XP, YM, YP, ZM, ZP = 991, 992, 993, 994, 995, 996
X_INTERFACE = 75.0

# boundary conditions: flow / heat (a, b) and mechanics (an, bn, at, bt)
NO_FLOW, DIRICHLET = (0.0, 1.0), (1.0, 0.0)
ROLLER = (1.0, 0.0, 0.0, 1.0)
LOAD = (0.0, 1.0, 0.0, 1.0)
STUCK_T_LOAD_N = (0.0, 1.0, 1.0, 0.0)
# a full anisotropic stiffness (Voigt, bar), the one of the poroelastic convergence model
ANISO_STIFFNESS = 1e4 * np.array(
    [
        [1.323, 0.0726, 0.263, 0.108, -0.08, -0.239],
        [0.0726, 1.276, -0.318, 0.383, 0.108, 0.501],
        [0.263, -0.318, 0.943, -0.183, 0.146, 0.182],
        [0.108, 0.383, -0.183, 1.517, -0.0127, -0.304],
        [-0.08, 0.108, 0.146, -0.0127, 1.209, -0.326],
        [-0.239, 0.501, 0.182, -0.304, -0.326, 1.373],
    ]
)


def _mesh_file(name):
    path = MESH_DIR / name
    if not path.is_file():
        pytest.skip(f"mesh {path} not found")
    return str(path)


def _lame(E, nu):
    return E * nu / (1 + nu) / (1 - 2 * nu), E / 2 / (1 + nu)


def _layers(biot=(0.9, 0.2), th_expn=(2.0, 20.0), perm=(1.0, 1.0)):
    """Two layers with contrasting stiffness, Biot coefficient and thermal expansion."""
    return {
        TOP: dict(E=1e4, nu=0.15, biot=biot[0], th_expn=th_expn[0], perm=perm[0]),
        BOTTOM: dict(E=2e4, nu=0.3, biot=biot[1], th_expn=th_expn[1], perm=perm[1]),
    }


def _bcs(xp_mech=LOAD, xp_flow=DIRICHLET, xp_heat=DIRICHLET):
    """Rollers with no-flow (and no heat flux) everywhere except X+."""
    bcs = {tag: (NO_FLOW, ROLLER, NO_FLOW) for tag in (XM, YM, YP, ZM, ZP)}
    bcs[XP] = (xp_flow, xp_mech, xp_heat)
    return bcs


class Discretization:
    """mech_discretizer on a mesh of the poroelastic analytics model, plus flux evaluation."""

    def __init__(
        self,
        mesh_name,
        props,
        bcs,
        thermal=False,
        stiffness=None,
        gravity=None,
        options=None,
    ):
        self.thermal = thermal
        self.nb = 5 if thermal else 4  # ux, uy, uz, p, (T)
        self.props = props
        mesh = dis.Mesh()
        mesh.gmsh_mesh_processing(
            _mesh_file(mesh_name),
            {
                dis.elem_loc.MATRIX: {TOP, BOTTOM},
                dis.elem_loc.FRACTURE: set(),
                dis.elem_loc.BOUNDARY: {XM, XP, YM, YP, ZM, ZP},
                dis.elem_loc.FRACTURE_BOUNDARY: set(),
            },
        )
        d = (
            dis.thermoporo_mech_discretizer()
            if thermal
            else dis.poro_mech_discretizer()
        )
        d.set_mesh(mesh)
        d.init()
        for key, value in (options or {}).items():
            setattr(d, key, value)
        if gravity is not None:
            from darts.engines import matrix

            d.grav_vec = matrix(list(gravity), 1, 3)

        self.n_cells = mesh.n_cells
        self.tags = np.array(mesh.tags)
        m0, m1 = mesh.region_ranges[dis.elem_loc.MATRIX]
        self.cell_tag = self.tags[m0:m1]
        for tag in self.cell_tag:
            p = props[tag]
            d.perms.append(dis.matrix33(p["perm"], p["perm"], p["perm"]))
            d.biots.append(dis.matrix33(p["biot"]))
            if stiffness is None:
                d.stfs.append(dis.Stiffness(*_lame(p["E"], p["nu"])))
            else:
                d.stfs.append(dis.Stiffness(np.asarray(stiffness).flatten().tolist()))
            if thermal:
                d.heat_conductions.append(dis.matrix33(1.0))
                d.thermal_expansions.append(dis.matrix33(p["th_expn"]))

        from darts.engines import value_vector

        b0, b1 = mesh.region_ranges[dis.elem_loc.BOUNDARY]
        self.bnd_tag = self.tags[b0:b1]
        coefs = np.array([np.concatenate(bcs[tag]) for tag in self.bnd_tag])
        self.bc_flow, self.bc_mech, self.bc_heat = (
            coefs[:, :2],
            coefs[:, 2:6],
            coefs[:, 6:],
        )
        thm = dis.THMBoundaryCondition()
        thm.flow.a = value_vector(coefs[:, 0])
        thm.flow.b = value_vector(coefs[:, 1])
        thm.mech_normal.a = value_vector(coefs[:, 2])
        thm.mech_normal.b = value_vector(coefs[:, 3])
        thm.mech_tangen.a = value_vector(coefs[:, 4])
        thm.mech_tangen.b = value_vector(coefs[:, 5])
        flow = dis.BoundaryCondition()
        flow.a = value_vector(coefs[:, 0])
        flow.b = value_vector(coefs[:, 1])
        if thermal:
            thm.thermal.a = value_vector(coefs[:, 6])
            thm.thermal.b = value_vector(coefs[:, 7])
            heat = dis.BoundaryCondition()
            heat.a = value_vector(coefs[:, 6])
            heat.b = value_vector(coefs[:, 7])
            d.reconstruct_pressure_temperature_gradients_per_cell(flow, heat)
        else:
            d.reconstruct_pressure_gradients_per_cell(flow)
        d.reconstruct_displacement_gradients_per_cell(thm)
        d.calc_interface_approximations()
        self.d, self.mesh = d, mesh

        # connection geometry, in the order of the flux arrays (cells, then their adjacency)
        self.centroids = np.array([c.values for c in mesh.centroids[: self.n_cells]])
        self.volumes = np.array(mesh.volumes)[: self.n_cells]
        adj_off, adj = np.array(mesh.adj_matrix_offset), np.array(mesh.adj_matrix)
        conns = mesh.conns  # converted to a list on every attribute access: fetch once
        normals, centers, areas = [], [], []
        for i in range(m1 - m0):
            for j in range(adj_off[i], adj_off[i + 1]):
                conn = conns[adj[j]]
                n, c = np.array(conn.n.values), np.array(conn.c.values)
                normals.append(n if n.dot(c - self.centroids[i]) > 0 else -n)
                centers.append(c)
                areas.append(conn.area)
        self.normals, self.centers = np.array(normals), np.array(centers)
        self.areas = np.array(areas)

        # flux arrays, one row per stencil entry
        self.cell_m, self.cell_p = np.array(d.cell_m), np.array(d.cell_p)
        self.offset = np.array(d.flux_offset)
        assert len(self.offset) == len(self.cell_m) + 1 == len(self.areas) + 1
        self.stencil = np.array(d.flux_stencil)
        self.entry_conn = np.repeat(np.arange(len(self.cell_m)), np.diff(self.offset))
        self.hooke = np.array(d.hooke).reshape(-1, 3, self.nb)
        self.biot = np.array(d.biot_traction).reshape(-1, 3)
        self.vol_strain = np.array(d.biot_vol_strain).reshape(-1, self.nb)
        self.thermal_traction = (
            np.array(d.thermal_traction).reshape(-1, 3) if thermal else None
        )

    def evaluate(self, state):
        """Face forces (hooke + biot + thermal) and Biot's volumetric-strain term of every
        connection; state has one row [ux, uy, uz, p, (T)] per cell and boundary element."""
        x = state[self.stencil]
        f = np.einsum("eij,ej->ei", self.hooke, x) + self.biot * x[:, 3:4]
        if self.thermal:
            f += self.thermal_traction * x[:, 4:5]
        forces = np.zeros((len(self.cell_m), 3))
        np.add.at(forces, self.entry_conn, f)
        vs = np.bincount(
            self.entry_conn,
            weights=np.einsum("ej,ej->e", self.vol_strain, x),
            minlength=len(self.cell_m),
        )
        return forces, vs

    def boundary_connections(self):
        return np.where(self.cell_p >= self.n_cells)[0]


def _layer_strain(props, stress, p0, t0):
    """Uniaxial strain of every layer carrying the total stress sigma_xx = stress."""
    eps = {}
    for tag, p in props.items():
        lam, mu = _lame(p["E"], p["nu"])
        eps[tag] = (stress + p["biot"] * p0 + p["th_expn"] * t0) / (lam + 2 * mu)
    return eps


def _patch_state(disc, eps, stress, p0, t0):
    """Piecewise-linear u_x(x), uniform p0 / T0, and consistent boundary values."""

    def u_x(x):
        below = eps[BOTTOM] * np.minimum(x, X_INTERFACE)
        return below + eps[TOP] * np.maximum(x - X_INTERFACE, 0.0)

    state = np.zeros((disc.n_cells + len(disc.bnd_tag), disc.nb))
    state[: disc.n_cells, 0] = u_x(disc.centroids[:, 0])
    state[: disc.n_cells, 3] = p0
    if disc.thermal:
        state[: disc.n_cells, 4] = t0
    for k in disc.boundary_connections():
        b = disc.cell_p[k] - disc.n_cells
        n, c = disc.normals[k], disc.centers[k]
        an, _, at, _ = disc.bc_mech[b]
        u = np.array([u_x(c[0]), 0.0, 0.0])
        t = (
            np.array([stress, 0.0, 0.0]) * n[0]
        )  # sigma n, only sigma_xx acts on x-faces
        proj = np.eye(3) - np.outer(n, n)
        rn = u @ n if an == 1.0 else t @ n
        rt = proj @ u if at == 1.0 else proj @ t
        state[disc.cell_p[k], :3] = rn * n + rt
        state[disc.cell_p[k], 3] = p0 if disc.bc_flow[b, 0] == 1.0 else 0.0
        if disc.thermal:
            state[disc.cell_p[k], 4] = t0 if disc.bc_heat[b, 0] == 1.0 else 0.0
    return state


@pytest.mark.parametrize("thermal", [False, True])
def test_traction_is_conservative(thermal):
    disc = Discretization("transfinite_two_layers.msh", _layers(), _bcs(), thermal)
    rng = np.random.default_rng(0)
    state = rng.standard_normal((disc.n_cells + len(disc.bnd_tag), disc.nb))
    forces, _ = disc.evaluate(state)
    index = {
        (m, p): k
        for k, (m, p) in enumerate(zip(disc.cell_m, disc.cell_p, strict=False))
    }
    worst = 0.0
    for k, (m, p) in enumerate(zip(disc.cell_m, disc.cell_p, strict=False)):
        if p < disc.n_cells:
            mismatch = forces[k] + forces[index[(p, m)]]
            worst = max(worst, np.linalg.norm(mismatch) / np.linalg.norm(forces[k]))
    assert worst < 1e-10


@pytest.mark.parametrize("thermal", [False, True])
def test_piecewise_linear_state_is_exact(thermal):
    props = _layers()
    disc = Discretization("transfinite_two_layers.msh", props, _bcs(), thermal)
    stress, p0, t0 = -100.0, 50.0, (5.0 if thermal else 0.0)
    eps = _layer_strain(props, stress, p0, t0)
    forces, vs = disc.evaluate(_patch_state(disc, eps, stress, p0, t0))

    net = np.zeros((disc.n_cells, 3))
    np.add.at(net, disc.cell_m, forces)
    face_force = abs(stress) * disc.areas.max()
    assert np.abs(net).max() < 1e-8 * face_force

    cell_vs = np.bincount(disc.cell_m, weights=vs, minlength=disc.n_cells)
    exact = (
        np.array([props[tag]["biot"] * eps[tag] for tag in disc.cell_tag])
        * disc.volumes
    )
    assert np.abs(cell_vs - exact).max() < 1e-8 * np.abs(exact).max()


def test_hydrostatic_interface_pressure():
    grav = 0.0981  # per unit density, along x across the layer interface
    props = _layers(biot=(1.0, 1.0), perm=(1.0, 100.0))
    disc = Discretization(
        "transfinite_two_layers.msh", props, _bcs(), gravity=(grav, 0.0, 0.0)
    )
    state = np.zeros((disc.n_cells + len(disc.bnd_tag), disc.nb))
    state[: disc.n_cells, 3] = grav * disc.centroids[:, 0]
    for k in disc.boundary_connections():
        if disc.bc_flow[disc.cell_p[k] - disc.n_cells, 0] == 1.0:
            state[disc.cell_p[k], 3] = grav * disc.centers[k][0]
    x = state[disc.stencil]
    pressure_force = np.zeros((len(disc.cell_m), 3))
    np.add.at(pressure_force, disc.entry_conn, disc.biot * x[:, 3:4])
    # the free term of the Biot traction is per unit density
    pressure_force += np.array(disc.d.biot_traction_rhs).reshape(-1, 3)
    across = (disc.centroids[disc.cell_m, 0] - X_INTERFACE) * (
        disc.centroids[np.minimum(disc.cell_p, disc.n_cells - 1), 0] - X_INTERFACE
    ) < 0
    across &= disc.cell_p < disc.n_cells
    assert across.any()
    p_face = pressure_force[across, 0] / (disc.normals[across, 0] * disc.areas[across])
    p_exact = grav * disc.centers[across, 0]
    assert np.abs(p_face - p_exact).max() < 1e-12 * np.abs(p_exact).max()


def test_roller_vol_strain_independent_of_cells():
    disc = Discretization(
        "transfinite_two_layers.msh", _layers(), _bcs(), stiffness=ANISO_STIFFNESS
    )
    worst, scale = 0.0, 0.0
    for k in disc.boundary_connections():
        b = disc.cell_p[k] - disc.n_cells
        if tuple(disc.bc_mech[b]) != ROLLER:
            continue
        entries = np.arange(disc.offset[k], disc.offset[k + 1])
        cells = entries[disc.stencil[entries] < disc.n_cells]
        worst = max(worst, np.abs(disc.vol_strain[cells]).max())
        scale = max(scale, np.abs(disc.vol_strain[entries]).max())
    assert scale > 0.0
    assert worst < 1e-12 * scale


def test_loaded_no_flow_face_carries_the_load():
    disc = Discretization(
        "wedge_two_layers.msh", _layers(), _bcs(xp_flow=NO_FLOW, xp_heat=NO_FLOW)
    )
    rng = np.random.default_rng(1)
    state = rng.standard_normal((disc.n_cells + len(disc.bnd_tag), disc.nb))
    forces, _ = disc.evaluate(state)
    loaded = [
        k
        for k in disc.boundary_connections()
        if disc.bnd_tag[disc.cell_p[k] - disc.n_cells] == XP
    ]
    assert loaded
    for k in loaded:
        prescribed = -state[disc.cell_p[k], :3] * disc.areas[k]
        assert np.abs(forces[k] - prescribed).max() < 1e-10 * np.abs(prescribed).max()


def test_gradients_exact_with_gravity_at_loaded_no_flow_face():
    """Hydrostatic pressure along y and a uniform strain on the hexahedral mesh with a
    loaded no-flow X+ face: the reconstructed displacement gradients are exact, which
    needs the gravity part of the tangential pressure gradient in the boundary rows."""
    grav, eps = 0.0981, -1e-3
    props = _layers()
    disc = Discretization(
        "hexahedron.msh",
        props,
        _bcs(xp_flow=NO_FLOW, xp_heat=NO_FLOW),
        gravity=(0.0, grav, 0.0),
    )
    p = props[TOP]  # the hexahedral mesh has a single region
    lam, mu = _lame(p["E"], p["nu"])
    elastic_stress = np.diag([(lam + 2 * mu) * eps, lam * eps, lam * eps])
    state = np.zeros((disc.n_cells + len(disc.bnd_tag), disc.nb))
    state[: disc.n_cells, 0] = eps * disc.centroids[:, 0]
    state[: disc.n_cells, 3] = grav * disc.centroids[:, 1]
    for k in disc.boundary_connections():
        an, _, at, _ = disc.bc_mech[disc.cell_p[k] - disc.n_cells]
        n, c = disc.normals[k], disc.centers[k]
        u = np.array([eps * c[0], 0.0, 0.0])
        t = (elastic_stress - p["biot"] * grav * c[1] * np.eye(3)) @ n
        proj = np.eye(3) - np.outer(n, n)
        rn = u @ n if an == 1.0 else t @ n
        rt = proj @ u if at == 1.0 else proj @ t
        state[disc.cell_p[k], :3] = rn * n + rt  # no-flow everywhere: zero flux values
    exact = np.zeros(9)
    exact[0] = eps
    for i in range(disc.n_cells):
        g = disc.d.u_grads[i]
        stencil = list(g.stencil)
        a = np.array(g.a.values).reshape(9, disc.nb * len(stencil))
        # the free term is per unit density
        grad = a @ state[stencil].ravel() + np.array(g.rhs.values)
        assert np.abs(grad - exact).max() < 1e-9 * abs(eps)


def test_face_count_without_neumann_rows():
    """A normal-load / tangential-stuck face is not pure Neumann: the displacement
    gradients must not change when pure Neumann faces are left out."""
    arrays = []
    for neumann in (True, False):
        disc = Discretization(
            "transfinite_two_layers.msh",
            _layers(),
            _bcs(xp_mech=STUCK_T_LOAD_N),
            options={"neumann_boundaries_grad_reconstruction": neumann},
        )
        arrays.append((disc.hooke.copy(), disc.vol_strain.copy()))
    for a, b in zip(arrays[0], arrays[1], strict=True):
        assert a.shape == b.shape
        assert np.array_equal(a, b)


def test_extended_stencil_keeps_gradient_stencils_aligned():
    disc = Discretization(
        "transfinite_two_layers.msh",
        _layers(),
        _bcs(),
        thermal=True,
        options={"gradients_extended_stencil": True},
    )
    d = disc.d
    for i in range(disc.n_cells):
        u_stencil = list(d.u_grads[i].stencil)
        assert list(d.p_grads[i].stencil) == u_stencil
        assert list(d.t_grads[i].stencil) == u_stencil
