"""Regression tests for the compiled nucleus-of-strain proxy (_proxygeomech).

The reference ``nucleus_of_strain_ref.npz`` holds the displacements computed for the
grid built by ``_make_case`` with the standalone proxy_geomech build, so these tests
pin the open-darts build of the same kernel to its results.
"""

import os

import numpy as np
import pytest

from darts.physics.geomechanics.nucleus_of_strain import _proxygeomech as proxy

REF_FILE = os.path.join(os.path.dirname(__file__), "nucleus_of_strain_ref.npz")
KEYS = ("ux_p", "uy_p", "uz_p", "ux_t", "uy_t", "uz_t")
POISSON, YOUNG, THERMAL_EXPANSION = 0.25, 3300.0, 1.3e-5


def _make_case(nx=8, ny=6, nz=3):
    """Prisms of a regular nx*ny*nz grid at 1000 m depth, surface points at half-cell spacing.

    Returns flat (y, x, z) evaluation points, flat prisms (y1, y2, x1, x2, z2, z1),
    and per-prism pressure and temperature changes.
    """
    dx, dy, dz, z0 = 75.0, 75.0, 20.0, 1000.0
    prisms = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                prisms += [
                    dy * j,
                    dy * (j + 1),
                    dx * i,
                    dx * (i + 1),
                    z0 + dz * (k + 1),
                    z0 + dz * k,
                ]
    points = []
    for j in range(2 * ny):
        for i in range(2 * nx):
            points += [dy * j * 0.5 + 1e-3, dx * i * 0.5 + 1e-3, 0.0]
    n = nx * ny * nz
    rng = np.random.default_rng(0)
    return (
        np.array(points),
        np.array(prisms),
        rng.uniform(1, 20, n),
        rng.uniform(-30, 30, n),
    )


def _compute():
    points, prisms, dp, dt = _make_case()
    vv = proxy.value_vector
    res = proxy.compute_geomech(
        vv(points), vv(prisms), vv(dp), POISSON, YOUNG, vv(dt), THERMAL_EXPANSION
    )
    return {k: np.array(res[k]) for k in KEYS}


def test_matches_reference():
    ref = np.load(REF_FILE)
    res = _compute()
    for key in KEYS:
        # Each displacement sums signed prism kernels that largely cancel, so a value
        # near a sign change carries the round-off of the whole field, which differs
        # between compilers. The absolute tolerance is therefore scaled by the field's
        # largest magnitude rather than by the value itself.
        atol = 1e-8 * np.abs(ref[key]).max()
        np.testing.assert_allclose(
            res[key], ref[key], rtol=1e-8, atol=atol, err_msg=key
        )


def test_thread_count_does_not_change_result():
    proxy.set_num_threads(1)
    serial = _compute()
    proxy.set_num_threads(4)
    parallel = _compute()
    for key in KEYS:
        np.testing.assert_array_equal(serial[key], parallel[key], err_msg=key)


def test_set_platform():
    proxy.set_platform("cpu")
    with pytest.raises(ValueError):
        proxy.set_platform("tpu")
    if not hasattr(proxy, "compute_geomech_cuda"):
        with pytest.raises(RuntimeError):
            proxy.set_platform("gpu")
