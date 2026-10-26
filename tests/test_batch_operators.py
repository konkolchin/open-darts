"""Vectorised supporting-point evaluation: the operator sets with a batch-capable property container
(``PropertyContainer.evaluate_batch``) must give bitwise the values of the per-point ``evaluate()``."""

import warnings

import numpy as np
import pytest

from darts.engines import value_vector
from darts.physics.base.operator_evaluator import (
    ReservoirOperators,
    WellCtrlOperators,
    WellOperators,
)
from darts.physics.base.property_container import PropertyBatch, PropertyContainer
from darts.physics.properties.basic import RockCompactionEvaluator

EPS_Z = 1e-11


class ImmiscibleBatchContainer(PropertyContainer):
    """
    Three components that each form their own phase (x = identity, nu = z). A phase is present where its component
    fraction exceeds 1e-3; densities, viscosities, relative permeabilities and the water capillary pressure are simple
    polynomials. ``evaluate_batch`` repeats ``evaluate`` with arrays and leaves states with z_w < 0.05 to ``evaluate``.

    :param with_batch: implement the batch protocol (False: ``evaluate_batch`` returns None)
    :type with_batch: bool
    """

    def __init__(self, with_batch: bool = True):
        """
        Container with the phases gas, oil and water, the components g, o and w, and a rock compaction evaluator.

        :param with_batch: implement the batch protocol (False: ``evaluate_batch`` returns None)
        :type with_batch: bool
        """
        super().__init__(
            phases_name=['gas', 'oil', 'wat'],
            components_name=['g', 'o', 'w'],
            Mw=np.array([16.0, 100.0, 18.0]),
            eps_z=EPS_Z,
            temperature=350.0,
        )
        self.rock_compr_ev = RockCompactionEvaluator(pref=100.0, compres=1e-5)
        self.with_batch = with_batch

    @staticmethod
    def phase_functions(j, p, sat):
        """
        Density [kg/m3], viscosity [cP] and relative permeability of phase j; scalars or arrays.

        :param j: phase index
        :type j: int
        :param p: pressure [bar]
        :type p: float | np.ndarray
        :param sat: phase saturation
        :type sat: float | np.ndarray
        :returns: (density, viscosity, relative permeability)
        :rtype: tuple
        """
        if j == 0:
            return 0.8 + 0.002 * p, 0.02 + 1e-5 * p, sat * sat
        if j == 1:
            return 750.0 + 0.05 * p, 2.0 - 1e-3 * p, sat * sat * sat
        return 1000.0 + 0.04 * p, 0.5 + 1e-4 * p, 0.8 * sat * sat

    def evaluate(self, state):
        """
        Properties of one state.

        :param state: [pressure, z_g, z_o]
        :type state: np.ndarray
        """
        state = np.asarray(state)
        pressure = state[0]
        zc = np.append(state[1:3], 1.0 - np.sum(state[1:3]))
        self.clean_arrays()
        self.ph = np.array([j for j in range(3) if zc[j] > 1e-3], dtype=int)
        self.x = np.eye(3)
        self.nu = zc.copy()
        for j in self.ph:
            self.dens[j], self.mu[j], _ = self.phase_functions(j, pressure, 0.0)
            self.dens_m[j] = self.dens[j] / np.sum(self.Mw * self.x[j])
        self.compute_saturation()
        for j in self.ph:
            self.kr[j] = self.phase_functions(j, pressure, self.sat[j])[2]
        self.pc[:] = [0.0, 0.0, 0.5 * (1.0 - self.sat[2])]

    def evaluate_batch(self, states):
        """
        Properties of many states, bitwise those of ``evaluate``.

        :param states: states, one per row: [pressure, z_g, z_o]
        :type states: np.ndarray
        :returns: properties of the states, or None without batch support
        :rtype: PropertyBatch | None
        """
        if not self.with_batch:
            return None
        st = np.asarray(states, dtype=float)
        p = st[:, 0]
        n = p.size
        zc = np.empty((n, 3))
        zc[:, :2] = st[:, 1:3]
        zc[:, 2] = 1.0 - ((0.0 + st[:, 1]) + st[:, 2])
        b = PropertyBatch(n, self)
        b.evaluated = ~(zc[:, 2] < 0.05)
        b.present[:] = zc > 1e-3
        b.x[:] = np.eye(3)
        with np.errstate(all='ignore'):
            vol = []
            for j in range(3):
                on = b.present[:, j]
                dens, mu, _ = self.phase_functions(j, p, 0.0)
                b.dens[:, j] = np.where(on, dens, 0.0)
                b.mu[:, j] = np.where(on, mu, 0.0)
                mass = np.zeros(n)
                for c in range(3):
                    mass = mass + self.Mw[c] * b.x[:, j, c]
                b.dens_m[:, j] = np.where(on, b.dens[:, j] / mass, 0.0)
                vol.append(np.where(on, zc[:, j] / b.dens_m[:, j], 0.0))
            total = np.zeros(n)
            for j in range(3):
                total = total + vol[j]
            for j in range(3):
                on = b.present[:, j]
                b.sat[:, j] = np.where(on, vol[j] / total, 0.0)
                b.kr[:, j] = np.where(
                    on, self.phase_functions(j, p, b.sat[:, j])[2], 0.0
                )
        b.pc[:, 2] = 0.5 * (1.0 - b.sat[:, 2])
        b.rock_compr[:] = self.rock_compr_ev.evaluate(p)
        return b


def random_states(n, seed=0):
    """
    Random states [p, z_g, z_o] inside the simplex, with phases absent, points the container leaves to evaluate()
    (z_w < 0.05) and points that need composition extrapolation (z_w < eps_z).

    :param n: number of states
    :type n: int
    :param seed: random seed
    :type seed: int
    :returns: states (n, 3)
    :rtype: np.ndarray
    """
    rng = np.random.default_rng(seed)
    z = rng.dirichlet([0.6, 1.0, 1.0], size=n)
    z[rng.random(n) < 0.2, 0] = 0.0  # no gas
    z /= z.sum(axis=1, keepdims=True)
    z[: n // 20, 2] = 0.01 * rng.random(n // 20)  # left to evaluate() by the container
    z[n // 20 : n // 10, :2] /= z[n // 20 : n // 10, :2].sum(
        axis=1, keepdims=True
    )  # z_w = 0: extrapolated
    p = rng.uniform(50.0, 400.0, n)
    return np.column_stack([p, z[:, 0], z[:, 1]])


def scalar_values(ops, states, n_ops):
    """
    Operator values evaluated point by point with ``evaluate()``.

    :param ops: operator set
    :type ops: OperatorsBase
    :param states: states (n, n_dims)
    :type states: np.ndarray
    :param n_ops: operators per point of the interpolator
    :type n_ops: int
    :returns: values (n, n_ops)
    :rtype: np.ndarray
    """
    out = np.zeros((states.shape[0], n_ops))
    for i, s in enumerate(states):
        vv = value_vector(np.zeros(n_ops))
        ops.evaluate(value_vector(s.copy()), vv)
        out[i] = np.asarray(vv)
    return out


def batch_values(ops, states, n_ops):
    """
    Operator values from ``evaluate_batch``, the entry point of the interpolators.

    :param ops: operator set
    :type ops: OperatorsBase
    :param states: states (n, n_dims)
    :type states: np.ndarray
    :param n_ops: operators per point of the interpolator
    :type n_ops: int
    :returns: values (n, n_ops)
    :rtype: np.ndarray
    """
    n = states.shape[0]
    values = value_vector(np.zeros(n * n_ops))
    ops.evaluate_batch(value_vector(states.ravel().copy()), n, values, n_ops)
    return np.asarray(values).reshape(n, n_ops).copy()


def assert_bitwise_equal(a, b):
    """
    Assert that two float arrays have identical 64-bit patterns (sign of zero included).

    :param a: first array
    :type a: np.ndarray
    :param b: second array
    :type b: np.ndarray
    """
    assert a.shape == b.shape
    bits_a = np.ascontiguousarray(a).view(np.uint64)
    bits_b = np.ascontiguousarray(b).view(np.uint64)
    mismatch = np.argwhere(bits_a != bits_b)
    assert mismatch.size == 0, (
        f'{len(mismatch)} values differ, first at {mismatch[:3].tolist()}'
    )


@pytest.mark.parametrize('cls', [ReservoirOperators, WellOperators, WellCtrlOperators])
@pytest.mark.parametrize('extra_ops', [0, 3])
def test_batch_matches_scalar_bitwise(cls, extra_ops):
    """
    The vector path writes bitwise the scalar values, including the zero tail of a wider interpolator.

    :param cls: operator-set class
    :type cls: type
    :param extra_ops: operator slots of the interpolator beyond those of the operator set
    :type extra_ops: int
    """
    ops = cls(ImmiscibleBatchContainer(), thermal=False, dz=0.01)
    states = random_states(400)
    n_ops = ops.n_ops + extra_ops
    vectorised = ops._evaluate_batch_vectorised(states, np.zeros((len(states), n_ops)))
    assert vectorised is not None and 0 < vectorised.sum() < len(states)
    assert_bitwise_equal(
        batch_values(ops, states, n_ops), scalar_values(ops, states, n_ops)
    )


def test_container_without_batch_support_uses_evaluate():
    """A container without the protocol keeps the per-point loop."""
    ops = ReservoirOperators(ImmiscibleBatchContainer(with_batch=False), False, dz=0.01)
    states = random_states(50, seed=1)
    assert ops._evaluate_batch_vectorised(states, np.zeros((50, ops.n_ops))) is None
    assert_bitwise_equal(
        batch_values(ops, states, ops.n_ops), scalar_values(ops, states, ops.n_ops)
    )


def test_overridden_evaluate_uses_evaluate():
    """A subclass that overrides evaluate() and a container whose evaluate is replaced on the instance fall back."""

    class ScaledOperators(ReservoirOperators):
        """Reservoir operators with doubled mobilities."""

        def evaluate(self, state, values):
            """
            Reservoir operators with the mobilities doubled.

            :param state: state
            :type state: value_vector
            :param values: operator values, written in place
            :type values: value_vector
            :returns: 0
            :rtype: int
            """
            super().evaluate(state, values)
            v = values.to_numpy()
            v[self.LAMBDA_OP : self.LAMBDA_OP + self.nph] *= 2.0
            return 0

    states = random_states(40, seed=2)
    ops = ScaledOperators(ImmiscibleBatchContainer(), False, dz=0.01)
    assert ops._evaluate_batch_vectorised(states, np.zeros((40, ops.n_ops))) is None
    assert_bitwise_equal(
        batch_values(ops, states, ops.n_ops), scalar_values(ops, states, ops.n_ops)
    )

    container = ImmiscibleBatchContainer()
    base_evaluate = container.evaluate

    def evaluate(state):
        """
        Container evaluate with all relative permeabilities halved.

        :param state: [pressure, z_g, z_o]
        :type state: np.ndarray
        """
        base_evaluate(state)
        container.kr *= 0.5

    container.evaluate = evaluate
    ops = ReservoirOperators(container, False, dz=0.01)
    assert ops._evaluate_batch_vectorised(states, np.zeros((40, ops.n_ops))) is None
    assert_bitwise_equal(
        batch_values(ops, states, ops.n_ops), scalar_values(ops, states, ops.n_ops)
    )

    ops = ReservoirOperators(ImmiscibleBatchContainer(), False, dz=0.01)
    base_ops_evaluate = ops.evaluate

    def ops_evaluate(state, values):
        """
        Reservoir operators with the pressure operator negated.

        :param state: state
        :type state: value_vector
        :param values: operator values, written in place
        :type values: value_vector
        :returns: 0
        :rtype: int
        """
        base_ops_evaluate(state, values)
        v = values.to_numpy()
        v[ops.PRES_OP] = -v[ops.PRES_OP]
        return 0

    ops.evaluate = ops_evaluate
    assert ops._evaluate_batch_vectorised(states, np.zeros((40, ops.n_ops))) is None
    assert_bitwise_equal(
        batch_values(ops, states, ops.n_ops), scalar_values(ops, states, ops.n_ops)
    )


class HalvedKrContainer(ImmiscibleBatchContainer):
    """The batch container with all relative permeabilities halved in ``evaluate`` only."""

    def evaluate(self, state):
        """
        Properties of one state, relative permeabilities halved.

        :param state: [pressure, z_g, z_o]
        :type state: np.ndarray
        """
        super().evaluate(state)
        self.kr *= 0.5


class HalvedKrBatchContainer(HalvedKrContainer):
    """``HalvedKrContainer`` with the matching batch form."""

    def evaluate_batch(self, states):
        """
        Properties of many states, relative permeabilities halved.

        :param states: states, one per row: [pressure, z_g, z_o]
        :type states: np.ndarray
        :returns: properties of the states
        :rtype: PropertyBatch
        """
        b = super().evaluate_batch(states)
        b.kr *= 0.5
        return b


def test_container_subclass_overriding_evaluate():
    """A container subclass that changes evaluate() uses the vector path only with its own evaluate_batch()."""
    states = random_states(60, seed=5)
    for container, vectorised in (
        (HalvedKrContainer(), False),
        (HalvedKrBatchContainer(), True),
    ):
        ops = ReservoirOperators(container, False, dz=0.01)
        mask = ops._evaluate_batch_vectorised(states, np.zeros((60, ops.n_ops)))
        assert (mask is not None) == vectorised
        assert_bitwise_equal(
            batch_values(ops, states, ops.n_ops), scalar_values(ops, states, ops.n_ops)
        )


def test_thermal_operators_use_evaluate():
    """Thermal operator sets keep the per-point loop."""
    ops = ReservoirOperators(ImmiscibleBatchContainer(), thermal=True, dz=0.01)
    states = np.column_stack([random_states(10, seed=3), np.full(10, 350.0)])
    assert ops._evaluate_batch_vectorised(states, np.zeros((10, ops.n_ops))) is None


class BatchOnlyContainer(ImmiscibleBatchContainer):
    """The batch container whose ``evaluate`` fails: only the vectorised path can evaluate its states."""

    def evaluate(self, state):
        """
        Fails: the tests that use this container must not reach the per-point path.

        :param state: [pressure, z_g, z_o]
        :type state: np.ndarray
        """
        raise RuntimeError('evaluate() of BatchOnlyContainer called')

    def evaluate_batch(self, states):
        """
        Properties of many states, those of ``ImmiscibleBatchContainer``.

        :param states: states, one per row: [pressure, z_g, z_o]
        :type states: np.ndarray
        :returns: properties of the states
        :rtype: PropertyBatch
        """
        return super().evaluate_batch(states)


def make_reservoir_operators():
    """
    Worker factory of the ParallelEvaluator tests: reservoir operators over the batch container.

    :returns: reservoir operators
    :rtype: ReservoirOperators
    """
    return ReservoirOperators(ImmiscibleBatchContainer(), False, dz=0.01)


def make_batch_only_operators():
    """
    Worker factory of the ParallelEvaluator tests: reservoir operators that only the vectorised path can evaluate.

    :returns: reservoir operators
    :rtype: ReservoirOperators
    """
    return ReservoirOperators(BatchOnlyContainer(), False, dz=0.01)


def vectorisable_states(n, seed):
    """
    Random states that the batch container evaluates and that need no extrapolation (z_w >= 0.05).

    :param n: number of states
    :type n: int
    :param seed: random seed
    :type seed: int
    :returns: states (n, 3)
    :rtype: np.ndarray
    """
    states = random_states(4 * n, seed=seed)
    z_w = 1.0 - (states[:, 1] + states[:, 2])
    return states[z_w >= 0.1][:n].copy()


@pytest.mark.parametrize('n_pts', [2, 301])
def test_parallel_evaluator_uses_batch_path(n_pts):
    """
    ParallelEvaluator gives the scalar values through its pool (n_pts > 2) and its in-process path (n_pts <= 2); the
    pool dispatch itself succeeds.

    :param n_pts: number of points
    :type n_pts: int
    """
    from darts.physics.base.parallel_evaluator import ParallelEvaluator

    states = random_states(n_pts, seed=4)
    ops = make_reservoir_operators()
    par = ParallelEvaluator(evaluator_factory=make_reservoir_operators, n_workers=2)
    try:
        values = value_vector(np.zeros(n_pts * ops.n_ops))
        par.evaluate_batch(
            value_vector(states.ravel().copy()), n_pts, values, ops.n_ops
        )
        pooled = np.zeros(n_pts * ops.n_ops)
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # a failed dispatch warns and returns False
            assert par._shared_pool.evaluate_batch(
                par._key, states.ravel().copy(), n_pts, pooled, 3, ops.n_ops
            )
    finally:
        par.shutdown()
    scalar = scalar_values(ops, states, ops.n_ops)
    assert_bitwise_equal(np.asarray(values).reshape(n_pts, ops.n_ops), scalar)
    assert_bitwise_equal(pooled.reshape(n_pts, ops.n_ops), scalar)


@pytest.mark.parametrize('n_pts', [2, 64])
def test_parallel_evaluator_vectorises(n_pts):
    """
    The pool workers and the in-process path take the vectorised path: with a container whose evaluate() fails the
    batch still succeeds and gives the values of the regular container.

    :param n_pts: number of points
    :type n_pts: int
    """
    from darts.physics.base.parallel_evaluator import ParallelEvaluator

    states = vectorisable_states(n_pts, seed=6)
    assert len(states) == n_pts
    ops = make_reservoir_operators()
    par = ParallelEvaluator(evaluator_factory=make_batch_only_operators, n_workers=2)
    try:
        values = value_vector(np.zeros(n_pts * ops.n_ops))
        with warnings.catch_warnings():
            warnings.simplefilter(
                'error'
            )  # a failed dispatch warns and then fails serially
            par.evaluate_batch(
                value_vector(states.ravel().copy()), n_pts, values, ops.n_ops
            )
    finally:
        par.shutdown()
    assert_bitwise_equal(
        np.asarray(values).reshape(n_pts, ops.n_ops),
        scalar_values(ops, states, ops.n_ops),
    )
