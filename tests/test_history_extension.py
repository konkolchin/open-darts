"""Test the standalone OBL history-state support object."""

from types import SimpleNamespace

import numpy as np

from darts.physics.base.history_extension import HistoryField, HistoryStateSupport
from darts.physics.base.physics import PhysicsBase


def test_history_support_extends_axes_and_updates_engine_buffers() -> None:
    """Keep axis ordering and engine-buffer access consistent after extraction."""
    history = HistoryStateSupport(
        [
            HistoryField("sg_max", axes_origin=0.0, axes_step=0.1, default=0.2),
            HistoryField("cycle", axes_origin=1.0, axes_step=1.0, default=3.0),
        ]
    )
    engine = SimpleNamespace(
        X=np.array([100.0, 0.1, 200.0, 0.2]),
        Xhistory=np.array([0.4, 4.0, 0.5, 5.0]),
    )

    assert history.extend_axes([1.0, 0.0], [10.0, 0.01]) == (
        [1.0, 0.0, 0.0, 1.0],
        [10.0, 0.01, 0.1, 1.0],
    )
    np.testing.assert_allclose(
        history.get_engine_history_array(engine, "sg_max"), [0.4, 0.5]
    )
    history.set_engine_history_array(engine, "cycle", 7.0)
    np.testing.assert_allclose(engine.Xhistory, [0.4, 7.0, 0.5, 7.0])
    np.testing.assert_allclose(
        history.get_interpolator_state(engine, n_vars=2),
        [100.0, 0.1, 0.4, 7.0, 200.0, 0.2, 0.5, 7.0],
    )


def test_history_support_propagates_defaults() -> None:
    """Populate boundary and well fallback values in cell-major order."""
    history = HistoryStateSupport(
        [HistoryField("sg_max", default=0.2), HistoryField("cycle", default=3.0)]
    )
    mesh = SimpleNamespace(n_bounds=2)
    control = SimpleNamespace()
    constraint = SimpleNamespace()
    well = SimpleNamespace(control=control, constraint=constraint)

    history.populate_boundary_defaults(mesh)
    history.populate_well_defaults([well])

    np.testing.assert_allclose(np.asarray(mesh.Xhistory_bounds), [0.2, 3.0, 0.2, 3.0])
    np.testing.assert_allclose(np.asarray(well.Xhistory_well_default), [0.2, 3.0])
    np.testing.assert_allclose(np.asarray(control.Xhistory_well_default), [0.2, 3.0])
    np.testing.assert_allclose(np.asarray(constraint.Xhistory_well_default), [0.2, 3.0])


def test_history_support_is_inert_without_fields() -> None:
    """An empty descriptor list disables every history operation without guards."""
    history = HistoryStateSupport()
    engine = SimpleNamespace(X=np.array([1.0, 2.0, 3.0, 4.0]))
    mesh = SimpleNamespace(n_bounds=2)
    well = SimpleNamespace()

    assert history.n_fields == 0
    assert history.labels == [] and history.defaults == []
    assert history.extend_axes([0.0], [1.0]) == ([0.0], [1.0])
    np.testing.assert_allclose(
        history.get_interpolator_state(engine, n_vars=2), [1.0, 2.0, 3.0, 4.0]
    )
    # No-ops rather than errors, so PhysicsBase never needs an `if history` guard.
    history.populate_boundary_defaults(mesh)
    history.populate_well_defaults([well])
    assert not hasattr(mesh, "Xhistory_bounds")
    assert not hasattr(well, "Xhistory_well_default")


def test_history_support_rejects_unknown_label_and_missing_fields() -> None:
    """Report a bad label and an unconfigured physics distinctly."""
    import pytest

    history = HistoryStateSupport([HistoryField("sg_max")])
    engine = SimpleNamespace(Xhistory=np.zeros(3))
    with pytest.raises(KeyError):
        history.default("nope")
    with pytest.raises(KeyError):
        history.get_engine_history_array(engine, "nope")

    empty = HistoryStateSupport()
    with pytest.raises(RuntimeError):
        empty.get_engine_history_array(engine, "sg_max")
    with pytest.raises(RuntimeError):
        empty.set_engine_history_array(engine, "sg_max", 1.0)


def test_history_field_validates_axis_metadata() -> None:
    """Reject descriptors that would produce an unusable OBL axis."""
    import pytest

    for kwargs in (
        {"label": ""},
        {"label": "h", "axes_step": 0.0},
        {"label": "h", "axes_step": float("inf")},
        {"label": "h", "axes_origin": float("nan")},
        {"label": "h", "default": float("inf")},
    ):
        with pytest.raises(ValueError):
            HistoryField(**kwargs)


def test_physics_history_fields_round_trip() -> None:
    """`PhysicsBase.history.fields` stays assignable, now that the `history_fields`
    setter moved from `PhysicsBase` onto `HistoryStateSupport` itself."""
    physics = PhysicsBase.__new__(PhysicsBase)
    physics.history = HistoryStateSupport()

    physics.history.fields = []
    assert physics.history.n_fields == 0
    assert physics.history.fields == []

    fields = [HistoryField("sg_max", default=0.2), HistoryField("cycle", default=3.0)]
    physics.history.fields = fields
    assert physics.history.n_fields == 2
    assert physics.history.labels == ["sg_max", "cycle"]
    assert physics.history.default("cycle") == 3.0

    physics.history.fields = None
    assert physics.history.n_fields == 0


def test_extrapolation_keeps_history_variables() -> None:
    """Boundary extrapolation of a state with OBL history variables: the history value is not
    a composition and every reference state carries it unchanged."""
    from darts.engines import value_vector
    from darts.physics.base.operator_evaluator import OperatorsBase

    class LinearOperators(OperatorsBase):
        """Two operators: a linear function of the compositions, and the history value."""

        def __init__(self, prop):
            super().__init__(prop, thermal=False, extrapolation_flag=True, dz=0.01)
            self.n_ops = 2
            self.reference_states = []

        def evaluate(self, state, values):
            if self.apply_extrapolation(state, values):
                return 0
            s = state.to_numpy()
            self.reference_states.append(s.copy())
            out = values.to_numpy()
            out[0] = 1.0 + 2.0 * s[1] + 3.0 * s[2]
            out[1] = s[-1]
            return 0

    prop = SimpleNamespace(nc=3, nph=2, eps_z=1e-10, n_history=1)
    ops = LinearOperators(prop)
    # z_last = 1 - 0.6 - 0.45 < 0: outside the composition simplex, history value 0.3
    state = value_vector([50.0, 0.6, 0.45, 0.3])
    values = value_vector([0.0, 0.0])
    ops.evaluate(state, values)

    np.testing.assert_allclose(values.to_numpy(), [1.0 + 2.0 * 0.6 + 3.0 * 0.45, 0.3])
    assert ops.reference_states
    for ref in ops.reference_states:
        assert ref.size == 4
        assert ref[-1] == 0.3
        assert 1.0 - ref[1] - ref[2] >= -1e-12


def test_three_phase_history_interpolator_templates() -> None:
    """Three-phase physics with one or two OBL history axes needs interpolators with
    N_DIMS = N_VARS + n_history at the operator count of N_VARS."""
    import darts.interpolators as itors

    n_phases = 3
    for n_vars in (2, 3, 4, 5):
        n_ops = n_vars * (2 * n_phases + 2) + 7 * n_phases + 3
        for n_history in (1, 2):
            name = f"multilinear_cpu_interpolator_d_{n_vars + n_history}_{n_ops}"
            assert hasattr(itors, name), name
