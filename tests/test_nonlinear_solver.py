"""Focused unit tests for the Python nonlinear-solver refactoring (MR327).

These exercise the pure-Python Newton driver (``darts.nonlinear_solvers``)
against a recording ``FakeEngine`` (see conftest.py) — no full simulation. They
lock in the behaviours fixed in MR327 review: residual hooks, pre/update/post
ordering, line-search accepted status, linear-solver failure return codes and
statistics, OBL-bounds wiring, fallback subclass preservation, spec validation,
TimestepControl->sim_params propagation, the legacy set_sim_params shim, and NaN/Inf
rejection.

``darts.engines`` is required only for the canonical enum constants that
``NonlinearSolverSpec.sync_to_engine`` reads; the engine's numerical kernels are
never used (FakeEngine stands in for them).
"""

import numpy as np
import pytest

# sync_to_engine reads sim_params enum constants; skip the whole module on a
# checkout without the compiled extension (CI builds it before pytest runs).
pytest.importorskip("darts.engines")

from darts.nonlinear_solvers import (  # noqa: E402
    ChopSpec,
    FallbackSpec,
    NewtonSolver,
    NewtonSpec,
    Norm,
    OBLBoundsSpec,
    RecoverableEvaluationError,
)


# ------------------------------------------------------ recoverable evaluation errors
def test_recoverable_error_in_assembly_fails_the_step(make_newton):
    # e.g. a flash failure raised by a property evaluator during OBL point
    # generation inside engine.assemble_linear_system of the 2nd iterate
    solver, model, engine, log = make_newton(res_seq=[1.0, 1e-1, 1e-9])
    assemble = engine.assemble_linear_system

    def failing_assemble(dt):
        if engine.n_apply_update == 1:
            raise RecoverableEvaluationError("flash did not converge")
        return assemble(dt)

    engine.assemble_linear_system = failing_assemble
    assert solver.run_timestep(1.0, 0.0) is False
    assert "flash did not converge" in solver.status.divergence_reason
    assert "recoverable evaluation error" in solver._failure_message(1.0)
    assert engine.n_apply_update == 1
    assert solver.stats.n_timesteps_wasted == 1
    assert log[-1] == "post_newtonloop:0"  # the ordinary failed-step path


def test_recoverable_error_in_newton_update_fails_the_step(make_newton):
    # e.g. a flash failure in OBL point generation of engine.correct_thermal
    # (thermal-variable correction of a PH/PS formulation) during the update
    solver, model, engine, log = make_newton(res_seq=[1.0, 1e-1, 1e-9])
    correct_thermal = engine.correct_thermal

    def failing_correct_thermal():
        raise RecoverableEvaluationError("thermal correction flash failure")

    engine.correct_thermal = failing_correct_thermal
    assert solver.run_timestep(1.0, 0.0) is False
    assert "thermal correction flash failure" in solver.status.divergence_reason
    assert engine.n_apply_update == 0
    assert log[-1] == "post_newtonloop:0"
    engine.correct_thermal = correct_thermal


def test_recoverable_error_in_pre_routine_fails_the_step(make_newton):
    spec = NewtonSpec()

    def reject(solver, dt, t, iteration):
        raise RecoverableEvaluationError("residual guard")

    spec.pre_routines.append(reject)
    solver, model, engine, log = make_newton(spec=spec, res_seq=[1.0, 1e-9])
    assert solver.run_timestep(1.0, 0.0) is False
    assert engine.n_apply_update == 0  # raised before the first update
    assert "residual guard" in solver.status.divergence_reason
    assert log[-1] == "post_newtonloop:0"


def test_stop_dangling_timers_stops_only_running_timers():
    # an exception unwinding through C++ assembly leaves its timers started;
    # the Newton driver must stop exactly those (once or repeatedly started)
    import time

    from darts.engines import timer_node
    from darts.nonlinear_solvers.newton import NewtonSolver

    root = timer_node()
    for name in ("idle", "once", "twice"):
        root.node[name] = timer_node()
    root.node["once"].node["child"] = timer_node()
    root.node["idle"].start()
    root.node["idle"].stop()
    root.node["once"].start()
    root.node["once"].node["child"].start()
    root.node["twice"].start()
    root.node["twice"].start()
    idle_before = root.node["idle"].get_timer()

    NewtonSolver._stop_dangling_timers(root)

    frozen = {k: root.node[k].get_timer() for k in ("idle", "once", "twice")}
    frozen["child"] = root.node["once"].node["child"].get_timer()
    time.sleep(0.01)
    assert root.node["idle"].get_timer() == idle_before
    for k in ("idle", "once", "twice"):
        assert root.node[k].get_timer() == frozen[k] >= 0.0
    assert root.node["once"].node["child"].get_timer() == frozen["child"] >= 0.0


def test_other_exceptions_still_propagate(make_newton):
    spec = NewtonSpec()

    def broken(solver, dt, t, iteration):
        raise ValueError("model bug")

    spec.pre_routines.append(broken)
    solver, model, engine, log = make_newton(spec=spec, res_seq=[1.0, 1e-9])
    with pytest.raises(ValueError, match="model bug"):
        solver.run_timestep(1.0, 0.0)


# ------------------------------------------------------ divergence guard
def test_guard_off_by_default_runs_to_the_budget(make_newton):
    # exploding residuals, no thresholds: the historical behaviour -- every
    # iteration is solved and applied until max_iterations
    spec = NewtonSpec(max_iterations=4)
    solver, model, engine, log = make_newton(
        spec=spec, res_seq=[1e-3, 1.0, 1e3, 1e6, 1e9]
    )
    assert solver.run_timestep(1.0, 0.0) is False
    assert solver.status.divergence_reason is None
    assert engine.n_apply_update == 4
    assert solver.stats.n_divergence_aborts == 0


def test_residual_growth_aborts_before_the_solve(make_newton):
    spec = NewtonSpec(max_residual_growth=1e3, residual_growth_floor=1e-6)
    solver, model, engine, log = make_newton(spec=spec, res_seq=[1e-3, 1e-1, 1e2])
    assert solver.run_timestep(1.0, 0.0) is False
    # iterates 1 and 2 were solved and applied (1e-1 is 100x, below 1e3);
    # iterate 3 is 1e5x the smallest: rejected without a linear solve
    assert log.count("solve_linear:0") == 2
    assert engine.n_apply_update == 2
    assert "smallest of this timestep" in solver.status.divergence_reason
    assert solver.status.linear_solver_rc == 0  # not a linear failure
    assert model._linear_solver_rc_last == 0
    assert "divergence guard" in solver._failure_message(1.0)
    assert solver.stats.n_timesteps_wasted == 1
    assert solver.stats.n_divergence_aborts == 1
    assert log[-1] == "post_newtonloop:0"  # the ordinary failed-step path


def test_residual_growth_floor_defaults_to_100x_tolerance(make_newton):
    # 5e-2 is 500x the first residual but below the floor 100 * 1e-3 = 0.1:
    # not judged, and the step then converges
    spec = NewtonSpec(tolerance=1e-3, max_residual_growth=1e2)
    solver, model, engine, log = make_newton(spec=spec, res_seq=[1e-4, 5e-2, 1e-9])
    assert solver.run_timestep(1.0, 0.0) is True
    assert solver.status.divergence_reason is None


def test_max_residual_is_absolute(make_newton):
    spec = NewtonSpec(max_residual=10.0)
    solver, model, engine, log = make_newton(spec=spec, res_seq=[1.0, 50.0])
    assert solver.run_timestep(1.0, 0.0) is False
    assert "max_residual" in solver.status.divergence_reason
    assert log.count("solve_linear:0") == 1


def test_converged_iterate_is_never_guarded(make_newton):
    # the convergence test precedes the guard: a residual below tolerance is
    # accepted even with an impossible growth bound
    spec = NewtonSpec(
        tolerance=1e-3, max_residual_growth=1.0, residual_growth_floor=0.0
    )
    solver, model, engine, log = make_newton(spec=spec, res_seq=[1e-3, 1e-9])
    assert solver.run_timestep(1.0, 0.0) is True


def test_update_ratio_rejects_the_update_before_it_is_applied(make_newton):
    spec = NewtonSpec(max_update_ratio=4.0)
    solver, model, engine, log = make_newton(spec=spec, res_seq=[1.0, 1.0], n_vars=2)
    model.ts_control.eta = np.array([1.0, 0.1])
    engine.X = np.array([[100.0, 0.5], [100.0, 0.5]]).ravel()
    engine.dX = np.array([[0.0, 0.5], [0.0, 0.05]]).ravel()  # 5x eta[1] in block 0
    assert solver.run_timestep(1.0, 0.0) is False
    assert log.count("solve_linear:0") == 1  # the solve happened ...
    assert engine.n_apply_update == 0  # ... the update was never applied
    assert "variable 1" in solver.status.divergence_reason
    assert "5x" in solver.status.divergence_reason
    assert solver.stats.n_divergence_aborts == 1


def test_update_ratio_per_variable_exempts_inf_and_none(make_newton):
    spec = NewtonSpec(max_update_ratio=[None, 4.0])
    solver, model, engine, log = make_newton(spec=spec, res_seq=[1.0, 1e-9], n_vars=2)
    model.ts_control.eta = np.array([1.0, 0.1])
    engine.X = np.array([100.0, 0.5])
    engine.dX = np.array([1e6, 0.1])  # pressure exempt, z within 1x eta
    assert solver.run_timestep(1.0, 0.0) is True
    assert engine.n_apply_update == 1


def test_hypercube_jump_judges_the_raw_direction_on_the_obl_grid(make_newton):
    spec = NewtonSpec(max_hypercube_jump=[None, 10])
    solver, model, engine, log = make_newton(spec=spec, res_seq=[1.0, 1e-9], n_vars=2)
    model.physics.axes_origin = [0.0, 0.0]
    model.physics.axes_step = [1.0, 0.01]
    engine.X = np.array([5.0, 0.5])
    engine.dX = np.array([0.0, -0.5])  # proposes z = 1.0: 50 cubes away
    assert solver.run_timestep(1.0, 0.0) is False
    assert "hop 50 OBL hypercubes along axis 1" in solver.status.divergence_reason
    assert engine.n_apply_update == 0
    # the obl_bounds box the engine clamps into does NOT hide the excursion: the
    # raw direction is judged (parking the block on the box face would make the
    # hop look like 10 cubes and let the diverging update through)
    spec2 = NewtonSpec(
        max_hypercube_jump=[None, 10],
        obl_bounds=OBLBoundsSpec(mode="obl_axes", axis_min=None, axis_max=[None, 0.6]),
    )
    solver, model, engine, log = make_newton(spec=spec2, res_seq=[1.0, 1e-9], n_vars=2)
    model.physics.axes_origin = [0.0, 0.0]
    model.physics.axes_step = [1.0, 0.01]
    engine.X = np.array([5.0, 0.5])
    engine.dX = np.array([0.0, -0.5])
    assert solver.run_timestep(1.0, 0.0) is False
    assert "hop 50 OBL hypercubes along axis 1" in solver.status.divergence_reason
    engine.dX = np.array([0.0, -0.1])  # 10 cubes: allowed (bound is inclusive)
    assert solver.run_timestep(1.0, 0.0) is True


def test_nonfinite_update_is_rejected(make_newton):
    spec = NewtonSpec(max_update_ratio=4.0)
    solver, model, engine, log = make_newton(spec=spec, res_seq=[1.0, 1.0], n_vars=2)
    engine.X = np.array([1.0, 0.5])
    engine.dX = np.array([np.nan, 0.0])
    assert solver.run_timestep(1.0, 0.0) is False
    assert "not finite" in solver.status.divergence_reason
    assert engine.n_apply_update == 0


def test_per_variable_bound_length_mismatch_raises(make_newton):
    spec = NewtonSpec(max_update_ratio=[1.0, 2.0, 3.0])
    solver, model, engine, log = make_newton(spec=spec, res_seq=[1.0, 1.0], n_vars=2)
    engine.X = np.array([1.0, 0.5])
    engine.dX = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="expected n_vars = 2"):
        solver.run_timestep(1.0, 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_residual=0.0),
        dict(max_residual_growth=-1.0),
        dict(residual_growth_floor=float("nan")),
        dict(max_update_ratio=[1.0, 0.0]),
        dict(max_hypercube_jump=-5),
    ],
)
def test_guard_spec_validation(kwargs):
    with pytest.raises(ValueError):
        NewtonSpec(**kwargs)


def test_guard_fields_are_keyword_only():
    with pytest.raises(TypeError):
        NewtonSpec(
            1e-3,
            100.0,
            20,
            1e-3,
            Norm.L2,
            1,
            [],
            [],
            [],
            ChopSpec(),
            OBLBoundsSpec(),
            10.0,
        )


def test_guard_spec_serializes():
    spec = NewtonSpec(max_residual_growth=1e3, max_update_ratio=[None, 4.0])
    d = spec.to_dict()
    assert d["max_residual_growth"] == 1e3
    assert d["max_update_ratio"] == [None, 4.0]
    assert spec.divergence_guard_active
    assert not NewtonSpec().divergence_guard_active


# ----------------------------------------------------------------- F3 hooks
def test_on_iteration_called_once_per_iteration(make_newton):
    calls = []

    class SpySolver(NewtonSolver):
        def on_iteration(self, iteration, dt, t):
            calls.append((iteration, dt, t))

    _, model, engine, _ = make_newton()  # build fake infra
    solver = SpySolver(NewtonSpec())
    solver.bind(model)
    solver.run_timestep(0.5, 10.0)
    # residuals [1e-3, 1e-9]: i=0 (no converge), i=1 (converge) -> 2 calls
    assert [c[0] for c in calls] == [0, 1]
    assert calls[0][1] == 0.5 and calls[0][2] == 10.0


def test_compute_residual_hooks_are_used(make_newton):
    """The loop must route residuals through the overridable hooks."""
    seen = {"res": 0, "well": 0}

    class SpySolver(NewtonSolver):
        def compute_reservoir_residual(self):
            seen["res"] += 1
            return super().compute_reservoir_residual()

        def compute_well_residual(self):
            seen["well"] += 1
            return super().compute_well_residual()

    _, model, engine, _ = make_newton()
    solver = SpySolver(NewtonSpec())
    solver.bind(model)
    solver.run_timestep(1.0, 0.0)
    assert seen["res"] >= 1 and seen["well"] >= 1


# --------------------------------------------------- F6 pre/update/post order
def test_pre_update_post_ordering(make_newton):
    spec = NewtonSpec()
    solver, model, engine, elog = make_newton(spec=spec, res_seq=[1.0, 1e-12])
    # share the engine's event log so pre/post routines interleave with kernels
    spec.pre_routines = [lambda s, dt, t, i: elog.append("pre")]
    spec.post_routines = [lambda s, dt, t, i: elog.append("post")]
    solver.run_timestep(1.0, 0.0)
    # one full update cycle at i=0, then i=1 converges & breaks before update
    order = [
        e for e in elog if e in ("pre", "correct_composition", "apply_update", "post")
    ]
    assert order == ["pre", "correct_composition", "apply_update", "post"]
    # default chop mode is 'local'; obl mode None => NO correct_obl_axes step (F2)
    assert "correct_chop_local" in elog
    assert "correct_obl_axes" not in elog


# ---------------------------------------------------- F5 linear-solve results
def test_linear_success_counts_n_linear(make_newton):
    solver, model, engine, _ = make_newton(res_seq=[1.0, 1e-12], solve_rcs=[0])
    solver.run_timestep(1.0, 0.0)
    assert solver.status.n_linear == engine.get_last_linear_iters()  # one solve


@pytest.mark.parametrize("rc,reason", [(1, "setup"), (2, "solve")])
def test_linear_failure_aborts_and_marks_wasted(make_newton, rc, reason):
    solver, model, engine, _ = make_newton(res_seq=[1.0, 1.0], solve_rcs=[rc])
    converged = solver.run_timestep(1.0, 0.0)
    assert converged is False
    assert solver.status.linear_solver_rc == rc
    assert model._linear_solver_rc_last == rc
    assert engine.n_apply_update == 0  # never applied a stale update
    assert reason in solver._failure_message(1.0)
    assert solver.stats.n_timesteps_wasted == 1


def test_linear_nonconvergence_accept_policy(make_newton):
    # rc 3 = "budget exhausted, iterate usable" (unified solve() convention).
    # Policy 'accept': the step is applied, iterations are counted, the
    # occurrence is recorded, and the timestep converges on the Newton gate.
    spec = NewtonSpec()
    spec.on_linear_nonconvergence = "accept"
    solver, model, engine, _ = make_newton(
        spec=spec, res_seq=[1.0, 1e-12], solve_rcs=[3]
    )
    converged = solver.run_timestep(1.0, 0.0)
    assert converged is True
    assert solver.status.linear_solver_rc == 0  # accepted: not a failure
    assert solver.status.n_linear_nonconverged == 1
    assert solver.status.n_linear == engine.get_last_linear_iters()
    assert engine.n_apply_update >= 1  # the inexact step WAS applied


def test_linear_nonconvergence_cut_by_default(make_newton):
    # Default policy 'cut': the non-converged solve fails the attempt and the
    # inexact update is never applied.
    solver, model, engine, _ = make_newton(res_seq=[1.0, 1.0], solve_rcs=[3])
    converged = solver.run_timestep(1.0, 0.0)
    assert converged is False
    assert solver.status.linear_solver_rc == 3
    assert solver.status.n_linear_nonconverged == 1
    assert engine.n_apply_update == 0  # never applied
    assert "did not converge" in solver._failure_message(1.0)


def test_on_linear_nonconvergence_validation():
    assert NewtonSpec().on_linear_nonconvergence == "cut"
    spec = NewtonSpec()
    spec.on_linear_nonconvergence = "abort"  # invalid
    with pytest.raises(ValueError, match="on_linear_nonconvergence"):
        spec.validate()


def test_on_linear_nonconvergence_is_keyword_only():
    # The field was appended to the BASE spec; making it positional would shift
    # every inherited positional argument (NewtonSpec.chop was the 10th) and
    # silently rebind existing call sites.
    spec = NewtonSpec(1e-3, 100.0, 20, 1e-3, Norm.L2, 1, [], [], [], ChopSpec())
    assert isinstance(spec.chop, ChopSpec)
    assert spec.on_linear_nonconvergence == "cut"
    assert (
        NewtonSpec(on_linear_nonconvergence="accept").on_linear_nonconvergence
        == "accept"
    )


@pytest.mark.parametrize("rc", [1, 2])
def test_hard_failure_ignores_accept_policy(make_newton, rc):
    # 'accept' must apply ONLY to the usable-iterate status (3). A setup (1) or
    # hard solve (2) failure still aborts, whatever the policy says.
    spec = NewtonSpec()
    spec.on_linear_nonconvergence = "accept"
    solver, model, engine, _ = make_newton(
        spec=spec, res_seq=[1.0, 1.0], solve_rcs=[rc]
    )
    assert solver.run_timestep(1.0, 0.0) is False
    assert solver.status.linear_solver_rc == rc
    assert engine.n_apply_update == 0
    assert solver.status.n_linear_nonconverged == 0


def test_nonconvergence_then_success_counts_both(make_newton):
    # A non-converged solve followed by a converged one: iterations from both
    # are accounted, and only the first is counted as non-converged. Needs the
    # 'accept' policy: under the default 'cut' the first solve ends the attempt.
    spec = NewtonSpec()
    spec.on_linear_nonconvergence = "accept"
    solver, model, engine, _ = make_newton(
        spec=spec, res_seq=[1.0, 1.0, 1e-12], solve_rcs=[3, 0]
    )
    assert solver.run_timestep(1.0, 0.0) is True
    assert solver.status.n_linear_nonconverged == 1
    assert solver.status.n_linear == 2 * engine.get_last_linear_iters()


# -------------------------------------------------------------- F2 OBL modes
def test_build_corrections_obl_modes(make_newton):
    # mode=None (default): no OBL step at all
    solver, model, engine, _ = make_newton(spec=NewtonSpec())
    names = [getattr(s, "__name__", "partial") for s in solver.build_corrections()]
    assert "correct_obl_axes" not in names

    # mode='obl_axes' without explicit bounds: the no-arg kernel is appended
    solver2, *_ = make_newton(
        spec=NewtonSpec(obl_bounds=OBLBoundsSpec(mode="obl_axes"))
    )
    names2 = [getattr(s, "__name__", "partial") for s in solver2.build_corrections()]
    assert "correct_obl_axes" in names2


def test_obl_axis_length_mismatch_raises(make_newton):
    solver, model, engine, _ = make_newton(
        spec=NewtonSpec(
            obl_bounds=OBLBoundsSpec(mode="obl_axes", axis_min=[0.0])  # n_vars=2
        ),
        n_vars=2,
    )
    with pytest.raises(ValueError):
        solver.build_corrections()


# ------------------------------------------------------- F7 fallback subclass
def test_fallback_preserves_runtime_subclass(make_newton):
    class MySolver(NewtonSolver):
        pass

    _, model, engine, _ = make_newton()
    solver = MySolver(NewtonSpec())
    solver.bind(model)
    # same-spec retry (solver=None) must reconstruct MySolver, not plain NewtonSolver
    same = solver._make_fallback_solver(FallbackSpec())
    assert type(same) is MySolver
    # explicit alternative spec builds that spec's default runtime class
    other = solver._make_fallback_solver(FallbackSpec(solver=NewtonSpec()))
    assert type(other) is NewtonSolver


def test_fallback_solve_timestep_succeeds(make_newton):
    spec = NewtonSpec(max_iterations=1, fallbacks=[FallbackSpec()])
    # primary consumes [1.0, 1.0] (fails at max_iter); fallback consumes
    # [1.0, 1e-12] (converges). Shared FakeEngine residual cursor advances.
    solver, model, engine, _ = make_newton(
        spec=spec, res_seq=[1.0, 1.0, 1.0, 1e-12], solve_rcs=[0, 0, 0, 0]
    )
    converged = solver.solve_timestep(1.0, 0.0)
    assert converged is True
    assert solver.stats.n_timesteps_wasted >= 1  # the primary attempt was wasted


# ------------------------------------------------------------ F9 NaN/Inf + serialization
@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_nonfinite_residual_rejected(make_newton, bad):
    solver, model, engine, _ = make_newton()
    solver.status.newton_residual = bad
    solver.status.well_residual = 0.0
    assert solver.converged() is False
    assert "not finite" in solver._failure_message(1.0)


def test_spec_to_dict_roundtrips_nested():
    spec = NewtonSpec(tolerance=1e-4, chop=ChopSpec(mode="global", factor=0.3))
    d = spec.to_dict()
    assert d["tolerance"] == 1e-4
    assert d["chop"]["mode"] == "global" and d["chop"]["factor"] == 0.3
    assert not hasattr(spec, "model_dump")  # stdlib dataclass, no pydantic today


# ------------------------------------------------------------- F12 validation
def test_spec_validation_errors():
    with pytest.raises(ValueError):
        ChopSpec(mode="bogus")
    with pytest.raises(ValueError):
        ChopSpec(factor=0.0)
    with pytest.raises(ValueError):
        NewtonSpec(tolerance=-1.0)
    with pytest.raises(ValueError):
        NewtonSpec(max_iterations=0)
    with pytest.raises(ValueError):
        NewtonSpec(coupled_well_res_norm_method=3)
    with pytest.raises(TypeError):
        NewtonSpec(norm="L2")  # must be a Norm, not a str
    with pytest.raises(ValueError):
        OBLBoundsSpec(mode="obl_axes", axis_min=[1.0, 5.0], axis_max=[2.0, 3.0])
    with pytest.raises(ValueError):
        OBLBoundsSpec(mode="bogus")


def test_validate_catches_post_construction_mutation():
    spec = NewtonSpec()
    spec.chop.factor = -1.0  # bypasses __post_init__
    with pytest.raises(ValueError):
        spec.validate()


# ---------------------------------------------------------- F13 enum sourcing
def test_sync_to_engine_uses_canonical_enum_ints(make_newton):
    from darts.engines import sim_params

    solver, model, engine, _ = make_newton(
        spec=NewtonSpec(norm=Norm.LINF, chop=ChopSpec(mode="global", factor=0.2))
    )
    solver.spec.sync_to_engine(engine)
    assert engine.residual_norm_type == int(sim_params.LINF)
    assert engine.newton_chop_mode == int(sim_params.newton_global_chop)


# ------------------------------------------- F1 linear-spec -> sim_params (!280)
def test_linear_spec_settings_reach_params():
    """After !280 the linear settings are owned by ``linear_solver.spec`` (they
    were transitional ``ts_control.linear_*`` attributes before) and are mirrored
    into sim_params by ``LinearSolver._sync_solver_to_sim_params``."""
    import types

    from darts.engines import sim_params
    from darts.linear_solvers import GMRESSolverSpec, LinearSolver

    # a minimal stand-in model: the sync path only reads model.params and
    # model.linear_solver_from_engine_factory (via getattr)
    m = types.SimpleNamespace(params=sim_params())
    m.linear_solver = LinearSolver(
        GMRESSolverSpec(tolerance=7.5e-9, max_iterations=777), model=m
    )
    m.linear_solver._sync_solver_to_sim_params()
    assert m.params.tolerance_linear == 7.5e-9
    assert m.params.max_i_linear == 777


# --------------------------------------------------------- F10 legacy shim
def test_set_sim_params_legacy_kwargs_map_and_warn():
    import types

    from darts.linear_solvers import LinearSolver

    m = types.SimpleNamespace(nonlinear_solver=NewtonSolver(NewtonSpec()))
    ls = LinearSolver(model=m)
    with pytest.warns(DeprecationWarning):
        ls._migrate_legacy_solver_kwargs(
            {
                "tol_newton": 1e-4,
                "it_newton": 7,
                "newton_type": 1,  # legacy int -> 'global'
                "newton_params": [0.25],
                "coupled_well_res_norm_method": 2,
            },
        )
    s = m.nonlinear_solver.spec
    assert s.tolerance == 1e-4
    assert s.max_iterations == 7
    assert s.chop.mode == "global"
    assert s.chop.factor == 0.25
    assert s.coupled_well_res_norm_method == 2
    # a genuine typo still fails loudly
    with pytest.raises(TypeError):
        ls._migrate_legacy_solver_kwargs({"bogus": 1})


# --------------------------------------------- MechanicsNewtonSolver (F4 Tier2)
def test_mechanics_per_component_convergence(make_mechanics):
    # default tol=1e-3; i=0 not converged (does an update), i=1 all components
    # below tol and well 0 < well_mult*tol -> converge
    solver, model, engine, _ = make_mechanics(
        dev_seq=[(1.0, 1.0), (1e-9, 1e-9)], well_seq=[0.0, 0.0]
    )
    converged = solver.run_timestep(1.0, 0.0)
    assert converged  # post_newtonloop passes the verdict through
    assert engine.n_apply_update == 1  # one update at i=0, then i=1 converges
    assert solver.status.n_newton == 1
    assert "apply_newton_update" in engine.log  # C++ composite, not the Python update


def test_mechanics_break_before_update_on_failed_solve(make_mechanics):
    solver, model, engine, _ = make_mechanics(
        dev_seq=[(1.0, 1.0)], well_seq=[0.0], solve_rcs=[1]
    )
    converged = solver.run_timestep(1.0, 0.0)
    assert not converged
    assert solver.status.linear_solver_rc == 1
    assert model._linear_solver_rc_last == 1
    assert engine.n_apply_update == 0  # no stale update after a failed solve


def test_mechanics_early_break_and_finalize_hooks(make_mechanics):
    from darts.nonlinear_solvers import MechanicsNewtonSolver

    events = []

    class FaultLike(MechanicsNewtonSolver):
        def check_early_break(self, i):
            events.append("early")
            return True  # fail the timestep before the linear solve

        def finalize_convergence(self, converged):
            events.append("finalize")
            return 0  # veto

    solver, model, engine, _ = make_mechanics(
        dev_seq=[(1.0, 1.0)], well_seq=[0.0], solver_cls=FaultLike
    )
    converged = solver.run_timestep(1.0, 0.0)
    assert not converged
    assert engine.n_apply_update == 0  # broke before the update
    assert "early" in events and "finalize" in events
    assert "solve_linear:0" not in engine.log  # never reached the solve


def test_mechanics_thermal_third_component(make_mechanics):
    model_thermo = True
    solver, model, engine, _ = make_mechanics(
        dev_seq=[(1e-9, 1e-9, 1.0), (1e-9, 1e-9, 1e-9)], well_seq=[0.0, 0.0]
    )
    model.reservoir.thermoporoelasticity = model_thermo
    solver.run_timestep(1.0, 0.0)
    # i=0: third=1.0 >= tol -> NOT converged (does an update); i=1: third tiny -> converge
    assert engine.n_apply_update == 1
    assert engine.dev_e == 1e-9  # thermal component recorded on the engine


def test_stats_line_goes_to_the_darts_stream_and_to_stdout(capsys, monkeypatch):
    """The totals of a run() invocation: print() reaches stdout only, so the log
    of a redirected run needs the darts output stream as well."""
    from darts.engines import logging
    from darts.nonlinear_solvers.base import SolverStats

    logged = []
    monkeypatch.setattr(logging, "info", logged.append)
    stats = SolverStats()
    stats.n_timesteps_total, stats.n_newton_total, stats.n_linear_total = 21, 28, 332
    line = "----- TS = 21(0), NI = 28(0), LI = 332(0) -----"

    assert stats.format() == line
    stats.log()
    assert logged == [line]  # darts output stream -> run.log
    assert capsys.readouterr().out == ""
    stats.print()
    assert capsys.readouterr().out == line + "\n"  # console
