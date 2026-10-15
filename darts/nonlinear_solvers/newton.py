"""Newton-Raphson nonlinear solver: spec and runtime driver.

Holds :class:`NewtonSpec` and the :class:`NewtonSolver` that drives the C++
per-iteration kernels (assembly, linear solve, dX corrections, residual norms).
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from darts.nonlinear_solvers.base import (
    ChopSpec,
    NonlinearSolver,
    NonlinearSolverSpec,
    OBLBoundsSpec,
    RecoverableEvaluationError,
    write_to_log,
)

# Python chop mode -> NAME of the canonical C++ enum member
# (sim_params::newton_solver_t). The integer written to engine.newton_chop_mode
# is read from the compiled enum at sync time (single source of truth; robust to
# a future C++ enum reordering).
_CHOP_MODE_TO_CPP_ENUM = {
    None: "newton_std",
    "global": "newton_global_chop",
    "local": "newton_local_chop",
}
# Reverse map (legacy int -> mode) used only by the set_sim_params(newton_type=..)
# deprecation shim in darts_model.py.
_ENUM_TO_CHOP_MODE = {0: None, 1: "global", 2: "local"}


# ------------------------------------------------------------------ specs


@dataclass
class NewtonSpec(NonlinearSolverSpec):
    """Newton-Raphson solver with the full analytic (OBL-derivative) Jacobian.

    :ivar chop: chopping strategy of the nonlinear update (:class:`ChopSpec`).
    :ivar obl_bounds: restraining of the Newton trajectory (:class:`OBLBoundsSpec`).

    **Divergence guard** (all off by default). A Newton iteration that has left
    the basin of attraction keeps producing iterates; each one costs a full
    assembly -- with adaptive OBL that means supporting-point evaluations at
    states nobody will ever visit -- and a linear solve on a Jacobian that means
    nothing, until ``max_iterations`` is spent. The guard fails the timestep as
    soon as the residual history or the *proposed* update says the iteration has
    diverged, so the step is cut before that cost is paid (the update is judged
    before it is applied). Set the thresholds alongside ``tolerance`` and
    ``max_iterations``; a guard abort goes through the ordinary failed-step path
    (engine rollback, timestep cut, ``stats.n_timesteps_wasted``) and is reported
    in ``status.divergence_reason`` / ``stats.n_divergence_aborts``.

    :ivar max_residual: fail the step when the reservoir residual exceeds this
        absolute value.
    :ivar max_residual_growth: fail the step when the reservoir residual exceeds
        this factor times the SMALLEST residual seen in the timestep so far.
        Against the minimum, not the previous iterate: a healthy early transient
        may overshoot by orders of magnitude before it recovers, and it is the
        monotone growth over several iterates that marks a diverging one. Only
        residuals above :attr:`residual_growth_floor` are judged.
    :ivar residual_growth_floor: residuals at or below this are never judged by
        growth (the ratio of two tiny residuals is noise). ``None``: 100 times
        ``tolerance``.
    :ivar max_update_ratio: bound on ``max|dX_i| / eta_i`` -- the largest change
        of variable *i* the proposed update makes in ANY block, in units of the
        change the timestep control tolerates over a whole timestep
        (``ts_control.eta[i]``). A scalar applies to every variable; a sequence
        gives one bound per state variable (``inf``/``None`` exempts one, e.g.
        the pressure when a boundary switch moves it by a fixed amount whatever
        ``dt`` is). Judged after the linear solve, before the update is applied.
    :ivar max_hypercube_jump: bound on the number of OBL hypercubes ANY block may
        hop along axis *i* in one update (lattice from ``physics.axes_origin`` /
        ``physics.axes_step``). The raw Newton direction is judged, not the update
        the engine applies after clamping it into the :attr:`obl_bounds` box:
        clamping first parks a diverging block on the box face and hides the
        excursion. Scalar or one entry per state variable (``inf``/``None``
        exempts an axis). Each newly entered
        hypercube costs up to ``2**n_vars`` supporting-point evaluations, so a
        wild update is expensive before its residual is even known. Judged
        before the update is applied.
    """

    chop: ChopSpec = field(default_factory=ChopSpec)
    obl_bounds: OBLBoundsSpec = field(default_factory=OBLBoundsSpec)
    # keyword-only (see NonlinearSolverSpec.on_linear_nonconvergence): appending
    # positional fields would rebind existing positional call sites
    max_residual: float | None = field(default=None, kw_only=True)
    max_residual_growth: float | None = field(default=None, kw_only=True)
    residual_growth_floor: float | None = field(default=None, kw_only=True)
    max_update_ratio: float | Sequence[float] | None = field(default=None, kw_only=True)
    max_hypercube_jump: float | Sequence[float] | None = field(
        default=None, kw_only=True
    )

    def validate(self):
        """Validate the base convergence fields and re-validate the sub-specs so
        post-construction mutation (e.g. ``spec.chop.factor = -1``) is caught by
        the per-timestep ``validate()`` call, not just at construction."""
        super().validate()
        # sub-spec __post_init__ bodies are pure validators (raise-or-pass)
        self.chop.__post_init__()
        self.obl_bounds.__post_init__()
        for name in ("max_residual", "max_residual_growth"):
            value = getattr(self, name)
            if value is not None and not value > 0:  # also rejects NaN
                raise ValueError(f"{name} must be > 0 or None, got {value!r}")
        floor = self.residual_growth_floor
        if floor is not None and not floor >= 0:  # 0 = judge every residual
            raise ValueError(
                f"residual_growth_floor must be >= 0 or None, got {floor!r}"
            )
        for name in ("max_update_ratio", "max_hypercube_jump"):
            value = getattr(self, name)
            if value is None:
                continue
            entries = value if isinstance(value, Sequence) else (value,)
            for entry in entries:
                if entry is not None and not entry > 0:
                    raise ValueError(
                        f"{name} entries must be > 0, inf or None, got {value!r}"
                    )

    @property
    def divergence_guard_active(self) -> bool:
        """True when any guard threshold is set."""
        return any(
            getattr(self, name) is not None
            for name in (
                "max_residual",
                "max_residual_growth",
                "max_update_ratio",
                "max_hypercube_jump",
            )
        )

    def per_variable_bound(self, name: str, n_vars: int) -> np.ndarray | None:
        """``max_update_ratio`` / ``max_hypercube_jump`` as an ``n_vars`` array of
        floats (``inf`` where an entry is exempt), or ``None`` when unset."""
        value = getattr(self, name)
        if value is None:
            return None
        if isinstance(value, Sequence):
            if len(value) != n_vars:
                raise ValueError(
                    f"NewtonSpec.{name} has {len(value)} entries, expected n_vars = {n_vars}"
                )
            return np.array([np.inf if v is None else float(v) for v in value])
        return np.full(n_vars, float(value))

    def make_solver(self, model=None) -> "NewtonSolver":
        if self.obl_bounds.mode == "physical":
            raise NotImplementedError(
                "Physical per-axis Newton bounds are not implemented yet"
            )
        return NewtonSolver(self, model=model)

    def sync_to_engine(self, engine):
        """Sync the residual norm (base) plus the Newton kernel controls."""
        super().sync_to_engine(engine)
        from darts.engines import sim_params

        # canonical int comes from the compiled enum (single source of truth)
        engine.newton_chop_mode = int(
            getattr(sim_params, _CHOP_MODE_TO_CPP_ENUM[self.chop.mode])
        )
        engine.newton_chop_factor = self.chop.factor
        engine.log_transform = 1 if self.chop.log_transform else 0


# ------------------------------------------------------------------ solver


class NewtonSolver(NonlinearSolver):
    """Newton-Raphson driver: the Python Newton loop over the C++ kernels.

    Constructed from a :class:`NewtonSpec` (or its keyword arguments) and
    detached from the model until :meth:`bind`; the driving Newton loop is
    ported from ``DartsModel.run_timestep`` and is behaviour-identical to the
    historic implementation.

    :param spec: the :class:`NewtonSpec` to run, or ``None`` to build one from
        the keyword arguments.
    :param model: optional model to bind to at construction (usually left
        ``None`` — the model binds the solver during ``init()``).
    """

    def __init__(self, spec: NewtonSpec = None, model=None, **spec_kwargs):
        if spec is None:
            spec = NewtonSpec(**spec_kwargs)
        elif spec_kwargs:
            raise TypeError(
                "NewtonSolver: pass either a NewtonSpec or its keyword "
                "arguments, not both"
            )
        elif not isinstance(spec, NewtonSpec):
            raise TypeError(
                f"NewtonSolver expects a NewtonSpec, got {type(spec).__name__}"
            )
        super().__init__(spec, model=model)

    def build_corrections(self) -> list:
        """Newton dX-correction pipeline prescribed by the spec, mirroring the
        legacy C++ ``apply_newton_update`` composite: composition correction,
        chop (per :class:`ChopSpec`), OBL-axes clamp (per :class:`OBLBoundsSpec`)
        and thermal-variable correction (self-guarded by the state
        specification).

        OBL-axes behaviour by ``obl_bounds.mode``:

        - ``None`` (default): NO OBL step is appended — the correction is
          skipped entirely, so it never consults engine ``op_axis`` state. (The
          legacy no-arg kernel was a no-op on a clean engine anyway, so default
          models are byte-identical; this additionally prevents a prior bounded
          solve — a fallback or live spec switch — from leaking its bounds into
          a later unbounded solve.)
        - ``'obl_axes'`` WITH ``axis_min``/``axis_max``: the bounds are pushed to
          the C++ kernel (which writes the engine's per-region op_axis bounds).
        - ``'obl_axes'`` WITHOUT bounds: the no-arg kernel is appended (reuse
          whatever op_axis bounds the engine already holds)."""
        engine, spec = self.engine, self.spec
        steps = [engine.correct_composition]
        if spec.chop.mode == "global":
            steps.append(engine.correct_chop_global)
        elif spec.chop.mode == "local":
            steps.append(engine.correct_chop_local)
        obl = spec.obl_bounds
        if obl.mode == "obl_axes" and (
            obl.axis_min is not None or obl.axis_max is not None
        ):
            from functools import partial

            from darts.engines import value_vector

            n_vars = self.model.physics.n_vars
            for name in ("axis_min", "axis_max"):
                bound = getattr(obl, name)
                if bound is not None and len(bound) != n_vars:
                    raise ValueError(
                        f"OBLBoundsSpec.{name} has {len(bound)} entries, "
                        f"expected n_vars = {n_vars}"
                    )
            # None entries leave an axis unbounded (+/-inf never clamps)
            inf = float("inf")
            lo = obl.axis_min if obl.axis_min is not None else [None] * n_vars
            hi = obl.axis_max if obl.axis_max is not None else [None] * n_vars
            axis_min = value_vector([-inf if v is None else v for v in lo])
            axis_max = value_vector([inf if v is None else v for v in hi])
            steps.append(partial(engine.correct_obl_axes, axis_min, axis_max))
        elif obl.mode == "obl_axes":
            # 'obl_axes' without explicit bounds: reuse the engine's op_axis box
            steps.append(engine.correct_obl_axes)
        # obl.mode is None -> append nothing (disabled; never touches engine state)
        steps.append(engine.correct_thermal)
        return steps

    def run_timestep(self, dt: float, t: float, verbose: int | None = None) -> bool:
        """
        Solve the Newton loop for the specified timestep.

        :param dt: Timestep size [days]
        :param t: Current time [days]
        :param verbose: Verbosity level; ``None`` inherits ``model.verbose``.
        :return: True if the timestep converged.
        """
        from darts.models.darts_model import DartsModel

        model = self.model
        engine = self.engine
        spec = self.spec
        verbose = model.verbose if verbose is None else verbose
        assert dt > 0, "Time step size must be a positive value!"
        # revalidate the spec (catches post-construction mutation of any field)
        spec.validate()

        max_newt = spec.max_iterations
        max_residual = np.zeros(max_newt + 1)
        self.status.reset()
        status = self.status
        model._linear_solver_rc_last = 0
        spec.sync_to_engine(engine)
        self._corrections = self.build_corrections()
        self.timer.node["simulation"].start()

        residual_history = self.status.residual_history
        for i in range(max_newt + 1):
            # A RecoverableEvaluationError raised by a property/flash evaluator (OBL
            # supporting-point generation) fails this attempt like the divergence guard.
            try:
                # Update well phase velocities and derivatives if DFM wells are used
                if model.has_dfm_well:
                    model.update_dfm_well_vels_and_ders(dt, t, i)

                # assemble Jacobian and residual of reservoir and well blocks
                engine.assemble_linear_system(dt)

                # apply RHS flux
                model.apply_rhs_flux(dt, t)
            except RecoverableEvaluationError as err:
                self._abort_recoverable(err)
                break

            # The device RHS is authoritative after assembly; re-upload only when a
            # host-side flux modification actually touched the host mirror (!280 --
            # an unconditional copy costs a device transfer every Newton iteration).
            if model.platform == "gpu" and (
                model.has_dfm_well
                or type(model).set_rhs_flux is not DartsModel.set_rhs_flux
            ):
                from darts.engines import copy_data_to_device

                copy_data_to_device(engine.RHS, engine.get_RHS_d())

            # reservoir residual (overridable via compute_reservoir_residual)
            status.newton_residual = self.compute_reservoir_residual()

            max_residual[i] = status.newton_residual
            counter = 0
            for j in range(i):
                denom = max(np.fabs(max_residual[i]), np.finfo(float).eps)
                if (
                    abs(max_residual[i] - max_residual[j]) / denom
                    < spec.stationary_point_tolerance
                ):
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            # well residual (overridable via compute_well_residual)
            status.well_residual = self.compute_well_residual()
            residual_history.append(
                (
                    status.newton_residual,  # matrix residual
                    status.well_residual,  # well residual
                    1.0,
                )
            )  # Newton update coefficient

            status.n_newton = i
            self.on_iteration(i, dt, t)

            #  check tolerance if it converges
            if (
                status.newton_residual < spec.tolerance
                and status.well_residual
                < spec.tolerance * spec.well_tolerance_multiplier
            ) or status.n_newton == max_newt:
                if i > 0:
                    break

            # divergence guard, residual side: do not even solve for an iterate
            # whose residual history says the iteration has diverged
            reason = self._residual_diverged(max_residual[: i + 1])
            if reason is not None:
                self._abort_diverged(reason)
                break

            rc = self._solve_linear()
            if rc != 0:
                # Abort the Newton loop on a failed linear solve without
                # burning the full max_newt budget on stale dX updates.
                status.linear_solver_rc = rc
                model._linear_solver_rc_last = rc
                break
            # divergence guard, update side: judge the proposed update BEFORE it
            # is applied -- its assembly (and the OBL supporting points it would
            # pull in) is the cost that must not be paid
            reason = self._update_diverges()
            if reason is not None:
                self._abort_diverged(reason)
                break
            # pre/post routines and the update (engine.correct_thermal generates OBL
            # points too) may raise RecoverableEvaluationError as well
            try:
                self.pre_iteration(dt, t, i)
                self.update(dt)
                self.post_iteration(dt, t, i)
            except RecoverableEvaluationError as err:
                self._abort_recoverable(err)
                break

        # End of newton loop: the convergence decision is made here and the
        # engine only commits (converged) or rolls back (failed) its state.
        converged = self.converged()
        if not converged:
            write_to_log(self._failure_message(dt))
        converged = engine.post_newtonloop(dt, t, converged)
        self.stats.update(converged, status)
        if converged and not engine.opt_history_matching:
            engine.print_timestep(
                t + dt,
                dt,
                status.n_newton,
                status.n_linear,
                status.newton_residual,
                status.well_residual,
            )

        model.ts_control.time.append(t)
        self.n_newton_iters.append(status.n_newton)
        model.ts_control.time_step_size.append(dt)

        self.timer.node["simulation"].stop()
        return converged

    # -------- divergence guard (NewtonSpec.max_residual & co.)

    def _abort_recoverable(self, err: RecoverableEvaluationError):
        """End the attempt on an evaluation failure of the trial state (e.g. a
        flash that does not converge at an overshooting iterate): same failed-step
        path as the divergence guard (engine rollback, timestep cut)."""
        self._abort_diverged(f"recoverable evaluation error: {err}")
        self._stop_timers_after_abort()

    def _stop_timers_after_abort(self):
        """Stop the timers an exception unwound past before their ``stop()``:
        engine/interpolator nodes of an aborted assembly (below "simulation",
        which itself keeps running until the end of ``run_timestep``) and the
        "newton update" node, whose ``engine.correct_thermal`` step also
        generates OBL supporting points (and can therefore raise)."""
        timers = self.timer.node
        if "simulation" in timers:
            self._stop_dangling_timers(timers["simulation"])
        if "newton update" in timers:
            self._stop_if_running(timers["newton update"])
            self._stop_dangling_timers(timers["newton update"])

    @staticmethod
    def _stop_if_running(timer, probe_seconds: float = 2e-5):
        """Stop ``timer`` while it is running.

        Left running, a timer accumulates wall time until its next ``stop()``
        (inflating the report) and turns negative once it is started again. The
        raw C++ value is not exposed and ``get_timer()`` of a timer started once
        is positive, so a timer is recognised as running when its
        ``get_timer()`` still advances over a short busy-wait; ``stop()`` is
        repeated while it does. Idle timers and GPU event timers (whose value
        changes only in ``stop_gpu()``) are left untouched."""
        get_timer = getattr(timer, "get_timer", None)
        if get_timer is None:
            return
        for _ in range(4):
            before = get_timer()
            deadline = time.perf_counter() + probe_seconds
            while time.perf_counter() < deadline:
                pass
            if get_timer() == before:
                return
            timer.stop()

    @staticmethod
    def _stop_dangling_timers(node, probe_seconds: float = 2e-5):
        """Stop every running descendant of ``node`` (not ``node`` itself)."""
        for child in getattr(node, "node", {}).values():
            NewtonSolver._stop_if_running(child, probe_seconds)
            NewtonSolver._stop_dangling_timers(child, probe_seconds)

    def _abort_diverged(self, reason: str):
        self.status.divergence_reason = reason
        write_to_log(f"divergence guard: {reason}\n")

    def _residual_diverged(self, residuals) -> str | None:
        """Reason to abandon the timestep given the reservoir residuals of the
        iterates so far (the last entry is the current one), or ``None``."""
        spec = self.spec
        current = float(residuals[-1])
        if not np.isfinite(current):
            return f"newton residual {current} is not finite"
        if spec.max_residual is not None and current > spec.max_residual:
            return (
                f"newton residual {current:.3e} > max_residual {spec.max_residual:.3g} "
                f"at iteration {len(residuals)}"
            )
        if spec.max_residual_growth is not None and len(residuals) > 1:
            floor = spec.residual_growth_floor
            if floor is None:
                floor = 100.0 * spec.tolerance
            smallest = float(np.min(residuals[:-1]))
            if (
                current > floor
                and smallest > 0
                and current > spec.max_residual_growth * smallest
            ):
                return (
                    f"newton residual {current:.3e} is {current / smallest:.3g}x the "
                    f"smallest of this timestep (max_residual_growth "
                    f"{spec.max_residual_growth:.3g}) at iteration {len(residuals)}"
                )
        return None

    def _update_diverges(self) -> str | None:
        """Reason to reject the proposed update ``X - dX`` before it is applied
        (see :attr:`NewtonSpec.max_update_ratio` / :attr:`NewtonSpec.max_hypercube_jump`),
        or ``None``. Works on the host copies of ``X`` and ``dX`` (the GPU engine
        refreshes the host ``dX`` after every solve and applies the update from the
        host ``X``)."""
        spec, model = self.spec, self.model
        if spec.max_update_ratio is None and spec.max_hypercube_jump is None:
            return None
        engine = self.engine
        n_vars = model.physics.n_vars
        ratio_bound = spec.per_variable_bound("max_update_ratio", n_vars)
        jump_bound = spec.per_variable_bound("max_hypercube_jump", n_vars)
        update = np.asarray(engine.dX, dtype=float).reshape(-1, n_vars)
        if not np.isfinite(update).all():
            return "the proposed update is not finite"
        if ratio_bound is not None and np.isfinite(ratio_bound).any():
            eta = np.asarray(model.ts_control.eta, dtype=float)
            if eta.size < n_vars:
                raise ValueError(
                    f"ts_control.eta has {eta.size} entries, expected n_vars = {n_vars} "
                    "(needed by NewtonSpec.max_update_ratio)"
                )
            ratio = np.abs(update).max(axis=0) / eta[:n_vars]
            hit = np.flatnonzero(ratio > ratio_bound)
            if hit.size:
                j = int(hit[np.argmax(ratio[hit] / ratio_bound[hit])])
                return (
                    f"update of iteration {self.status.n_newton + 1} rejected: variable "
                    f"{j} would change by {ratio[j]:.3g}x its per-timestep target "
                    f"ts_control.eta[{j}] in one iteration (max_update_ratio "
                    f"{ratio_bound[j]:.3g})"
                )
        if jump_bound is not None and np.isfinite(jump_bound).any():
            physics = model.physics
            origin = np.asarray(physics.axes_origin, dtype=float)[:n_vars]
            step = np.asarray(physics.axes_step, dtype=float)[:n_vars]
            state = np.asarray(engine.X, dtype=float).reshape(-1, n_vars)
            n_blocks = min(state.shape[0], update.shape[0])
            state, update = state[:n_blocks], update[:n_blocks]
            # Judge the RAW Newton direction, not the update the engine will apply
            # after clamping it into the obl_bounds box: a block that wants to
            # leave the box is parked on its face by the clamp and the hop then
            # looks small, so the divergence goes unnoticed.
            proposed = state - update
            hops = np.abs(
                np.floor((proposed - origin) / step) - np.floor((state - origin) / step)
            ).max(axis=0)
            hit = np.flatnonzero(hops > jump_bound)
            if hit.size:
                j = int(hit[np.argmax(hops[hit] / jump_bound[hit])])
                return (
                    f"update of iteration {self.status.n_newton + 1} rejected: a block "
                    f"would hop {int(hops[j])} OBL hypercubes along axis {j} "
                    f"(max_hypercube_jump {jump_bound[j]:.3g})"
                )
        return None
