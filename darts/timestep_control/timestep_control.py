import warnings

import numpy as np


class TimestepControl:
    """Timestep-control parameters.

    Holds ONLY the timestep controls (``dt_first``/``dt_min``/``dt_mult``/
    ``dt_max``/``eta``) plus the total ``runtime``. Neither solver's settings
    are mirrored here — each lives at its own single source of truth:
    ``DartsModel.nonlinear_solver.spec`` (a
    :class:`darts.nonlinear_solvers.NonlinearSolverSpec`, !327) and
    ``DartsModel.linear_solver.spec`` (a
    :class:`darts.linear_solvers.LinearSolverSpec`, !280 — the transitional
    ``linear_*`` attributes this structure carried are now removed).

    This is the timestepping analogue of the validated, serializable nonlinear
    and linear solver specs: :meth:`validate` and :meth:`to_dict` mirror
    ``NonlinearSolverSpec``'s, so the effective timestepping configuration is
    serializable and inspectable (see :meth:`DartsModel.print_config`).
    """

    #: per-DOF ``eta`` value meaning "this variable does not restrict the timestep"
    ETA_UNCONSTRAINED = 1e20

    _FIELDS = (
        "eta",
        "dt_first",
        "dt_min",
        "dt_mult",
        "dt_max",
        "runtime",
        "regrowth_hold_steps",
        "regrowth_factor",
        "regrowth_cap",
    )

    def __init__(self, n_vars=0):
        # timestep control (owned by this structure)
        self.eta = (
            self.ETA_UNCONSTRAINED * np.ones(n_vars)
        )  # controls the timestep by the variable change from the previous newton iteration
        # dX = Xn - X. Eta has a size of number of DOFs per cell. Set to a large value by default, so doesn't affect the timestep choice
        self.dt_first = 1.0  # initial timestep [days]
        # minimal allowed timestep [days] = the divergence-abort floor. NOTE: a
        # known discrepancy remains -- set_sim_params(min_ts=) defaults to 1e-15,
        # so a model configured through set_sim_params (without an explicit min_ts)
        # floors lower than one that constructs TimestepControl directly. This value is left
        # at 1e-12 deliberately (behaviour-preserving); the *effective* value is now
        # surfaced by DartsModel.print_config(). Unifying the two paths is a follow-up
        # that requires re-baselining the models that ride the abort floor.
        self.dt_min = 1e-12
        self.dt_mult = 2.0  # timestep multiplier, affects the next timestep choice
        self.dt_max = 10.0  # maximal allowed timestep [days]
        self.runtime = (
            1000.0  # total runtime [days]; read by run() when days is not given
        )

        # Regrowth limiter after a FAILED timestep (off by default). The plain
        # controller retries at dt/dt_mult and, once that is accepted, regrows by
        # dt_mult straight back into the size that just failed -- a fail/accept
        # alternation where every failed attempt costs a full nonlinear solve.
        # With regrowth_hold_steps > 0 the next `regrowth_hold_steps` accepted
        # steps stay below regrowth_cap * dt_failed and grow by at most
        # regrowth_factor per accepted step. dt_max itself is never modified:
        # DartsModel.run() takes the effective bound from effective_dt_max().
        self.regrowth_hold_steps = 0
        self.regrowth_factor = 1.25
        self.regrowth_cap = 0.75
        self._hold_left = 0  # accepted steps still under the limiter
        self._failed_dt = 0.0  # dt of the last failed attempt
        self._ref_dt = 0.0  # growth reference since that failure (see notify_result)

        # Per-timestep run history (not configuration -- excluded from _FIELDS /
        # to_dict() / validate() / print()). Appended once per accepted timestep by
        # NewtonSolver.run_timestep(); read back for post-run plotting/diagnostics.
        self.time = []  # accepted timestep end times [days]
        self.time_step_size = []  # accepted timestep sizes [days]

    def resize(self, n_vars: int):
        """Grow :attr:`eta` to ``n_vars`` entries; never shrinks.

        ``__init__`` sizes ``eta`` from its ``n_vars`` argument, but a structure
        built before the physics exists (``DartsModel.__init__`` constructs one
        with ``n_vars=0``) only learns the real DOF count later. This is that
        deferred sizing: ``DartsModel.init()`` calls it once ``physics.n_vars`` is
        known, before and after ``set_solver()``.

        Entries already set are carried over rather than discarded, so a
        ``ts_control.eta[i] = ...`` written before the structure was sized
        survives; the new tail defaults to :attr:`ETA_UNCONSTRAINED`. Idempotent.
        """
        eta = np.asarray(self.eta, dtype=float)
        if eta.size >= n_vars:
            return
        grown = self.ETA_UNCONSTRAINED * np.ones(n_vars)
        grown[: eta.size] = eta
        self.eta = grown

    def notify_result(self, dt: float, converged: bool, clipped: bool = False):
        """Record how the attempt at ``dt`` ended; drives the regrowth limiter
        (called by ``DartsModel.run()`` after every attempt). ``clipped`` marks an
        accepted step that was shortened to end exactly at the stop time of
        ``run()``: it counts as an accepted step but never lowers the growth
        reference, so the next interval still regrows from the last full-size step."""
        if self.regrowth_hold_steps <= 0:
            self._hold_left = 0
            return
        if not converged:
            self._hold_left = int(self.regrowth_hold_steps)
            self._failed_dt = float(dt)
            self._ref_dt = 0.0
        elif self._hold_left > 0:
            self._hold_left -= 1
            # regrow by regrowth_factor from the step just accepted; a step
            # clipped to the stop time keeps the previous, full-size reference
            self._ref_dt = max(self._ref_dt, float(dt)) if clipped else float(dt)

    def effective_dt_max(self) -> float:
        """:attr:`dt_max`, lowered while the regrowth limiter holds the step down
        after a failure (see :meth:`notify_result`)."""
        if self._hold_left <= 0:
            return self.dt_max
        cap = self.regrowth_cap * self._failed_dt
        if self._ref_dt > 0.0:
            cap = min(cap, self.regrowth_factor * self._ref_dt)
        return min(self.dt_max, cap)

    def validate(self):
        """Raise ``ValueError`` on an obviously-invalid timestepping configuration
        (mirror of ``NonlinearSolverSpec.validate()``); called at ``init()``."""
        if self.dt_first <= 0:
            raise ValueError(f"ts_control.dt_first must be > 0, got {self.dt_first}")
        if self.dt_min <= 0:
            raise ValueError(f"ts_control.dt_min must be > 0, got {self.dt_min}")
        if self.dt_max < self.dt_min:
            raise ValueError(
                f"ts_control.dt_max ({self.dt_max}) must be >= dt_min ({self.dt_min})"
            )
        if self.dt_mult < 1.0:
            raise ValueError(f"ts_control.dt_mult must be >= 1, got {self.dt_mult}")
        if self.runtime <= 0:
            raise ValueError(f"ts_control.runtime must be > 0, got {self.runtime}")
        if (
            int(self.regrowth_hold_steps) != self.regrowth_hold_steps
            or self.regrowth_hold_steps < 0
        ):
            raise ValueError(
                f"ts_control.regrowth_hold_steps must be a non-negative integer, got {self.regrowth_hold_steps}"
            )
        if self.regrowth_hold_steps > 0:
            if not self.regrowth_factor > 1.0:
                raise ValueError(
                    f"ts_control.regrowth_factor must be > 1, got {self.regrowth_factor}"
                )
            if not 0.0 < self.regrowth_cap <= 1.0:
                raise ValueError(
                    f"ts_control.regrowth_cap must be in (0, 1], got {self.regrowth_cap}"
                )

    def to_dict(self):
        """Serializable snapshot (mirror of ``NonlinearSolverSpec.to_dict()``)."""
        return {
            "eta": np.asarray(self.eta).tolist(),
            "dt_first": self.dt_first,
            "dt_min": self.dt_min,
            "dt_mult": self.dt_mult,
            "dt_max": self.dt_max,
            "runtime": self.runtime,
            "regrowth_hold_steps": self.regrowth_hold_steps,
            "regrowth_factor": self.regrowth_factor,
            "regrowth_cap": self.regrowth_cap,
        }

    def print(self):
        print("Timestepping parameters:")
        for k in self._FIELDS:
            print("\t", k, "=", getattr(self, k))

    def set_sim_params(
        self,
        model,
        first_ts: float = None,
        mult_ts: float = None,
        min_ts: float = 1e-15,
        max_ts: float = None,
        runtime: float = 1000,
        **legacy,
    ):
        """
        Deprecated: set the timestep controls on this structure (installing it as
        ``model.ts_control``) and, for one deprecation cycle, migrate removed
        nonlinear/linear solver keyword arguments onto ``model``'s solvers.

        The nonlinear solver parameters are NOT set here anymore — specify them
        on ``model.nonlinear_solver`` (a :class:`darts.nonlinear_solvers.NewtonSolver`),
        typically in a ``set_solver()`` override::

            self.nonlinear_solver = NewtonSolver(tolerance=1e-4, max_iterations=15,
                                                 chop=ChopSpec(mode='local', factor=0.2))

        For one deprecation cycle the removed nonlinear keyword arguments
        (``tol_newton``, ``it_newton``, ``newton_type``, ``newton_params``,
        ``line_search``, ``coupled_well_res_norm_method``) are still accepted:
        they emit a :class:`DeprecationWarning` and are mapped onto
        ``model.nonlinear_solver.spec``. Any other unexpected keyword still raises
        :class:`TypeError`.

        :param first_ts: First timestep
        :type first_ts: float
        :param mult_ts: Timestep multiplier
        :type mult_ts: float
        :param max_ts: Maximum timestep
        :type max_ts: float
        :param runtime: Total runtime in days, default is 1000
        :type runtime: float
        Linear-solver parameters are NOT set here either: the linear solver is
        configured through ``model.linear_solver.spec`` / ``.spec.tolerance`` /
        ``.max_iterations`` in :meth:`DartsModel.set_solver`. For one deprecation
        cycle the removed linear keyword arguments (``tol_linear``, ``it_linear``)
        are accepted on the same terms as the nonlinear ones above.

        .. deprecated::
            Set timestep controls on ``model.ts_control``, the nonlinear solver on
            ``model.nonlinear_solver`` and the linear solver on
            ``model.linear_solver.spec`` instead.
        """
        warnings.warn(
            "set_sim_params() is deprecated; set timestep controls on "
            "DartsModel.ts_control and specify DartsModel.nonlinear_solver in set_solver()",
            DeprecationWarning,
            stacklevel=3,  # blame the set_solver() override, not the LinearSolver delegator
        )
        # Solver settings are NOT set here -- they live on model.nonlinear_solver /
        # model.linear_solver. One-cycle migration of legacy solver kwargs: apply
        # them now if the solvers already exist, otherwise DEFER to set_solver(),
        # which materializes the defaults. Deferring (rather than materializing
        # here) keeps set_sim_params() from re-entering the overridable
        # set_solver() hook -- models call set_sim_params() from their set_solver()
        # override, so calling back would recurse infinitely.
        if legacy:
            # BOTH solvers must exist: the migration reads model.nonlinear_solver.spec
            # and dispatches through model.linear_solver. Reached with linear_solver
            # still None via the direct model.ts_control.set_sim_params(model, ...)
            # entry point, which does not go through the LinearSolver delegator.
            if (
                getattr(model, "nonlinear_solver", None) is not None
                and getattr(model, "linear_solver", None) is not None
            ):
                model.linear_solver._migrate_legacy_solver_kwargs(legacy)
            else:
                model._pending_legacy_solver_kwargs = legacy

        # Time stepping parameters. if None, the current value is kept.
        self.dt_first = first_ts if first_ts is not None else self.dt_first
        self.dt_min = min_ts if min_ts is not None else self.dt_min
        self.dt_max = max_ts if max_ts is not None else self.dt_max
        self.dt_mult = mult_ts if mult_ts is not None else self.dt_mult
        self.runtime = runtime
        model.ts_control = self
        return self
