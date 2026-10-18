import numpy as np

from darts.engines import ms_well, timer_node, value_vector
from darts.models.darts_model import DartsModel


class DFMWellModel(DartsModel):
    """
    Base class for models with drift-flux model (DFM) wells.

    In a DFM well (``ms_well.MS_Type.DFM``), the phase velocities at the segment
    interfaces come from a drift-flux model that a :class:`~darts.pipes.pipe.Pipe`
    object evaluates in Python, one pipe per DFM well. Before every assembly of the
    linear system, the Newton solver calls :meth:`update_dfm_well_vels_and_ders`,
    which passes these velocities and their derivatives to the engine. After every
    converged timestep, :meth:`after_converged_timestep` stores the state of each
    pipe, from which the next timestep starts. :meth:`DartsModel.init` raises
    ``TypeError`` for a model with DFM wells that does not derive from this class.

    A derived model adds its DFM wells in :meth:`set_wells` and stores the pipe of each
    of them in :attr:`wells` under the well name::

        def set_wells(self):
            self.wells = {"I1": Pipe("I1", geometry, self.physics, self.reservoir, ic)}
            self.reservoir.add_well("I1", ms_well.MS_Type.DFM, well_geometry=geometry)

    DFM wells run only on the CPU platform. To combine DFM wells with live plots, list
    :class:`~darts.pipes.viz.plot_live.DartsModelWithLivePlots` first among the base
    classes: ``class Model(DartsModelWithLivePlots, DFMWellModel)``.

    :ivar wells: Pipe of each DFM well, keyed by well name
    """

    def __init__(self):
        """
        Initialize the model and the timer of the DFM velocity evaluation.
        """
        super().__init__()
        self.wells = {}
        self.timer.node["simulation"].node["dfm_well_velocity_calculation"] = (
            timer_node()
        )

    def init(self, discr_type: str = "tpfa", platform: str = "cpu", *args, **kwargs):
        """
        Initialize the model as :meth:`DartsModel.init` does, which takes the same
        arguments.

        :param discr_type: Discretization type, passed to :meth:`DartsModel.init`
        :param platform: Platform; must be ``'cpu'``, the only one with DFM wells
        :raises ValueError: If ``platform`` is not ``'cpu'``
        """
        if platform != "cpu":
            raise ValueError(
                f"DFM wells run only on the CPU platform, not on platform={platform!r}."
            )
        super().init(discr_type, platform, *args, **kwargs)

    def set_output(self, *args, **kwargs):
        """
        Set up the output as :meth:`DartsModel.set_output` does, which takes the same
        arguments, and give it the pipes of the DFM wells, whose geometry
        :meth:`~darts.models.output.Output.well_output_to_vtp` writes.
        """
        super().set_output(*args, **kwargs)
        self.output.wells = self.wells

    def after_converged_timestep(self):
        """
        Store the pipe states of the converged timestep, then run
        :meth:`DartsModel.after_converged_timestep`.

        A derived model that overrides this method must call
        ``super().after_converged_timestep()``.
        """
        self.accept_pipe_states()
        super().after_converged_timestep()

    def update_dfm_well_vels_and_ders(self, dt: float, t: float, iter_counter: int):
        """
        Evaluate the phase velocities and their derivatives in every DFM well and pass
        them to the engine.

        The Newton solver calls this method before each assembly of the linear system.

        :param dt: Timestep size [day]
        :param t: Simulation time at the start of the timestep [day]
        :param iter_counter: Newton iteration within the timestep, starting from 0
        """
        self.timer.node["simulation"].node["dfm_well_velocity_calculation"].start()
        for w in self.reservoir.wells:
            if w.ms_type == ms_well.MS_Type.DFM:
                start = w.well_head_idx * self.physics.n_vars
                stop = (w.well_head_idx + w.num_segments) * self.physics.n_vars

                Xn_dfm_well = np.asarray(self.physics.engine.Xn)[start:stop]
                X_dfm_well = np.asarray(self.physics.engine.X)[start:stop]
                well_phase_v, well_phase_v_d = self.wells[
                    w.name
                ].eval_phase_vels_and_ders(Xn_dfm_well, X_dfm_well, dt, t, iter_counter)
                w.phases_vels = value_vector(well_phase_v)
                w.phases_vels_ders = value_vector(well_phase_v_d)
        self.timer.node["simulation"].node["dfm_well_velocity_calculation"].stop()

    def accept_pipe_states(self):
        """
        Store the current state of every DFM pipe as its accepted state.

        The velocity evaluation of the next timestep starts from the accepted state, so
        this method must run after a timestep converges and never after a failed
        attempt; :meth:`after_converged_timestep` calls it.
        """
        if not self.has_dfm_well:
            return

        for w in self.reservoir.wells:
            if w.ms_type == ms_well.MS_Type.DFM:
                self.wells[w.name].accept_pipe_state()
