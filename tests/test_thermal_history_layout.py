"""
Check primary-only thermal states against history-aware property containers.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from darts.interpolators import value_vector
from darts.physics.base.operator_evaluator import OperatorsBase, ThermalVarOperator


@pytest.mark.parametrize("n_history", [0, 1, 2])
@pytest.mark.parametrize("is_pt", [False, True])
@pytest.mark.parametrize("composition", [(0.2, 0.3), (0.6, 0.45)])
def test_thermal_state_excludes_container_history(
    n_history: int, is_pt: bool, composition: tuple[float, float]
) -> None:
    """
    Preserve temperature and primary-only reference states during extrapolation.
    """
    reference_states = []

    def enthalpy(state_pt: value_vector) -> float:
        """
        Record the PT input and return a linear mock enthalpy.
        """
        state = state_pt.to_numpy().copy()
        reference_states.append(state)
        assert len(state) == 4
        assert state[3] == 350.0
        assert state[1] + state[2] <= 1.0 + 1e-12
        return 2.0 * state[3] + state[1] + 3.0 * state[2]

    prop = SimpleNamespace(
        nc=3,
        nph=2,
        eps_z=1e-10,
        n_history=n_history,
        compute_total_enthalpy=enthalpy,
    )
    thermal = ThermalVarOperator(prop, thermal=True, is_pt=is_pt, dz=0.01)
    regular = OperatorsBase(prop, thermal=True, dz=0.01)
    assert regular.n_history == n_history
    assert thermal.n_history == 0
    values = value_vector([0.0])
    thermal.evaluate(value_vector([50.0, *composition, 350.0]), values)
    expected = 350.0 if is_pt else 700.0 + composition[0] + 3.0 * composition[1]
    np.testing.assert_allclose(values.to_numpy(), [expected])
    assert prop.n_history == n_history
    if not is_pt:
        assert reference_states
