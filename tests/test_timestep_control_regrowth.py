"""Unit tests of the dt regrowth limiter of TimestepControl (pure Python)."""

import pytest

from darts.timestep_control.timestep_control import TimestepControl


def _ts(hold=8, factor=1.25, cap=0.75, dt_max=1e-3):
    ts = TimestepControl(n_vars=2)
    ts.dt_max = dt_max
    ts.regrowth_hold_steps, ts.regrowth_factor, ts.regrowth_cap = hold, factor, cap
    ts.validate()
    return ts


def test_off_by_default():
    ts = TimestepControl(n_vars=2)
    ts.validate()
    ts.notify_result(1e-3, converged=False)
    assert ts.effective_dt_max() == ts.dt_max


def test_failure_caps_then_grows_then_releases():
    ts = _ts(hold=3)
    ts.notify_result(1e-3, converged=False)
    assert ts.effective_dt_max() == pytest.approx(0.75e-3)
    assert ts.dt_max == 1e-3  # never overwritten
    ts.notify_result(0.5e-3, converged=True)  # the retry at dt/2
    assert ts.effective_dt_max() == pytest.approx(min(0.75e-3, 1.25 * 0.5e-3))
    ts.notify_result(0.625e-3, converged=True)
    assert ts.effective_dt_max() == pytest.approx(0.75e-3)  # cap binds first
    ts.notify_result(0.75e-3, converged=True)  # third accepted step: released
    assert ts.effective_dt_max() == 1e-3


def test_growth_reference_is_the_step_just_accepted():
    # the bound follows the accepted step both ways: a controller-driven
    # decrease (dt_mult < 1 from eta) lowers it, so regrowth stays gradual
    ts = _ts(hold=8, dt_max=1e-2)
    ts.notify_result(1e-3, converged=False)
    ts.notify_result(0.5e-3, converged=True)
    ts.notify_result(0.2e-3, converged=True)
    assert ts.effective_dt_max() == pytest.approx(1.25 * 0.2e-3)
    ts.notify_result(0.25e-3, converged=True)
    assert ts.effective_dt_max() == pytest.approx(1.25 * 0.25e-3)


def test_clipped_step_does_not_shrink_the_bound():
    # a step shortened to end at the stop time of run() (tiny dt) is an accepted
    # step (the hold counts down) but keeps the previous, full-size reference
    ts = _ts(hold=5)
    ts.notify_result(1e-3, converged=False)
    ts.notify_result(0.5e-3, converged=True)
    ts.notify_result(1e-6, converged=True, clipped=True)  # remainder of an interval
    assert ts.effective_dt_max() == pytest.approx(min(0.75e-3, 1.25 * 0.5e-3))
    assert ts._hold_left == 3
    ts.notify_result(1e-6, converged=True)  # the same step NOT flagged as clipped
    assert ts.effective_dt_max() == pytest.approx(1.25e-6)


def test_clipped_first_step_after_a_failure_is_the_reference():
    ts = _ts(hold=5)
    ts.notify_result(1e-3, converged=False)
    ts.notify_result(0.4e-3, converged=True, clipped=True)  # nothing bigger to keep
    assert ts.effective_dt_max() == pytest.approx(min(0.75e-3, 1.25 * 0.4e-3))


def test_second_failure_restarts_the_hold_at_the_new_dt():
    ts = _ts(hold=2)
    ts.notify_result(1e-3, converged=False)
    ts.notify_result(0.5e-3, converged=True)
    ts.notify_result(0.6e-3, converged=False)
    assert ts.effective_dt_max() == pytest.approx(0.75 * 0.6e-3)


def test_user_dt_max_change_applies_immediately_when_lower():
    ts = _ts(hold=4)
    ts.notify_result(1e-3, converged=False)
    ts.dt_max = 1e-4  # plain assignment, no hand-over needed
    assert ts.effective_dt_max() == 1e-4
    ts.dt_max = 5e-3
    assert ts.effective_dt_max() == pytest.approx(0.75e-3)  # the hold still binds


@pytest.mark.parametrize(
    "hold, factor, cap",
    [
        (-1, 1.25, 0.75),
        (2.5, 1.25, 0.75),
        (3, 1.0, 0.75),
        (3, 1.25, 0.0),
        (3, 1.25, 1.5),
    ],
)
def test_validation(hold, factor, cap):
    ts = TimestepControl(n_vars=1)
    ts.regrowth_hold_steps, ts.regrowth_factor, ts.regrowth_cap = hold, factor, cap
    with pytest.raises(ValueError):
        ts.validate()


def test_to_dict_and_print_include_the_limiter(capsys):
    ts = _ts(hold=3)
    d = ts.to_dict()
    assert (
        d["regrowth_hold_steps"] == 3
        and d["regrowth_factor"] == 1.25
        and d["regrowth_cap"] == 0.75
    )
    ts.print()
    assert "regrowth_hold_steps" in capsys.readouterr().out
