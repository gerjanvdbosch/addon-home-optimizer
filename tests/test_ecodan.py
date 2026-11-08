from features.ecodan import setpoint_bias_k


def test_the_bias_is_what_the_controller_ran_at_on_5_november():
    """The auto-adaptive law inverted for the heat the floor took (flow x
    rise): at 14:10 the bias was -0.8 (room 19.88, setpoint 21, 9 degC); at
    12:00 -0.6 (room 19.31, setpoint 20.5, 8 degC), where rounding up for at
    least the heat gives -0.5."""

    water_w_per_k = 18.0 / 60.0 * 0.998 * 4186.0

    assert setpoint_bias_k(2.6 * water_w_per_k, 18.0, 9.0, 19.88, 21.0, 21.5) == -0.8
    assert setpoint_bias_k(3.4 * water_w_per_k, 18.0, 8.0, 19.31, 20.5, 21.5) == -0.5


def test_the_bias_never_aims_past_the_comfort_band_nor_the_number():
    # The room target at the band's maximum: bias = maximum - setpoint.
    assert setpoint_bias_k(4500.0, 18.0, 8.0, 20.4, 20.0, 20.5) == 0.5
    # A setpoint far below: the number's limit.
    assert setpoint_bias_k(4500.0, 18.0, 8.0, 19.0, 15.0, 21.0) == 2.5


def test_less_than_the_cold_least_rise_aims_just_below_the_room():
    """Past its target the controller asks for the base rise, not the
    cold-widened least (4.1 kW at 2 degC, 18 L/min): 3 kW is asked for with the
    target one step under the room."""

    bias = setpoint_bias_k(3000.0, 18.0, 2.0, 19.18, 21.0, 21.5)

    assert bias == -1.9
    assert 19.18 - 0.1 <= 21.0 + bias < 19.18


def test_the_heat_pumps_least_heat_aims_just_below_the_room():
    """8 Nov 2026, 14:20 (room 19.35, setpoint 20.5, 9 degC, 16 L/min): the
    plan at the compressor's least heat - a fraction of a watt from the solver
    either side of it - is asked for below the room, the base rise; more heat
    through the curve."""

    for heat_w in (2976.0 - 1e-4, 2976.0 + 1e-4):
        bias = setpoint_bias_k(heat_w, 16.0, 9.0, 19.35, 20.5, 21.5, 2976.0)
        assert bias == -1.2
        assert 19.35 - 0.1 <= 20.5 + bias < 19.35

    assert setpoint_bias_k(2976.0, 16.0, 9.0, 19.35, 20.5, 21.5) == -0.7
    assert setpoint_bias_k(3500.0, 16.0, 9.0, 19.35, 20.5, 21.5, 2976.0) == -0.6
