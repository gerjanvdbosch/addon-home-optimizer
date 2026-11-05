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
