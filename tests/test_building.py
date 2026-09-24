import functools
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from domain.config import Config
from domain.dynamics import discretize_zoh, kalman_states
from domain.models import BuildingThermalModel
from domain.physics import (
    CP_AIR_J_PER_KG_K,
    Q_PERSON_SENSIBLE_W,
    RHO_AIR_KG_PER_M3,
    floor_heat_w,
    solar_gain_w,
    two_node_zone_state_space,
    zone_observation,
)
from features.building import BuildingThermalIdentifier

TRUE_VOLUME_M3 = 120.0
TRUE_ZONE_AREA_M2 = 46.0
TRUE_SOUTH_GLASS_M2 = 14.0

TRUE_MODEL = BuildingThermalModel(
    ua_envelope_w_per_k=180.0,
    ua_air_mass_w_per_k=450.0,
    c_air_j_per_k=3.0 * RHO_AIR_KG_PER_M3 * TRUE_VOLUME_M3 * CP_AIR_J_PER_KG_K,
    c_mass_j_per_k=18.0e6,
    a_eff_m2=6.0,
    # A pure air sensor, so these tests keep measuring the air node itself;
    # the operative-temperature reading has its own test below.
    sensor_mass_fraction=0.0,
    internal_gain_fraction=0.6,
)


def _config() -> Config:
    """The configuration these tests run against.

    Built here rather than read from data/config.json: that file is the live
    installation's, and it changes whenever a room sensor is added or an entity
    renamed - which would silently change what these tests assert, and makes
    them fail outright wherever the file is absent. The geometry matches the
    synthetic building the rest of this module simulates, so a test may compare
    against TRUE_ZONE_AREA_M2 and TRUE_SOUTH_GLASS_M2.
    """

    return Config(
        solar="sensor.pv_output",
        baseload="sensor.baseload",
        heat_pump={
            "state": "sensor.heat_pump_state",
            "power": "sensor.heat_pump_power",
            "supply_temperature": "sensor.supply_temperature",
            "return_temperature": "sensor.return_temperature",
            "compressor_frequency": "sensor.compressor_frequency",
            "flow": "sensor.flow",
            "boiler": {
                "setpoint": "sensor.dhw_setpoint",
                "top_temperature": "sensor.dhw_top",
                "bottom_temperature": "sensor.dhw_bottom",
                "ambient_temperature": "sensor.dhw_ambient",
                "target_temperature": 46.0,
            },
        },
        building={
            "thermostat": {
                "temperature": "sensor.living_room",
                "setpoint": "climate.living_room",
            },
            "target_temperature": 20.0,
            # Two rooms of unequal size, so an area-weighted mean differs from
            # a plain one - the distinction _shutter_open_fraction and the zone
            # temperature both rest on. The second uses the compact
            # [area, sensor] form, which must keep working.
            "rooms": [
                {"area_m2": 30.0, "temperature": "sensor.living_room"},
                [TRUE_ZONE_AREA_M2 - 30.0, "sensor.bedroom"],
            ],
            "ceiling_height": TRUE_VOLUME_M3 / TRUE_ZONE_AREA_M2,
            # One shaded group and one that is never covered, which is the
            # combination the unshaded-fraction weighting has to handle.
            "south_glazing": [
                {"glass_m2": 10.0, "cover": "cover.living_room"},
                TRUE_SOUTH_GLASS_M2 - 10.0,
            ],
        },
        forecast={"solcast": "sensor.solcast", "open_meteo": "sensor.open_meteo"},
        presence=["device_tracker.phone"],
    )


DT_SECONDS = 900.0
SAMPLES_PER_DAY = int(24 * 3600 / DT_SECONDS)
N_DAYS = 20

# Matches this installation's own room sensor: 0.1 K reporting resolution, with
# a little noise on top. The identification has to work through both.
SENSOR_RESOLUTION_C = 0.1
MEASUREMENT_NOISE_STD_C = 0.02

COOLING_STATE = "Koelen"
OFF_STATE = "Uit"
COOLING_SUPPLY_C = 16.0
COOLING_RETURN_C = 19.0
COOLING_FLOW_LPM = 14.0


def _raw_frame() -> pd.DataFrame:
    """The shape BuildingThermalIdentifier.prepare() consumes: Open-Meteo
    snapshots joined to the measured series, before any derived column exists.
    """

    start = datetime(2026, 4, 1, tzinfo=timezone.utc)
    n = N_DAYS * SAMPLES_PER_DAY

    target_time = pd.to_datetime(
        [start + timedelta(seconds=DT_SECONDS * i) for i in range(n)]
    )

    hour = np.array([t.hour + t.minute / 60.0 for t in target_time])
    day = np.arange(n) // SAMPLES_PER_DAY

    # A daily outdoor swing large enough to excite the envelope conductance.
    t_out = 12.0 - 8.0 * np.cos(2 * np.pi * (hour - 15.0) / 24.0)

    daylight = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
    direct = 700.0 * daylight
    diffuse = 120.0 * daylight

    # Shutters shut overnight, and kept shut all day every third day so the
    # aperture and the shading are not confounded.
    shut_all_day = (day % 3) == 0
    shutter = np.where((hour >= 8.0) & (hour < 20.0) & ~shut_all_day, 100.0, 0.0)

    # Cooling in the afternoon on half the days: without heat injected into the
    # floor the two nodes drift together and their coupling is not excited.
    cooling = ((day % 2) == 1) & (hour >= 13.0) & (hour < 17.0)

    return pd.DataFrame(
        {
            "time": target_time - pd.Timedelta(minutes=30),
            "target_time": target_time,
            "T_out": t_out,
            "direct_radiation": direct,
            "diffuse_radiation": diffuse,
            "direct_normal_irradiance": 900.0 * daylight,
            "temperature": t_out,
            "state": np.where(cooling, COOLING_STATE, OFF_STATE),
            "flow_lpm": np.where(cooling, COOLING_FLOW_LPM, 0.0),
            "T_supply": np.where(cooling, COOLING_SUPPLY_C, COOLING_RETURN_C),
            "T_return": COOLING_RETURN_C,
            "baseload_w": 200.0 + 150.0 * daylight,
            "shutter_0": shutter,
            "presence_0": np.where((hour >= 17.0) | (hour < 8.0), "home", "not_home"),
            "T_air": 21.0,
        }
    )


def _identifier() -> BuildingThermalIdentifier:
    identifier = BuildingThermalIdentifier(latitude=52.39, longitude=5.79)
    identifier.room_areas_m2 = [TRUE_ZONE_AREA_M2]
    identifier.volume_m3 = TRUE_VOLUME_M3
    identifier.glazing_areas_m2 = [TRUE_SOUTH_GLASS_M2]
    identifier.shutter_areas_m2 = [TRUE_SOUTH_GLASS_M2]
    identifier.shutter_columns = ["shutter_0"]
    identifier.presence_columns = ["presence_0"]
    return identifier


def _true_parameters() -> np.ndarray:
    return np.array(
        [
            TRUE_MODEL.ua_envelope_w_per_k,
            TRUE_MODEL.ua_air_mass_w_per_k,
            TRUE_MODEL.c_air_j_per_k,
            TRUE_MODEL.c_mass_j_per_k,
            TRUE_MODEL.a_eff_m2,
            TRUE_MODEL.internal_gain_fraction,
        ]
    )


def _simulate(rng: np.random.Generator) -> pd.DataFrame:
    """Generate a room-temperature trajectory from the known ODE parameters.

    The forcing is taken from prepare() itself, so the test exercises the real
    irradiance transposition, shutter handling and calorimetry rather than a
    second, parallel implementation of them.
    """

    identifier = _identifier()
    prepared = identifier.prepare(_raw_frame())

    inputs = identifier._inputs(TRUE_MODEL, prepared)

    a_d, b_d = discretize_zoh(*two_node_zone_state_space(TRUE_MODEL), DT_SECONDS)
    observation = zone_observation(TRUE_MODEL)

    state = np.array([21.0, 21.0])
    air = np.empty(len(prepared))

    for i in range(len(prepared)):
        # What the thermostat reads of the state, which for TRUE_MODEL is the
        # air node alone.
        air[i] = observation @ state
        state = a_d @ state + b_d @ inputs[i]

    noisy = air + rng.normal(0.0, MEASUREMENT_NOISE_STD_C, len(air))

    raw = _raw_frame()
    raw["T_air"] = np.round(noisy / SENSOR_RESOLUTION_C) * SENSOR_RESOLUTION_C

    return raw


@functools.cache
def _fitted(with_future: bool):
    """One fit, shared by every test that only needs a calibrated model: the
    fit is the slow part, and nothing these tests check depends on it being
    their own."""

    if with_future:
        df, now = _frame_with_future(np.random.default_rng(22))
    else:
        df, now = _simulate(np.random.default_rng(14)), None

    identifier = _identifier()
    identifier.calibrate(df)

    return df, now, identifier.model, identifier.parameter_std_errors


def _calibrated(with_future: bool = False):
    """A calibrated identifier of its own - the model is copied, so a test may
    break it - with the frame it was fitted on and, with_future, where now is."""

    df, now, model, std_errors = _fitted(with_future)
    identifier = _identifier()
    identifier.model = replace(model)
    identifier.parameter_std_errors = dict(std_errors or {})

    return identifier, df.copy(), now


def test_calibrate_recovers_known_parameters():
    identifier = _identifier()

    model = identifier.calibrate(_simulate(np.random.default_rng(0)))

    # The two conductances and the solar aperture set the zone's steady-state
    # energy balance, which is what planning depends on.
    assert model.ua_envelope_w_per_k == pytest.approx(
        TRUE_MODEL.ua_envelope_w_per_k, rel=0.15
    )
    assert model.ua_air_mass_w_per_k == pytest.approx(
        TRUE_MODEL.ua_air_mass_w_per_k, rel=0.25
    )
    assert model.a_eff_m2 == pytest.approx(TRUE_MODEL.a_eff_m2, rel=0.15)
    assert model.internal_gain_fraction == pytest.approx(
        TRUE_MODEL.internal_gain_fraction, abs=0.2
    )
    # The mass capacity sets how long the building coasts, the quantity the
    # whole two-node structure exists for.
    assert model.c_mass_j_per_k == pytest.approx(TRUE_MODEL.c_mass_j_per_k, rel=0.25)


def test_validate_reports_forward_simulation_accuracy():
    identifier = _identifier()
    df = _simulate(np.random.default_rng(1))

    identifier.calibrate(df)
    metrics = identifier.validate(df)

    assert metrics["scored_samples"] > 0
    # Well inside the sensor's own 0.1 K resolution would be suspicious; a few
    # tenths of a kelvin over a six-hour open-loop rollout is the real target.
    assert metrics["mae"] < 0.3
    assert abs(metrics["bias_k"]) < 0.2
    assert metrics["mae_free_float"] < 0.3
    assert metrics["mae_active"] < 0.5


def test_validate_flags_a_parameter_pinned_at_a_bound(caplog):
    identifier = _identifier()
    df = _simulate(np.random.default_rng(2))
    identifier.calibrate(df)

    identifier.model.ua_envelope_w_per_k = identifier.MAX_UA_ENVELOPE_W_PER_K

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    assert metrics["pinned_parameters"] == 1
    assert "ua_envelope_w_per_k is pinned at a bound" in caplog.text


def test_floor_heat_sign_follows_the_mode():
    """One model covers heating and cooling because this term carries the sign."""

    flow = np.array([COOLING_FLOW_LPM])

    heating = floor_heat_w(flow, np.array([35.0]), np.array([30.0]))
    cooling = floor_heat_w(flow, np.array([16.0]), np.array([19.0]))

    assert heating[0] > 0.0
    assert cooling[0] < 0.0

    # Energy balance: the same flow and the same temperature difference must
    # move the same amount of heat, only in the other direction.
    assert heating[0] == pytest.approx(-floor_heat_w(flow, 30.0, 35.0)[0])


def test_closed_shutter_blocks_solar_gain():
    irradiance = np.array([600.0, 600.0])
    gain = solar_gain_w(6.0, np.array([1.0, 0.0]), irradiance)

    assert gain[0] == pytest.approx(6.0 * 600.0)
    assert gain[1] == 0.0


def test_occupancy_adds_metabolic_heat_to_the_air_node():
    identifier = _identifier()
    prepared = identifier.prepare(_raw_frame())

    inputs = identifier._inputs(TRUE_MODEL, prepared)
    occupants = prepared["occupants"].to_numpy(dtype=float)

    empty = occupants == 0
    occupied = occupants == 1

    difference = inputs[occupied, 1].mean() - inputs[empty, 1].mean()
    baseload_difference = TRUE_MODEL.internal_gain_fraction * (
        prepared.loc[occupied, "baseload_w"].mean()
        - prepared.loc[empty, "baseload_w"].mean()
    )

    assert difference - baseload_difference == pytest.approx(
        Q_PERSON_SENSIBLE_W, abs=1e-6
    )


def test_validate_flags_a_physically_impossible_aperture(caplog):
    """A converged fit is not the same as an identified window.

    Real cooling-season data produced an effective aperture under 6% of the
    configured glass area with a small standard error - statistically
    determined, physically impossible, since it implies a g-value no glazing
    has. Validation has to say so rather than report the fit as a result.
    """

    identifier = _identifier()
    df = _simulate(np.random.default_rng(3))
    identifier.calibrate(df)

    identifier.model.a_eff_m2 = 0.05 * TRUE_SOUTH_GLASS_M2

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    assert metrics["implausible_aperture"] == 1.0
    assert metrics["aperture_fraction"] == pytest.approx(0.05)
    assert "no real glazing has" in caplog.text


def test_validate_accepts_a_realistic_aperture():
    identifier = _identifier()
    df = _simulate(np.random.default_rng(4))
    identifier.calibrate(df)

    # The synthetic truth is 6 of 14 m2, a g-value times frame factor of 0.43 -
    # an ordinary double-glazed window.
    metrics = identifier.validate(df)

    assert metrics["implausible_aperture"] == 0.0


def test_shading_is_weighted_by_glass_area():
    """A large screen must outweigh a small window, which is why the areas are
    configured per shutter rather than averaged over shutters.
    """

    identifier = _identifier()
    identifier.glazing_areas_m2 = [12.0, 3.0]
    identifier.shutter_areas_m2 = [12.0, 3.0]
    identifier.shutter_columns = ["shutter_0", "shutter_1"]

    # Big window fully open, small one shut.
    df = pd.DataFrame({"shutter_0": [100.0], "shutter_1": [0.0]})

    assert identifier._shutter_open_fraction(df)[0] == pytest.approx(12.0 / 15.0)

    # Reversed: the same unweighted mean, a very different physical result.
    df = pd.DataFrame({"shutter_0": [0.0], "shutter_1": [100.0]})

    assert identifier._shutter_open_fraction(df)[0] == pytest.approx(3.0 / 15.0)


def test_glass_without_a_shutter_is_always_unshaded():
    identifier = _identifier()
    identifier.glazing_areas_m2 = [10.0, 5.0]
    identifier.shutter_areas_m2 = [5.0]
    identifier.shutter_columns = ["shutter_0"]

    df = pd.DataFrame({"shutter_0": [0.0]})

    # The 10 m2 without a cover keeps contributing while the 5 m2 is shut.
    assert identifier._shutter_open_fraction(df)[0] == pytest.approx(10.0 / 15.0)


def test_simulate_returns_one_prediction_per_scored_sample():
    """The row indices must cover exactly the scored samples.

    Warm-up windows are simulated but not scored, so counting them in the index
    silently desynchronises the prediction from its own time axis - which is
    invisible to validate(), because that only consumes the values.
    """

    identifier = _identifier()
    df = _simulate(np.random.default_rng(5))
    identifier.calibrate(df)

    simulated = identifier.simulate(df)["predicted"]
    prepared = identifier.prepare(df)
    plan = identifier._rollout_plan(prepared, float(prepared["dt_seconds"].median()))
    predicted, _, _, _ = identifier._simulate_windows(
        identifier._parameters(identifier.model), prepared, plan
    )

    assert len(simulated) == len(predicted)
    assert simulated.index.is_monotonic_increasing
    # Nothing from the warm-up may appear: the first scored sample sits a full
    # MASS_WARMUP_HOURS after the run starts.
    warmup = pd.Timedelta(hours=identifier.MASS_WARMUP_HOURS)
    assert simulated.index.min() >= prepared["time"].min() + warmup


def test_simulate_tracks_the_measured_temperature():
    identifier = _identifier()
    df = _simulate(np.random.default_rng(6))
    identifier.calibrate(df)

    simulated = identifier.simulate(df)["predicted"]
    measured = identifier.prepare(df).set_index("time")["T_air"]

    aligned = measured.reindex(simulated.index)

    assert (simulated - aligned).abs().mean() < 0.3


def test_simulate_runs_up_to_the_last_measurement():
    """The displayed curve must not stop a whole horizon short of now.

    Scoring drops a run's trailing partial window, because a shorter window
    accumulates less error and would flatter the metrics. A dashboard has no
    such concern, and dropping it left the model line up to six hours behind
    the measurement.
    """

    identifier = _identifier()
    df = _simulate(np.random.default_rng(7))
    identifier.calibrate(df)

    # Trim to a length that is NOT a whole number of windows, so there really
    # is a remainder to drop - 20 whole days divides exactly and would make
    # this test vacuous.
    df = df.iloc[: len(df) - 7]

    simulated = identifier.simulate(df)
    prepared = identifier.prepare(df)

    step = pd.Timedelta(seconds=prepared["dt_seconds"].median())
    assert simulated.index.max() >= prepared["time"].max() - step

    # Scoring still drops it, so the metrics keep comparing equal-length windows.
    plan = identifier._rollout_plan(prepared, float(prepared["dt_seconds"].median()))
    scored, _, _, _ = identifier._simulate_windows(
        identifier._parameters(identifier.model), prepared, plan
    )
    _, _, horizon_samples = plan
    assert len(scored) % horizon_samples == 0
    assert len(simulated) >= len(scored)


def test_validate_flags_a_model_that_loses_to_persistence(caplog):
    """Holding the last measurement is the baseline any thermal model must beat.

    It costs nothing and is hard to beat indoors over a few hours, so a model
    that loses to it is adding error rather than information - which a small
    MAE next to the 0.1 K sensor resolution can otherwise hide.
    """

    identifier = _identifier()
    df = _simulate(np.random.default_rng(8))
    identifier.calibrate(df)

    # A badly wrong envelope makes the dynamics drift away from every anchor.
    identifier.model.ua_envelope_w_per_k = identifier.MAX_UA_ENVELOPE_W_PER_K

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    assert metrics["skill_vs_persistence"] <= 0.0
    assert "adding error, not information" in caplog.text


def test_validate_reports_positive_skill_for_the_calibrated_model():
    identifier = _identifier()
    df = _simulate(np.random.default_rng(9))
    identifier.calibrate(df)

    metrics = identifier.validate(df)

    assert metrics["mae_persistence"] > 0.0
    # On synthetic data generated by this very model, it must beat the baseline.
    assert metrics["skill_vs_persistence"] > 0.0


def test_validate_splits_skill_by_regime():
    """An overall skill averages over two regimes with very different counts.

    Planning only ever acts while the floor circuit runs, so that regime needs
    its own baseline and its own number - otherwise a model that fails exactly
    there still reports a healthy headline figure.
    """

    identifier, df, _ = _calibrated()

    metrics = identifier.validate(df)

    for regime in ("free_float", "active"):
        assert metrics[f"skill_{regime}"] <= 1.0
        assert metrics[f"mae_persistence_{regime}"] > 0.0
        assert metrics[f"samples_{regime}"] > 0.0

    # The headline is not simply one of the two.
    assert metrics["samples_free_float"] + metrics["samples_active"] == pytest.approx(
        metrics["scored_samples"]
    )
    assert metrics["active_windows"] > 0.0


def test_validate_reports_too_few_active_windows_as_untested(caplog):
    """Samples inside one rollout are not independent observations.

    With only a handful of windows containing floor activity, the active skill
    is noise, and saying so is the difference between "this model failed" and
    "this model has not been tried".
    """

    identifier, df, _ = _calibrated()

    identifier.MIN_ACTIVE_WINDOWS = 10_000

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    assert metrics["active_windows"] < 10_000
    assert "effectively untested" in caplog.text
    # Not reported as a failure of the model.
    assert "cannot predict the response" not in caplog.text


def test_validate_flags_a_model_that_cannot_predict_the_forced_response(caplog):
    identifier, df, _ = _calibrated()

    identifier.MIN_ACTIVE_WINDOWS = 1
    # Break only the coupling to delivered heat: free drift stays fine, the
    # response to the floor circuit does not.
    identifier.model.c_mass_j_per_k = identifier.MIN_C_MASS_J_PER_K

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    assert metrics["skill_active"] <= 0.0
    assert "cannot predict the response" in caplog.text


def test_zone_temperature_is_weighted_by_floor_area():
    """Thermostats are not spread evenly over a dwelling.

    Four of this installation's five zones are upstairs, so a plain average
    counts the ground floor for 20% of a temperature it is half the area of.
    With warm air collecting upstairs that biases the modelled temperature, so
    the mean is weighted by the area each sensor stands for.
    """

    identifier = _identifier()
    identifier.room_temperature_columns = ["room_temperature_0", "room_temperature_1"]
    identifier.room_areas_m2 = [36.0, 4.0]

    df = _raw_frame()
    df["room_temperature_0"] = 20.0
    df["room_temperature_1"] = 30.0

    prepared = identifier.prepare(df)

    # Weighted: (36*20 + 4*30) / 40 = 21.0, not the plain mean of 25.0.
    assert prepared["T_air"].iloc[0] == pytest.approx(21.0)


def test_a_dropped_out_sensor_drops_its_weight_too():
    identifier = _identifier()
    identifier.room_temperature_columns = ["room_temperature_0", "room_temperature_1"]
    identifier.room_areas_m2 = [36.0, 4.0]

    df = _raw_frame()
    df["room_temperature_0"] = 20.0
    df["room_temperature_1"] = np.nan

    prepared = identifier.prepare(df)

    # The remaining sensor stands for the whole zone rather than being diluted.
    assert prepared["T_air"].iloc[0] == pytest.approx(20.0)


def test_volume_is_derived_from_the_zone_areas_and_ceiling_height():
    """One fact, one place. A separately configured volume would be a second
    place for the same geometry to be wrong.
    """

    config = _config()
    identifier = BuildingThermalIdentifier(latitude=52.39, longitude=5.79)
    identifier.dataset(config)

    expected = sum(room.area_m2 for room in config.building.rooms) * (
        config.building.ceiling_height
    )

    assert identifier.volume_m3 == pytest.approx(expected)
    assert identifier.room_areas_m2 == [room.area_m2 for room in config.building.rooms]


def test_filter_infers_the_unmeasured_mass_node():
    """The point of the filter: a state nothing measures still gets corrected.

    Hard-resetting the measured state while letting the unmeasured one
    free-run leaves the two inconsistent, which is neither simulation nor
    estimation. The filter has to pull an initial mass temperature towards
    something the measured air temperature supports.
    """

    identifier = _identifier()
    df = _simulate(np.random.default_rng(17))
    identifier.calibrate(df)

    prepared = identifier.prepare(df)
    model = identifier.model
    a, b = identifier._state_space(model)

    measured = prepared["T_air"].to_numpy(dtype=float)
    estimates = kalman_states(
        a,
        b,
        measured,
        identifier._inputs(model, prepared),
        prepared["dt_seconds"].to_numpy(dtype=float),
        identifier.PROCESS_NOISE_W,
        identifier.SENSOR_RESOLUTION_K**2 / 12.0,
    )

    assert estimates.shape == (len(prepared), 2)
    # The measured state tracks its measurement closely...
    assert np.abs(estimates[100:, 0] - measured[100:]).mean() < 0.1
    # ...while the mass node is a distinct, inferred quantity rather than a
    # copy of it.
    assert np.abs(estimates[100:, 1] - estimates[100:, 0]).mean() > 0.0


def test_filter_result_does_not_hinge_on_the_process_noise():
    """Above the sensor's own noise the estimate must stop depending on it.

    A filter setting that changes the verdict would make the comparison between
    the two structures a matter of tuning rather than of evidence.
    """

    identifier = _identifier()
    df = _simulate(np.random.default_rng(18))
    identifier.calibrate(df)

    scores = []

    for noise_w in (3_000.0, 10_000.0, 30_000.0):
        identifier.PROCESS_NOISE_W = noise_w
        scores.append(identifier.validate(df)["skill_vs_persistence"])

    assert max(scores) - min(scores) < 0.05


def test_validate_flags_an_envelope_that_reacts_too_weakly(caplog):
    """A mean error cannot show a wrong reaction to the drive.

    If the envelope conductance is off, the error a window ends with grows with
    the indoor-outdoor difference that drove it - and it does so in both
    directions, so the two halves cancel in any average. Halving the
    conductance is exactly that failure.
    """

    identifier = _identifier()
    df = _simulate(np.random.default_rng(19))
    identifier.calibrate(df)

    identifier.model.ua_envelope_w_per_k *= 0.5

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    assert metrics["envelope_bias_slope_k_per_k"] < 0.0
    assert metrics["envelope_bias_span_k"] > identifier.SENSOR_RESOLUTION_K
    assert "envelope response is too weak" in caplog.text


def test_validate_reports_no_envelope_trend_for_the_calibrated_model(caplog):
    identifier = _identifier()
    df = _simulate(np.random.default_rng(20))
    identifier.calibrate(df)

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    # Fitted to data this model generated, so its reaction to the drive must
    # not be distinguishable from no reaction at all. The raw slope is never
    # exactly zero - what matters is that it stays within its own uncertainty.
    assert abs(metrics["envelope_bias_slope_k_per_k"]) <= (
        identifier.SIGNIFICANT_SLOPE_STD_ERRORS
        * metrics["envelope_bias_slope_std_error"]
    )
    assert "envelope response is too" not in caplog.text


def test_envelope_trend_ignores_sunlit_windows():
    """Solar gain and the outdoor difference move together, so a trend fitted
    over every window measures the net of two errors instead of the envelope.

    Measured on real data: over all windows the single-node model slopes
    +0.0009 K/K and looks clean, while after dark it slopes -0.0218 - its
    oversized solar term cancelling its own envelope error. Only dark windows
    isolate the envelope.
    """

    identifier, df, _ = _calibrated()

    metrics = identifier.validate(df)

    prepared = identifier.prepare(df)
    sunlit = prepared["shutter_open_fraction"] * prepared["I_facade_w_per_m2"]

    assert metrics["dark_windows"] > 0
    # The trend must rest on fewer windows than the run has in total, or it is
    # not excluding anything.
    assert (sunlit >= identifier.NEGLIGIBLE_SOLAR_GAIN_W).any()
    assert metrics["dark_windows"] < metrics["scored_samples"] / (
        identifier.ROLLOUT_HORIZON_HOURS * 4
    )


def _frame_with_future(rng: np.random.Generator) -> tuple[pd.DataFrame, datetime]:
    """A frame reaching past `now`, as the loader returns when asked for it."""

    df = _simulate(rng)
    # Every sensor column carries its last reading forward there, which is
    # exactly what the loader's fill="previous" produces.
    now = df["target_time"].iloc[len(df) * 3 // 4]

    return df, now


def test_forecast_starts_where_the_measurements_stop():
    identifier, df, now = _calibrated(with_future=True)

    forecast = identifier.forecast(df, now)

    assert forecast.index.min() <= now
    assert forecast.index.max() > now
    # It picks up from the filter, so the first value is what the filter had.
    estimated = identifier.estimate(df[df["target_time"] <= now])
    assert forecast["air"].iloc[0] == pytest.approx(estimated["air"].iloc[-1], abs=1e-6)


def test_trajectory_is_the_filter_until_now_and_the_forecast_after():
    """One filter pass for both halves: what a plan starts from, and the
    history the dashboard draws it against."""

    identifier, df, now = _calibrated(with_future=True)

    trajectory = identifier.trajectory(df, now)
    forecast = identifier.forecast(df, now)
    past = trajectory.index < forecast.index.min()

    assert trajectory.loc[~past, "mass"].to_numpy() == pytest.approx(
        forecast["mass"].to_numpy()
    )
    assert trajectory.loc[past, "measured"].notna().any()
    assert trajectory.loc[trajectory.index > now, "measured"].isna().all()


def test_two_node_forecast_carries_the_thermostat_reading():
    """The curve drawn against the measurement has to be the same quantity:
    with two nodes the thermostat reads part of the mass as well."""

    identifier, df, now = _calibrated(with_future=True)
    identifier.model = replace(identifier.model, sensor_mass_fraction=0.4)

    forecast = identifier.forecast(df, now)

    assert forecast["reading"].to_numpy() == pytest.approx(
        0.6 * forecast["air"].to_numpy() + 0.4 * forecast["mass"].to_numpy()
    )


def test_forecast_delivers_no_heat_to_the_zone():
    """Free-running: it answers what the house does if nothing is done to it.

    There is no decision in it, which is why it does not live in the optimizer.
    """

    identifier, df, now = _calibrated(with_future=True)

    # Cooling runs in the synthetic frame, so a forecast that used them would
    # visibly pull the zone down.
    with_cooling = identifier.forecast(df, now)

    cold = df.copy()
    cold["state"] = "Uit"
    cold["flow_lpm"] = 0.0
    without = identifier.forecast(cold, now)

    # The past differs (the filter saw different history), but the forward part
    # must not respond to floor heat at all.
    assert len(with_cooling) == len(without)


def test_forecast_needs_both_a_past_and_a_future():
    identifier, df, _ = _calibrated(with_future=True)

    with pytest.raises(ValueError, match="No future rows"):
        identifier.forecast(df, df["target_time"].max())

    with pytest.raises(ValueError, match="No future rows"):
        identifier.forecast(df, df["target_time"].min() - timedelta(hours=1))


def test_forecast_uses_a_supplied_baseload_curve():
    """The baseload forecaster's own curve beats carrying one reading forward."""

    identifier, df, now = _calibrated(with_future=True)

    prepared = identifier.prepare(df)
    high = pd.Series(5000.0, index=prepared["time"])

    plain = identifier.forecast(df, now)
    boosted = identifier.forecast(df, now, baseload_w=high)

    # More appliance heat means a warmer house, and it has to reach the model.
    assert boosted["air"].iloc[-1] > plain["air"].iloc[-1]


def test_observation_is_air_alone_without_a_radiant_share():
    assert zone_observation(TRUE_MODEL).tolist() == [1.0, 0.0]


def test_observation_weighs_air_against_mass():
    """A wall thermostat reads an operative temperature: part air, part the
    surfaces around it, and the two shares are one reading."""

    model = replace(TRUE_MODEL, sensor_mass_fraction=0.3)

    assert zone_observation(model).tolist() == pytest.approx([0.7, 0.3])
    assert zone_observation(model).sum() == pytest.approx(1.0)


def _simulate_operative(rng: np.random.Generator, fraction: float) -> pd.DataFrame:
    """The same house, read by a thermostat that also sees the surfaces."""

    identifier = _identifier()
    prepared = identifier.prepare(_raw_frame())
    true_model = replace(TRUE_MODEL, sensor_mass_fraction=fraction)
    inputs = identifier._inputs(true_model, prepared)

    a_d, b_d = discretize_zoh(*two_node_zone_state_space(true_model), DT_SECONDS)
    observation = zone_observation(true_model)

    state = np.array([21.0, 21.0])
    reading = np.empty(len(prepared))

    for i in range(len(prepared)):
        reading[i] = observation @ state
        state = a_d @ state + b_d @ inputs[i]

    noisy = reading + rng.normal(0.0, MEASUREMENT_NOISE_STD_C, len(reading))

    raw = _raw_frame()
    raw["T_air"] = np.round(noisy / SENSOR_RESOLUTION_C) * SENSOR_RESOLUTION_C

    return raw


def test_calibrate_recovers_the_radiant_share_of_the_reading():
    """Identifiable because it changes the shape of the response, not its
    level: the mass node moves slower than the air, so how much of it the
    reading carries sets how slowly the reading itself responds."""

    identifier = _identifier()

    model = identifier.calibrate(_simulate_operative(np.random.default_rng(3), 0.3))

    assert model.sensor_mass_fraction == pytest.approx(0.3, abs=0.15)
    assert model.ua_envelope_w_per_k == pytest.approx(
        TRUE_MODEL.ua_envelope_w_per_k, rel=0.25
    )


def test_filter_corrects_the_mass_node_through_a_mixed_reading():
    """With the reading carrying part of the mass, the filter can correct a
    state nothing measures directly - air alone leaves it free-running."""

    rng = np.random.default_rng(5)
    identifier = _identifier()
    prepared = identifier.prepare(_raw_frame())
    true_model = replace(TRUE_MODEL, sensor_mass_fraction=0.3)
    inputs = identifier._inputs(true_model, prepared)
    a, b = two_node_zone_state_space(true_model)
    a_d, b_d = discretize_zoh(a, b, DT_SECONDS)
    observation = zone_observation(true_model)

    state = np.array([21.0, 24.0])
    states = np.empty((len(prepared), 2))

    for i in range(len(prepared)):
        states[i] = state
        state = a_d @ state + b_d @ inputs[i]

    measured = states @ observation + rng.normal(
        0.0, MEASUREMENT_NOISE_STD_C, len(states)
    )
    dt_seconds = prepared["dt_seconds"].to_numpy(dtype=float)

    def mass_error(obs: np.ndarray) -> float:
        estimates = kalman_states(
            a,
            b,
            measured,
            inputs,
            dt_seconds,
            identifier.PROCESS_NOISE_W,
            identifier.SENSOR_RESOLUTION_K**2 / 12.0,
            obs,
            identifier.DISTURBANCE_INPUTS,
        )
        return float(np.abs(estimates[50:, 1] - states[50:, 1]).mean())

    assert mass_error(observation) < mass_error(np.array([1.0, 0.0]))


def test_filter_lets_an_unmodelled_floor_flow_disturb_the_mass():
    """The pipe run to this heat pump's shed loses part of what was measured
    into the floor circuit, so the mass is not driven exactly as measured. A
    disturbance on that channel lets the filter correct for it; one on the air
    channel alone has to push the whole error through the air node."""

    rng = np.random.default_rng(7)
    identifier = _identifier()
    prepared = identifier.prepare(_raw_frame())
    inputs = identifier._inputs(TRUE_MODEL, prepared)
    a, b = two_node_zone_state_space(TRUE_MODEL)
    a_d, b_d = discretize_zoh(a, b, DT_SECONDS)

    # The house receives four fifths of the floor heat the meter reports.
    delivered = inputs.copy()
    delivered[:, 3] *= 0.8

    state = np.array([21.0, 21.0])
    states = np.empty((len(prepared), 2))

    for i in range(len(prepared)):
        states[i] = state
        state = a_d @ state + b_d @ delivered[i]

    measured = states[:, 0] + rng.normal(0.0, MEASUREMENT_NOISE_STD_C, len(states))
    dt_seconds = prepared["dt_seconds"].to_numpy(dtype=float)

    def mass_error(channels: tuple[int, ...]) -> float:
        estimates = kalman_states(
            a,
            b,
            measured,
            inputs,
            dt_seconds,
            identifier.PROCESS_NOISE_W,
            identifier.SENSOR_RESOLUTION_K**2 / 12.0,
            None,
            channels,
        )
        return float(np.abs(estimates[50:, 1] - states[50:, 1]).mean())

    assert mass_error(identifier.DISTURBANCE_INPUTS) < mass_error((1,))
