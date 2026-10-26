import functools
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from pvlib import irradiance, solarposition

from domain.config import Config, HeatPumpStates, SouthGlazing
from domain.dynamics import ScheduledSystem, discretize_zoh, kalman_states
from domain.models import BuildingThermalModel
from domain.physics import (
    CP_AIR_J_PER_KG_K,
    Q_PERSON_SENSIBLE_W,
    RHO_AIR_KG_PER_M3,
    envelope_conductance_w_per_k,
    night_share,
    extension_shaded_fraction,
    floor_heat_moved_w,
    floor_heat_w,
    floor_living_fraction,
    floor_loops_conductance_w_per_k,
    solar_gain_w,
    zone_observation,
    zone_observations,
    zone_state_space,
)
from domain.time import LOCAL_TIMEZONE, to_local_time
from features.building import BuildingThermalIdentifier, facade_irradiance_w_per_m2

TRUE_VOLUME_M3 = 120.0
TRUE_ZONE_AREA_M2 = 46.0
TRUE_SOUTH_GLASS_M2 = 14.0
# The thermostat's room and its glazing (see _config).
TRUE_LIVING_AREA_M2 = 30.0
TRUE_LIVING_GLASS_M2 = 10.0

TRUE_MODEL = BuildingThermalModel(
    ua_envelope_w_per_k=180.0,
    ua_air_mass_w_per_k=450.0,
    c_air_j_per_k=3.0 * RHO_AIR_KG_PER_M3 * TRUE_VOLUME_M3 * CP_AIR_J_PER_KG_K,
    c_mass_j_per_k=18.0e6,
    a_eff_m2=4.5,
    # A pure air sensor, so these tests keep measuring the room nodes
    # themselves; the operative-temperature reading has its own test below.
    sensor_mass_fraction=0.0,
    internal_gain_fraction=0.6,
    a_eff_rest_m2=1.5,
    ua_rooms_w_per_k=60.0,
    living_envelope_fraction=0.6,
    living_area_fraction=TRUE_LIVING_AREA_M2 / TRUE_ZONE_AREA_M2,
    closed_shutter_gain_fraction=0.0,
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
            "target_temperature": [["00:00", 20.0, 1.0, 1.0]],
            # Two rooms of unequal size, so an area-weighted mean differs from
            # a plain one - the distinction _shutter_open_fraction and the zone
            # temperature both rest on. The second uses the compact
            # [area, sensor] form, which must keep working.
            "rooms": [
                {"area_m2": TRUE_LIVING_AREA_M2, "temperature": "sensor.living_room"},
                [TRUE_ZONE_AREA_M2 - TRUE_LIVING_AREA_M2, "sensor.bedroom"],
            ],
            "ceiling_height": TRUE_VOLUME_M3 / TRUE_ZONE_AREA_M2,
            # One shaded group and one that is never covered, which is the
            # combination the unshaded-fraction weighting has to handle.
            "south_glazing": [
                {
                    "glass_m2": TRUE_LIVING_GLASS_M2,
                    "cover": "cover.living_room",
                    "thermostat_room": True,
                },
                TRUE_SOUTH_GLASS_M2 - TRUE_LIVING_GLASS_M2,
            ],
        },
        forecast={"solcast": "sensor.solcast", "open_meteo": "sensor.open_meteo"},
        presence=["device_tracker.phone"],
    )


DT_SECONDS = 900.0
# Air leaking through a closed envelope in wind: ~0.1 air changes per hour per
# m/s through TRUE_VOLUME_M3 (rho * c_p ~ 1200 J/m3K).
INFILTRATION_W_PER_K_M_S = 4.0
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
            "temperature": t_out,
            "state": np.where(cooling, COOLING_STATE, OFF_STATE),
            "flow_lpm": np.where(cooling, COOLING_FLOW_LPM, 0.0),
            "T_supply": np.where(cooling, COOLING_SUPPLY_C, COOLING_RETURN_C),
            "T_return": COOLING_RETURN_C,
            "baseload_w": 200.0 + 150.0 * daylight,
            "shutter_0": shutter,
            "presence_0": np.where((hour >= 17.0) | (hour < 8.0), "home", "not_home"),
            "T_air": 21.0,
            "room_temperature_0": 21.0,
            "room_temperature_1": 21.0,
        }
    )


def _identifier() -> BuildingThermalIdentifier:
    identifier = BuildingThermalIdentifier(latitude=52.39, longitude=5.79)
    identifier.room_areas_m2 = [
        TRUE_LIVING_AREA_M2,
        TRUE_ZONE_AREA_M2 - TRUE_LIVING_AREA_M2,
    ]
    identifier.room_temperature_columns = ["room_temperature_0", "room_temperature_1"]
    identifier.thermostat_room_index = 0
    identifier.volume_m3 = TRUE_VOLUME_M3
    # The living room's glass behind its shutter, the rest's never covered.
    identifier.glazing_areas_m2 = [
        TRUE_LIVING_GLASS_M2,
        TRUE_SOUTH_GLASS_M2 - TRUE_LIVING_GLASS_M2,
    ]
    identifier.glazing_in_thermostat_room = [True, False]
    identifier.shutter_areas_m2 = [TRUE_LIVING_GLASS_M2]
    identifier.shutter_columns = ["shutter_0"]
    identifier.shutter_in_thermostat_room = [True]
    identifier.presence_columns = ["presence_0"]
    return identifier


def _simulate(
    rng: np.random.Generator,
    wind_kmh: np.ndarray | None = None,
    rest_loss_w: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generate a room-temperature trajectory from the known ODE parameters.

    The forcing is taken from prepare() itself, so the test exercises the real
    irradiance transposition, shutter handling and calorimetry rather than a
    second, parallel implementation of them.

    With wind, the zone also loses INFILTRATION_W_PER_K_M_S per K of
    indoor-outdoor difference per m/s of it - physics TRUE_MODEL lacks; with
    rest_loss_w, the rest loses that heat (W) as well, an open window.
    """

    identifier = _identifier()
    raw = _raw_frame()

    if wind_kmh is not None:
        raw["wind_speed"] = wind_kmh

    prepared = identifier.prepare(raw)

    inputs = identifier._inputs(TRUE_MODEL, prepared)

    a_d, b_d = discretize_zoh(*zone_state_space(TRUE_MODEL), DT_SECONDS)
    observations = zone_observations(TRUE_MODEL)
    share = TRUE_MODEL.living_area_fraction

    state = np.full(4, 21.0)
    rooms = np.empty((len(prepared), 2))

    wind = prepared.get("wind_m_per_s", pd.Series(0.0, index=prepared.index))

    for i in range(len(prepared)):
        # What the thermometers read of the state, which for TRUE_MODEL is
        # each room node alone.
        rooms[i] = observations @ state
        forcing = inputs[i].copy()
        # Into the rooms, like the other internal flows (index 1).
        zone = share * state[0] + (1.0 - share) * state[1]
        forcing[1] -= INFILTRATION_W_PER_K_M_S * wind.iloc[i] * (zone - forcing[0])
        if rest_loss_w is not None:
            forcing[3] -= rest_loss_w[i]
        state = a_d @ state + b_d @ forcing

    noisy = rooms + rng.normal(0.0, MEASUREMENT_NOISE_STD_C, rooms.shape)
    read = np.round(noisy / SENSOR_RESOLUTION_C) * SENSOR_RESOLUTION_C

    raw["room_temperature_0"] = read[:, 0]
    raw["room_temperature_1"] = read[:, 1]

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
    identifier, _, _ = _calibrated()
    model = identifier.model

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
    identifier, df, _ = _calibrated()
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
    # The synthetic sensor reads pure air, which is its own lower bound: the
    # fit sits on it by construction, so count only what pinning adds.
    before = identifier.validate(df)["pinned_parameters"]

    identifier.model.ua_envelope_w_per_k = identifier.MAX_UA_ENVELOPE_W_PER_K

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    assert metrics["pinned_parameters"] == before + 1
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


def test_a_step_that_began_on_hot_water_brings_the_floor_only_its_own_heat():
    """The floor's heat is formed per sample (see dataset): a step whose first
    five minutes still heated the tank brings the floor the other two samples'
    heat, not the tank's with it."""

    states = HeatPumpStates()
    frame = _raw_frame().head(4)
    samples = {
        "state": [states.dhw, states.heating, states.heating],
        "flow_lpm": [18.0, 18.0, 18.0],
        "T_supply": [54.0, 25.0, 25.0],
        "T_return": [49.0, 22.0, 22.0],
    }

    for name, values in samples.items():
        frame = frame.drop(columns=name)

        for k, value in enumerate(values):
            frame[f"{name}_{k}"] = value

    prepared = _identifier().prepare(frame)
    heating = floor_heat_w(np.array([18.0]), np.array([25.0]), np.array([22.0]))[0]

    assert prepared["Q_floor_w"].to_numpy() == pytest.approx(2.0 / 3.0 * heating)
    # The step itself as before: the state at its end, its readings' mean.
    assert (prepared["state"] == states.heating).all()
    assert prepared["T_supply"].to_numpy() == pytest.approx(104.0 / 3.0)


def test_closed_shutter_blocks_solar_gain():
    irradiance = np.array([600.0, 600.0])
    gain = solar_gain_w(6.0, np.array([1.0, 0.0]), irradiance)

    assert gain[0] == pytest.approx(6.0 * 600.0)
    assert gain[1] == 0.0


def test_a_shut_shutter_still_passes_part_of_the_sun():
    """A cover reading its closed position is shut, and the glass behind it
    gets the closed-shutter share of the open gain - not none."""

    identifier = _identifier()
    identifier.shutter_closed_positions = [18.0] * len(identifier.shutter_columns)
    frame = _raw_frame()
    frame[identifier.shutter_columns] = 18.0
    shut = identifier.prepare(frame)
    frame[identifier.shutter_columns] = 100.0
    open_ = identifier.prepare(frame)

    assert (shut["open_living"] == 0.0).all()

    sunny = open_["I_facade_w_per_m2"].to_numpy() > 100.0
    model = replace(TRUE_MODEL, closed_shutter_gain_fraction=0.1)
    ratio = (
        identifier._inputs(model, shut)[sunny, 2]
        / identifier._inputs(model, open_)[sunny, 2]
    )

    assert ratio == pytest.approx(0.1)


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


def test_presence_not_yet_recorded_is_the_usual_not_nobody_home():
    """Before its tracker was recorded the household is home as it usually is
    at that time of day, not away."""

    identifier = _identifier()
    raw = _raw_frame()
    measured = identifier.prepare(raw)
    unrecorded = raw["time"] < raw["time"].min() + pd.Timedelta(days=1)
    raw.loc[unrecorded, "presence_0"] = np.nan
    prepared = identifier.prepare(raw)

    early = prepared["time"] < prepared["time"].min() + pd.Timedelta(days=1)
    np.testing.assert_array_equal(
        prepared.loc[early, "occupants"], measured.loc[early, "occupants"]
    )


def test_asleep_the_occupants_heat_moves_to_the_bedrooms():
    identifier = _identifier()
    prepared = identifier.prepare(_raw_frame())

    by_area = identifier._inputs(TRUE_MODEL, prepared)
    upstairs = identifier._inputs(
        replace(TRUE_MODEL, asleep_in_rest_fraction=1.0), prepared
    )

    local = pd.DatetimeIndex(prepared["time"]).tz_convert(LOCAL_TIMEZONE)
    asleep = night_share(
        local.hour + local.minute / 60.0,
        TRUE_MODEL.asleep_from_hour,
        TRUE_MODEL.asleep_until_hour,
        TRUE_MODEL.asleep_spread_h,
    )
    # Asleep in the small hours, awake in the afternoon.
    assert asleep[local.hour == 3].min() > 0.99
    assert asleep[local.hour == 15].max() < 0.01
    people_w = prepared["occupants"].to_numpy(dtype=float) * Q_PERSON_SENSIBLE_W
    moved = by_area[:, 2] - upstairs[:, 2]

    # The thermostat's room's floor-area share of them, asleep only; the
    # dwelling keeps all of it.
    assert moved.max() > 0.0
    np.testing.assert_allclose(
        moved, asleep * TRUE_MODEL.living_area_fraction * people_w, atol=1e-9
    )
    np.testing.assert_allclose(
        upstairs[:, 2] + upstairs[:, 3], by_area[:, 2] + by_area[:, 3], atol=1e-9
    )
    np.testing.assert_array_equal(upstairs[:, 1], by_area[:, 1])


def test_validate_flags_a_physically_impossible_aperture(caplog):
    """A converged fit is not the same as an identified window.

    Real cooling-season data produced an effective aperture under 6% of the
    configured glass area with a small standard error - statistically
    determined, physically impossible, since it implies a g-value no glazing
    has. Validation has to say so rather than report the fit as a result.
    """

    identifier, df, _ = _calibrated()

    identifier.model.a_eff_m2 = 0.05 * TRUE_LIVING_GLASS_M2
    identifier.model.a_eff_rest_m2 = 0.05 * (TRUE_SOUTH_GLASS_M2 - TRUE_LIVING_GLASS_M2)

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    assert metrics["implausible_aperture"] == 1.0
    assert metrics["aperture_fraction"] == pytest.approx(0.05)
    assert "no real glazing has" in caplog.text


def test_validate_accepts_a_realistic_aperture():
    identifier, df, _ = _calibrated()

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


def test_validate_flags_a_model_that_loses_to_persistence(caplog):
    """Holding the last measurement is the baseline any thermal model must beat.

    It costs nothing and is hard to beat indoors over a few hours, so a model
    that loses to it is adding error rather than information - which a small
    MAE next to the 0.1 K sensor resolution can otherwise hide.
    """

    identifier, df, _ = _calibrated()

    # A badly wrong envelope makes the dynamics drift away from every anchor.
    identifier.model.ua_envelope_w_per_k = identifier.MAX_UA_ENVELOPE_W_PER_K

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    assert metrics["skill_vs_persistence"] <= 0.0
    assert "adding error, not information" in caplog.text


def test_validate_reports_positive_skill_for_the_calibrated_model():
    identifier, df, _ = _calibrated()

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
    # Break the balance delivered heat enters: with five times the envelope
    # loss it leaks out far faster than it does. Breaking the slab alone no
    # longer does it, since the sun warms the room node and is still predicted
    # in the same windows.
    identifier.model.ua_envelope_w_per_k *= 5.0

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

    # The remaining sensor stands for the whole zone rather than being diluted,
    # and the rest of the house going unmeasured does not void the step.
    assert prepared["T_air"].iloc[0] == pytest.approx(20.0)
    assert prepared["T_living"].iloc[0] == pytest.approx(20.0)
    assert prepared["T_rest"].isna().all()


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


def test_a_room_may_name_the_thermostat_that_opens_its_loops():
    """[area, thermometer, thermostat mode]: the mode is loaded per sample
    beside the heat it splits (see prepare); a room without one is open."""

    config = _config()
    rooms = config.building.rooms
    rooms[1] = type(rooms[1]).model_validate(
        [rooms[1].area_m2, "sensor.bedroom", ["climate.bedroom", "state"]]
    )
    identifier = BuildingThermalIdentifier(latitude=52.39, longitude=5.79)
    definition = identifier.dataset(config)
    loaded = {d.name: d for d in definition.definitions}

    assert rooms[0].climate is None
    # Without an attribute, its mode rather than Home Assistant's numeric value.
    plain = type(rooms[1]).model_validate([7.0, "sensor.bath", "climate.bath"])
    assert plain.climate.attribute == "state"
    assert identifier.room_climate_columns == [None, "room_climate_1"]
    assert loaded["room_climate_1"].sensor.entity_id == "climate.bedroom"
    assert loaded["room_climate_1"].sensor.attribute == "state"


def test_filter_infers_the_unmeasured_mass_node():
    """The point of the filter: a state nothing measures still gets corrected.

    Hard-resetting the measured states while letting the unmeasured ones
    free-run leaves them inconsistent, which is neither simulation nor
    estimation. The filter has to pull the slabs towards something the
    measured rooms support.
    """

    identifier, df, _ = _calibrated()

    prepared = identifier.prepare(df)
    model = identifier.model
    a, b = zone_state_space(model)

    measured = prepared[["T_living", "T_rest"]].to_numpy(dtype=float)
    estimates = kalman_states(
        a,
        b,
        measured,
        identifier._inputs(model, prepared),
        prepared["dt_seconds"].to_numpy(dtype=float),
        identifier.PROCESS_NOISE_W,
        identifier.SENSOR_RESOLUTION_K**2 / 12.0,
        zone_observations(model),
        identifier.DISTURBANCE_INPUTS,
    )

    assert estimates.shape == (len(prepared), 4)
    # The measured states track their measurements closely...
    assert np.abs(estimates[100:, :2] - measured[100:]).mean() < 0.1
    # ...while each slab is a distinct, inferred quantity rather than a copy
    # of its room.
    assert np.abs(estimates[100:, 2] - estimates[100:, 0]).mean() > 0.0


def test_filter_result_does_not_hinge_on_the_process_noise():
    """Above the sensor's own noise the estimate must stop depending on it.

    A filter setting that changes the verdict would make the comparison between
    the two structures a matter of tuning rather than of evidence.
    """

    identifier, df, _ = _calibrated()

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

    identifier, df, _ = _calibrated()

    identifier.model.ua_envelope_w_per_k *= 0.5

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    assert metrics["envelope_bias_slope_k_per_k"] < 0.0
    assert metrics["envelope_bias_span_k"] > identifier.SENSOR_RESOLUTION_K
    assert "envelope response is too weak" in caplog.text


def test_validate_reports_no_envelope_trend_for_the_calibrated_model(caplog):
    identifier, df, _ = _calibrated()

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

    Measured on real data: over all windows the single-node model this
    replaced slopes +0.0009 K/K and looks clean, while after dark it slopes
    -0.0218 - its oversized solar term cancelling its own envelope error. Only
    dark windows
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


def _windy_days() -> np.ndarray:
    """Wind (km/h), steady within a day and random between days: independent
    of the daily indoor-outdoor cycle, and of this module's own day-periodic
    cooling and shutter patterns, which a periodic wind would line up with."""

    day = np.arange(N_DAYS * SAMPLES_PER_DAY) // SAMPLES_PER_DAY

    return np.random.default_rng(1).uniform(0.0, 40.0, N_DAYS)[day]


def test_validate_flags_heat_lost_to_wind_the_model_lacks(caplog):
    """Wind takes heat the model has no term for, and validation says so.

    Whether as wind or as envelope is the data's to settle: the loss scales
    with wind times the indoor-outdoor difference, which over these days
    correlates 0.85 with the difference alone, so most of it - the mean wind's
    share - reads as an envelope conducting more than the model, and only the
    day-to-day variation as wind. Both mean the zone loses heat the model
    keeps."""

    identifier, _, _ = _calibrated()
    windy = _simulate(np.random.default_rng(14), wind_kmh=_windy_days())

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(windy)

    assert metrics["wind_bias_slope_k_per_k_m_s"] < 0.0
    assert metrics["envelope_bias_slope_k_per_k"] < 0.0
    assert (
        "loses more heat with the wind" in caplog.text
        or "the envelope response is too weak" in caplog.text
    )


def test_validate_reports_no_wind_trend_where_wind_changes_nothing(caplog):
    identifier, df, _ = _calibrated()
    # The same trajectory, now with wind the zone never felt.
    df["wind_speed"] = _windy_days()

    with caplog.at_level("WARNING"):
        metrics = identifier.validate(df)

    assert abs(metrics["wind_bias_slope_k_per_k_m_s"]) <= (
        identifier.SIGNIFICANT_SLOPE_STD_ERRORS * metrics["wind_bias_slope_std_error"]
    )
    assert "with the wind" not in caplog.text


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
    assert forecast["living"].iloc[0] == pytest.approx(
        estimated["living"].iloc[-1], abs=1e-6
    )


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
    the thermostat reads part of its room's slab as well."""

    identifier, df, now = _calibrated(with_future=True)
    identifier.model = replace(identifier.model, sensor_mass_fraction=0.4)

    forecast = identifier.forecast(df, now)

    assert forecast["reading"].to_numpy() == pytest.approx(
        0.6 * forecast["living"].to_numpy() + 0.4 * forecast["slab_living"].to_numpy()
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


def test_forecast_takes_the_rooms_unexplained_heat_by_time_of_day():
    """A window opened upstairs every night, which no sensor sees, shows in
    the filter and recurs in the forecast - only at night, only upstairs."""

    loss_w = 300.0
    raw = _raw_frame()
    hour = pd.DatetimeIndex(raw["target_time"]).tz_convert(LOCAL_TIMEZONE).hour
    night = np.asarray(hour < 6)
    identifier = _identifier()
    identifier.model = TRUE_MODEL
    now = raw["target_time"].iloc[len(raw) * 3 // 4]

    zones = []
    for loss in (0.0, loss_w):
        df = _simulate(np.random.default_rng(7), rest_loss_w=night * loss)
        zones.append(
            identifier.trajectory(
                df, now, room_heat_w=identifier.room_heat_habit(df, now)
            )
        )
    ahead = zones[0].index > now
    added = (zones[1] - zones[0]).loc[ahead]
    asleep = pd.DatetimeIndex(added.index).tz_convert(LOCAL_TIMEZONE).hour < 6

    assert added.loc[asleep, "rest_gain_w"].mean() == pytest.approx(-loss_w, rel=0.2)
    assert added.loc[~asleep, "rest_gain_w"].abs().mean() < 0.1 * loss_w
    assert added["living_gain_w"].abs().mean() < 0.1 * loss_w


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
    assert boosted["living"].iloc[-1] > plain["living"].iloc[-1]


def test_forecast_shutters_hold_for_the_day_then_follow_the_time_of_day():
    """A shutter opened just now stays open for the rest of the day - someone
    is in - but says nothing about noon tomorrow: from then on it stands as it
    did on that weekday before."""

    identifier, df, now = _calibrated(with_future=True)
    identifier.model = replace(identifier.model, closed_shutter_gain_fraction=0.1)
    df = df.copy()
    df["shutter_0"] = 0.0
    shut = identifier.forecast(df, now)
    # Shut throughout, opened only in the reading at now.
    df["shutter_0"] = np.where(df["target_time"] >= now, 100.0, 0.0)

    forecast = identifier.forecast(df, now)
    ahead = forecast.index > now
    today = np.array(
        [to_local_time(t).date() == to_local_time(now).date() for t in forecast.index]
    )

    assert (
        forecast.loc[ahead & today, "living_gain_w"]
        > shut.loc[ahead & today, "living_gain_w"]
    ).any()
    np.testing.assert_allclose(
        forecast.loc[ahead & ~today, "living_gain_w"],
        shut.loc[ahead & ~today, "living_gain_w"],
    )


def test_forecast_shutters_follow_their_weekday_habit_while_shut():
    """Shut now, a shutter opens as it did on the same weekday the weeks
    before - the household's week, not the frame's last days."""

    identifier, df, now = _calibrated(with_future=True)
    identifier.model = replace(identifier.model, closed_shutter_gain_fraction=0.1)
    df = df.copy()
    df["shutter_0"] = 0.0
    weeks = pd.date_range(now - timedelta(days=21), now, freq="15min")
    weekday = np.array([to_local_time(t).weekday() for t in weeks])

    def forecast(position: np.ndarray) -> pd.DataFrame:
        history = pd.DataFrame(
            {"time": weeks, "shutter_0": position, "presence_0": "not_home"}
        )
        return identifier.forecast(df, now, habits=history)

    shut = forecast(np.zeros(len(weeks)))
    # Open all of today's weekday, shut on every other.
    habit = forecast(np.where(weekday == to_local_time(now).weekday(), 100.0, 0.0))
    ahead = habit.index > now
    today = np.array(
        [to_local_time(t).date() == to_local_time(now).date() for t in habit.index]
    )

    assert (
        habit.loc[ahead & today, "living_gain_w"]
        > shut.loc[ahead & today, "living_gain_w"]
    ).any()
    np.testing.assert_allclose(
        habit.loc[ahead & ~today, "living_gain_w"],
        shut.loc[ahead & ~today, "living_gain_w"],
    )


def test_forecast_presence_follows_its_weekday_habit():
    """Who is home ahead is who was home on that weekday at that time of day
    the weeks before - the household's week, not the frame's last days, from
    which a weekend morning was forecast as a weekday."""

    identifier, df, now = _calibrated(with_future=True)
    weeks = pd.date_range(now - timedelta(days=21), now, freq="15min")
    weekday = np.array([to_local_time(t).weekday() for t in weeks])

    def forecast(home: np.ndarray) -> pd.DataFrame:
        history = pd.DataFrame(
            {
                "time": weeks,
                "shutter_0": 0.0,
                "presence_0": np.where(home, "home", "not_home"),
            }
        )
        return identifier.forecast(df, now, habits=history)

    away = forecast(np.zeros(len(weeks), dtype=bool))
    # Home all of today's weekday, away on every other.
    habit = forecast(weekday == to_local_time(now).weekday())
    ahead = habit.index > now
    today = np.array(
        [to_local_time(t).date() == to_local_time(now).date() for t in habit.index]
    )

    np.testing.assert_allclose(
        habit.loc[ahead & today, "internal_gain_w"]
        - away.loc[ahead & today, "internal_gain_w"],
        Q_PERSON_SENSIBLE_W,
    )
    np.testing.assert_allclose(
        habit.loc[ahead & ~today, "internal_gain_w"],
        away.loc[ahead & ~today, "internal_gain_w"],
    )


def test_forecast_takes_the_sky_ahead_from_the_pv_forecast():
    """The array is the facade's sun meter ahead as it is behind: forecast to
    give twice what it gives at Open-Meteo's irradiance on its plane, it puts
    twice the sun on the zone."""

    identifier, df, now = _calibrated(with_future=True)
    identifier.pv_power_per_irradiance_m2 = 2.0
    # Asleep by floor area, as by day: the room's gain is its sun alone, also
    # around waking on a sunny morning.
    model = identifier.get_model()
    identifier.model = replace(
        model, asleep_in_rest_fraction=1.0 - model.living_area_fraction
    )
    df = df.copy()
    df["global_tilted_irradiance"] = df["direct_radiation"] + df["diffuse_radiation"]
    pv = pd.Series(
        2.0
        * identifier.pv_power_per_irradiance_m2
        * df["global_tilted_irradiance"].to_numpy(),
        index=df["target_time"],
    )

    plain = identifier.forecast(df, now)
    sunnier = identifier.forecast(df, now, pv_w=pv)
    lit = (plain.index > now) & (
        df.set_index("target_time")["global_tilted_irradiance"].reindex(plain.index)
        >= identifier.PV_CORRECTION_MIN_IRRADIANCE_W_PER_M2
    ).to_numpy()

    assert lit.any()
    np.testing.assert_allclose(
        sunnier.loc[lit, "living_gain_w"], 2.0 * plain.loc[lit, "living_gain_w"]
    )


def test_forecast_presence_follows_the_time_of_day_not_the_last_reading():
    """Home just now says nothing about tomorrow's afternoon: ahead the
    household is in as it was at that time of day before."""

    identifier, df, now = _calibrated(with_future=True)
    usual = identifier.forecast(df, now)
    df = df.copy()
    # Home in the reading at now and all of the carried-forward rows after it.
    df["presence_0"] = np.where(df["target_time"] >= now, "home", df["presence_0"])

    forecast = identifier.forecast(df, now)
    ahead = forecast.index > now

    np.testing.assert_allclose(
        forecast.loc[ahead, "internal_gain_w"], usual.loc[ahead, "internal_gain_w"]
    )


def test_a_rooms_sun_warms_that_room_before_the_rest():
    """Each room's glazing warms its own room: the house only follows through
    the coupling between them, so for the first hour the thermostat's room
    runs ahead."""

    a_d, b_d = discretize_zoh(*zone_state_space(TRUE_MODEL), 3600.0)
    sun_in_living = np.array([20.0, 0.0, 2000.0, 0.0, 0.0, 0.0])

    state = a_d @ np.full(4, 20.0) + b_d @ sun_in_living

    assert state[0] - 20.0 > 2.0 * (state[1] - 20.0) > 0.0


def test_glazing_takes_its_room_in_the_compact_form():
    glazing = SouthGlazing.model_validate(
        [12.0, ["cover.woonkamer", "current_position"], 18.0, True]
    )

    assert glazing.thermostat_room is True
    assert glazing.closed_position == 18.0
    assert SouthGlazing.model_validate([2.0, "cover.bedroom"]).thermostat_room is False


def test_extensions_are_depths_over_the_glazings_width():
    """Each side's depth over the glazing's width, the glazing floor to
    ceiling; configured, they are held rather than fitted."""

    config = _config()
    config.building.south_glazing[0] = SouthGlazing.model_validate(
        [TRUE_LIVING_GLASS_M2, "cover.living_room", 0.0, True, [2.4, 0.0]]
    )
    identifier = BuildingThermalIdentifier(latitude=52.39, longitude=5.79)
    identifier.dataset(config)
    width_m = TRUE_LIVING_GLASS_M2 / config.building.ceiling_height

    assert identifier.extensions == {
        "east": pytest.approx(2.4 / width_m),
        "west": 0.0,
    }
    lower, _, upper = identifier._bounds()
    assert lower[11] == upper[11] == pytest.approx(2.4 / width_m)
    assert lower[12] == upper[12] == 0.0


def test_extensions_belong_to_the_thermostats_room():
    config = _config()
    config.building.south_glazing[1] = SouthGlazing.model_validate(
        {"glass_m2": 2.0, "extensions": [2.4, 2.4]}
    )

    with pytest.raises(ValueError, match="thermostat's room"):
        BuildingThermalIdentifier(latitude=52.39, longitude=5.79).dataset(config)


def test_configured_extensions_are_held_through_the_fit():
    identifier = _identifier()
    identifier.extensions = {"east": 0.4, "west": 0.7}
    model = identifier.calibrate(_simulate(np.random.default_rng(14)))

    assert model.extension_east_depth_ratio == 0.4
    assert model.extension_west_depth_ratio == 0.7
    assert identifier.parameter_std_errors["extension_west_depth_ratio"] == 0.0


def test_the_zone_needs_the_thermostats_own_room():
    config = _config()
    config.building.rooms = config.building.rooms[1:]

    with pytest.raises(ValueError, match="thermostat's own room"):
        BuildingThermalIdentifier(latitude=52.39, longitude=5.79).dataset(config)


def test_the_upper_slab_warms_the_living_room_through_its_ceiling():
    """What the rest's slab gives off through the ceiling, the living room
    takes up - and nothing else changes."""

    model = replace(TRUE_MODEL, ua_ceiling_w_per_k=120.0)
    a, b = zone_state_space(model)
    without, _ = zone_state_space(replace(model, ua_ceiling_w_per_k=0.0))
    share = model.living_area_fraction
    c_living = model.c_air_j_per_k * share
    c_slab_rest = model.c_mass_j_per_k * (1.0 - share)

    assert c_living * a[0, 3] == pytest.approx(120.0)
    assert c_slab_rest * a[3, 0] == pytest.approx(120.0)
    assert c_living * (without[0, 0] - a[0, 0]) == pytest.approx(120.0)
    assert c_slab_rest * (without[3, 3] - a[3, 3]) == pytest.approx(120.0)
    changed = ~np.isclose(a, without)
    assert {(i, j) for i, j in zip(*np.nonzero(changed), strict=True)} == {
        (0, 0),
        (0, 3),
        (3, 0),
        (3, 3),
    }


def test_the_floors_heat_lands_only_under_the_loops_that_are_open():
    """With the rest's loops closed the floor's heat all enters the living
    room's screed, the zone storing the same heat; with every loop open
    nothing is moved."""

    model = replace(
        TRUE_MODEL, screed_capacity_j_per_k=6.0e6, screed_to_slab_w_per_k=1900.0
    )
    a, b = zone_state_space(model)
    share = model.living_area_fraction
    shares = np.array([share, 1.0 - share])
    capacity = np.concatenate(
        [
            model.c_air_j_per_k * shares,
            (model.c_mass_j_per_k - 6.0e6) * shares,
            6.0e6 * shares,
        ]
    )
    living = floor_living_fraction(share, 0.0, share)
    moved = floor_heat_moved_w(1000.0, living, share)
    heated = capacity * (b @ np.array([0.0, 0.0, 0.0, 0.0, 1000.0, moved]))

    assert heated.sum() == pytest.approx(1000.0)
    assert heated[4] == pytest.approx(1000.0)
    assert heated[5] == pytest.approx(0.0, abs=1e-9)
    every_loop = floor_living_fraction(share, 1.0 - share, share)
    assert floor_heat_moved_w(1000.0, every_loop, share) == pytest.approx(0.0)


def test_a_plan_takes_the_loops_the_last_run_held_longest():
    """The rooms switch at a run's ends, one at a time: the last run's loops
    are the set it held longest, at its flow then - not its last step, and
    not an earlier run's."""

    columns = ["floor_open_living", "floor_open_rest", "floor_flow_lpm"]
    nan = (np.nan, np.nan, np.nan)
    every, two, bath = (0.5, 0.5, 18.0), (0.5, 0.07, 16.0), (0.0, 0.07, 9.0)
    zone = pd.DataFrame(
        [two, two, two, nan, every, every, two, two, two, bath, nan], columns=columns
    )

    assert BuildingThermalIdentifier.last_run_loops(zone) == (0.5, 0.07, 16.0)
    assert BuildingThermalIdentifier.last_run_loops(zone.iloc[:6]) == (0.5, 0.5, 18.0)
    assert BuildingThermalIdentifier.last_run_loops(zone.iloc[[3, 10]]) is None


def test_a_room_that_closes_its_loops_takes_none_of_the_floors_heat():
    """Per sample, as the heat: a step whose last two samples ran with the
    rest's loops closed moves those samples' rest share of the heat under the
    living room, and was open over the rest a third of it."""

    states = HeatPumpStates()
    identifier = _identifier()
    identifier.room_climate_columns = [None, "room_climate_1"]
    frame = _raw_frame().head(4)
    samples = {
        "state": [states.heating] * 3,
        "flow_lpm": [18.0] * 3,
        "T_supply": [25.0] * 3,
        "T_return": [22.0] * 3,
        "room_climate_1": ["heat", "off", "off"],
    }

    for name, values in samples.items():
        frame = frame.drop(columns=name, errors="ignore")

        for k, value in enumerate(values):
            frame[f"{name}_{k}"] = value

    prepared = identifier.prepare(frame)
    heating = floor_heat_w(np.array([18.0]), np.array([25.0]), np.array([22.0]))[0]
    rest = 1.0 - identifier.living_area_fraction

    assert prepared["Q_floor_w"].to_numpy() == pytest.approx(heating)
    assert prepared["Q_floor_moved_w"].to_numpy() == pytest.approx(
        2.0 / 3.0 * heating * rest
    )
    assert prepared["floor_open_rest"].to_numpy() == pytest.approx(rest / 3.0)
    assert prepared["floor_open_living"].to_numpy() == pytest.approx(1.0 - rest)
    assert prepared["floor_flow_lpm"].to_numpy() == pytest.approx(18.0)


def test_a_screed_holds_part_of_the_slabs_heat_at_the_floors_surface():
    """With a screed each floor's top layer is it: the floor's heat enters it
    alone, it passes the heat up to its room and down to the concrete, and
    the zone stores the same heat - the screed's capacity taken from the
    slabs'. Without one, the four states as before."""

    model = replace(
        TRUE_MODEL, screed_capacity_j_per_k=6.0e6, screed_to_slab_w_per_k=1900.0
    )
    a, b = zone_state_space(model)
    share = model.living_area_fraction
    shares = np.array([share, 1.0 - share])
    capacity = np.concatenate(
        [
            model.c_air_j_per_k * shares,
            (model.c_mass_j_per_k - 6.0e6) * shares,
            6.0e6 * shares,
        ]
    )
    uniform = np.full(6, 20.0)

    stored = capacity @ (a @ uniform + b @ np.array([20.0, 0.0, 0.0, 0.0, 1000.0, 0.0]))
    assert stored == pytest.approx(1000.0)
    assert np.nonzero(b[:, 4])[0].tolist() == [4, 5]
    # Surface up to the room, conductance down to the concrete.
    assert capacity[0] * a[0, 4] == pytest.approx(model.ua_air_mass_w_per_k * share)
    assert capacity[4] * a[4, 2] == pytest.approx(1900.0 * share)
    assert zone_observations(model)[0, 4] == model.sensor_mass_fraction
    assert zone_state_space(TRUE_MODEL)[0].shape == (4, 4)


def test_the_screed_is_read_off_how_runs_fall_off():
    """A run's heat at one supply falls as its screed warms: from the start
    and how fast and how far it falls, the screed's capacity and its
    conductance down to the slabs - also from runs through half the loops,
    which heat half the screed from water the pump drives faster through
    them."""

    capacity, down, pipes = 9.0e6, 1900.0, 1600.0
    surface = TRUE_MODEL.ua_air_mass_w_per_k
    identifier = _identifier()
    identifier.model = TRUE_MODEL
    share = identifier.living_area_fraction
    rng = np.random.default_rng(5)
    rows, slabs = [], []
    time = pd.Timestamp("2026-07-01T00:00Z")

    for run in range(8):
        supply = rng.uniform(15.0, 18.0) if run % 2 else 25.0
        slab, room = rng.uniform(19.0, 22.0), rng.uniform(20.0, 23.0)
        open_, flow = (0.5, 16.0) if run % 4 > 1 else (1.0, 18.0)
        loops = float(floor_loops_conductance_w_per_k(pipes, open_, flow))
        water = loops / open_
        total = water + down + surface
        balance = (water * supply + down * slab + surface * room) / total
        decay = np.exp(-total / capacity * DT_SECONDS)
        mean = (1.0 - decay) / (total / capacity * DT_SECONDS)
        screed = slab

        for step in range(12 + 4):
            on = step < 12
            heat = loops * (supply - balance - (screed - balance) * mean) if on else 0.0
            screed = balance + (screed - balance) * decay if on else slab
            rows.append(
                {
                    "time": time,
                    "state": "Koelen" if run % 2 else "Verwarmen",
                    "T_supply": supply if on else np.nan,
                    "Q_floor_w": heat + (rng.normal(0.0, 30.0) if on else 0.0),
                    "T_living": room,
                    "T_rest": room,
                    "dt_seconds": DT_SECONDS,
                    "floor_open_living": open_ * share if on else np.nan,
                    "floor_open_rest": open_ * (1.0 - share) if on else np.nan,
                    "floor_flow_lpm": flow if on else np.nan,
                    "flow_lpm": flow if on else 0.0,
                }
            )
            if not on:
                rows[-1]["state"] = "Uit"
            slabs.append(slab)
            time += pd.Timedelta(seconds=DT_SECONDS)

    states = np.zeros((len(slabs), 4))
    states[:, 2] = states[:, 3] = slabs
    identifier._filtered = lambda model, df: states
    fitted, fitted_down = identifier._fit_screed(pd.DataFrame(rows))

    assert fitted == pytest.approx(capacity, rel=0.15)
    assert fitted_down == pytest.approx(down, rel=0.15)


def test_a_schedule_is_exact_at_whole_values_and_linear_between():
    """Each step's system is the exact discretization at a whole value, and
    between two the blend of them - within 1e-5 of the exact one."""

    schedule = ScheduledSystem(
        lambda ua: zone_state_space(TRUE_MODEL, ua), np.array([180.0, 180.4])
    )
    a_d, b_d = schedule.steps(np.array([DT_SECONDS, DT_SECONDS]))
    exact = discretize_zoh(*zone_state_space(TRUE_MODEL, 180.4), DT_SECONDS)

    assert a_d[0] == pytest.approx(
        discretize_zoh(*zone_state_space(TRUE_MODEL), DT_SECONDS)[0]
    )
    assert a_d[1] == pytest.approx(exact[0], abs=1e-6)
    assert b_d[1] == pytest.approx(exact[1], rel=1e-5, abs=1e-12)


def test_a_constant_schedule_filters_as_the_constant_system():
    """An envelope that does not move gives the filter of before."""

    rng = np.random.default_rng(3)
    a, b = zone_state_space(TRUE_MODEL)
    n = 50
    inputs = np.column_stack(
        [rng.uniform(0, 15, n), rng.uniform(0, 500, n)] + [np.zeros(n)] * 4
    )
    measured = 20.0 + rng.normal(0, 0.1, (n, 2))
    dt = np.full(n, DT_SECONDS)
    arguments = (
        measured,
        inputs,
        dt,
        30.0,
        1e-3,
        zone_observations(TRUE_MODEL),
        (2, 3, 4),
    )
    schedule = ScheduledSystem(
        lambda ua: zone_state_space(TRUE_MODEL, ua),
        np.full(n, TRUE_MODEL.ua_envelope_w_per_k),
    )

    assert kalman_states(a, b, *arguments, schedule=schedule) == pytest.approx(
        kalman_states(a, b, *arguments)
    )


def test_the_windows_open_overnight_on_warm_days_only():
    """Half the nights open at the half day, none on cold days, every night
    on hot ones - half open at the opening and closing hours, shut by day."""

    model = replace(
        TRUE_MODEL,
        ua_window_w_per_k=60.0,
        window_half_day_c=17.0,
        window_spread_k=1.0,
        window_open_hour=22.0,
        window_close_hour=6.0,
        window_hour_spread_h=0.25,
    )
    ua = TRUE_MODEL.ua_envelope_w_per_k

    assert envelope_conductance_w_per_k(
        model, [17.0, 0.0, 30.0, 30.0, 30.0, 30.0], [2.0, 2.0, 2.0, 14.0, 22.0, 6.0]
    ) == pytest.approx([ua + 30.0, ua, ua + 60.0, ua, ua + 30.0, ua + 30.0], abs=1e-3)
    assert envelope_conductance_w_per_k(TRUE_MODEL, 25.0, 2.0) == pytest.approx(ua)


def test_the_day_the_windows_follow_is_the_last_24_hours():
    prepared = _identifier().prepare(_raw_frame())
    expected = prepared.set_index("time")["T_out"].rolling("24h").mean()

    assert prepared["T_out_day"].to_numpy() == pytest.approx(expected.to_numpy())
    # A whole day of this frame's swing averages out to its mean.
    assert prepared["T_out_day"].iloc[200] == pytest.approx(12.0, abs=0.1)


def test_observation_is_air_alone_without_a_radiant_share():
    assert zone_observation(TRUE_MODEL).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_observation_weighs_air_against_mass():
    """A wall thermostat reads an operative temperature: part its room, part
    the surfaces around it, and the two shares are one reading."""

    model = replace(TRUE_MODEL, sensor_mass_fraction=0.3)

    assert zone_observation(model).tolist() == pytest.approx([0.7, 0.0, 0.3, 0.0])
    assert zone_observation(model).sum() == pytest.approx(1.0)


def _simulate_operative(rng: np.random.Generator, fraction: float) -> pd.DataFrame:
    """The same house, read by thermometers that also see the surfaces."""

    identifier = _identifier()
    prepared = identifier.prepare(_raw_frame())
    true_model = replace(TRUE_MODEL, sensor_mass_fraction=fraction)
    inputs = identifier._inputs(true_model, prepared)

    a_d, b_d = discretize_zoh(*zone_state_space(true_model), DT_SECONDS)
    observations = zone_observations(true_model)

    state = np.full(4, 21.0)
    readings = np.empty((len(prepared), 2))

    for i in range(len(prepared)):
        readings[i] = observations @ state
        state = a_d @ state + b_d @ inputs[i]

    noisy = readings + rng.normal(0.0, MEASUREMENT_NOISE_STD_C, readings.shape)
    read = np.round(noisy / SENSOR_RESOLUTION_C) * SENSOR_RESOLUTION_C

    raw = _raw_frame()
    raw["room_temperature_0"] = read[:, 0]
    raw["room_temperature_1"] = read[:, 1]

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
    """With the readings carrying part of the slabs, the filter can correct a
    state nothing measures directly - the rooms alone leave it free-running."""

    rng = np.random.default_rng(5)
    identifier = _identifier()
    prepared = identifier.prepare(_raw_frame())
    true_model = replace(TRUE_MODEL, sensor_mass_fraction=0.3)
    inputs = identifier._inputs(true_model, prepared)
    a, b = zone_state_space(true_model)
    a_d, b_d = discretize_zoh(a, b, DT_SECONDS)
    observations = zone_observations(true_model)

    state = np.array([21.0, 21.0, 24.0, 24.0])
    states = np.empty((len(prepared), 4))

    for i in range(len(prepared)):
        states[i] = state
        state = a_d @ state + b_d @ inputs[i]

    measured = states @ observations.T + rng.normal(
        0.0, MEASUREMENT_NOISE_STD_C, (len(states), 2)
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
        return float(np.abs(estimates[50:, 2] - states[50:, 2]).mean())

    rooms_alone = zone_observations(replace(true_model, sensor_mass_fraction=0.0))

    assert mass_error(observations) < mass_error(rooms_alone)


def test_filter_lets_an_unmodelled_floor_flow_disturb_the_mass():
    """The pipe run to this heat pump's shed loses part of what was measured
    into the floor circuit, so the slabs are not driven exactly as measured. A
    disturbance on that channel lets the filter correct for it; one on the
    rooms' channel alone has to push the whole error through the rooms."""

    rng = np.random.default_rng(7)
    identifier = _identifier()
    prepared = identifier.prepare(_raw_frame())
    inputs = identifier._inputs(TRUE_MODEL, prepared)
    a, b = zone_state_space(TRUE_MODEL)
    a_d, b_d = discretize_zoh(a, b, DT_SECONDS)

    # The house receives four fifths of the floor heat the meter reports.
    delivered = inputs.copy()
    delivered[:, 4] *= 0.8

    state = np.full(4, 21.0)
    states = np.empty((len(prepared), 4))

    for i in range(len(prepared)):
        states[i] = state
        state = a_d @ state + b_d @ delivered[i]

    measured = states[:, :2] + rng.normal(
        0.0, MEASUREMENT_NOISE_STD_C, (len(states), 2)
    )
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
            zone_observations(TRUE_MODEL),
            channels,
        )
        return float(np.abs(estimates[50:, 2] - states[50:, 2]).mean())

    assert mass_error(identifier.DISTURBANCE_INPUTS) < mass_error((1,))


def test_facade_irradiance_follows_from_the_horizontal_components():
    """The beam's normal intensity comes from its horizontal share by
    geometry, so the facade sees the sun from those two components alone."""

    noon = pd.Series([pd.Timestamp("2026-08-11T11:30Z")])
    zenith = solarposition.get_solarposition(
        pd.DatetimeIndex(noon), latitude=52.39, longitude=5.79
    )["apparent_zenith"].to_numpy()
    expected = irradiance.get_total_irradiance(
        surface_tilt=90.0,
        surface_azimuth=155.0,
        solar_zenith=zenith,
        solar_azimuth=solarposition.get_solarposition(
            pd.DatetimeIndex(noon), latitude=52.39, longitude=5.79
        )["azimuth"].to_numpy(),
        dni=500.0 / np.cos(np.radians(zenith)),
        ghi=np.array([700.0]),
        dhi=np.array([200.0]),
        model="isotropic",
    )["poa_global"]

    facade, _, _ = facade_irradiance_w_per_m2(
        noon,
        direct_horizontal=np.array([500.0]),
        diffuse_horizontal=np.array([200.0]),
        latitude=52.39,
        longitude=5.79,
        azimuth_deg=155.0,
    )

    assert facade[0] > 300.0
    assert facade[0] == pytest.approx(float(np.asarray(expected)[0]), rel=1e-6)


def test_the_facade_sun_follows_what_the_pv_array_measured():
    """An array of 1.8 W per W/m2 measuring 900 W under a forecast 250 W/m2
    on its plane had twice the forecast's sun; without a measurement, or under
    too little forecast light to divide by, the forecast stands."""

    identifier = BuildingThermalIdentifier(latitude=52.0, longitude=5.0)
    frame = pd.DataFrame(
        {
            "global_tilted_irradiance": [250.0, 250.0, 20.0, 400.0],
            "pv_w": [900.0, np.nan, 100.0, 720.0],
        }
    )

    identifier.pv_power_per_irradiance_m2 = identifier._pv_power_per_irradiance(frame)

    assert identifier.pv_power_per_irradiance_m2 == pytest.approx(
        np.median([900.0 / 250.0, 720.0 / 400.0])
    )

    identifier.pv_power_per_irradiance_m2 = 1.8

    assert identifier._measured_sky(frame) == pytest.approx([2.0, 1.0, 1.0, 1.0])


def test_an_extension_shades_its_side_of_the_day_by_its_depth():
    """A west extension half as deep as the glazing is wide covers it fully
    once the sun is 63.4 degrees west of the facade's normal (tan = 2), half of
    it at 45, and none of it with the sun on the facade or in the east."""

    shaded = extension_shaded_fraction(
        np.array([-45.0, 0.0, 45.0, np.degrees(np.arctan(2.0)), 80.0, 120.0]),
        east_depth_ratio=0.0,
        west_depth_ratio=0.5,
    )

    assert shaded == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
