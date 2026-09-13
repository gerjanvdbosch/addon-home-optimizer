from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from domain.types import HeatPumpCOPModel
from features.cop import HeatPumpCOPIdentifier

TRUE_ETA_CARNOT = 0.45
TRUE_DELTA_T_COND = 5.0
TRUE_DELTA_T_EVAP = 4.0

SWW_STATE = "SWW"
HEATING_STATE = "Verwarmen"

N = 400
FLOW_LPM = 12.0


def _simulate(rng: np.random.Generator, state: str, t_supply: float) -> pd.DataFrame:
    """Synthetic data generated from the exact COP formula being fit, with a
    realistic outdoor-temperature range and a small amount of measurement
    noise on the electrical power (the only noisy input - flow/temperatures
    are treated as exact for this test).
    """

    T_outdoor = rng.uniform(-5.0, 15.0, size=N)
    T_return = t_supply - rng.uniform(3.0, 8.0, size=N)

    parameters = np.array([TRUE_ETA_CARNOT, TRUE_DELTA_T_EVAP])
    T_supply = np.full(N, t_supply)
    cop = HeatPumpCOPIdentifier._predict_cop(
        parameters, T_outdoor, T_supply, TRUE_DELTA_T_COND
    )

    delta_t_water = T_supply - T_return
    q_th_w = (1.0 / 60.0) * 4186.0 * FLOW_LPM * delta_t_water
    p_el = q_th_w / cop

    noise = rng.normal(0.0, 10.0, size=N)
    p_el_measured = p_el + noise

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time = pd.to_datetime(
        [start + timedelta(minutes=5 * i) for i in range(N)], utc=True
    )

    return pd.DataFrame(
        {
            "time": time,
            "T_outdoor": T_outdoor,
            "T_supply": T_supply,
            "T_return": T_return,
            "flow_lpm": FLOW_LPM,
            "P_el": p_el_measured,
            "state": state,
        }
    )


def test_calibrate_recovers_known_parameters_and_ignores_other_modes():
    rng = np.random.default_rng(3)

    sww_df = _simulate(rng, SWW_STATE, t_supply=50.0)

    # Heating operates at a much lower supply temperature and, deliberately,
    # a different (wrong-for-SWW) COP relationship - if mode filtering were
    # broken, mixing this in would bias the fit.
    heating_df = _simulate(rng, HEATING_STATE, t_supply=35.0)
    heating_df["P_el"] = heating_df["P_el"] * 3.0  # obviously inconsistent

    df = pd.concat([sww_df, heating_df], ignore_index=True)

    identifier = HeatPumpCOPIdentifier(mode=SWW_STATE, key="dhw")
    model = identifier.calibrate(df)

    assert model.eta_carnot == pytest.approx(TRUE_ETA_CARNOT, rel=0.1)
    assert model.delta_t_cond == identifier.FIXED_DELTA_T_COND
    assert model.delta_t_evap == pytest.approx(TRUE_DELTA_T_EVAP, rel=0.3)
    # The synthetic SWW data uses a constant T_supply=50.0 - its own 95th
    # percentile must be that same value, not (say) HEATING_STATE's T_supply
    # leaking in from the mode filter being broken.
    assert model.reference_supply_temperature_c == pytest.approx(50.0)
    # Constant T_supply=50.0 falls outside both reference windows (30+/-5,
    # 60+/-5) - both fields must fall back to this mode's overall median
    # Q_th, not the (broken, always-0.0) default.
    assert model.q_th_at_power_fit_low_w == pytest.approx(
        model.q_th_at_power_fit_high_w
    )
    assert model.q_th_at_power_fit_low_w > 0.0


def test_median_near_uses_local_window_not_global_median():
    df = pd.DataFrame(
        {
            "T_supply": [28.0, 29.0, 32.0, 58.0, 59.0, 62.0],
            # An outlier in each window - if this were a mean, it would pull
            # the result well away from the other two, ordinary readings.
            "Q_th": [3000.0, 3200.0, 9000.0, 6000.0, 6400.0, 100.0],
        }
    )

    identifier = HeatPumpCOPIdentifier(mode=SWW_STATE, key="dhw")

    # POWER_FIT_WINDOW_C=5 degC - only the first three rows fall within 5
    # degC of 30, only the last three within 5 degC of 60.
    low = identifier._median_near(df, "Q_th", 30.0)
    high = identifier._median_near(df, "Q_th", 60.0)

    assert low == pytest.approx(3200.0)
    assert high == pytest.approx(6000.0)
    # A mean would have been pulled noticeably higher/lower by the outlier.
    assert low != pytest.approx((3000.0 + 3200.0 + 9000.0) / 3)
    assert high != pytest.approx((6000.0 + 6400.0 + 100.0) / 3)


def test_median_near_falls_back_to_overall_median_when_window_is_empty():
    df = pd.DataFrame(
        {
            "T_supply": [45.0, 46.0, 47.0],
            "Q_th": [5000.0, 5200.0, 20000.0],
        }
    )

    identifier = HeatPumpCOPIdentifier(mode=SWW_STATE, key="dhw")

    # Nothing in this data falls within 5 degC of 30.
    result = identifier._median_near(df, "Q_th", 30.0)

    assert result == pytest.approx(5200.0)


def test_prepare_excludes_rows_where_the_booster_heater_is_active():
    """A resistive backup/booster heater's electrical draw follows entirely
    different physics from the compressor (no refrigerant cycle, COP
    trivially ~1), so any row where it is confirmed active must not be
    treated as heat-pump-only data - regardless of how normal the row
    otherwise looks.
    """

    times = pd.date_range("2026-01-01T10:00:00Z", periods=2, freq="5min")
    df = pd.DataFrame(
        {
            "time": times,
            "T_outdoor": [10.0, 10.0],
            "T_supply": [45.0, 45.0],
            "T_return": [40.0, 40.0],
            "flow_lpm": [12.0, 12.0],
            "P_el": [1500.0, 1500.0],
            "state": [SWW_STATE, SWW_STATE],
            "booster": ["off", "on"],
        }
    )

    identifier = HeatPumpCOPIdentifier(mode=SWW_STATE, key="dhw")
    prepared = identifier.prepare(df)

    assert len(prepared) == 1
    assert prepared.iloc[0]["time"] == times[0]


def test_prepare_does_not_filter_on_booster_when_not_configured():
    """Installations without a separate booster sensor (no "booster" column
    in the fetched dataset - see HeatPumpConfig.booster) must not have any
    rows filtered on this basis.
    """

    times = pd.date_range("2026-01-01T10:00:00Z", periods=2, freq="5min")
    df = pd.DataFrame(
        {
            "time": times,
            "T_outdoor": [10.0, 10.0],
            "T_supply": [45.0, 45.0],
            "T_return": [40.0, 40.0],
            "flow_lpm": [12.0, 12.0],
            "P_el": [1500.0, 1500.0],
            "state": [SWW_STATE, SWW_STATE],
        }
    )

    identifier = HeatPumpCOPIdentifier(mode=SWW_STATE, key="dhw")
    prepared = identifier.prepare(df)

    assert len(prepared) == 2



def test_prepare_filters_to_the_configured_mode_only():
    rng = np.random.default_rng(5)

    sww_df = _simulate(rng, SWW_STATE, t_supply=50.0)
    heating_df = _simulate(rng, HEATING_STATE, t_supply=35.0)
    df = pd.concat([sww_df, heating_df], ignore_index=True)

    identifier = HeatPumpCOPIdentifier(mode=SWW_STATE, key="dhw")
    prepared = identifier.prepare(df)

    assert (prepared["state"] == SWW_STATE).all()
    assert len(prepared) <= len(sww_df)


def test_prepare_bridges_a_brief_reporting_gap_while_active_but_zeroes_when_idle():
    """Regression test for two real findings on this exact installation: (1) a
    genuine InfluxDB reporting gap in P_el/flow_lpm (see dataset(): fetched
    with no fill at all) must not be assumed to mean "still running at its
    last known rate" once the compressor is confirmed idle - the bug already
    found and fixed for flow_lpm in boiler.py, which applies equally to P_el
    (a frozen, bit-identical P_el reading persisted for 20+ minutes into a
    real idle period); but (2) a brief gap while state still confirms the
    compressor is running must be bridged forward, not force-zeroed - real
    data showed T_supply/T_return rising smoothly through exactly such a gap
    in flow_lpm during a DHW ramp-up.
    """

    times = pd.date_range("2026-01-01T10:00:00Z", periods=4, freq="5min")
    df = pd.DataFrame(
        {
            "time": times,
            "T_outdoor": [10.0] * 4,
            "T_supply": [30.0, 35.0, 40.0, 20.0],
            "T_return": [25.0, 28.0, 32.0, 20.0],
            # A reporting gap (NaN) at index 1 while still SWW - must bridge
            # to the prior reading. A gap at index 3, once state has gone
            # idle - must resolve to 0 despite the earlier nonzero reading.
            "flow_lpm": [12.0, np.nan, 12.0, np.nan],
            "P_el": [500.0, np.nan, 700.0, np.nan],
            "state": ["SWW", "SWW", "SWW", "Uit"],
        }
    )

    # Tested directly against _bridge_reporting_gaps rather than through the
    # full prepare(): the idle row's resolved value is exactly 0, which
    # prepare()'s own validity filter (flow_lpm/P_el must be > 0) would
    # otherwise strip before it could be inspected.
    identifier = HeatPumpCOPIdentifier(mode=SWW_STATE, key="dhw")
    bridged = identifier._bridge_reporting_gaps(df)

    active_gap = bridged[bridged["time"] == times[1]].iloc[0]
    assert active_gap["flow_lpm"] == pytest.approx(12.0)
    assert active_gap["P_el"] == pytest.approx(500.0)

    idle_gap = bridged[bridged["time"] == times[3]].iloc[0]
    assert idle_gap["flow_lpm"] == 0.0
    assert idle_gap["P_el"] == 0.0


def test_validate_reports_low_error_on_matching_synthetic_data():
    rng = np.random.default_rng(7)

    df = _simulate(rng, SWW_STATE, t_supply=50.0)

    identifier = HeatPumpCOPIdentifier(mode=SWW_STATE, key="dhw")
    identifier.calibrate(df)
    metrics = identifier.validate(df)

    assert metrics["r2"] > 0.9
    assert metrics["mae"] < 0.3


def test_validate_flags_delta_t_pinned_at_bound(caplog):
    """Regression test for a real finding: eta_carnot and delta_t_evap can be
    poorly separable on real data (delta_t_cond is fixed, not fitted, so it
    can no longer be pinned) - a real calibration landed delta_t_evap close
    to MAX_DELTA_T_EVAP. validate() must surface this rather than reporting a
    precise-looking approach temperature.
    """

    rng = np.random.default_rng(11)
    df = _simulate(rng, SWW_STATE, t_supply=50.0)

    identifier = HeatPumpCOPIdentifier(mode=SWW_STATE, key="dhw")
    identifier.calibrate(df)

    # Force the exact pinned-at-bound condition directly, rather than
    # constructing a data scenario that reproduces the real identifiability
    # issue end-to-end.
    identifier.model = HeatPumpCOPModel(
        eta_carnot=identifier.model.eta_carnot,
        delta_t_cond=identifier.model.delta_t_cond,
        delta_t_evap=identifier.MAX_DELTA_T_EVAP,
        reference_supply_temperature_c=identifier.model.reference_supply_temperature_c,
    )

    with caplog.at_level("WARNING", logger="features.cop"):
        result = identifier.validate(df)

    assert result["implausible_delta_t"] == 1.0
    assert any(
        "pinned at or near its sanity bound" in record.message
        for record in caplog.records
    )


def test_validate_does_not_flag_delta_t_away_from_bounds():
    rng = np.random.default_rng(11)
    df = _simulate(rng, SWW_STATE, t_supply=50.0)

    identifier = HeatPumpCOPIdentifier(mode=SWW_STATE, key="dhw")
    identifier.calibrate(df)
    result = identifier.validate(df)

    # The synthetic data is generated with TRUE_DELTA_T_COND/EVAP comfortably
    # inside the bounds, so a correct fit should not flag them.
    assert result["implausible_delta_t"] == 0.0


def test_name_and_label_are_key_specific():
    sww = HeatPumpCOPIdentifier(mode=SWW_STATE, key="dhw")
    heating = HeatPumpCOPIdentifier(mode=HEATING_STATE, key="heating")

    assert sww.name != heating.name
    assert sww.label != heating.label
    assert sww.name == "cop_dhw"
