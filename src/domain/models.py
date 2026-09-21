"""The models identified from measurements.

Parameters only - every one with a physical meaning and a unit. The balances
they appear in live in domain/physics.py, so a calibrated model can be
evaluated by the planner without dragging in the identification that produced
it.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

HeatPumpMode = Literal["heat", "cool"]

ForecasterType = Literal["baseload", "tap"]
IdentificationType = Literal[
    "boiler",
    "cop_dhw",
    "cop_heating",
    "solar",
    "building",
    "building_lumped",
]


@dataclass
class BoilerThermalModel:
    volume_l: float
    ua_top_w_per_k: float
    ua_bottom_w_per_k: float
    ua_mix_idle_w_per_k: float
    ua_mix_active_w_per_k: float
    q_in_nominal_w: float
    # Identified from heating runs where the booster heater took over (see
    # BoilerThermalIdentifier._identify_booster): the highest tank temperature
    # the heat pump reaches on its own, the tank's maximum (where the thermostat
    # cuts the booster out - the setpoint those runs used), and the booster's
    # heat input (W). None until such a run has been observed - the optimizer
    # then plans without the booster, and with a default tank maximum (see
    # optimizer.DEFAULT_MAX_TANK_TEMPERATURE_C).
    heat_pump_max_tank_temperature_c: float | None = None
    max_tank_temperature_c: float | None = None
    booster_heat_w: float | None = None


@dataclass
class BuildingThermalModel:
    """Two-node (2R2C) grey-box model of one thermal zone.

    x = [T_air, T_mass]:

        C_air  dT_air/dt  = UA_env (T_out - T_air) + UA_am (T_mass - T_air) + Q_int
        C_mass dT_mass/dt = UA_am  (T_air - T_mass) + Q_sol + Q_floor

    Q_floor and Q_sol enter the mass node rather than the air node because that
    is where the physics puts them: the floor circuit runs inside the screed,
    and air is effectively transparent to shortwave radiation, which is
    absorbed by the floor and furnishings. Q_int (metabolic and appliance
    heat) is released convectively into the air.

    One model serves both heating and cooling: none of these parameters
    describes the heat pump. Q_floor is a measured calorimetric input carrying
    its own sign, so cooling is simply a negative Q. Mode-dependence lives in
    the COP model and in the condensation limit, not in this balance.

    Simplification, stated explicitly: the mass node has no direct path to
    outdoors, so envelope mass is lumped with the air node rather than given a
    third node. With a single measured zone temperature a third capacity is not
    identifiable, and inventing one would be a fit term without evidence.
    """

    ua_envelope_w_per_k: float
    ua_air_mass_w_per_k: float
    c_air_j_per_k: float
    c_mass_j_per_k: float
    # Effective solar aperture of the zone's south glazing (m2): glass area
    # times g-value times an incidence/soiling factor. Identified as one
    # lumped parameter because those three factors only ever appear as their
    # product in the heat balance, and the g-value is not separately measured.
    a_eff_m2: float
    # Fraction of the house-wide baseload electrical power that is released as
    # heat inside this zone. The baseload sensor measures the whole house; the
    # modelled zone is only part of it.
    internal_gain_fraction: float


@dataclass
class BuildingLumpedModel:
    """Single-node (1R1C) model of the same zone: C dT/dt = UA (T_out - T) +
    Q_internal + Q_solar + Q_floor.

    One capacity covering everything that stores heat - air, furnishings,
    screed, internal walls together - and one conductance to outdoors. Its
    parameters are NOT a reduction of BuildingThermalModel's: both are fitted
    to the same data and land on genuinely different values, because a single
    node has to account for the whole response with one time constant.

    It exists because the two-node model's mass node is never measured, and on
    cooling-season data that hidden state was the largest single error source:
    dropping it took the six-hour rollout error from 0.36 K to 0.11 K and
    turned an effective solar aperture of 6% of the glass area - which no
    glazing can have - into a perfectly ordinary 50%. Here the state IS the
    measurement, so nothing has to be inferred.

    The cost is real and physical: this cannot represent the floor being warmer
    than the air, so it cannot describe charging the screed as thermal storage.
    That is what the two-node model is for, once data with daily floor-circuit
    transitions can actually identify it - see BuildingLumpedIdentifier.
    """

    ua_w_per_k: float
    c_j_per_k: float
    a_eff_m2: float
    internal_gain_fraction: float


# Exact by definition of the Kelvin scale (0 degC = 273.15 K) - used
# wherever a Celsius temperature must enter a formula (like COP) that is
# only valid on an absolute temperature scale.
KELVIN_OFFSET_C = 273.15


@dataclass
class HeatPumpCOPModel:
    # COP=1 is the theoretical floor for any heat pump (no better than pure
    # resistive heating), and COP=10 a generous ceiling far above what a real
    # residential compressor achieves even at its most favorable operating
    # point. Together they clamp what this model may predict (see
    # clamped_cop). They are deliberately NOT a filter on measurements: a
    # measured COP around 1 is exactly what booster-heater rows look like
    # (real data: 0.99), so filtering on the floor silently dropped some of
    # them and let others through - those rows are excluded explicitly
    # instead (see HeatPumpCOPIdentifier.prepare).
    MIN_COP: ClassVar[float] = 1.0
    MAX_COP: ClassVar[float] = 10.0

    # The two supply temperatures this model reports a thermal output for, and
    # the reference points MPCOptimizer fits its linear electrical-power-vs-
    # tank-temperature approximation through (see
    # MPCOptimizer._power_line_coefficients) - the normal active-heating
    # operating range for a DHW cycle on this installation. They belong with
    # the model because q_th_at_power_fit_low_w/high_w below are defined AT
    # them: the fields have no meaning without these two numbers.
    POWER_FIT_T_LOW_C: ClassVar[float] = 30.0
    POWER_FIT_T_HIGH_C: ClassVar[float] = 60.0

    eta_carnot: float
    delta_t_cond: float
    delta_t_evap: float
    # The 95th percentile of this mode's own observed supply temperature
    # (see HeatPumpCOPIdentifier.calibrate()) - planning's stand-in for the
    # heat pump's actual supply temperature, which it has no forecast for
    # (see MPCOptimizer._power_line_coefficients). Not read by cop() itself
    # (default 0.0 is harmless there, e.g. for a trial fit's parameter
    # vector - see HeatPumpCOPIdentifier._predict_cop).
    reference_supply_temperature_c: float = 0.0
    # Thermal output (W) at POWER_FIT_T_LOW_C/HIGH_C above, from
    # a line fitted to real calorimetric data so that it reproduces measured
    # electrical power (see HeatPumpCOPIdentifier._fit_q_th_line()) - used
    # by MPCOptimizer instead of BoilerThermalModel's fixed q_in_nominal_w
    # when estimating electrical power: real data confirmed Q_th is not
    # constant across a compressor run (it rises from a low start, peaks
    # mid-cycle, then falls as the compressor modulates down approaching
    # setpoint), so q_in_nominal_w (calibrated for the tank's temperature
    # *trajectory*, a different purpose) understated real electrical draw
    # through the middle of a cycle. Not read by cop() itself (default 0.0
    # is harmless there, e.g. for a trial fit's parameter vector).
    q_th_at_power_fit_low_w: float = 0.0
    q_th_at_power_fit_high_w: float = 0.0

    def cop(self, T_outdoor: float, T_supply: float) -> float:
        """Carnot COP scaled by eta_carnot - see
        HeatPumpCOPIdentifier's class docstring (features/cop.py) for the
        physical derivation. Kept on the model itself, not just the
        identifier, so planning code (MPCOptimizer) can evaluate a
        calibrated model directly without depending on the identifier that
        produced it. Works equally on scalars or numpy arrays (T_outdoor,
        T_supply is plain arithmetic, no numpy-specific code needed) - used
        both by MPCOptimizer (scalars, one step at a time) and by
        HeatPumpCOPIdentifier._predict_cop (arrays, one call per fit
        iteration).
        """

        T_cond_K = T_supply + self.delta_t_cond + KELVIN_OFFSET_C
        T_evap_K = T_outdoor - self.delta_t_evap + KELVIN_OFFSET_C

        return self.eta_carnot * T_cond_K / (T_cond_K - T_evap_K)

    def clamped_cop(self, T_outdoor, T_supply):
        """cop() held inside the [MIN_COP, MAX_COP] sanity range, so an
        outdoor/supply combination outside anything the model was fitted on
        cannot turn into an absurd power estimate. Scalars or arrays.
        """

        return np.clip(self.cop(T_outdoor, T_supply), self.MIN_COP, self.MAX_COP)

    def planned_power_at_reference_points(self, T_outdoor):
        """Electrical power (W) at POWER_FIT_T_LOW_C and POWER_FIT_T_HIGH_C
        supply temperature - the two points MPCOptimizer's linear planning
        power line passes through (see MPCOptimizer._power_line_coefficients).
        Shared with HeatPumpCOPIdentifier.validate() so it scores exactly the
        line planning costs with. Scalars or arrays.
        """

        return (
            self.q_th_at_power_fit_low_w
            / self.clamped_cop(T_outdoor, self.POWER_FIT_T_LOW_C),
            self.q_th_at_power_fit_high_w
            / self.clamped_cop(T_outdoor, self.POWER_FIT_T_HIGH_C),
        )
