"""The models identified from measurements.

Parameters only - every one with a physical meaning and a unit. The balances
they appear in live in domain/physics.py, so a calibrated model can be
evaluated by the planner without dragging in the identification that produced
it.
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

ForecasterType = Literal["baseload", "tap"]
# The tank sensors report in steps of this (K; every reading on real data lies
# on a 0.5 K grid), and at rest they flip between two neighbouring steps: a
# difference of one step is no stratification the sensors can show.
TANK_SENSOR_RESOLUTION_K = 0.5
# In the order they are calibrated (see app.bootstrap): space_heating is
# fitted against the building model's estimate of the floor's mass.
IdentificationType = Literal[
    "boiler",
    "cop_dhw",
    "cop_heating",
    "cop_cooling",
    "building",
    "solar",
    "space_heating",
    "space_cooling",
    "dew_point",
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
    # How far the settled tank ends above the SWW setpoint of a heat pump run
    # that stopped by itself on it (K; see
    # BoilerThermalIdentifier._identify_setpoint_overshoot): the heat the run
    # still puts in beyond its setpoint, which planning counts. None until such
    # a run has been observed.
    setpoint_overshoot_k: float | None = None
    # The heat the compressor puts into the tank once it is up to speed (W),
    # and how long it takes to get there after a start (s). Measured
    # calorimetrically rather than fitted to the temperature trajectory (see
    # BoilerThermalIdentifier._identify_heat_input_ramp): a compressor
    # modulates up over the first minutes of a run, so one constant either
    # overstates the start or understates the rest. Planning uses both; the
    # identification ODE keeps using q_in_nominal_w wherever the calorimetric
    # input is missing. None until runs have shown them.
    q_in_steady_w: float | None = None
    q_in_ramp_seconds: float | None = None
    # How much hotter the heat pump's supply runs than the tank it charges, at
    # the tank's energy temperature, once past a run's first step (K; see
    # BoilerThermalIdentifier._identify_supply_margin). None until runs with a
    # supply temperature have been observed.
    supply_margin_k: float | None = None
    # The layer the two sensors do not show (see
    # BoilerThermalIdentifier._identify_cold_layer): after tapping, cold water
    # fills the tank from the bottom, so two point sensors in the stratified
    # tank misstate its mean. Its share of the volume, its temperature (deg C,
    # an effective one: the cold water warms against the tank above it), and
    # the stratification (top minus bottom sensor as tank_stratification_k holds
    # it, beyond one sensor step, K) at which it is full.
    # None until runs have shown it; the sensors' average then stands for the
    # tank.
    cold_layer_fraction: float | None = None
    cold_water_temperature_c: float | None = None
    cold_layer_spread_k: float | None = None

    def mixed_temperature(self, top_c, bottom_c, stratification_k=None):
        """The tank's mean temperature (deg C): what mixing it gives, and so
        what a plan starts from. The layer counts in proportion to the
        stratification, up to full at cold_layer_spread_k: held since the tank
        was last mixed (see physics.tank_stratification_k), or without that
        history the sensors' own difference now. Only beyond one sensor step
        (TANK_SENSOR_RESOLUTION_K): the sensors of a tank at rest flip a step
        apart, which read as stratification dropped the tank 1.4 K on every flip
        (real data, 28 September 2026). Scalars or arrays."""

        average = (top_c + bottom_c) / 2.0

        if stratification_k is None:
            stratification_k = top_c - bottom_c

        if (
            self.cold_layer_fraction is None
            or self.cold_water_temperature_c is None
            or not self.cold_layer_spread_k
        ):
            return average

        share = np.clip(
            (stratification_k - TANK_SENSOR_RESOLUTION_K) / self.cold_layer_spread_k,
            0.0,
            1.0,
        )

        return average - self.cold_layer_fraction * share * (
            average - self.cold_water_temperature_c
        )


@dataclass
class DewPointModel:
    """How the indoor dew point follows the outdoor one (see
    physics.indoor_dew_point_c)."""

    # How long ventilation takes to carry a change of outdoor humidity
    # indoors, stretched by what walls and furnishings buffer (hours).
    time_constant_hours: float
    # The vapour pressure the occupants' own moisture keeps indoors above the
    # outdoor air's (Pa).
    moisture_surplus_pa: float


@dataclass
class BuildingThermalModel:
    """Two-node (2R2C) grey-box model of one thermal zone.

    x = [T_air, T_mass]:

        C_air  dT_air/dt  = UA_env (T_out - T_air) + UA_am (T_mass - T_air) + Q_int
        C_mass dT_mass/dt = UA_am  (T_air - T_mass) + Q_sol + Q_floor

    The air node is the room: its air, furnishings and internal walls, which
    follow one another closely. The mass node is the floor slab - screed and
    the concrete floor it lies on - with the floor circuit in it, coupled to
    the room only through its surface. Q_floor and Q_sol enter the slab because
    that is where the physics puts them: the pipes run in the screed, and air is
    effectively transparent to shortwave radiation, which the floor absorbs.
    Q_int (metabolic and appliance heat) is released convectively into the air.

    All of the measured floor heat enters the slab: every room is in the zone,
    the pipe run from the shed is insulated and buried, and the run through the
    attic exchanges too little with it to matter (real data: the attic 9 K
    warmer than the water, against some 2 kW missing). With the internal walls
    lumped with the slab instead, a fit had to discard half of the measured
    cooling to match the rooms. The slab takes it: a floor at 4-5 K below the
    room passes 1.5-2 kW through its surface (EN 1264), the rest of a 4.4 kW
    run cools the slab and returns to the room hours later (real data,
    cross-validated on 19 cooling runs: -0.37 K per 10 kWh of cooling before,
    +0.03 to +0.04 after).

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
    # Share of the thermostat's reading that follows the slab rather than the
    # room. A wall-mounted sensor exchanges longwave radiation with the floor
    # as well as the walls, so what it reports is an operative temperature -
    # 0 reads the room node alone, 0.5 the floor as much as the room. It
    # belongs to the sensor, not to the balance: no heat flows because of it.
    sensor_mass_fraction: float
    # Fraction of the house-wide baseload electrical power that is released as
    # heat inside this zone. The baseload sensor measures the whole house; the
    # modelled zone is only part of it.
    internal_gain_fraction: float


# Exact by definition of the Kelvin scale (0 degC = 273.15 K) - used
# wherever a Celsius temperature must enter a formula (like COP) that is
# only valid on an absolute temperature scale.
KELVIN_OFFSET_C = 273.15


@dataclass
class FloorCircuitModel:
    """How the heat pump runs the floor circuit by itself, heating or cooling
    (see features.floor): the supply temperature it chooses, the heat
    that brings into the floor, and the least it can bring."""

    # Its heating curve, supply = a + b * T_outdoor (deg C, K per K).
    supply_at_zero_outdoor_c: float
    supply_per_outdoor_k: float
    # From the supply water to the building's thermal mass (W/K), both the
    # screed's uptake and the water cooling through the loop.
    conductance_w_per_k: float
    # The least heat a settled run moves (W, a magnitude): the compressor at
    # its lowest speed. Asked for less, the heat pump cannot turn down further
    # and takes the water past its setpoint instead (real cooling data: 3.4-3.6
    # kW at 22-30 Hz, the water a median 0.5 K colder than set). 0.0 for a
    # model from before it was identified.
    min_heat_w: float = 0.0

    def supply_c(self, outdoor_c):
        return self.supply_at_zero_outdoor_c + self.supply_per_outdoor_k * outdoor_c

    def heat_w(self, outdoor_c, mass_c):
        """The heat a run delivers into the floor (W), negative cooling."""

        return self.conductance_w_per_k * (self.supply_c(outdoor_c) - mass_c)


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
    # them: the fields have no meaning without these two numbers. DHW only: a
    # heating model leaves these fields at their defaults (see
    # HeatPumpCOPIdentifier.calibrate()).
    POWER_FIT_T_LOW_C: ClassVar[float] = 30.0
    POWER_FIT_T_HIGH_C: ClassVar[float] = 60.0

    eta_carnot: float
    # The approaches of the water-side plate exchanger and the outdoor coil
    # (K), named for heating, where the plate condenses and the coil
    # evaporates. Cooling reverses both roles on the same two exchangers.
    delta_t_cond: float
    delta_t_evap: float
    # Whether this describes cooling: the useful output is then the heat
    # taken from the chilled water, so cop() is the cooling EER.
    cooling: bool = False
    # The most efficient the heat pump was seen to be (see clamped_cop):
    # MAX_COP until a fit sets it. Beyond its measured lift the Carnot form
    # keeps rising while the machine does not - fan, pump and electronics draw
    # the same however small the lift - so a cooling plan at 1 K of lift was
    # priced at an EER of 7.6 where the heat pump never showed more than 6.3.
    max_cop: float = MAX_COP
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
    #
    # At 0 degC outdoor, rising by q_th_per_outdoor_w_per_k per K warmer: a
    # compressor at a fixed speed moves a fixed volume of refrigerant vapour,
    # and warmer outdoor air evaporates it at a higher pressure, so denser -
    # more refrigerant, more heat (real data at 57 Hz: 6.45 kW at 15-18 degC
    # outdoor, 7.1 kW above 21 degC). Left out, a line fitted on summer runs
    # planned autumn runs 5% too high at their real supply temperature.
    q_th_at_power_fit_low_w: float = 0.0
    q_th_at_power_fit_high_w: float = 0.0
    q_th_per_outdoor_w_per_k: float = 0.0
    # The heat a DHW run puts into the tank once past its start-up (W), at
    # 0 degC outdoor and per K warmer: measured calorimetrically over every such
    # reading, modulation included (see HeatPumpCOPIdentifier._fit_heat_input),
    # rather than fitted through the electrical power like the line above - it
    # sets how fast the tank heats, so how long a run lasts. It rises with the
    # outdoor temperature for the same reason as that line (real data: 6.3 kW at
    # 17 degC, +55 W per K); a summer median of 6.6 kW planned autumn runs
    # minutes too short. 0.0 until calibrated: planning then uses the boiler's
    # own q_in_steady_w.
    q_in_at_zero_outdoor_w: float = 0.0
    q_in_per_outdoor_w_per_k: float = 0.0
    # The electrical power over a DHW run's first planning step (W), measured
    # (see HeatPumpCOPIdentifier._start_up): the compressor comes up
    # to speed and the loop warms, so the heat lags the power, and the steady
    # COP above does not hold there (real data: 0.75 kW for 3.0 kW of heat, a
    # COP of 4 where the steady model gave 5.3). 0.0 until calibrated: planning
    # then costs that step on the power line too.
    start_step_power_w: float = 0.0
    # How long the compressor takes to reach its heat after a run in this mode
    # starts (s), read off the energy the start-up leaves short (see
    # BoilerThermalIdentifier._identify_heat_input_ramp) - from minute
    # readings, timed from the run's own start. The boiler's own estimate from
    # 5-minute readings timed runs up to 5 minutes early, from the start of the
    # interval they switched on in (real data: 1005 s against 790-830 s per
    # minute). 0.0 until calibrated: DHW planning then uses the boiler's, a
    # floor run none (real cooling data: 530 s over 30 runs).
    start_ramp_seconds: float = 0.0

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

        if self.cooling:
            T_evap_K = T_supply - self.delta_t_cond + KELVIN_OFFSET_C
            T_cond_K = T_outdoor + self.delta_t_evap + KELVIN_OFFSET_C

            return self.eta_carnot * T_evap_K / (T_cond_K - T_evap_K)

        T_cond_K = T_supply + self.delta_t_cond + KELVIN_OFFSET_C
        T_evap_K = T_outdoor - self.delta_t_evap + KELVIN_OFFSET_C

        return self.eta_carnot * T_cond_K / (T_cond_K - T_evap_K)

    def clamped_cop(self, T_outdoor, T_supply):
        """cop() held inside [MIN_COP, max_cop], so an outdoor/supply
        combination outside anything the model was fitted on cannot turn into
        an absurd power estimate. Scalars or arrays.
        """

        return np.clip(self.cop(T_outdoor, T_supply), self.MIN_COP, self.max_cop)

    def planned_power_at_reference_points(self, T_outdoor):
        """Electrical power (W) at POWER_FIT_T_LOW_C and POWER_FIT_T_HIGH_C
        supply temperature - the two points MPCOptimizer's linear planning
        power line passes through (see MPCOptimizer._power_line_coefficients).
        Shared with HeatPumpCOPIdentifier.validate() so it scores exactly the
        line planning costs with. Scalars or arrays.
        """

        outdoor_w = self.q_th_per_outdoor_w_per_k * T_outdoor

        return (
            (self.q_th_at_power_fit_low_w + outdoor_w)
            / self.clamped_cop(T_outdoor, self.POWER_FIT_T_LOW_C),
            (self.q_th_at_power_fit_high_w + outdoor_w)
            / self.clamped_cop(T_outdoor, self.POWER_FIT_T_HIGH_C),
        )
