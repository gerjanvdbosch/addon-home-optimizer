# Building thermal model

A grey-box model of the dwelling's temperature, in `features/building.py`. It covers
space heating and space cooling with **one** set of parameters: the heat pump's
contribution is a measured calorimetric term carrying its own sign, so cooling is
simply a negative heat input and nothing here describes the heat pump. Mode-dependence
belongs in the COP model and in the condensation limit on floor cooling, not in this
heat balance.

Calibrate with `"target": "building"` (see the Calibrate API in the
[README](../README.md)), and once there are heating runs, `"target": "space_heating"`
after it - that model is fitted against this one's estimate of the floor's mass.

## Configuration

All of it lives in the `building` block of the config payload.

### `rooms`

The rooms belonging to the zone, as `[floor_area_m2, sensor]` pairs, averaged
into one representative temperature weighted by those areas. A whole-dwelling balance
weighs house-wide delivered heat and baseload against it, so a single room would be an
arbitrary sample.

The weighting matters because thermostats are not spread evenly over a dwelling. On this
installation four of five rooms are upstairs, so a plain average counted the ground
floor for 20% of a temperature it is half the area of, biasing it by +0.085 K (p95
0.42 K) as warm air collects upstairs. Weighting by area brings that to +0.008 K.

Leave out any unheated space: the attic here runs 3.5 K above the heated rooms and
correlates 0.65 with outdoor temperature, so it tracks the weather rather than the zone.

### `ceiling_height`

Net floor-to-ceiling height. The zone's air volume is derived from this and the areas
above rather than configured separately, so the same geometry is stated once. It covers
only rooms that have a sensor, slightly undercounting hall and landing - which is fine,
because the volume only bounds a heat capacity.

### `south_glazing`

The south-facing windows as `[glass_m2, cover]` pairs, one per shutter. Use
`[glass_m2, null]`, or just `glass_m2`, for glass with no shutter.

Read the areas off the floor plan as structural opening width times height, since a
roller shutter covers the whole opening. On this installation the living room front is
2760 + 1500 + 1000 mm wide at 2.35 m high (12.36 m2), and upstairs a 3230 mm opening at
1.5 m split over two shutters (2.42 m2 each) plus 975 + 1055 mm on one shutter
(3.05 m2). Split one opening across however many shutters cover it, so each cover
carries the area it actually shades.

Shading is weighted by these areas, which matters whenever the shutters do not move
together - here they are almost uncorrelated, and an unweighted average would be off by
a median of 9 percentage points.

The total is the upper bound on the identified effective aperture
`a_eff_m2 = area x g-value x frame factor`, so the g-value does not have to be known.
Leave the list out and solar gain is bounded to zero.

### What needs no configuration

**Internal gains.** Appliance heat comes from the existing `baseload` sensor (household
electricity ends up as heat indoors) and occupancy from the existing `presence`
trackers, at a fixed 75 W sensible per person.

**Outdoor temperature.** Taken from Open-Meteo's `temperature_2m`. The heat pump's own
outdoor sensor can be used instead via `heat_pump.outdoor_temperature`, but only if the
unit really stands outdoors: here it sits in a shed and reads 1.8 K warm on average with
a strongly diurnal bias (+2.8 K at night against +0.9 K at midday), so it is
deliberately left unset.

## Two nodes

**`building`** has two nodes: room air and thermal mass (screed, internal walls),
coupled to each other, with solar and floor heat entering the mass. Floor heating
really does reach the room with a lag, and only a mass node can describe charging the
screed as storage - which is what planning it as a buffer needs. Its mass temperature
is never measured directly, but the thermostats see part of it, which is what
`sensor_mass_fraction` below is about.

A single-node model - one capacity for everything that stores heat, its state the
measurement itself - used to be calibrated beside it for comparison, and has been
removed: it cannot represent the floor being warmer than the air, so it cannot plan the
screed as storage whatever it scores. On this installation's cooling-season data (43
days, both recalibrated on it) the comparison stood at:

|                        | two nodes       | one node          |
|------------------------|-----------------|-------------------|
| MAE over 6 h rollouts  | 0.134 K         | **0.128 K**       |
| persistence baseline   | 0.153 K         | 0.160 K           |
| `skill_vs_persistence` | +0.122          | **+0.200**        |
| `aperture_fraction`    | 0.168           | **0.647**         |

The two-node model used to score below persistence here (-0.005) with the hidden mass
state as its largest single error source. Reading the thermostats as an operative
temperature and letting the filter disturb the mass node closed most of that gap, and
took `ua_air_mass` off its bound. The single node is still ahead, and still finds the
more ordinary solar aperture, on data with the floor circuit active in 2.4% of quarter
hours.

### What the thermostat reads

A wall-mounted sensor exchanges longwave radiation with the surfaces around it, so it
reports an operative temperature between air and mass rather than air alone.
`sensor_mass_fraction` is that share: 0 a pure air sensor, 0.5 the textbook average in
still air, which is the bound here because the sensor sits in moving room air. It is a
measurement equation - it moves no heat - and it is what the comfort target in planning
is held to, since that is the quantity a setpoint is set in.

It earns its place twice over. On this data it takes the rollout error from 0.153 to
0.144 K on its own, and it makes the mass node partly observable: the filter can
correct a state nothing measures directly, which is what planning the screed as storage
needs.

### What can disturb which node

The filter carries one unmodelled heat flow per node (`DISTURBANCE_INPUTS`): through
the air, as ventilation, a stove or a visitor; through the mass, as heat that was
measured into the floor circuit but never reached the screed - this heat pump stands in
a shed, so part of the supply-to-return difference happens in the pipe run. Disturbing
the air alone, as before, left the filter certain of a mass node driven by a flow it
cannot fully trust. Adding the second channel took the rollout error from 0.144 to
0.134 K.

## Reading `validate()`

Metrics beyond the usual MAE and bias, each answering a question a mean error cannot.

### `skill_vs_persistence`, and per regime

How much better the model is than simply holding the last reading for the length of a
rollout. That baseline is free, needs no parameters, and is hard to beat indoors, so a
model at or below zero skill is adding error rather than information.

It is reported **per regime**, because one regime normally dominates by count: with the
floor circuit running in 1.3% of quarter hours, a headline skill is almost entirely the
free-floating one, while planning only ever acts on the other. `active_windows` says how
many independent rollouts the active figure rests on - samples inside one rollout are
not independent observations, so below ten windows it is reported as untested rather
than as a verdict.

### `implausible_aperture`

`a_eff_m2` as a fraction of the configured glass. From the glazing alone this has a
floor around 0.17: even solar-control glass has a g-value near 0.25, and a frame factor
below 0.7 would mean more frame than glass. The angle of incidence is **not** in that
product - the transposition to the facade already accounts for it.

The threshold sits below that floor at 0.15, because `a_eff_m2` also absorbs facade
shading the transposition knows nothing about. It therefore tests for values no
combination of glass and shade could produce, not for good glass: a result between 0.15
and 0.17 passes only if there is real shading to point at.

### `envelope_bias_slope_k_per_k`

Whether the model reacts correctly to the thing that drives it. If the envelope
conductance is off, the error a window ends with grows with the indoor-outdoor
difference - in both directions, so the two halves cancel in any average.

Fitted over **dark windows only**. Solar gain and the outdoor difference both peak in
the afternoon and correlate about +0.4 here, so a trend fitted over every window
measures the net of two errors and can read clean while both are large: over all windows
the single-node model slopes +0.0009 K/K, but after dark it slopes -0.0328, its oversized
solar term cancelling its own envelope error.

Judged against its own standard error rather than a fixed threshold, since how well a
slope is determined depends on how spread out the conditions happened to be.

## What this data cannot settle

Everything below needs a heating season. The floor circuit ran in 2.4% of quarter hours
over the cooling season, giving three to four scored windows with any floor activity.

**How the mass couples to the air.** What a second node adds is the transient after a
step in floor heat, and there is almost none of it here: `c_air` sits at its ceiling.

**The envelope conductance.** The model slopes negative after dark (-0.018 K/K), so it
responds too weakly, and three independent estimates put the true time constant near
90 h against the ~120 h it holds. The slope does not clear two standard errors on 45-49
windows, so it is suggestive, not established. In summer the indoor-outdoor
difference is 2-5 K and a window's decay signal sits under the sensor resolution; in
winter it is 20-30 K.

**How much of the reading is radiant.** `sensor_mass_fraction` runs into its 0.5 ceiling
on this data, and releasing that ceiling keeps improving the fit until the reading is
almost entirely the mass - at which point the solar aperture falls below what any glass
can have and the active bias grows, the signature of a fit rather than a physical
value. What it does say is that these thermostats follow the structure at least as
closely as they follow the air, which is also why `c_air` sits at its ceiling. A heating season, where the floor drives the mass hard, is what can settle
the split.

**Whether the floor coupling is mode-dependent.** A warm floor drives a buoyant plume
and a cold one leaves stable stratification, so the combined heat transfer coefficient
is roughly 11 W/m2K heating against 7 cooling. `ua_air_mass` is currently one parameter
for both modes - the first thing in this model that may genuinely need splitting, and
for a physical reason rather than a better fit.

**Whether delivered heat is overstated.** The model runs cold during floor operation by
almost pure bias (93% of its active error), worth 13% of the measured cooling. The heat pump stands in a shed, so part of the measured
supply-to-return difference happens in the pipe run rather than in the screed. Heating
flips the sign of that test: a positive `bias_active_k` of the same order would confirm
it, and the fix would be one efficiency factor on the delivered heat.

## Acceptance test for the next calibration

After the first weeks of heating, recalibrate and check, in this order:

1. `active_windows` >= 10 - enough independent rollouts to judge anything.
2. `skill_active` > 0 - the model can predict the response to heating, which is the only
   thing planning needs.
3. `implausible_aperture` == 0 - the solar aperture is one a real window could have.
4. `envelope_bias_slope_k_per_k` within two standard errors of zero.

If it passes, the mass node is identified from real heating - and only then can the
screed be planned as thermal storage rather than in the shadow plan alone.
