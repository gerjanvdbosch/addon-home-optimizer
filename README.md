# Home Optimizer

> ⚠️ **Work in Progress**
>
> This project is currently under development and should be considered experimental.
> Features, configuration, and API endpoints may change at any time. It is not yet
> recommended for production use.

A Home Assistant addon that learns your home's thermal behavior and uses model
predictive control to schedule your heat pump around solar production and electricity
prices.

## Requirements

Home Optimizer requires the [InfluxDB addon](https://github.com/vistalba/addon-influxdb)
for Home Assistant to store and retrieve historical sensor data used for training and
optimization.

## Home Assistant Setup

Add the following configurations to your Home Assistant `configuration.yaml` to enable
communication and prepare the required forecast sensors.

### 1. REST Command

This command allows Home Assistant to send automations and payloads to the Home
Optimizer API.

```yaml
rest_command:
  home_optimizer_api:
    url: "http://127.0.0.1:8099/api/{{ endpoint }}"
    method: POST
    headers:
      content-type: "application/json"
    payload: "{{ payload }}"
```

### 2. Open-Meteo Forecast (REST Sensor)

This sensor fetches 15-minute interval weather metrics (irradiance, temperature, wind,
and clouds) from the Open-Meteo API.

```yaml
rest:
  - resource: "https://api.open-meteo.com/v1/forecast\
      ?latitude=YOUR_LATITUDE\
      &longitude=YOUR_LONGITUDE\
      &tilt=YOUR_TILT\
      &azimuth=YOUR_AZIMUTH\
      &minutely_15=\
        is_day,\
        temperature_2m,\
        relative_humidity_2m,\
        global_tilted_irradiance,\
        direct_radiation,\
        direct_normal_irradiance,\
        diffuse_radiation,\
        precipitation,\
        wind_speed_10m,\
        wind_direction_10m,\
        cloud_cover_low,\
        cloud_cover_mid,\
        cloud_cover_high\
      &forecast_days=2\
      &timezone=UTC"
    scan_interval: 1800
    sensor:
      - name: "Open-Meteo Forecast"
        unique_id: open_meteo_forecast
        value_template: "{{ now() }}"
        device_class: timestamp
        json_attributes_path: "$.minutely_15"
        json_attributes:
          - time
          - is_day
          - temperature_2m
          - relative_humidity_2m
          - global_tilted_irradiance
          - direct_radiation
          - direct_normal_irradiance
          - diffuse_radiation
          - precipitation
          - wind_speed_10m
          - wind_direction_10m
          - cloud_cover_low
          - cloud_cover_mid
          - cloud_cover_high
```

### 3. Solcast PV Forecast (Template Sensor)

This sensor aggregates the Solcast today and tomorrow forecasts.

```yaml
template:
  - sensor:
      - name: "Solcast PV Forecast"
        unique_id: solcast_pv_forecast
        device_class: timestamp
        state: "{{ states('sensor.solcast_pv_forecast_api_last_polled') }}"
        attributes:
          time: >
            {% set forecast =
              state_attr('sensor.solcast_pv_forecast_forecast_today', 'detailedForecast')
              + state_attr('sensor.solcast_pv_forecast_forecast_tomorrow', 'detailedForecast')
            %}
            {{ forecast 
              | map(attribute='period_start') 
              | map('as_timestamp') 
              | map('timestamp_custom', '%Y-%m-%dT%H:%M:%S%z')
              | list 
            }}
          pv_estimate: >
            {% set forecast =
              state_attr('sensor.solcast_pv_forecast_forecast_today', 'detailedForecast')
              + state_attr('sensor.solcast_pv_forecast_forecast_tomorrow', 'detailedForecast')
            %}
            {{ forecast 
              | map(attribute='pv_estimate') 
              | map('multiply', 1000.0) 
              | list 
            }}
          pv_estimate10: >
            {% set forecast =
              state_attr('sensor.solcast_pv_forecast_forecast_today', 'detailedForecast')
              + state_attr('sensor.solcast_pv_forecast_forecast_tomorrow', 'detailedForecast')
            %}
            {{ forecast 
              | map(attribute='pv_estimate10') 
              | map('multiply', 1000.0) 
              | list 
            }}
          pv_estimate90: >
            {% set forecast =
              state_attr('sensor.solcast_pv_forecast_forecast_today', 'detailedForecast')
              + state_attr('sensor.solcast_pv_forecast_forecast_tomorrow', 'detailedForecast')
            %}
            {{ forecast
              | map(attribute='pv_estimate90')
              | map('multiply', 1000.0) 
              | list 
            }}
```

### 4. InfluxDB Integration

Configure the InfluxDB integration in Home Assistant.

See the [InfluxDB documentation](https://www.home-assistant.io/integrations/influxdb/)
for configuration.

```yaml
influxdb:
  host: f6484555-influxdb-vistalba
  port: 8086
  database: home_assistant
  username: YOUR_USERNAME
  password: YOUR_PASSWORD
  include:
    entities:
      - sensor.solcast_pv_forecast
      - sensor.open_meteo_forecast
      - ...
```

## Endpoints

### Config API

The `/api/config` endpoint is called from a Home Assistant automation using a
`rest_command`. It registers the Home Assistant sensor mappings for later use by the
training and optimization endpoints.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: config
      payload: |
        {{ {
          "solar": "sensor.pv_output",
          "baseload": "sensor.stroomverbruik_base_load",
          "heat_pump": {
            "state": "sensor.ecodan_heatpump_ca09ec_status_bedrijf",
            "power": "sensor.warmtepomp_geschat_vermogen",
            "supply_temperature": "sensor.ecodan_heatpump_ca09ec_aanvoer_temp",
            "return_temperature": "sensor.ecodan_heatpump_ca09ec_retour_temp",
            "flow": "sensor.ecodan_heatpump_ca09ec_flow_waarde",
            "compressor_frequency": "sensor.ecodan_heatpump_compressor_frequentie",
            "booster": "binary_sensor.ecodan_heatpump_ca09ec_status_booster_heater", 
            "boiler": {
              "setpoint": "sensor.ecodan_heatpump_ca09ec_sww_setpoint_waarde",
              "top_temperature": "sensor.ecodan_heatpump_ca09ec_sww_2e_temp_sensor",
              "bottom_temperature": "sensor.ecodan_heatpump_ca09ec_sww_huidige_temp",
              "ambient_temperature": "sensor.xiaomi_sensor_3_temperatuur",
              "volume": 200,
              "target_temperature": [
                ["18:00", 45.0],
                ["19:00", 10.0]
              ]
            }
          },
          "climate": {
            "temperature": "sensor.danfoss_15_temperature",
            "setpoint": ["climate.danfoss_icon_woonkamer", "temperature"],
            "target_temperature": [
              ["18:00", 20.0],
              ["22:00", 19.0]
            ],
            "zone_temperatures": [
              [6.0, "sensor.danfoss_0_temperature"],
              [10.0, "sensor.danfoss_1_temperature"],
              [19.0, "sensor.danfoss_2_temperature"],
              [7.0, "sensor.danfoss_3_temperature"],
              [42.0, "sensor.danfoss_15_temperature"]
            ],
            "ceiling_height": 2.6,
            "south_glazing": [
              [12.0, ["cover.woonkamer", "current_position"]],
              [2.0, ["cover.slaapkamer_groot_1", "current_position"]],
              [2.0, ["cover.slaapkamer_groot_2", "current_position"]],
              [2.0, ["cover.slaapkamer_achter_3", "current_position"]]
            ]
          },
          "forecast": {
            "solcast": "sensor.solcast_pv_forecast",
            "open_meteo": "sensor.open_meteo_forecast"
          },
          "presence": [
            "device_tracker.iphone_gerjan",
            "device_tracker.phone_partner"
          ]
        } | to_json }}
```

#### Building thermal model

The `climate` block also configures the zone thermal model (`features/building.py`),
which covers both space heating and space cooling with one set of parameters:

- `zone_temperatures` - the room sensors belonging to the zone as
  `[floor_area_m2, sensor]` pairs, averaged into one representative temperature weighted
  by those areas. A whole-dwelling balance weighs house-wide delivered heat and baseload
  against it, so a single room would be an arbitrary sample; averaging also suppresses
  the sensors' 0.1 K reporting steps. The weighting matters because thermostats are not
  spread evenly: on this installation four of five zones are upstairs, so a plain
  average counted the ground floor for 20% of a dwelling temperature it is half the area
  of, biasing it by +0.085 K (p95 0.42 K) as warm air collects upstairs. Weighting by
  area brings that to +0.008 K. Leave out any unheated space - an attic tracks outdoor
  temperature, not the zone.
- `ceiling_height` - net floor-to-ceiling height. The zone's air volume is derived from
  it and the areas above rather than configured separately, so the same geometry is
  stated once. It covers only rooms that have a sensor, slightly undercounting hall and
  landing, which is fine because the volume only bounds a heat capacity - and for
  `building_lumped` it does not bind at all.
- `south_glazing` - the south-facing windows as `[glass_m2, cover]` pairs, one per
  shutter (use `[glass_m2, null]`, or just `glass_m2`, for glass with no shutter). Read
  the areas off the floor plan as structural opening width times height, since a roller
  shutter covers the whole opening: on this installation the living room front is 2760 +
  1500 + 1000 mm wide at 2.35 m high (12.36 m2), and upstairs a 3230 mm opening at 1.5 m
  split over two shutters (2.42 m2 each) plus 975 + 1055 mm on one shutter (3.05 m2).
  Split one opening across however many shutters cover it, so that each cover carries
  the area it actually shades.

  The total is the upper bound on the identified effective aperture
  `a_eff_m2 = area x g-value x frame factor`, so the g-value does not have to be known;
  leave the list out and solar gain is bounded to zero. The angle of incidence is
  **not** in that product - the transposition to the facade already accounts for it -
  which is why `validate()` reports `implausible_aperture` when the identified
  `a_eff_m2` falls below 15% of the glass area: no real glazing has a g-value that low,
  so such a fit has not identified the window however well it converged.

  Shading is weighted by these areas, which matters whenever the shutters do not move
  together - on this installation they are almost uncorrelated, and an unweighted
  average would be off by a median of 9 percentage points. Outdoor air temperature comes
  from Open-Meteo's `temperature_2m`. The heat pump's own outdoor sensor can be used
  instead via
  `heat_pump.outdoor_temperature`, but only if the unit really stands outdoors - on this
  installation it sits in a shed and reads on average 1.8 K warm with a strongly diurnal
  bias, so it is deliberately left unset.

Internal gains need no configuration: appliance heat comes from the existing
`baseload` sensor (household electricity ends up as heat indoors) and occupancy from the
existing `presence` trackers.

#### Two structures, calibrated side by side

The same `climate` configuration feeds two models, and both are calibrated against the
same data on purpose:

- **`building`** - two nodes: room air and thermal mass (screed, internal walls),
  coupled to each other, with solar and floor heat entering the mass. Physically the
  more faithful of the two, because floor heating really does reach the room with a lag.
  Its mass temperature is never measured.
- **`building_lumped`** - one node: a single capacity covering everything that stores
  heat, and one conductance to outdoors. Cruder - it cannot represent the floor being
  warmer than the air, so it cannot describe charging the screed as storage - but its
  state *is* the measurement, so nothing has to be inferred.

Which one to trust is not an assumption but a measurement. `validate()` reports
`skill_vs_persistence`: how much better the model is than simply holding the last
reading for the length of a rollout. That baseline is free and hard to beat indoors, so
a model at or below zero skill is adding error rather than information, and
`validate()` warns. On this installation's cooling-season data:

|                        | `building`         | `building_lumped` |
|------------------------|--------------------|-------------------|
| MAE over 6 h rollouts  | 0.362 K            | **0.139 K**       |
| persistence baseline   | 0.161 K            | 0.161 K           |
| `skill_vs_persistence` | **-1.24**          | **+0.14**         |
| `aperture_fraction`    | 0.065 (impossible) | **0.419**         |

The two-node model's hidden mass state was the largest single error source, and it also
swallowed the solar gain - the same data gives an ordinary double-glazing g-value once
that node is gone. This is not evidence that two nodes are wrong, but that data with the
floor circuit active in 2.4% of quarter hours cannot identify them. Re-test once a
heating season has produced daily floor-circuit transitions.

Calibrate them with `"target": "building"` and `"target": "building_lumped"`. There is
no cooling COP model: the heat pump's efficiency while cooling needs an EER formulation
with the roles of water and outdoor air reversed, which `HeatPumpCOPIdentifier` does not
implement (see its class docstring).

### Update API

The `/api/update` endpoint is called from a Home Assistant automation using a
`rest_command`.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: update
      payload: |
        {{ {} | to_json }}
```

### Fit API

The `/api/fit` endpoint is called from a Home Assistant automation using a
`rest_command`.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: fit
      payload: |
        {{ {
          "target": "baseload"
          "days": 90,
        } | to_json }}
```

### Predict API

The `/api/predict` endpoint is called from a Home Assistant automation using a
`rest_command`.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: predict
      payload: |
        {{ {
          "target": "baseload",
          "steps": 192
        } | to_json }}
```

### Backtest API

The `/api/backtest` endpoint is called from a Home Assistant automation using a
`rest_command`.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: backtest
      payload: |
        {{ {
          "target": "baseload",
          "days": 90,
          "steps": 192
        } | to_json }}
```

### Tune API

The `/api/tune` endpoint is called from a Home Assistant automation using a
`rest_command`.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: tune
      payload: |
        {{ {
          "target": "baseload",
          "days": 90,
          "trails": 5
        } | to_json }}
```

### Calibrate API

The `/api/calibrate` endpoint is called from a Home Assistant automation using a
`rest_command`.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: calibrate
      payload: |
        {{ {
          "target": "boiler"
          "days": 90,
        } | to_json }}
```

### Validate API

The `/api/validate` endpoint is called from a Home Assistant automation using a
`rest_command`.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: validate
      payload: |
        {{ {
          "target": "boiler"
          "days": 90,
        } | to_json }}
```

### Optimize API

The `/api/optimize` endpoint is called from a Home Assistant automation using a
`rest_command`.

Example automation action:

```yaml
actions:
  - action: rest_command.home_optimizer_api
    data:
      endpoint: optimize
      payload: |
        {{ {
          "steps": 192
        } | to_json }}
```

## Sensors

Home Optimizer writes the following entities to Home Assistant:

| Entity                                    | Description                                                |
|-------------------------------------------|------------------------------------------------------------|
| `binary_sensor.home_optimizer_dhw_status` | `on` while the current quarter hour is planned to heat DHW |
| `sensor.home_optimizer_dhw_start`         | Start of the next planned DHW run (`unknown` if none)      |
| `sensor.home_optimizer_dhw_setpoint`      | DHW setpoint (°C) for that run (`unknown` if none)         |

## Development

To run Home Optimizer locally:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e ".[dev]"
  ./run.sh
```
