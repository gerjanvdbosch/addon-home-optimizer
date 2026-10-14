# Home Optimizer

A Home Assistant addon that learns your home's thermal behavior and uses model
predictive control to schedule your heat pump around solar production and electricity
prices.

> ⚠️ **Work in Progress**
>
> This project is currently under development and should be considered experimental.
> Features, configuration, and API endpoints may change at any time. It is not yet
> recommended for production use.

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
        dew_point_2m,\
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
          - dew_point_2m
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

### 4. Dynamic Prices (Template Sensor)

With a dynamic contract, this sensor lists the prices published by the
[EPEX Spot](https://github.com/brambo123/ha_epex_spot) integration.

```yaml
template:
  - sensor:
      - name: "Epex Energy Prices"
        unique_id: epex_energy_prices
        state: "{{ states('sensor.epex_spot_data_import_price') }}"
        attributes:
          time: >
            {{ state_attr('sensor.epex_spot_data_import_price', 'data')
              | map(attribute='start_time')
              | map('as_timestamp')
              | map('timestamp_custom', '%Y-%m-%dT%H:%M:%S%z')
              | list
            }}
          import: >
            {{ state_attr('sensor.epex_spot_data_import_price', 'data')
              | map(attribute='price_per_kwh')
              | list
            }}
          export: >
            {{ state_attr('sensor.epex_spot_data_export_price', 'data')
              | map(attribute='price_per_kwh')
              | list
            }}
```

### 5. InfluxDB Integration

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
      - sensor.epex_energy_prices
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
            "mode": "select.ecodan_heatpump_ca09ec_gekozen_werking_mode",
            "state": "sensor.ecodan_heatpump_ca09ec_status_bedrijf",
            "power": "sensor.warmtepomp_geschat_vermogen",
            "supply_temperature": "sensor.ecodan_heatpump_ca09ec_aanvoer_temp",
            "return_temperature": "sensor.ecodan_heatpump_ca09ec_retour_temp",
            "flow": "sensor.ecodan_heatpump_ca09ec_flow_waarde",
            "compressor_frequency": "sensor.ecodan_heatpump_compressor_frequentie",
            "booster": "binary_sensor.ecodan_heatpump_ca09ec_status_booster_heater", 
            "defrost": "binary_sensor.ecodan_heatpump_ca09ec_status_defrost", 
            "boiler": {
              "setpoint": "sensor.ecodan_heatpump_ca09ec_sww_setpoint_waarde",
              "top_temperature": "sensor.ecodan_heatpump_ca09ec_sww_2e_temp_sensor",
              "bottom_temperature": "sensor.ecodan_heatpump_ca09ec_sww_huidige_temp",
              "ambient_temperature": "sensor.ecodan_heatpump_ca09ec_zone_1_ruimte_temp",
              "volume": 200,
              "target_temperature": [
                ["18:00", 45.0],
                ["19:00", 10.0]
              ],
              "legionella": {
                "temperature": 60.0,
                "interval": 14
              }
            }
            "states": {
              "off": "Uit", 
              "dhw": "SWW", 
              "heating": "Verwarmen", 
              "cooling": "Koelen"
            },
          },
          "building": {
            "thermostat": {
              "temperature": "sensor.danfoss_15_temperature",
              "setpoint": ["climate.danfoss_icon_woonkamer", "temperature"]
            },
            "target_temperature": {
              "heating": [
                ["18:00", 20.0, 1.0, 1.5],
                ["22:00", 19.0, 1.0, 2.5]
              ],
              "cooling": [
                ["00:00", 24.0, 2.0, 1.0]
              ]
            },
            "comfort_weight": 0.5,
            "rooms": [
              [6.0, "sensor.danfoss_0_temperature"],
              [10.0, "sensor.danfoss_1_temperature"],
              [19.0, "sensor.danfoss_2_temperature"],
              [7.0, "sensor.danfoss_3_temperature"],
              [42.0, "sensor.danfoss_15_temperature"]
            ],
            "ceiling_height": 2.6,
            "south_glazing": [
              [6.5, ["cover.woonkamer", "current_position"], 20.0, true, [2.4, 2.4]],
              [0.9, ["cover.slaapkamer_groot_1", "current_position"]],
              [0.9, ["cover.slaapkamer_groot_2", "current_position"]],
              [0.9, ["cover.slaapkamer_achter_3", "current_position"]]
            ],
            "facade_azimuth": 155.0,
            "insulated_pipes": false,
            "dhw_after_cooling": true,
            "dew_point_margin": 1.0,
            "dew_points": [
              "sensor.xiaomi_sensor_1_condensatie_temperatuur", 
              "sensor.xiaomi_sensor_2_condensatie_temperatuur"
            ]
          },
          "prices": {
            "import": {
              "weekdays": [
                ["07:00", 0.213], 
                ["23:00", 0.225]
              ],
              "weekend": 0.225
            },
            "export": {
              "weekdays": [
                ["07:00", 0.077], 
                ["23:00", 0.089]
              ],
              "weekend": 0.089
            }
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
          "days": 365,
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
          "days": 365,
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
          "days": 365,
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
          "days": 365,
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
          "days": 365,
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

| Entity                                     | Description                                                                 |
|--------------------------------------------|-----------------------------------------------------------------------------|
| `binary_sensor.home_optimizer_dhw_status`  | `on` while the current quarter hour is planned to heat DHW                  |
| `sensor.home_optimizer_dhw_start`          | Start of the next planned DHW run (`unknown` if none)                       |
| `sensor.home_optimizer_dhw_setpoint`       | DHW setpoint (°C) for that run (`unknown` if none)                          |
| `binary_sensor.home_optimizer_zone_status` | `on` while the current quarter hour is planned to heat or cool the zone     |
| `sensor.home_optimizer_zone_start`         | Start of the zone run under way or the next one planned (`unknown` if none) |
| `sensor.home_optimizer_zone_supply`        | Supply setpoint (°C) while a zone run is under way (`unknown` otherwise)    |

## Development

To run Home Optimizer locally:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e ".[dev]"
  ./run.sh
```
