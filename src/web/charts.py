from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from domain.models import BoilerThermalModel
from domain.mpc import MPCConfig
from domain.state import BacktestResult, BoilerMeasurement, SeriesPoint, State
from domain.time import local_day_start, to_local_series, to_local_time


def add_series(
    fig: go.Figure,
    name: str,
    points: list,
    line: dict | None = None,
    legendgroup: str | None = None,
    showlegend: bool = True,
    visible: str | None = None,
    decimal: int | None = 0,
    unit: str | None = "",
    row: int | None = None,
    col: int | None = None,
    connectgaps: bool = True,
) -> None:
    if line is None:
        line = dict(width=2)

    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in points],
            y=[p.value for p in points],
            mode="lines",
            name=name,
            line=line,
            legendgroup=legendgroup,
            showlegend=showlegend,
            visible=visible,
            connectgaps=connectgaps,
            hovertemplate=f"%{{y:.{decimal}f}} {unit}<extra>%{{fullData.name}}</extra>",
        ),
        row=row,
        col=col,
    )


def broken_between_runs(points: list, step: timedelta) -> list:
    """Points that exist only while something runs, with a gap (NaN) a step
    after each run ends, so a line drawn through them stops there rather than
    bridging the time nothing ran."""

    broken = []

    for previous, point in zip([None, *points], points, strict=False):
        if previous is not None and point.time - previous.time > step:
            broken.append(SeriesPoint(time=previous.time + step, value=float("nan")))
        broken.append(point)

    return broken


def joined(measured: list, planned: list) -> list:
    """One line of both: the measured points, then the planned ones after the
    last of them."""

    if not measured:
        return planned

    return measured + [p for p in planned if p.time > measured[-1].time]


def continued(measured: list, planned: list) -> list:
    """The planned points from where the measured ones end, starting at the
    last measurement: a plan starts from the state it was made in, which the
    measurements have moved on from since, so drawn whole it overlaps them with
    a line of its own instead of running on from what really happened."""

    if not measured:
        return planned

    last = measured[-1]

    return [last] + [point for point in planned if point.time > last.time]


def read_at(points: list[SeriesPoint], updated: datetime) -> list[SeriesPoint]:
    """Each ended quarter's last reading at the quarter's end, when it was
    read. InfluxDB labels the quarter at its start (see
    StateManager._dataset), where a heating tank was drawn a quarter early -
    flat through the running quarter, and then jumping to the plan. The
    running quarter is left to the plan, as in ended(): drawn at the update
    it put a point between the quarters, and the line bent there."""

    quarter = timedelta(hours=MPCConfig().step_hours)

    return [
        SeriesPoint(time=p.time + quarter, value=p.value)
        for p in ended(points, updated)
    ]


def ended(points: list[SeriesPoint], updated: datetime) -> list[SeriesPoint]:
    """Only the quarters that had ended at the update. The running quarter's
    mean covers only its first minutes, and a power sensor that has not
    reported yet in it is filled with 0 (see StateManager._dataset): a run
    that had just started was drawn at 0 W for that quarter, in place of the
    plan, which does cover it."""

    quarter = timedelta(hours=MPCConfig().step_hours)

    return [p for p in points if p.time + quarter <= updated]


def mixed_tank(
    boiler: BoilerMeasurement, model: BoilerThermalModel | None
) -> list[SeriesPoint]:
    """The measured tank as the optimizer plans with it: mixed (see
    BoilerThermalModel.mixed_temperature), or the two sensors' average without
    a calibrated model."""

    bottom_by_time = {p.time: p.value for p in boiler.bottom_temperature}
    stratification_by_time = {p.time: p.value for p in boiler.stratification}

    return [
        SeriesPoint(
            time=p.time,
            value=model.mixed_temperature(
                p.value, bottom_by_time[p.time], stratification_by_time.get(p.time)
            )
            if model is not None
            else (p.value + bottom_by_time[p.time]) / 2.0,
        )
        for p in boiler.top_temperature
        if p.time in bottom_by_time
    ]


def dashboard_chart(
    state: State, boiler_model: BoilerThermalModel | None = None
) -> str:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Power", "Climate", "Boiler"),
        row_heights=[0.4, 0.3, 0.3],
    )

    # Exactly today and tomorrow in local time. Points outside are dropped, not
    # just scrolled out of view: plotly's y autorange would otherwise still
    # scale to them (e.g. a baseload prediction running past tomorrow).
    now = datetime.now(timezone.utc)
    window_start = local_day_start(now)
    window_end = local_day_start(now, days=2)

    def in_window(points: list) -> list:
        return [p for p in points if window_start <= p.time < window_end]

    def series(name: str, points: list, **kwargs) -> None:
        add_series(fig, name, in_window(points), **kwargs)

    solcast_p10 = in_window(state.forecast.solcast.p10)
    solcast_p90 = in_window(state.forecast.solcast.p90)

    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in solcast_p10],
            y=[p.value for p in solcast_p10],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            legendgroup="solar",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in solcast_p90],
            y=[p.value for p in solcast_p90],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 161, 90, 0.1)",
            showlegend=False,
            legendgroup="solar",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    series(
        "Solcast",
        state.forecast.solcast.p50,
        line=dict(
            width=1, color="rgba(255, 161, 90, 0.45)", dash="dot", shape="spline"
        ),
        legendgroup="solar",
        showlegend=False,
        unit="W",
        row=1,
        col=1,
    )

    series(
        "Solar",
        joined(state.measurements.solar, state.predictions.solar),
        line=dict(width=1.5, color="#FFA15A", shape="spline"),
        legendgroup="solar",
        unit="W",
        row=1,
        col=1,
    )

    series(
        "Baseload",
        joined(state.measurements.baseload, state.predictions.baseload),
        unit="W",
        row=1,
        col=1,
        line=dict(width=1, color="rgba(239, 85, 59, 0.5)", shape="spline"),
        legendgroup="baseload",
    )

    # The upper edge of the band the optimizer plans the surplus with (see
    # MPCInput.baseload_p10_w): where the load may run up to.
    series(
        "Baseload p90",
        state.predictions.baseload_p90,
        line=dict(width=0.8, color="rgba(239, 85, 59, 0.45)", dash="dot"),
        legendgroup="baseload",
        showlegend=False,
        unit="W",
        row=1,
        col=1,
    )

    series(
        "Heat pump",
        joined(
            ended(state.measurements.heat_pump.power, state.updated),
            state.schedule.heat_pump.power,
        ),
        unit="W",
        row=1,
        col=1,
        line=dict(width=2, color="#AB63FA", shape="hv"),
    )

    # The heat it delivers: into the tank, and into or out of the floor.
    series(
        "Heat pump heat",
        state.schedule.heat_pump.heat,
        unit="W",
        row=1,
        col=1,
        line=dict(width=1, color="#FECB52", shape="hv", dash="dash"),
        visible="legendonly",
    )

    series(
        "Climate target",
        state.schedule.building.target_temperature,
        row=2,
        col=1,
        line=dict(width=1, color="#FECB52", shape="hv", dash="dot"),
        visible="legendonly",
        unit="°C",
        decimal=1,
    )

    series(
        "Climate temp",
        read_at(state.measurements.building.temperature, state.updated),
        row=2,
        col=1,
        line=dict(width=1.5, color="#FECB52", shape="spline"),
        unit="°C",
        decimal=2,
    )

    # The zone average the model actually predicts - `Climate temp` above is
    # the single thermostat the setpoint refers to, which is a different
    # quantity and would make the model look biased against it - measured, then
    # where the plan takes it: the plan predicts the same reading.
    series(
        "Zone temp",
        joined(
            state.measurements.building.zone_temperature,
            state.schedule.building.temperatures,
        ),
        row=2,
        col=1,
        line=dict(width=1, color="#00CC96", shape="spline"),
        unit="°C",
        decimal=2,
    )

    # The building's thermal mass - screed and internal walls - as the filter
    # infers it. No sensor measures this, so unlike the air trace it shows
    # something the other lines cannot: it lags the air by hours and swings
    # about a third less, which is the storage an MPC would be charging.
    series(
        "Thermal mass",
        state.predictions.thermal_mass,
        row=2,
        col=1,
        line=dict(width=1, color="#FFA15A", shape="spline"),
        unit="°C",
        decimal=2,
    )

    # What a cooled floor, and uninsulated pipes, must stay above: measured,
    # then as the plan forecast it.
    series(
        "Dew point",
        joined(state.measurements.building.dew_point, state.predictions.dew_point),
        row=2,
        col=1,
        line=dict(width=1, color="#B6E880", shape="spline"),
        visible="legendonly",
        unit="°C",
        decimal=1,
    )

    # The supply the plan runs the floor at - while cooling, the
    # setpoint it would give the heat pump - only while a run holds it.
    series(
        "Supply plan",
        broken_between_runs(
            state.schedule.building.supply, timedelta(hours=MPCConfig().step_hours)
        ),
        row=2,
        col=1,
        line=dict(width=1, color="#636EFA", shape="hv"),
        visible="legendonly",
        unit="°C",
        decimal=1,
        connectgaps=False,
    )

    series(
        "Outside",
        state.forecast.open_meteo.temperature,
        row=2,
        col=1,
        line=dict(width=1, color="rgba(255, 255, 255, 0.4)"),
        visible="legendonly",
        unit="°C",
        decimal=1,
    )

    # The tank mixed - the temperature the optimizer plans from (see
    # BoilerThermalModel.mixed_temperature) - so the measured line continues
    # straight into the planned "Boiler temperature" line. The sensors' own
    # average without a calibrated model, which then is the same thing. The two
    # sensors are drawn beside it: their stratification is what the mixed
    # temperature is read from.
    measured = state.measurements.heat_pump.boiler
    boiler = BoilerMeasurement(
        top_temperature=read_at(measured.top_temperature, state.updated),
        bottom_temperature=read_at(measured.bottom_temperature, state.updated),
        stratification=read_at(measured.stratification, state.updated),
    )
    measured_average = mixed_tank(boiler, boiler_model)

    for name, points in (
        ("Boiler bottom", boiler.bottom_temperature),
        ("Boiler top", boiler.top_temperature),
    ):
        series(
            name,
            points,
            row=3,
            col=1,
            line=dict(width=1, color="#636EFA", dash="dot"),
            legendgroup="boiler_temperature",
            showlegend=False,
            unit="°C",
            decimal=1,
        )

    series(
        "Boiler temperature",
        continued(measured_average, state.schedule.heat_pump.boiler.temperatures),
        row=3,
        col=1,
        line=dict(width=2, color="#19D3F3", shape="spline"),
        legendgroup="boiler_temperature",
        unit="°C",
        decimal=2,
    )

    series(
        "Boiler target",
        state.schedule.heat_pump.boiler.target_temperature,
        row=3,
        col=1,
        line=dict(width=1, color="#19D3F3", shape="hv", dash="dot"),
        unit="°C",
        decimal=1,
    )

    series(
        "Boiler temperature",
        measured_average,
        row=3,
        col=1,
        line=dict(width=2, color="#19D3F3", shape="spline"),
        legendgroup="boiler_temperature",
        showlegend=False,
        unit="°C",
        decimal=1,
    )

    fig.update_layout(
        title=dict(
            text="Dashboard",
            x=0.01,
            y=0.97,
            font=dict(
                size=22,
                color="#eeeeee",
            ),
        ),
        template="plotly_dark",
        paper_bgcolor="#2b2b2b",
        plot_bgcolor="#2b2b2b",
        margin=dict(
            l=70,
            r=20,
            t=60,
            b=10,
        ),
        height=700,
        font=dict(
            color="#cccccc",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(size=12),
        ),
        xaxis2=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(size=12),
        ),
        xaxis3=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Power (W)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        yaxis2=dict(
            title="Temp (°C)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        yaxis3=dict(
            title="Temp (°C)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            y=-0.10,
            x=0.5,
            xanchor="center",
            font=dict(size=12),
        ),
        hovermode="x unified",
    )

    # Points mark the start of their step, so tomorrow's last one is at 23:45:
    # ending the axis there makes it equal to the data's own extent, so plotly's
    # double-click autosize lands on exactly the same view.
    fig.update_xaxes(
        range=[window_start, window_end - timedelta(hours=MPCConfig().step_hours)],
        hoverformat="%b %-d, %Y, %H:%M",
    )

    fig.add_vline(
        x=to_local_time(datetime.now(timezone.utc)).timestamp() * 1000,
        line_width=1,
        line_color="#ffffff",
        layer="above",
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )


def backtest_chart(result: BacktestResult | None) -> str:
    if result is None:
        return ""

    fig = go.Figure()

    for bp in result.points:
        df = pd.DataFrame(bp.points)
        df["x_time"] = to_local_series(pd.to_datetime(df["time"]))

        fig.add_trace(
            go.Scatter(
                x=df["x_time"],
                y=df["value"],
                mode="lines",
                name=bp.label,
                legendgroup=bp.group,
                showlegend=not bp.group,
                line=dict(width=1, color=bp.color),
                visible=True,
                connectgaps=True,
                hovertemplate=(
                    f"%{{y:.1f}} {result.unit}<extra>%{{fullData.name}}</extra>"
                ),
            )
        )

    groups: set[str] = set()

    for bp in result.points:
        if bp.group not in groups and bp.group:
            groups.add(bp.group)

            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=bp.group,
                    legendgroup=bp.group,
                    showlegend=True,
                    line=dict(width=1, color=bp.color),
                )
            )

    fig.update_layout(
        title=dict(
            text=(
                f"{result.name.capitalize()} backtest - "
                f"MAE {result.mae:.3f}, RMSE {result.rmse:.3f}, R2 {result.r2:.3f}"
            ),
            x=0.01,
            y=0.95,
            font=dict(
                size=22,
                color="#eeeeee",
            ),
        ),
        template="plotly_dark",
        paper_bgcolor="#2b2b2b",
        plot_bgcolor="#2b2b2b",
        margin=dict(
            l=70,
            r=20,
            t=60,
            b=10,
        ),
        font=dict(
            color="#cccccc",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=f"{result.label} ({result.unit})",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor="center",
            font=dict(size=12),
            groupclick="togglegroup",
        ),
        hovermode="x unified",
    )

    return fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
    )
