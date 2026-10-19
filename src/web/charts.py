from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from domain.config import HeatPumpStates
from domain.models import BoilerThermalModel
from domain.mpc import MPCConfig
from domain.state import (
    BacktestResult,
    BoilerMeasurement,
    Schedule,
    SeriesPoint,
    State,
)
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


def handed_over(schedule: Schedule) -> list[SeriesPoint]:
    """The heat pump's planned power, the quarter the tank's block ends in with
    the zone's run planned from the next as it is run: the tank's part, then
    the zone for the rest of the quarter (see Optimization.publish_zone). The
    plan serves one demand per quarter, which left the rest of that quarter
    idle, a dip the heat pump does not make. The tank's part is its heat over
    the quarter before's, which it ran in full; the zone's power is its
    first quarter's. Heating only: cooling stops between the two."""

    power = schedule.heat_pump.power
    heat = {p.time: p.value for p in schedule.heat_pump.heat}
    tank = {p.time: p.value for p in schedule.heat_pump.boiler.on}
    zone = {p.time: p.value for p in schedule.building.on}
    drawn = list(power)

    for i in range(1, len(power) - 1):
        before, now, after = (power[j].time for j in (i - 1, i, i + 1))

        if not (
            tank.get(now)
            and not tank.get(after)
            and zone.get(after)
            and not zone.get(now)
            and heat.get(after, 0.0) > 0.0
        ):
            continue

        share = (
            min(heat.get(now, 0.0) / heat[before], 1.0)
            if tank.get(before) and heat.get(before, 0.0) > 0.0
            else 1.0
        )
        drawn[i] = SeriesPoint(
            time=now, value=power[i].value + (1.0 - share) * power[i + 1].value
        )

    return drawn


def spans(changes: list[SeriesPoint], end: datetime) -> list[tuple]:
    """(start, end, value) of each change's value, held until the next change
    and the last until `end`."""

    ends = [p.time for p in changes[1:]] + [end]

    return [
        (p.time, until, p.value)
        for p, until in zip(changes, ends, strict=False)
        if until > p.time
    ]


def joined(measured: list, planned: list) -> list:
    """One line of both: the measured points, then the planned ones after the
    last of them."""

    if not measured:
        return planned

    return measured + [p for p in planned if p.time > measured[-1].time]


def tank_line(measured: list, state: State) -> list:
    """The tank measured, then as planned (see joined). Through a DHW run
    under way as the state has it (see BoilerSchedule.run), in place of the
    sensors, which lag the heat while the tank heats."""

    run = state.schedule.heat_pump.boiler.run

    if run:
        measured = [p for p in measured if p.time < run[0].time] + run

    return joined(measured, state.schedule.heat_pump.boiler.temperatures)


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
    state: State,
    boiler_model: BoilerThermalModel | None = None,
    states: HeatPumpStates | None = None,
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

    def band(
        lower: list,
        upper: list,
        fillcolor: str,
        legendgroup: str,
        row: int,
        edge_line: dict | None = None,
    ) -> None:
        """The area between two series, toggled with the line it belongs to
        and left out of the hover."""

        for points, fill in ((lower, None), (upper, "tonexty")):
            points = in_window(points)
            fig.add_trace(
                go.Scatter(
                    x=[to_local_time(p.time) for p in points],
                    y=[p.value for p in points],
                    mode="lines",
                    line=edge_line or dict(width=0),
                    fill=fill,
                    fillcolor=fillcolor,
                    showlegend=False,
                    legendgroup=legendgroup,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )

    band(
        state.forecast.solcast.p10,
        state.forecast.solcast.p90,
        fillcolor="rgba(255, 161, 90, 0.08)",
        legendgroup="solar",
        row=1,
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
            handed_over(state.schedule),
        ),
        unit="W",
        row=1,
        col=1,
        line=dict(width=2, color="#AB63FA", shape="hv"),
    )

    # The comfort band the plan keeps the thermostat's room in, a shade off
    # the background: hard at its edges, the wanted temperature dotted in the
    # room's colour within it.
    # Straight between the quarters, not stepped: a change slopes over the
    # quarter before its time and holds from that time, as set.
    band(
        state.schedule.building.minimum_temperature,
        state.schedule.building.maximum_temperature,
        fillcolor="rgba(255, 255, 255, 0.03)",
        legendgroup="climate",
        row=2,
        edge_line=dict(width=0.5, color="rgba(255, 255, 255, 0.2)"),
    )

    # The slab under the thermostat's room, as the filter infers it. No sensor
    # measures this, so unlike the room it shows something the other lines
    # cannot: it lags the room by hours and swings less, which is the storage
    # an MPC would be charging.
    series(
        "Thermal mass",
        state.predictions.thermal_mass,
        row=2,
        col=1,
        line=dict(width=1, color="#E58606", shape="spline"),
        unit="°C",
        decimal=2,
    )

    series(
        "Climate target",
        state.schedule.building.target_temperature,
        row=2,
        col=1,
        line=dict(width=1, color="rgba(254, 203, 82, 0.5)", dash="dash"),
        legendgroup="climate",
        unit="°C",
        decimal=1,
    )

    # The thermostat's room, which comfort is set in and the plan is judged
    # on: measured, then where the plan takes it (see zone_observation).
    series(
        "Living room",
        joined(
            state.measurements.building.zone_temperature,
            state.schedule.building.temperatures,
        ),
        row=2,
        col=1,
        line=dict(width=2, color="#FECB52", shape="spline"),
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
        line=dict(width=1, color="#636EFA", shape="spline"),
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
        line=dict(width=1, color="#AB63FA", shape="hv"),
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
        line=dict(width=1, color="#8C9BAB"),
        visible="legendonly",
        unit="°C",
        decimal=1,
    )

    # The tank mixed - the temperature the optimizer plans from (see
    # BoilerThermalModel.mixed_temperature) - so measured and planned are one
    # "Boiler temperature" line. The sensors' own
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
        "Boiler target",
        state.schedule.heat_pump.boiler.target_temperature,
        row=3,
        col=1,
        line=dict(width=1, color="rgba(25, 211, 243, 0.5)", shape="hv", dash="dash"),
        unit="°C",
        decimal=1,
    )

    series(
        "Boiler temperature",
        tank_line(measured_average, state),
        row=3,
        col=1,
        line=dict(width=2, color="#19D3F3", shape="spline"),
        legendgroup="boiler_temperature",
        unit="°C",
        decimal=2,
    )

    # What the heat pump serves, behind what it acts on: the tank's runs
    # behind the boiler, the floor's behind the climate. Drawn after the
    # traces, as plotly leaves a shape out of a subplot without any. Measured
    # from its state's changes, to the second, then as planned. Only
    # the configured states (see HeatPumpStates): any other, a legionella run
    # say, is none of these. A defrost on top of the run it interrupts: it
    # takes its heat from that run's water.
    states = states or HeatPumpStates()
    quarter = timedelta(hours=MPCConfig().step_hours)
    mode = state.measurements.heat_pump.mode
    cooling = bool(mode) and str(mode[-1].value).startswith(states.cooling)
    dhw = ("DHW", "#19D3F3")
    zone = {
        states.heating: ("Heating", "#FECB52"),
        states.cooling: ("Cooling", "#FECB52"),
    }
    shown: set[str] = set()

    def shade(
        runs: list[tuple], name: str, color: str, row: int, opacity: float = 0.06
    ) -> None:
        for start, end in runs:
            start, end = max(start, window_start), min(end, window_end)

            if start >= end:
                continue

            fig.add_vrect(
                x0=to_local_time(start),
                x1=to_local_time(end),
                fillcolor=color,
                opacity=opacity,
                line_width=0,
                layer="below",
                name=name,
                legendgroup=name,
                showlegend=name not in shown,
                row=row,
                col=1,
            )
            shown.add(name)

    def planned(on: list[SeriesPoint]) -> list[tuple]:
        """The plan's runs from the update on, consecutive quarters as one."""

        changes = [p for i, p in enumerate(on) if i == 0 or p.value != on[i - 1].value]

        return [
            (max(start, state.updated), end)
            for start, end, value in spans(
                changes, on[-1].time + quarter if on else now
            )
            if value and end > state.updated
        ]

    measured_states = spans(state.measurements.heat_pump.state_changes, state.updated)

    rows = {states.dhw: 3, states.heating: 2, states.cooling: 2}

    for value, (name, color) in {states.dhw: dhw, **zone}.items():
        shade(
            [(start, end) for start, end, v in measured_states if v == value],
            name,
            color,
            row=rows[value],
        )

    shade(planned(state.schedule.heat_pump.boiler.on), *dhw, row=3)
    shade(
        planned(state.schedule.building.on),
        *zone[states.cooling if cooling else states.heating],
        row=2,
    )

    for start, end, on in spans(state.measurements.heat_pump.defrost, state.updated):
        serving = [v for s, e, v in measured_states if s <= start < e]

        # Both panels where the state does not say which run it was in.
        for row in [rows[serving[0]]] if serving and serving[0] in rows else [2, 3]:
            if on:
                shade([(start, end)], "Defrost", "#FFFFFF", row=row, opacity=0.25)

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
