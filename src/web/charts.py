from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from domain.time import to_local_series, to_local_time
from domain.types import BacktestResult, State


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
            connectgaps=True,
            hovertemplate=f"%{{y:.{decimal}f}} {unit}<extra>%{{fullData.name}}</extra>",
        ),
        row=row,
        col=col,
    )


def dashboard_chart(state: State) -> str:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Power", "Climate", "Boiler"),
        row_heights=[0.5, 0.25, 0.25],
    )

    fig.add_trace(
        go.Scatter(
            x=[to_local_time(p.time) for p in state.forecast.solcast.p10],
            y=[p.value for p in state.forecast.solcast.p10],
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
            x=[to_local_time(p.time) for p in state.forecast.solcast.p90],
            y=[p.value for p in state.forecast.solcast.p90],
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

    add_series(
        fig,
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

    add_series(
        fig,
        "Solar",
        state.measurements.solar,
        line=dict(width=1.5, color="#FFA15A", shape="spline"),
        legendgroup="solar",
        showlegend=False,
        unit="W",
        row=1,
        col=1,
    )

    add_series(
        fig,
        "Solar",
        state.predictions.solar,
        line=dict(width=1.5, color="#FFA15A", shape="spline"),
        legendgroup="solar",
        unit="W",
        row=1,
        col=1,
    )

    add_series(
        fig,
        "Baseload",
        state.measurements.baseload,
        unit="W",
        row=1,
        col=1,
        line=dict(width=1, color="rgba(239, 85, 59, 0.5)", shape="spline"),
        legendgroup="baseload",
    )

    add_series(
        fig,
        "Baseload",
        state.predictions.baseload,
        line=dict(width=1, color="rgba(239, 85, 59, 0.5)", shape="spline"),
        legendgroup="baseload",
        showlegend=False,
        unit="W",
        row=1,
        col=1,
    )

    add_series(
        fig,
        "Heat pump",
        state.measurements.heat_pump.power,
        unit="W",
        row=1,
        col=1,
        line=dict(width=2, color="#AB63FA", shape="hv"),
        legendgroup="heat_pump",
        showlegend=False,
    )

    add_series(
        fig,
        "Heat pump",
        state.schedule.heat_pump.power,
        unit="W",
        row=1,
        col=1,
        line=dict(width=2, color="#AB63FA", shape="hv"),
        legendgroup="heat_pump",
    )

    add_series(
        fig,
        "Climate target",
        state.schedule.climate.target_temperature,
        row=2,
        col=1,
        line=dict(width=1, color="#FECB52", shape="hv", dash="dot"),
        unit="°C",
        decimal=1,
    )

    add_series(
        fig,
        "Climate temp",
        state.measurements.climate.temperature,
        row=2,
        col=1,
        line=dict(width=2, color="#FECB52", shape="spline"),
        unit="°C",
        decimal=2,
    )

    # add_series(
    #     fig,
    #     "Climate setpoint",
    #     state.measurements.climate.setpoint,
    #     row=2,
    #     col=1,
    #     line=dict(width=1, color="#FF6692", shape="hv", dash="dot"),
    #     unit="°C",
    #     decimal=1,
    # )

    add_series(
        fig,
        "Outside",
        state.forecast.open_meteo.temperature,
        row=2,
        col=1,
        line=dict(width=1, color="rgba(255, 255, 255, 0.4)"),
        visible="legendonly",
        unit="°C",
        decimal=1,
    )

    add_series(
        fig,
        "Boiler temperature",
        state.schedule.heat_pump.boiler.temperatures,
        row=3,
        col=1,
        line=dict(width=2, color="#19D3F3", shape="spline"),
        legendgroup="boiler_temperature",
        unit="°C",
        decimal=2,
    )

    add_series(
        fig,
        "Boiler target",
        state.schedule.heat_pump.boiler.target_temperature,
        row=3,
        col=1,
        line=dict(width=1, color="#19D3F3", shape="hv", dash="dot"),
        unit="°C",
        decimal=1,
    )

    add_series(
        fig,
        "Boiler bottom",
        state.measurements.heat_pump.boiler.bottom_temperature,
        row=3,
        col=1,
        line=dict(width=1, color="#636EFA", shape="spline"),
        legendgroup="boiler_bottom",
        showlegend=False,
        unit="°C",
        decimal=1,
    )

    add_series(
        fig,
        "Boiler top",
        state.measurements.heat_pump.boiler.top_temperature,
        row=3,
        col=1,
        line=dict(width=2, color="#19D3F3", shape="spline"),
        legendgroup="boiler_top",
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

    for i, bp in enumerate(result.points):
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
