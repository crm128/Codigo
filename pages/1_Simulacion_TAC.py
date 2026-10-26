"""Pagina Streamlit para simulacion de arranque en lazo abierto.

Responsabilidades:
- capturar parametros de operacion desde sidebar,
- ejecutar simulacion de planta sin PID,
- presentar resultados numericos y graficos,
- exportar datos de la corrida a CSV.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from app_core.simulate import (
    run_open_loop_simulation,
    SimulationResult
)
from app_core.types import (
    ReactorConditions, 
    SimulationConfig 
)
from app_core.plant import (
    compute_energy_balance,
    build_energy_balance_dataframe,
)

st.set_page_config(page_title="Arranque TAC")
st.title("Simulación de Arranque - Reactor TAC")

FLOW_UNIT = "lb-mol/h"
TEMP_UNIT = "°F"
UA_UNIT = "BTU/h·°F"
CONC_UNIT = "lb-mol/ft³"

V_FT3 = (1 / 7.484) * 500

default_unit = {
    "Fa0": 80.0,
    "Fb0": 1000.0,
    "Fm0": 100.0,
    "mc": 1000.0,
    "T0": 75.0,
    "Ta1": 60.0,
    "UA": 16000.0,
}

st.sidebar.header("Parámetros de operación")

Fa0 = st.sidebar.number_input(
    f"Flujo A ({FLOW_UNIT})",
    min_value=1.0,
    max_value=200.0,
    value=default_unit["Fa0"],
)
Fb0 = st.sidebar.number_input(
    f"Flujo B ({FLOW_UNIT})",
    min_value=1.0,
    max_value=2000.0,
    value=default_unit["Fb0"],
)
Fm0 = st.sidebar.number_input(
    f"Flujo Inerte ({FLOW_UNIT})",
    min_value=0.0,
    max_value=500.0,
    value=default_unit["Fm0"],
)
mc = st.sidebar.number_input(
    f"Flujo refrigerante ({FLOW_UNIT})",
    min_value=100.0,
    max_value=5000.0,
    value=default_unit["mc"],
)
T0 = st.sidebar.slider(
    f"Temperatura inicial ({TEMP_UNIT})",
    min_value=30,
    max_value=130,
    value=int(default_unit["T0"]),
)
Ta1 = st.sidebar.slider(
    f"Temperatura refrigerante ({TEMP_UNIT})",
    min_value=20,
    max_value=200,
    value=int(default_unit["Ta1"]),
)
UA = st.sidebar.number_input(
    f"UA ({UA_UNIT})",
    min_value=1000.0,
    max_value=50000.0,
    value=default_unit["UA"],
)

t_final = st.sidebar.slider("Tiempo de simulación (h)", 1.0, 12.0, 4.0)
n_points = st.sidebar.slider("Puntos de evaluación", 100, 600, 300, step=25)

conditions = ReactorConditions(
    Fa0=Fa0,
    Fb0=Fb0,
    Fm0=Fm0,
    mc=mc,
    UA=UA,
    V=V_FT3,
    T0=T0,
    Ta1=Ta1,
)

sim_cfg = SimulationConfig(t_start=0.0, t_end=t_final, n_points=n_points)

try:
    result = run_open_loop_simulation(conditions, sim_cfg)
except Exception as exc:
    st.error(f"No fue posible ejecutar la simulación: {exc}")
    st.stop()

safety_limit = 180.0

steady_time = result.steady_time


def build_dataframe(sim_res: SimulationResult) -> pd.DataFrame:
    """Arma el DataFrame principal para visualizacion/exportacion."""
    data = {
        "t (h)": sim_res.t,
        f"Ca ({CONC_UNIT})": sim_res.Ca,
        f"Cb ({CONC_UNIT})": sim_res.Cb,
        f"Cc ({CONC_UNIT})": sim_res.Cc,
        f"Cm ({CONC_UNIT})": sim_res.Cm,
        f"T ({TEMP_UNIT})": sim_res.T,
        "X (-)": sim_res.conversion,
        f"mc ({FLOW_UNIT})" : sim_res.mc,
    }
    return pd.DataFrame(data)


df = build_dataframe(result)

tab_info, tab_results, tab_plots, tab_config = st.tabs(
    ["Funcionamiento", "Resultados numéricos", "Gráficas", "Configuración"]
)

with tab_info:
    st.markdown(
        """
        ### Descripción
        El simulador resuelve los balances de materia y energía del TAC durante el arranque.
        Todas las entradas y resultados se muestran en unidades imperiales (sistema inglés).
        """
    )
    st.markdown(
        f"- Volumen del reactor: {V_FT3:.2f} ft³\n"
        f"- Tiempo de simulación: {t_final:.2f} h"
    )

with tab_results:
    st.subheader("Resumen al final del intervalo simulado")
    st.write(f"**Ca final:** {result.Ca[-1]:.3f} {CONC_UNIT}")
    st.write(f"**Cb final:** {result.Cb[-1]:.3f} {CONC_UNIT}")
    st.write(f"**Cc final:** {result.Cc[-1]:.3f} {CONC_UNIT}")
    st.write(f"**Cm final:** {result.Cm[-1]:.3f} {CONC_UNIT}")
    st.write(f"**T final:** {result.T[-1]:.2f} {TEMP_UNIT}")
    st.write(f"**Conversión final de A:** {result.conversion[-1]:.3f}")
    st.write(f"**Tiempo de estabilización estimado:** {steady_time:.2f}")
    
    # Calculo del balance de energia en el estado final (o estacionario)
    eb = compute_energy_balance(
        Ca=result.Ca[-1],
        T=result.T[-1],
        Fa0=Fa0,
        Fb0=Fb0,
        Fm0=Fm0,
        V=V_FT3,
        UA=UA,
        Ta1=Ta1,
        mc=mc,
        T0=T0,
    )
    df_eb = build_energy_balance_dataframe(eb)

    st.subheader("Cierre Numérico del Balance de Energía (Estado Estacionario)")
    st.markdown(
        "Verificación formal de la conservación de energía: "
        "$\\dot{Q}_{\\text{gen}} + \\dot{Q}_{\\text{camisa}} + \\dot{Q}_{\\text{corrientes}} = \\text{Residuo} \\approx 0$."
    )
    st.dataframe(df_eb, hide_index=True, use_container_width=True)

    col_eb1, col_eb2, col_eb3 = st.columns(3)
    col_eb1.metric("Calor Generado (Químico)", f"{eb['Q_gen_kw']:.2f} kW", f"{eb['Q_gen_btu']:,.0f} BTU/h")
    col_eb2.metric("Retiro por Camisa + Corrientes", f"{(eb['Q_camisa_kw'] + eb['Q_corrientes_kw']):.2f} kW", delta=f"{eb['Q_camisa_pct'] + eb['Q_corrientes_pct']:.1f}%")
    col_eb3.metric("Residuo de Cierre", f"{eb['residuo_kw']:.4f} kW", delta=f"{eb['residuo_pct']:.2f}%", delta_color="normal")

    st.subheader("Datos detallados de la corrida dinámica")
    st.dataframe(df)
    st.download_button(
        "Descargar datos (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"simulacion_tac.csv",
        mime="text/csv",
    )

with tab_plots:
    st.subheader("Evolución temporal")
    fig_ca = go.Figure()
    fig_ca.add_trace(
        go.Scatter(
            x=result.t,
            y=result.Ca,
            mode="lines",
            name="Ca",
        )
    )
    fig_ca.update_layout(
        title="Concentración de A vs tiempo",
        xaxis_title="Tiempo (h)",
        yaxis_title=f"Ca ({CONC_UNIT})",
    )
    if steady_time is not None:
        fig_ca.add_vline(
            x=steady_time,
            line=dict(color="blue", dash="dot"),
            annotation_text=f"Estabilización {steady_time:.2f} h",
            annotation_position="top right",
        )
    st.plotly_chart(fig_ca, width="stretch")

    fig_temp = go.Figure()
    fig_temp.add_trace(
        go.Scatter(
            x=result.t,
            y=result.T,
            mode="lines",
            name="T",
            line=dict(color="red"),
        )
    )
    fig_temp.add_hline(
        y=safety_limit,
        line=dict(color="red", dash="dash"),
        annotation_text=f"Límite {safety_limit:.1f} {TEMP_UNIT}",
        annotation_position="top left",
    )
    fig_temp.update_layout(
        title="Temperatura vs tiempo",
        xaxis_title="Tiempo (h)",
        yaxis_title=f"Temperatura ({TEMP_UNIT})",
    )
    if steady_time is not None:
        fig_temp.add_vline(
            x=steady_time,
            line=dict(color="blue", dash="dot"),
            annotation_text=f"Estabilización {steady_time:.2f} h",
            annotation_position="top right",
        )
    st.plotly_chart(fig_temp, width="stretch")

    fig_phase = go.Figure()
    fig_phase.add_trace(
        go.Scatter(
            x=result.T,
            y=result.Ca,
            mode="lines",
            name="Trayectoria",
            line=dict(color="green"),
        )
    )
    fig_phase.add_vline(
        x=safety_limit,
        line=dict(color="red", dash="dash"),
        annotation_text=f"Límite {safety_limit:.1f} {TEMP_UNIT}",
        annotation_position="top left",
    )
    fig_phase.update_layout(
        title="Diagrama Ca - T",
        xaxis_title=f"T ({TEMP_UNIT})",
        yaxis_title=f"Ca ({CONC_UNIT})",
    )
    st.plotly_chart(fig_phase, width="stretch")

with tab_config:
    st.subheader("Parámetros utilizados (entradas)")
    st.json(
        {
            "Fa0": Fa0,
            "Fb0": Fb0,
            "Fm0": Fm0,
            "mc": mc,
            "UA": UA,
            "T0": T0,
            "Ta1": Ta1,
            "V": V_FT3,
            "t_final": t_final,
            "n_points": n_points,
        }
    )
