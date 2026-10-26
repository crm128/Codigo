"""Pagina Streamlit para simulacion con control PID de temperatura.

Incluye:
- parametrizacion de planta y controlador,
- ejecucion de corridas comparativas (incluyendo barrido de Kp),
- calculo de metricas (IAE, overshoot, TVu),
- visualizacion y exportacion de resultados.
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from dataclasses import replace as dc_replace

from app_core.types import (
    ReactorConditions,
    SimulationConfig, 
    SimulationResult, 
    PidConfig 
)
from app_core.simulate import (
    run_pid_control_simulation,
    run_open_loop_simulation,
    build_sp_fun,
)
from app_core.plant import (
    compute_energy_balance,
    build_energy_balance_dataframe,
)
from app_core.metrics import (
    compute_metrics,
    compute_steady_state_report,
)
from app_core.parameters import NOMINAL_PARAMS

st.set_page_config(page_title="Control PID TAC")
st.title("Control PID de Temperatura - Reactor TAC")

FLOW_UNIT = "lb-mol/h"
TEMP_UNIT = "°F"
UA_UNIT = "BTU/h·°F"
CONC_UNIT = "lb-mol/ft³"

V_FT3 = (1 / 7.484) * 500

default = {
    "Fa0": NOMINAL_PARAMS.Fa0,
    "Fb0": NOMINAL_PARAMS.Fb0,
    "Fm0": NOMINAL_PARAMS.Fm0,
    "mc": NOMINAL_PARAMS.mc,
    "T0": NOMINAL_PARAMS.T0,
    "Ta1": NOMINAL_PARAMS.Ta1,
    "UA": NOMINAL_PARAMS.UA,
    "T_set": NOMINAL_PARAMS.T_SET_NOMINAL,
    "mc_min": NOMINAL_PARAMS.mc_min,
    "mc_max": NOMINAL_PARAMS.mc_max,
    "mc_bias": NOMINAL_PARAMS.mc_bias,
}



#Comienzo del sidebar

st.sidebar.header("Condiciones del reactor")

Fa0 = st.sidebar.slider(f"Flujo A ({FLOW_UNIT})", 50.0, 100.0, default["Fa0"], 0.1)

st.sidebar.write("**Relación cte de alimentación**")
Fb0 = Fa0 * 1000 / 80
st.sidebar.write(f"**Flujo B:** {Fb0} {FLOW_UNIT}")
Fm0 = Fa0 * 100 / 80
st.sidebar.write(f"**Flujo Inerte:** {Fm0} {FLOW_UNIT}")

mc = default["mc"]
T0 = default["T0"]
Ta1 = default["Ta1"]
UA = default["UA"]

Kp_def, Ki_def, Kd_def, tau_d_def = 200, 5.0, 0.0, 0.0
Kp2_def, Ki2_def, Kd2_def, tau_d2_def = 200, 5.0, 0.0, 0.0


# Inputs manuales de planta para calcular sintoni­a (2º orden)
beta_value = 0.5
derivative_N = 10.0
rate_limit = 0.0


st.sidebar.header("Controlador PID")

# Checkboxes iniciales
T_set = st.sidebar.number_input("Setpoint de temperatura (°F)", 40.0, 200.0, (default["T_set"]))
mc_bias = st.sidebar.number_input("mc base (lb-mol/h)", 10.0, 6000.0, default["mc_bias"])
mc_min = st.sidebar.number_input("mc mínimo (lb-mol/h)", 10.0, mc_bias, default["mc_min"])
mc_max = st.sidebar.number_input("mc máximo (lb-mol/h)", mc_bias, 10000.0, default["mc_max"])


# Perfil de setpoint durante el arranque (servo)
sp_profile_label ="Lineal" 
sp_profile_map = {
    "Constante": "constant",
    "Lineal": "linear",
}
sp_profile = sp_profile_map.get(sp_profile_label, "constant")
sp_t_ramp = 0.0
if sp_profile != "constant":
    sp_t_ramp = float(
        st.sidebar.number_input("Tiempo de rampa (h)", 0.1, 12.0, 2.0, step=0.1)
    )

t_final = st.sidebar.slider("Tiempo de simulación (h)", 1.0, 40.0, 4.0)
n_points = st.sidebar.slider("Puntos de evaluación", 200, 500, 300, step=25)
tolerance_input = st.sidebar.slider("Tolerancia al setpoint (°F)", 0, 10, 2)
do_kp_sweep = st.sidebar.checkbox("Realizar barrido de Kp (lento)", value=False)
rate_limit_value = float(rate_limit)
rate_limit_cfg = rate_limit_value if rate_limit_value > 0.0 else None

st.sidebar.subheader("Ganancias PID - Servo")
Kp = st.sidebar.number_input("Kp (servo)", -1000.0, 1000.0, float(Kp_def), step=1.0)
Ki = st.sidebar.number_input("Ki (servo)", -1000.0, 1000.0, float(Ki_def), step=0.1)
Kd = st.sidebar.number_input("Kd (servo)", 0.0, 1000.0, float(Kd2_def), step=0.1)
tau_d = tau_d_def

st.sidebar.subheader("Ganancias PID - Regulación")
Kp2 = st.sidebar.number_input("Kp (regulación)", -1_000_000.0, 1_000_000.0, float(Kp2_def), step=1.0)
Ki2 = st.sidebar.number_input("Ki (regulación)", -1_000_000.0, 1_000_000.0, float(Ki2_def), step=0.1)
Kd2 = st.sidebar.number_input("Kd (regulación)", 0.0, 1_000_000.0, float(Kd2_def), step=0.1)
tau_d2 = tau_d2_def

st.session_state["saved_tunings"] = []         
saved_entries = st.session_state["saved_tunings"]
## Fin del sidebar


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

t_switch = sp_t_ramp

sp_fun_ref = build_sp_fun(sp_profile, T0, float(T_set), sp_t_ramp)

pid_cfg = PidConfig(
    T_set=T_set,
    mc_bias=mc_bias,
    mc_min=mc_min,
    mc_max=mc_max,
    Kp=Kp,
    Ki=Ki,
    Kd=Kd,
    tau_d=tau_d,
    sp_profile=sp_profile,
    sp_t_ramp=sp_t_ramp,
    rate_max=rate_limit_cfg,
    anti_windup_gain=NOMINAL_PARAMS.Kb,
    dual_gains=True,
    Kp2=Kp2,
    Ki2=Ki2,
    Kd2=Kd2,
    tau_d2=tau_d2,
    t_switch=float(t_switch),
)


sim_cfg = SimulationConfig(
    t_start=0.0,
    t_end=t_final,
    n_points=n_points,
)


@st.cache_data(show_spinner=False)
def get_cached_pid_control(cond, p_cfg, s_cfg):
    return run_pid_control_simulation(cond, p_cfg, s_cfg)

@st.cache_data(show_spinner=False)
def get_cached_open_loop(cond, s_cfg):
    return run_open_loop_simulation(cond, s_cfg)

tab_summary, tab_plots, tab_long, tab_data = st.tabs(["Resumen", "Gráficas", "Horizonte largo", "Datos"])

with st.spinner("Ejecutando simulaciones del lazo PID..."):
    pid_result = get_cached_pid_control(conditions, pid_cfg, sim_cfg)
    
    # Corrida base sin controlador para referencia comparativa.
    pid_display_base = get_cached_open_loop(conditions, sim_cfg)
    pid_display = pid_result
    kp_sweep_results = []
    # Barrido simple de Kp para comparar sensibilidad del lazo.
    if do_kp_sweep:
        for kp_gain in range(25, 250, 20):
            kp_variant_cfg = dc_replace(
                pid_cfg,
                Kp=float(kp_gain),
                Kp2=float(kp_gain),
            )
            try:
                sim_variant = get_cached_pid_control(conditions, kp_variant_cfg, sim_cfg)
                kp_sweep_results.append((f"Kp={kp_gain}", sim_variant))
            except Exception as exc:
                st.warning(f"No se pudo simular Kp={kp_gain}: {exc}")
    
    auto_simulation_results = []
    
    comparison_results = []
    for entry in saved_entries:
        saved_cfg = PidConfig(**entry["pid_cfg"])
        comparison_result = get_cached_pid_control(conditions, saved_cfg, sim_cfg)
        comparison_results.append((entry["label"], comparison_result))
    
    comparison_results.append(("Actual", pid_display))
    comparison_results.extend(kp_sweep_results)


with tab_summary:
    st.subheader("Indicadores finales")
    st.write(f"**Ca final:** {pid_display.Ca[-1]:.3f} {CONC_UNIT}")
    st.write(f"**T final:** {pid_display.T[-1]:.2f} {TEMP_UNIT}")
    st.write(f"**Tasa de fraccion aparente de desaparicion:** {pid_display.conversion[-1]:.3f}")
    st.write(f"**mc final:** {pid_display.mc[-1]:.2f} {FLOW_UNIT}")
    
    st.subheader("Cierre de Balance de Energía en Régimen Controlado")
    eb_pid = compute_energy_balance(
        Ca=pid_display.Ca[-1],
        T=pid_display.T[-1],
        Fa0=Fa0,
        Fb0=Fb0,
        Fm0=Fm0,
        V=V_FT3,
        UA=UA,
        Ta1=Ta1,
        mc=pid_display.mc[-1],
        T0=T0,
    )
    df_eb_pid = build_energy_balance_dataframe(eb_pid)
    st.dataframe(df_eb_pid, hide_index=True, width='stretch')
    
    performance_rows = []
    series_frames = []

    def collect_metrics(label: str, result: SimulationResult) -> None:
        time_series = np.asarray(result.t, dtype=float)
        y_series = np.asarray(result.T, dtype=float)
        # Misma referencia Tsp(t) para todas las filas, incluida "Sin PID".
        ysp_series = np.array([sp_fun_ref(ti) for ti in time_series], dtype=float)
        u_series = np.asarray(result.mc, dtype=float)
        metrics = compute_metrics(time_series, y_series, ysp_series, u_series)
        performance_rows.append({
            "Sintonía": label,
            "IAE_seg (°F·h)": metrics["IAE_seg"],
            "IAE_fin (°F·h)": metrics["IAE_fin"],
            "ΔT_max (°F)": metrics["dT_max"],
            "Mp (%)": metrics["overshoot"],
            "TVu (lb-mol/h)": metrics["TVu"],
        })
        series_frames.append(pd.DataFrame({
            "sintonia": label,
            "t_h": time_series,
            "Tsp_F": ysp_series,
            "T_F": y_series,
            "mc_lbmol_h": u_series,
        }))

    collect_metrics("Sin PID", pid_display_base)
    for label, result in comparison_results:
        collect_metrics(label, result)

    performance_df = pd.DataFrame(performance_rows)
    series_df = pd.concat(series_frames, ignore_index=True)

    T_set_display = T_set
    mc_min_display = mc_min
    mc_max_display = mc_max
    tolerance_display = float(tolerance_input)

    time_to_sp = None
    # Primer tiempo en banda de tolerancia contra setpoint.
    if getattr(pid_display_base, "T_set_series", None) is not None:
        for ti, temp, tsp in zip(pid_display_base.t, pid_display_base.T, pid_display_base.T_set_series):
            if abs(temp - tsp) <= tolerance_display:
                time_to_sp = ti
                break
    else:
        for ti, temp in zip(pid_display_base.t, pid_display_base.T):
            if abs(temp - T_set_display) <= tolerance_display:
                time_to_sp = ti
                break


    def build_dataframe(result: SimulationResult) -> pd.DataFrame:
        """Arma tabla de datos de salida para la vista y descarga CSV."""        
        st.caption(
            "IAE_seg: ∫|Tsp(t)−T|dt (métrica de comparación). IAE_fin: ∫|Tsp,f−T|dt. "
            "Mp = ΔT_max/(Tsp,f−T0)·100. TVu = Σ|Δmc|. Intervalo [0, t_final], "
            f"{n_points} puntos, misma referencia Tsp(t) para todas las filas."
        )
        data = {
            "t (h)": result.t,
            f"Ca ({CONC_UNIT})": result.Ca,
            f"T ({TEMP_UNIT})": result.T,
            f"Tsp ({TEMP_UNIT})": result.T_set_series,
            f"mc ({FLOW_UNIT})": result.mc,
            f"Error ({TEMP_UNIT})": result.error,
        }
        return pd.DataFrame(data)


    df = build_dataframe(pid_display_base)

    if not performance_df.empty:
        st.subheader("Comparación de sintonías")
        st.dataframe(performance_df.set_index("Sintonía"))
        st.download_button(
            "Descargar comparación (CSV)",

            data=performance_df.to_csv(index=False, sep ='\t', decimal='.').encode("utf-8"),
            file_name="comparacion_sintonias.csv",
            mime="text/csv",
        )

        st.caption(
            "IAE_seg: ∫|Tsp(t)−T|dt (métrica de comparación). IAE_fin: ∫|Tsp,f−T|dt. "
            "Mp = ΔT_max/(Tsp,f−T0)·100. TVu = Σ|Δmc|. Intervalo [0, t_final], "
            f"{n_points} puntos, misma referencia Tsp(t) para todas las filas."
        )
    
   
with tab_plots:
    fig_temp = go.Figure()
    fig_temp.add_trace(
        go.Scatter(
            x=pid_display_base.t,
            y=pid_display_base.T,
            mode="lines",
            name="Temperatura (Sin PID)",
        )
    )
    for label, result in comparison_results:
        fig_temp.add_trace(
            go.Scatter(
                x=result.t,
                y=result.T,
                mode="lines",
                name=f"Temperatura ({label})",
            )
        )
    if getattr(pid_display_base, "T_set_series", None) is not None:
        fig_temp.add_trace(
            go.Scatter(
                x=pid_display_base.t,
                y=pid_display_base.T_set_series,
                mode="lines",
                name="Setpoint",
                line=dict(color="orange", dash="dash"),
            )
        )
    else:
        fig_temp.add_hline(
            y=T_set_display,
            line=dict(color="orange", dash="dash"),
            annotation_text="Setpoint",
            annotation_position="top right",
        )
    fig_temp.update_layout(
        title="Seguimiento de temperatura",
        xaxis_title="Tiempo (h)",
        yaxis_title=f"Temperatura ({TEMP_UNIT})",
    )


    
    fig_temp.add_vline(x=float(t_switch), line=dict(color="red", dash="dash"))
    fig_temp.add_annotation(
        x=float(t_switch),
        y=1.02,
        xref="x",
        yref="paper",
        text="t_switch",
        showarrow=False,
        font=dict(color="red"),
    )
    st.plotly_chart(fig_temp, width="stretch")

    fig_mc = go.Figure()
    fig_mc.add_trace(
        go.Scatter(
            x=pid_display_base.t,
            y=pid_display_base.mc,
            mode="lines",
            name="mc",
        )
    )
    for label, result in comparison_results:
        if result.mc is None:
            continue
        fig_mc.add_trace(
            go.Scatter(
                x=result.t,
                y=result.mc,
                mode="lines",
                name=f"mc ({label})",
            )
        )
    fig_mc.add_hline(
        y=mc_min_display,
        line=dict(color="gray", dash="dot"),
        annotation_text="mc min",
        annotation_position="top left",
    )
    fig_mc.add_hline(
        y=mc_max_display,
        line=dict(color="gray", dash="dot"),
        annotation_text="mc max",
        annotation_position="top left",
    )
    fig_mc.update_layout(
        title="Perfil de mc",
        xaxis_title="Tiempo (h)",
        yaxis_title=f"mc ({FLOW_UNIT})",
    )


    fig_mc.add_vline(x=float(t_switch), line=dict(color="red", dash="dash"))
    fig_mc.add_annotation(
        x=float(t_switch),
        y=1.02,
        xref="x",
        yref="paper",
        text="t_switch",
        showarrow=False,
        font=dict(color="red"),
    )

        
    st.plotly_chart(fig_mc, width="stretch")

    if getattr(pid_display_base, "error", None) is not None:
        fig_error = go.Figure()
        fig_error.add_trace(
            go.Scatter(
                x=pid_display_base.t,
                y=pid_display_base.error,
                mode="lines",
                name="Error (actual)",
            )
        )
        for label, result in comparison_results:
            if getattr(result, "error", None) is None:
                continue
            fig_error.add_trace(
                go.Scatter(
                    x=result.t,
                    y=result.error,
                    mode="lines",
                    name=f"Error ({label})",
                )
            )
        fig_error.update_layout(
            title="Error de temperatura",
            xaxis_title="Tiempo (h)",
            yaxis_title=f"Error ({TEMP_UNIT})",
        )

        st.plotly_chart(fig_error, width="stretch")

with tab_long:
    st.subheader("Verificación del error estacionario (horizonte prolongado)")
    st.caption(
        f"Sintonía actual: Kp = {Kp:g}, Ki = {Ki:g} → Ti = Kp/Ki = "
        + (f"{Kp / Ki:.1f} h" if Ki > 0 else "∞ (sin acción integral)")
    )
    t_long = st.number_input("Horizonte prolongado (h)", 4.0, 500.0, 200.0, step=10.0)
    t_check = st.number_input("Instante de control (h)", 0.5, float(t_long), 4.0, step=0.5)
    sim_cfg_long = SimulationConfig(t_start=0.0, t_end=float(t_long), n_points=int(20 * t_long) + 1)
    with st.spinner("Simulando horizonte prolongado..."):
        long_res = get_cached_pid_control(conditions, pid_cfg, sim_cfg_long)

    ysp_long = np.asarray(long_res.T_set_series, dtype=float)
    rep = compute_steady_state_report(long_res.t, long_res.T, ysp_long, long_res.mc, t_check=float(t_check))
    XA_check = float(np.interp(t_check, long_res.t, long_res.conversion))

    def _fmt_t(x):
        return "no se alcanza en la corrida" if not np.isfinite(x) else f"{x:.1f} h"

    rep_df = pd.DataFrame([
        {"Magnitud": f"T({t_check:g} h)", "Valor": f"{rep['T_check']:.2f} {TEMP_UNIT}"},
        {"Magnitud": f"e({t_check:g} h) = Tsp - T", "Valor": f"{rep['e_check']:+.2f} {TEMP_UNIT}"},
        {"Magnitud": f"mc({t_check:g} h)", "Valor": f"{rep['u_check']:.1f} {FLOW_UNIT}"},
        {"Magnitud": f"XA({t_check:g} h)", "Valor": f"{XA_check:.3f}"},
        {"Magnitud": f"T({rep['t_end']:g} h)", "Valor": f"{rep['T_end']:.3f} {TEMP_UNIT}"},
        {"Magnitud": f"|e|({rep['t_end']:g} h)", "Valor": f"{abs(rep['e_end']):.3f} {TEMP_UNIT}"},
        {"Magnitud": f"mc({rep['t_end']:g} h)", "Valor": f"{rep['u_end']:.1f} {FLOW_UNIT}"},
        {"Magnitud": f"XA({rep['t_end']:g} h)", "Valor": f"{long_res.conversion[-1]:.3f}"},
        {"Magnitud": "t_s banda ±0.5 °F (medido)", "Valor": _fmt_t(rep["t_s(0.5)"])},
        {"Magnitud": "t_s banda ±0.1 °F (medido)", "Valor": _fmt_t(rep["t_s(0.1)"])},
        {"Magnitud": "τ_I modo lento (ajuste de la cola)", "Valor": _fmt_t(rep["tau_I"])},
        {"Magnitud": "t banda ±0.5 °F (estimado)", "Valor": _fmt_t(rep["t_est(0.5)"])},
        {"Magnitud": "t banda ±0.1 °F (estimado)", "Valor": _fmt_t(rep["t_est(0.1)"])},
        {"Magnitud": "Saturación del actuador", "Valor": (
            "sí" if (np.min(long_res.mc) <= mc_min + 1e-6 or np.max(long_res.mc) >= mc_max - 1e-6) else "no"
        )},
    ]).set_index("Magnitud")
    st.dataframe(rep_df, width="stretch")
    st.caption(
        "t_s: instante a partir del cual |e| permanece dentro de la banda hasta el final de la corrida. "
        "τ_I: constante de tiempo de ln|e| = a − t/τ ajustada para t ≥ instante de control."
    )

    fig_long = go.Figure()
    fig_long.add_trace(go.Scatter(x=long_res.t, y=long_res.T, mode="lines", name="T"))
    fig_long.add_trace(go.Scatter(x=long_res.t, y=ysp_long, mode="lines", name="Setpoint",
                                  line=dict(color="orange", dash="dash")))
    fig_long.add_hrect(y0=T_set - 0.5, y1=T_set + 0.5, fillcolor="green", opacity=0.10, line_width=0,
                       annotation_text="±0.5 °F", annotation_position="top right")
    fig_long.add_hrect(y0=T_set - 0.1, y1=T_set + 0.1, fillcolor="green", opacity=0.20, line_width=0)
    fig_long.add_vline(x=float(t_check), line=dict(color="red", dash="dot"),
                       annotation_text=f"{t_check:g} h", annotation_position="top")
    fig_long.update_layout(title="Temperatura en horizonte prolongado",
                           xaxis_title="Tiempo (h)", yaxis_title=f"Temperatura ({TEMP_UNIT})")
    st.plotly_chart(fig_long, width="stretch")

    fig_long_mc = go.Figure()
    fig_long_mc.add_trace(go.Scatter(x=long_res.t, y=long_res.mc, mode="lines", name="mc"))
    fig_long_mc.add_vline(x=float(t_check), line=dict(color="red", dash="dot"))
    fig_long_mc.update_layout(title="Caudal de refrigerante en horizonte prolongado",
                              xaxis_title="Tiempo (h)", yaxis_title=f"mc ({FLOW_UNIT})")
    st.plotly_chart(fig_long_mc, width="stretch")

    df_long = pd.DataFrame({
        "t (h)": long_res.t,
        f"T ({TEMP_UNIT})": long_res.T,
        f"Error ({TEMP_UNIT})": long_res.error,
        f"mc ({FLOW_UNIT})": long_res.mc,
        "XA": long_res.conversion,
    })
    st.download_button(
        "Descargar corrida larga (CSV)",
        data=df_long.to_csv(index=False, sep='\t', decimal=',').encode("utf-8"),
        file_name="control_pid_horizonte_largo.csv",
        mime="text/csv",
    )

with tab_data:
    df_actual = build_dataframe(pid_result)
    st.dataframe(df_actual)
    st.download_button(
        "Descargar datos (CSV)",
        data=df_actual.to_csv(index=False, sep ='\t', decimal=',').encode("utf-8"),
        file_name=f"control_pid.csv",
        mime="text/csv",
    )

