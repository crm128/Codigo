"""Pagina Streamlit para validar P/PI/PID ante perturbacion tipo escalon.

La pagina usa el banco simplificado de `app_core.classic_pid` y compara:
- desviacion maxima post-escalon,
- IAE post-perturbacion,
- error estacionario final.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_core.classic_pid import (
    ClassicPlantConfig,
    DisturbanceStepConfig,
    run_classic_pid_disturbance_simulation,
)
from app_core.types import PidConfig, SimulationConfig

st.set_page_config(page_title="PID Clásico - Perturbación")
st.title("Validación Clásica del PID ante Perturbación Escalón")


def post_step_metrics(t, y, y_set, error, step_time, sat_high, sat_low, tol=0.02):
    """Calcula métricas del transitorio una vez aplicada la perturbación.

    Parametros
    ----------
    t, y, y_set, error:
        Series de tiempo de una corrida.
    step_time:
        Instante de inicio de la perturbación.
    sat_high, sat_low:
        Banderas de saturación de actuador (reservadas para futuras métricas).
    tol:
        Banda absoluta usada para una métrica de recuperación (actualmente no
        expuesta en la tabla final, pero mantenida para extensión).
    """
    idx = int(np.searchsorted(t, step_time))
    t_post = t[idx:] - step_time
    y_post = y[idx:]
    ysp_post = y_set[idx:]
    e_post = error[idx:]
    peak = float(np.max(np.abs(y_post - ysp_post)))
    iae = float(np.trapezoid(np.abs(e_post), t_post))
    window = max(10, len(e_post) // 10)
    offset = float(np.mean(e_post[-window:]))
    sat = float(np.mean(np.logical_or(sat_high[idx:], sat_low[idx:])))

    recovery = np.nan
    inside = np.abs(y_post - ysp_post) <= tol
    for i in range(len(inside)):
        if inside[i] and np.all(inside[i:]):
            recovery = float(t_post[i])
            break

    return {
        "Pico de desviación": peak,
        "IAE post-perturbación": iae,
        "Error estacionario": offset
    }


st.markdown(
    """
Ensayo dinámico clásico: planta de primer orden y perturbación escalón de carga.

Objetivo: contrastar `P`, `PI` y `PID` con expectativas bibliográficas:
- `P` deja offset.
- `PI` elimina offset estacionario.
- `PID` mejora el transitorio respecto de `PI` (menor desviación pico, según sintonía).
"""
)

st.sidebar.header("Planta clásica (1er orden)")
K_process = st.sidebar.number_input("Ganancia de proceso Kp_proc", -5.0, 5.0, 1.0, step=0.05)
tau_process = st.sidebar.number_input("Constante de tiempo tau (h)", 0.1, 20.0, 1.8, step=0.1)

st.sidebar.header("Perturbación")
step_time = st.sidebar.number_input("Tiempo de escalón (h)", 0.0, 50.0, 2.0, step=0.1)
step_value = st.sidebar.number_input("Magnitud de escalon", -5.0, 5.0, 1.0, step=0.1)

st.sidebar.header("Controlador base")
y_set = st.sidebar.number_input("Setpoint", -5.0, 5.0, 0.0, step=0.1)
u_bias = st.sidebar.number_input("Bias de actuador", -10.0, 10.0, 0.0, step=0.1)
u_min = st.sidebar.number_input("u minimo", -20.0, 20.0, -5.0, step=0.1)
u_max = st.sidebar.number_input("u maximo", u_min, 20.0, 5.0, step=0.1)
Kp = st.sidebar.number_input("Kp", 0.0, 200.0, 2.0, step=0.1)
Ki = st.sidebar.number_input("Ki", 0.0, 200.0, 2.0, step=0.1)
Kd = st.sidebar.number_input("Kd", 0.0, 200.0, 0.2, step=0.05)
tau_d = st.sidebar.number_input("Tau derivativo (h)", 0.0, 5.0, 0.08, step=0.01)
aw_gain = st.sidebar.number_input("Ganancia anti-windup Kb", 0.01, 500.0, 2.0, step=0.1)
rate_max = st.sidebar.number_input("Rate limit |du/dt|", 0.0, 200.0, 0.0, step=0.5)

st.sidebar.header("Simulación")
t_end = st.sidebar.number_input("Tiempo final (h)", 1.0, 80.0, 20.0, step=0.5)
n_points = st.sidebar.slider("Puntos", 200, 3000, 1500, step=100)
tol_recovery = st.sidebar.number_input("Banda de recuperación (abs)", 0.001, 1.0, 0.02, step=0.001, format="%.3f")

plant_cfg = ClassicPlantConfig(K_process=float(K_process), tau_process=float(tau_process))
dist_cfg = DisturbanceStepConfig(step_time=float(step_time), step_value=float(step_value))
sim_cfg = SimulationConfig(t_start=0.0, t_end=float(t_end), n_points=int(n_points))
rate_cfg = float(rate_max) if rate_max > 0.0 else None

base = dict(
    T_set=float(y_set),
    mc_bias=float(u_bias),
    mc_min=float(u_min),
    mc_max=float(u_max),
    tau_d=float(tau_d),
    sp_profile="constant",
    sp_t_ramp=0.0,
    dual_gains=False,
    anti_windup_gain=float(aw_gain),
    rate_max=rate_cfg,
)

cfg_p = PidConfig(Kp=float(Kp), Ki=0.0, Kd=0.0, **base)
cfg_pi = PidConfig(Kp=float(Kp), Ki=float(Ki), Kd=0.0, **base)
cfg_pid = PidConfig(Kp=float(Kp), Ki=float(Ki), Kd=float(Kd), **base)

# Simulaciones comparativas con exactamente la misma planta/perturbacion.
res_p = run_classic_pid_disturbance_simulation(plant_cfg, cfg_p, sim_cfg, dist_cfg)
res_pi = run_classic_pid_disturbance_simulation(plant_cfg, cfg_pi, sim_cfg, dist_cfg)
res_pid = run_classic_pid_disturbance_simulation(plant_cfg, cfg_pid, sim_cfg, dist_cfg)

# Metricas enfocadas en el comportamiento post-escalon.
metrics_p = post_step_metrics(
    res_p.t, res_p.y, res_p.y_set, res_p.error, dist_cfg.step_time, res_p.sat_high, res_p.sat_low, 
)
metrics_pi = post_step_metrics(
    res_pi.t, res_pi.y, res_pi.y_set, res_pi.error, dist_cfg.step_time, res_pi.sat_high, res_pi.sat_low, 
)
metrics_pid = post_step_metrics(
    res_pid.t, res_pid.y, res_pid.y_set, res_pid.error, dist_cfg.step_time, res_pid.sat_high, res_pid.sat_low, 
)

st.subheader("Comparación P vs PI vs PID")
metrics_df = pd.DataFrame(
        [
            {"Control": "P", **metrics_p},
            {"Control": "PI", **metrics_pi},
            {"Control": "PID", **metrics_pid},
        ]
    ).set_index("Control")
st.dataframe(metrics_df)

fig_dist = go.Figure()
fig_dist.add_trace(go.Scatter(x=res_pid.t, y=res_pid.disturbance, mode="lines", name="d(t)"))
fig_dist.update_layout(title="Perturbación aplicada", xaxis_title="Tiempo (h)", yaxis_title="d")
st.plotly_chart(fig_dist, width="stretch")

fig_y = go.Figure()
fig_y.add_trace(go.Scatter(x=res_p.t, y=res_p.y, mode="lines", name="y (P)"))
fig_y.add_trace(go.Scatter(x=res_pi.t, y=res_pi.y, mode="lines", name="y (PI)"))
fig_y.add_trace(go.Scatter(x=res_pid.t, y=res_pid.y, mode="lines", name="y (PID)"))
fig_y.add_trace(
        go.Scatter(x=res_pid.t, y=res_pid.y_set, mode="lines", name="Setpoint", line=dict(color="black", dash="dash"))
    )
fig_y.add_vline(x=dist_cfg.step_time, line=dict(color="red", dash="dot"))
fig_y.update_layout(title="Salida del proceso y(t)", xaxis_title="Tiempo (h)", yaxis_title="y")
st.plotly_chart(fig_y, width="stretch")

fig_u = go.Figure()
fig_u.add_trace(go.Scatter(x=res_p.t, y=res_p.u_act, mode="lines", name="u_act (P)"))
fig_u.add_trace(go.Scatter(x=res_pi.t, y=res_pi.u_act, mode="lines", name="u_act (PI)"))
fig_u.add_trace(go.Scatter(x=res_pid.t, y=res_pid.u_act, mode="lines", name="u_act (PID)"))
fig_u.add_vline(x=dist_cfg.step_time, line=dict(color="red", dash="dot"))
fig_u.update_layout(title="Acción de control", xaxis_title="Tiempo (h)", yaxis_title="u")
st.plotly_chart(fig_u, width="stretch")

fig_pid_terms = go.Figure()
fig_pid_terms.add_trace(go.Scatter(x=res_pid.t, y=res_pid.p_term, mode="lines", name="Termino P"))
fig_pid_terms.add_trace(go.Scatter(x=res_pid.t, y=res_pid.i_term, mode="lines", name="Termino I"))
fig_pid_terms.add_trace(go.Scatter(x=res_pid.t, y=res_pid.d_term, mode="lines", name="Termino D"))
fig_pid_terms.add_vline(x=dist_cfg.step_time, line=dict(color="red", dash="dot"))
fig_pid_terms.update_layout(
        title="Descomposición PID (escenario PID)",
        xaxis_title="Tiempo (h)",
        yaxis_title="Contribución",
    )
st.plotly_chart(fig_pid_terms, width="stretch")
