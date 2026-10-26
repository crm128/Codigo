from __future__ import annotations

"""Simulador clasico para validar un PID frente a una perturbacion escalon.

Este modulo implementa un banco de prueba simplificado:
- planta de primer orden (ganancia + constante de tiempo),
- perturbacion de carga tipo escalon,
- controlador PID con saturacion, anti-windup y rate limit opcional.

Se utiliza principalmente en la pagina `2_Perturbacion_Clasica_PID.py` para
comparar respuestas P vs PI vs PID en un escenario didactico.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .controller_pid import pid_command
from .types import PidConfig, SimulationConfig


@dataclass(frozen=True)
class ClassicPlantConfig:
    """Parametros de la planta lineal de primer orden.

    Modelo:
    `tau * dy/dt + y = -K_process*u + d`
    """

    #: Ganancia de proceso.
    K_process: float = -1.0
    #: Constante de tiempo de la planta.
    tau_process: float = 1.8
    #: Condicion inicial de salida.
    y0: float = 0.0


@dataclass(frozen=True)
class DisturbanceStepConfig:
    """Definicion de perturbacion tipo escalon."""

    #: Instante de aplicacion del escalon.
    step_time: float = 2.0
    #: Magnitud de perturbacion aplicada.
    step_value: float = 1.0


@dataclass
class ClassicSimulationResult:
    """Contenedor de series para analisis posterior de la corrida clasica."""

    #: Tiempo.
    t: np.ndarray
    #: Salida del proceso.
    y: np.ndarray
    #: Setpoint aplicado.
    y_set: np.ndarray
    #: Error de seguimiento.
    error: np.ndarray
    #: Perturbacion aplicada.
    disturbance: np.ndarray
    #: Comando PID no saturado.
    u_cmd: np.ndarray
    #: Comando PID saturado.
    u_sat: np.ndarray
    #: Senal efectiva de actuador.
    u_act: np.ndarray
    #: Aporte proporcional.
    p_term: np.ndarray
    #: Aporte integral.
    i_term: np.ndarray
    #: Aporte derivativo.
    d_term: np.ndarray
    #: Bandera de saturacion alta.
    sat_high: np.ndarray
    #: Bandera de saturacion baja.
    sat_low: np.ndarray


def _disturbance_value(t: float, cfg: DisturbanceStepConfig) -> float:
    """Evalua la perturbacion escalon en el instante `t`."""
    return float(cfg.step_value if t >= cfg.step_time else 0.0)


def _sp_value(t: float, pid_cfg: PidConfig, y0: float) -> float:
    """Calcula setpoint instantaneo (constante o rampa lineal)."""
    if pid_cfg.sp_profile == "linear" and pid_cfg.sp_t_ramp > 0.0:
        frac = max(0.0, min(1.0, t / pid_cfg.sp_t_ramp))
        return float(y0 + (pid_cfg.T_set - y0) * frac)
    return float(pid_cfg.T_set)


def _classic_ode(
    t: float,
    state: np.ndarray,
    plant_cfg: ClassicPlantConfig,
    pid_cfg: PidConfig,
    dist_cfg: DisturbanceStepConfig,
) -> np.ndarray:
    """Dinamica acoplada de planta + actuador + estados internos del PID.

    Estado interno:
    - `y`: salida del proceso,
    - `u_state`: estado de actuador cuando hay rate limit,
    - `i_term`: estado integrador,
    - `d_state`: filtro derivativo de primer orden.
    """
    y, u_state, i_term, d_state = state
    y_set = _sp_value(t, pid_cfg, plant_cfg.y0)

    pid_vals = pid_command(
        T=y,
        I_term=i_term,
        d_state=d_state,
        T_set=y_set,
        mc_bias=pid_cfg.mc_bias,
        Kp=pid_cfg.Kp,
        Kd=pid_cfg.Kd,
        mc_min=pid_cfg.mc_min,
        mc_max=pid_cfg.mc_max,
    )

    error = pid_vals["error"]
    u_cmd = pid_vals["u_cmd"]
    u_sat = pid_vals["u_sat"]
    sat_high = pid_vals["saturated_high"]
    sat_low = pid_vals["saturated_low"]

    rate_active = pid_cfg.rate_max is not None and np.isfinite(pid_cfg.rate_max) and pid_cfg.rate_max > 0.0
    if rate_active:
        du_dt = float(np.clip(u_sat - u_state, -pid_cfg.rate_max, pid_cfg.rate_max))
        u_act = u_state
    else:
        du_dt = 0.0
        u_act = u_sat

    disturbance = _disturbance_value(t, dist_cfg)
    tau = max(float(plant_cfg.tau_process), 1e-6)
    dy_dt = (-plant_cfg.K_process * u_act + disturbance - y) / tau

    if abs(pid_cfg.Ki) <= 1e-12:
        di_dt = 0.0
    else:
        kb_eff = (
            pid_cfg.anti_windup_gain
            if pid_cfg.anti_windup_gain is not None and np.isfinite(pid_cfg.anti_windup_gain) and pid_cfg.anti_windup_gain > 0.0
            else max(abs(pid_cfg.Kp), 1.0)
        )
        # Reverse acting control law: u = bias - Kp*e + I + Kd*d.
        freeze_integral = (sat_high and error < 0.0) or (sat_low and error > 0.0)
        if freeze_integral:
            di_dt = 0.0
        else:
            di_dt = pid_cfg.Ki * (-error + (u_act - u_cmd) / kb_eff)

    filt_tau = max(float(pid_cfg.tau_d), 1e-6)
    dd_dt = (dy_dt - d_state) / filt_tau
    return np.array([dy_dt, du_dt, di_dt, dd_dt], dtype=float)


def run_classic_pid_disturbance_simulation(
    plant_cfg: ClassicPlantConfig,
    pid_cfg: PidConfig,
    sim_cfg: SimulationConfig,
    dist_cfg: DisturbanceStepConfig,
) -> ClassicSimulationResult:
    """Ejecuta la simulacion clasica y reconstruye series derivadas.

    Ademas de integrar la ODE, recompone para cada muestra:
    - setpoint, error y perturbacion,
    - accion de control (comando, saturada y efectiva),
    - terminos P/I/D y banderas de saturacion.
    """
    t_eval = sim_cfg.time_grid()
    u0 = float(np.clip(pid_cfg.mc_bias, pid_cfg.mc_min, pid_cfg.mc_max))
    state0 = np.array([plant_cfg.y0, u0, 0.0, 0.0], dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: _classic_ode(t, y, plant_cfg, pid_cfg, dist_cfg),
        t_span=(sim_cfg.t_start, sim_cfg.t_end),
        y0=state0,
        t_eval=t_eval,
        method=sim_cfg.method,
    )
    if not sol.success:
        raise RuntimeError(f"Fallo en integracion clasica: {sol.message}")

    y, u_state, i_term, d_state = sol.y
    rate_active = pid_cfg.rate_max is not None and np.isfinite(pid_cfg.rate_max) and pid_cfg.rate_max > 0.0

    y_set_series = []
    error_series = []
    disturbance_series = []
    u_cmd_series = []
    u_sat_series = []
    u_act_series = []
    p_term_series = []
    i_term_series = []
    d_term_series = []
    sat_high_series = []
    sat_low_series = []

    for t_i, y_i, i_i, d_i, u_i in zip(sol.t, y, i_term, d_state, u_state):
        y_set_i = _sp_value(float(t_i), pid_cfg, plant_cfg.y0)
        pid_vals = pid_command(
            T=float(y_i),
            I_term=float(i_i),
            d_state=float(d_i),
            T_set=y_set_i,
            mc_bias=pid_cfg.mc_bias,
            Kp=pid_cfg.Kp,
            Kd=pid_cfg.Kd,
            mc_min=pid_cfg.mc_min,
            mc_max=pid_cfg.mc_max,
        )
        u_act_i = float(u_i if rate_active else pid_vals["u_sat"])

        y_set_series.append(y_set_i)
        error_series.append(pid_vals["error"])
        disturbance_series.append(_disturbance_value(float(t_i), dist_cfg))
        u_cmd_series.append(pid_vals["u_cmd"])
        u_sat_series.append(pid_vals["u_sat"])
        u_act_series.append(u_act_i)
        p_term_series.append(-pid_cfg.Kp * pid_vals["error"])
        i_term_series.append(float(i_i))
        d_term_series.append(pid_cfg.Kd * float(d_i))
        sat_high_series.append(pid_vals["saturated_high"])
        sat_low_series.append(pid_vals["saturated_low"])

    return ClassicSimulationResult(
        t=sol.t,
        y=y,
        y_set=np.asarray(y_set_series, dtype=float),
        error=np.asarray(error_series, dtype=float),
        disturbance=np.asarray(disturbance_series, dtype=float),
        u_cmd=np.asarray(u_cmd_series, dtype=float),
        u_sat=np.asarray(u_sat_series, dtype=float),
        u_act=np.asarray(u_act_series, dtype=float),
        p_term=np.asarray(p_term_series, dtype=float),
        i_term=np.asarray(i_term_series, dtype=float),
        d_term=np.asarray(d_term_series, dtype=float),
        sat_high=np.asarray(sat_high_series, dtype=bool),
        sat_low=np.asarray(sat_low_series, dtype=bool),
    )


__all__ = [
    "ClassicPlantConfig",
    "DisturbanceStepConfig",
    "ClassicSimulationResult",
    "run_classic_pid_disturbance_simulation",
]
