from __future__ import annotations

"""Rutinas de simulacion del reactor TAC en lazo abierto y lazo cerrado.

Este modulo integra:
- modelo de planta (balances de materia y energia),
- modelo de controlador PID (incluyendo anti-windup),
- gestion de setpoint constante/rampa,
- soporte opcional de doble tramo de ganancias.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Optional

from .controller_pid import apply_rate_limit, pid_command
from .plant import cstr_odes, compute_conversion, compute_vol, detect_steady_state
from .types import PidConfig, ReactorConditions, SimulationConfig, SimulationResult


def _pack_pid_state(plant_state: np.ndarray, pid_state: np.ndarray) -> np.ndarray:
    """Empaqueta estado completo para el integrador.

    Orden interno usado en este modulo:
    - planta: `[Ca, Cb, Cc, Cm, T, u_state]`
    - PID: `[I_term, d_state]`
    """
    return np.concatenate([plant_state, pid_state]).astype(float, copy=False)


def _split_pid_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Desempaqueta vector de estado completo en planta y PID."""
    plant_state = state[:6]
    pid_state = state[6:8]
    return plant_state, pid_state

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def build_sp_fun(profile: str, T0: float, T_final: float, t_ramp: float) -> Callable[[float], float]:
    """Construye funcion de setpoint Tsp(t) para seguimiento servo.

    Es la referencia unica usada por el lazo (error del PID) y por las
    metricas (compute_metrics), de modo que ambos miden contra lo mismo.
    """
    if profile == "constant" or t_ramp <= 0.0:
        return lambda t: T_final
    return lambda t: T0 + (T_final - T0) * _clamp01(t / t_ramp)

def cstr_odes_pid_aw(
    t: float,
    state: np.ndarray,
    Fa0: float,
    Fb0: float,
    Fm0: float,
    V: float,
    UA: float,
    Ta1: float,
    T0: float,
    sp_fun: Callable[[float], float],
    mc_bias: float,
    mc_min: float,
    mc_max: float,
    Kp: float,
    Ki: float,
    Kd: float,
    tau_d: float,
    mc_disturbance: Optional[Callable[[float], float]],
    rate_max: Optional[float],
    Kb: Optional[float],
) -> np.ndarray:
    """Ecuaciones acopladas planta + PID con anti-windup.

    El controlador opera sobre temperatura:
    - variable controlada: `T`
    - manipulada: `mc` (caudal de refrigerante)
    """
    plant_state, pid_state = _split_pid_state(state)
    Ca, Cb, Cc, Cm, T, u_state = plant_state
    I_term, d_state = pid_state

    T_set = sp_fun(t)
    pid = pid_command(T, I_term, d_state, T_set, mc_bias, Kp, Kd, mc_min, mc_max)
    error = pid["error"]
    u_cmd = pid["u_cmd"]
    u_sat = pid["u_sat"]
    saturated_high = pid["saturated_high"]
    saturated_low = pid["saturated_low"]

    rate_active = rate_max is not None and np.isfinite(rate_max) and rate_max > 0.0
    if rate_active:
        # Si hay rate limit, `u_state` es la salida del actuador y se integra.
        u_limited = apply_rate_limit(u_sat, u_state, rate_max, 1.0)
        du_dt = float(u_limited - u_state)
        u_actuator = u_state
    else:
        du_dt = 0.0
        u_actuator = u_sat

    if mc_disturbance is not None:
        try:
            disturbance = float(mc_disturbance(t))
        except Exception:
            disturbance = 0.0
        mc_effective = u_actuator * max(0.0, 1.0 + disturbance)
    else:
        mc_effective = u_actuator
    mc_effective = max(mc_effective, 1e-6)

    R = 1.987
    k = 16.96e12 * np.exp(-32400 / (R * (T + 460)))
    ra = -k * Ca
    rb = -k * Ca
    rc = k * Ca
    rm = 0.0

    v0 = Fa0 / 0.923 + Fb0 / 3.45 + Fm0 / 1.54
    tau = V / v0

    Ca0 = Fa0 / v0
    Cb0 = Fb0 / v0
    Cm0 = Fm0 / v0

    dCa_dt = (Ca0 - Ca) / tau + ra
    dCb_dt = (Cb0 - Cb) / tau + rb
    dCc_dt = (0.0 - Cc) / tau + rc
    dCm_dt = (Cm0 - Cm) / tau + rm

    ThetaCp = 35 + Fb0 / Fa0 * 18 + Fm0 / Fa0 * 19.5
    Q = mc_effective * 18 * (Ta1 - (T - (T - Ta1) * np.exp(-UA / (18 * mc_effective))))
    NCp = Ca * V * 35 + Cb * V * 18 + Cc * V * 46 + Cm * V * 19.5
    dT_dt = (Q - Fa0 * ThetaCp * (T - T0) + (-36000) * ra * V) / (NCp + 1e-8)

    # Integral con anti-windup por retrocalculo + congelamiento por saturacion.
    if Ki <= 0.0:
        dI_dt = 0.0
    else:
        Kb_eff = Kb if Kb is not None and np.isfinite(Kb) and Kb > 0.0 else max(abs(Kp), 1.0)
        # Ley inversa u = bias - Kp*e + I: la saturacion alta ocurre con e < 0, y alli
        # el integrador seguiria creciendo; se congela. El retrocalculo descarga I.
        freeze_integral = (saturated_high and error < 0.0) or (saturated_low and error > 0.0)
        if freeze_integral:
            dI_dt = 0.0
        else:
            dI_dt = Ki * (-error + (u_actuator - u_cmd) / Kb_eff)

    # Derivativo implementado como filtro de primer orden de dT/dt.
    filt_tau = max(tau_d, 1e-6)
    dFilt_dt = (dT_dt - d_state) / filt_tau

    plant_derivs = np.array([dCa_dt, dCb_dt, dCc_dt, dCm_dt, dT_dt, du_dt], dtype=float)
    pid_derivs = np.array([dI_dt, dFilt_dt], dtype=float)
    return _pack_pid_state(plant_derivs, pid_derivs)


def run_open_loop_simulation(conditions: ReactorConditions, sim_cfg: SimulationConfig) -> SimulationResult:
    """Simula el arranque del reactor sin lazo de control."""
    y0 = [0.0, 3.45, 0.0, 0.0, conditions.T0]
    t_eval = sim_cfg.time_grid()

    ode_fun = lambda t, y: cstr_odes(
        t,
        y,
        conditions.Fa0,
        conditions.Fb0,
        conditions.Fm0,
        conditions.V,
        conditions.UA,
        conditions.Ta1,
        conditions.mc,
        conditions.T0,
    )

    sol = solve_ivp(
        ode_fun,
        (sim_cfg.t_start, sim_cfg.t_end),
        y0,
        t_eval=t_eval,
        method=sim_cfg.method,
        rtol=sim_cfg.rtol,
        atol=sim_cfg.atol,
    )
    if not sol.success:
        raise RuntimeError(f"Fallo en la integracion: {sol.message}")

    Ca, Cb, Cc, Cm, T = sol.y
    v0 = compute_vol(conditions.Fa0, conditions.Fb0, conditions.Fm0, conditions.V)
    conversion = compute_conversion(conditions.Fa0, Ca, v0)
    steady_time = detect_steady_state(sol.t, Ca, T)
    mc_series = np.full_like(sol.t, conditions.mc)

    return SimulationResult(
        t=sol.t,
        Ca=Ca,
        Cb=Cb,
        Cc=Cc,
        Cm=Cm,
        T=T,
        conversion=conversion,
        steady_time=steady_time,
        mc=mc_series,
    )


def run_pid_control_simulation(
    conditions: ReactorConditions,
    pid_cfg: PidConfig,
    sim_cfg: SimulationConfig,
) -> SimulationResult:
    """Simula el lazo cerrado con PID para seguimiento y regulacion.

    Comportamiento general:
    - genera `sp_fun(t)` (constante o rampa lineal),
    - integra en 1 o 2 segmentos segun `dual_gains`,
    - reconstruye series de `mc`, error y setpoint,
    - calcula conversion y tiempo de cuasi-estacionario.
    """
    mc_dist_fn = sim_cfg.mc_disturbance

    u0 = float(np.clip(pid_cfg.mc_bias, pid_cfg.mc_min, pid_cfg.mc_max))
    y0 = np.array([0.0, 3.45, 0.0, 0.0, conditions.T0, u0], dtype=float)
    pid_state0 = np.array([0.0, 0.0], dtype=float)
    state0 = _pack_pid_state(y0, pid_state0)

    
    sp_fun = build_sp_fun(pid_cfg.sp_profile, conditions.T0, pid_cfg.T_set, pid_cfg.sp_t_ramp)

    def _solve_segment(
        t0: float,
        t1: float,
        state_init: np.ndarray,
        Kp: float,
        Ki: float,
        Kd: float,
        tau_d: float,
        n_points: int,
    ):
        """Integra un tramo con ganancias fijas y fallback de integrador."""
        if t1 <= t0:

            class _Dummy:
                def __init__(self, t0, y0):
                    self.t = np.array([t0], dtype=float)
                    self.y = np.array(y0, dtype=float).reshape(-1, 1)
                    self.success = True
                    self.message = "skipped zero-length segment"

            return _Dummy(t0, state_init)

        t_eval_seg = np.linspace(t0, t1, max(2, int(n_points)))

        def _run(method: str):
            return solve_ivp(
                cstr_odes_pid_aw,
                (t0, t1),
                state_init,
                t_eval=t_eval_seg,
                method=method,
                rtol=sim_cfg.rtol,
                atol=sim_cfg.atol,
                args=(
                    conditions.Fa0,
                    conditions.Fb0,
                    conditions.Fm0,
                    conditions.V,
                    conditions.UA,
                    conditions.Ta1,
                    conditions.T0,
                    sp_fun,
                    pid_cfg.mc_bias,
                    pid_cfg.mc_min,
                    pid_cfg.mc_max,
                    Kp,
                    Ki,
                    Kd,
                    tau_d,
                    mc_dist_fn,
                    pid_cfg.rate_max,
                    pid_cfg.anti_windup_gain,
                ),
            )

        sol_try = _run(sim_cfg.method)
        if not sol_try.success:
            sol_try = _run("Radau")
        return sol_try

    use_dual = bool(getattr(pid_cfg, "dual_gains", False)) and (
        pid_cfg.t_switch > sim_cfg.t_start and pid_cfg.t_switch < sim_cfg.t_end
    )

    if not use_dual:
        sol = _solve_segment(
            sim_cfg.t_start,
            sim_cfg.t_end,
            state0,
            pid_cfg.Kp,
            pid_cfg.Ki,
            pid_cfg.Kd,
            pid_cfg.tau_d,
            sim_cfg.n_points,
        )
        if not sol.success:
            raise RuntimeError(f"Fallo en la integracion: {sol.message}")

        plant_states, pid_states = _split_pid_state(sol.y)
        Ca, Cb, Cc, Cm, T, u_state = plant_states
        I_term, filt_dT = pid_states
        t_all = sol.t
        gains_flag = np.zeros_like(t_all, dtype=int)
    else:
        t0 = sim_cfg.t_start
        t1 = float(pid_cfg.t_switch)
        t2 = sim_cfg.t_end

        total_dur = max(t2 - t0, 1e-9)
        frac1 = max((t1 - t0) / total_dur, 0.0)
        n1 = max(2, int(round(sim_cfg.n_points * frac1)))
        n2 = max(2, sim_cfg.n_points - n1 + 1)

        sol1 = _solve_segment(
            t0,
            t1,
            state0,
            pid_cfg.Kp,
            pid_cfg.Ki,
            pid_cfg.Kd,
            pid_cfg.tau_d,
            n1,
        )
        if not sol1.success:
            raise RuntimeError(f"Fallo en la integracion (segmento 1): {sol1.message}")

        y_init2 = sol1.y[:, -1]
        try:
            # Re-seeding suave del integrador para continuidad de accion al cambiar ganancias.
            T_last = float(y_init2[4])
            u_state_last = float(y_init2[5])
            I_last = float(y_init2[6])
            D_last = float(y_init2[7])
            pid_last = pid_command(
                T_last,
                I_last,
                D_last,
                float(sp_fun(t1)),
                pid_cfg.mc_bias,
                pid_cfg.Kp,
                pid_cfg.Kd,
                pid_cfg.mc_min,
                pid_cfg.mc_max,
            )
            e_last = pid_last["error"]
            rate_active = pid_cfg.rate_max is not None and np.isfinite(pid_cfg.rate_max) and pid_cfg.rate_max > 0.0
            mc_last = u_state_last if rate_active else pid_last["u_sat"]
            if abs(pid_cfg.Ki2) > 1e-12:
                I_new = mc_last - pid_cfg.mc_bias + pid_cfg.Kp2 * e_last - pid_cfg.Kd2 * D_last
                y_init2 = y_init2.copy()
                y_init2[6] = I_new
        except Exception:
            # Si falla el ajuste de continuidad, se mantiene el estado original.
            pass

        sol2 = _solve_segment(
            t1,
            t2,
            y_init2,
            pid_cfg.Kp2,
            pid_cfg.Ki2,
            pid_cfg.Kd2,
            pid_cfg.tau_d2,
            n2,
        )
        if not sol2.success:
            raise RuntimeError(f"Fallo en la integracion (segmento 2): {sol2.message}")

        t_all = np.concatenate([sol1.t, sol2.t[1:]])
        Y_all = np.concatenate([sol1.y, sol2.y[:, 1:]], axis=1)
        plant_states, pid_states = _split_pid_state(Y_all)
        Ca, Cb, Cc, Cm, T, u_state = plant_states
        I_term, filt_dT = pid_states
        gains_flag = np.concatenate(
            [np.zeros_like(sol1.t, dtype=int), np.ones_like(sol2.t[1:], dtype=int)]
        )

    mc_series = []
    error_series = []
    T_set_series = []
    rate_active = pid_cfg.rate_max is not None and np.isfinite(pid_cfg.rate_max) and pid_cfg.rate_max > 0.0
    for idx, (t_i, T_i, I_i, D_i, u_i) in enumerate(zip(t_all, T, I_term, filt_dT, u_state)):
        T_sp_i = float(sp_fun(t_i))
        if use_dual and gains_flag[idx] == 1:
            Kp_i, Ki_i, Kd_i = pid_cfg.Kp2, pid_cfg.Ki2, pid_cfg.Kd2
        else:
            Kp_i, Ki_i, Kd_i = pid_cfg.Kp, pid_cfg.Ki, pid_cfg.Kd

        pid_vals = pid_command(
            T_i,
            I_i,
            D_i,
            T_sp_i,
            pid_cfg.mc_bias,
            Kp_i,
            Kd_i,
            pid_cfg.mc_min,
            pid_cfg.mc_max,
        )
        u_actuator = u_i if rate_active else pid_vals["u_sat"]
        error_series.append(pid_vals["error"])
        T_set_series.append(T_sp_i)

        u_effective = u_actuator
        if mc_dist_fn is not None:
            try:
                disturbance = float(mc_dist_fn(t_i))
            except Exception:
                disturbance = 0.0
            u_effective = u_actuator * max(0.0, 1.0 + disturbance)
        mc_series.append(u_effective)

    mc_array = np.array(mc_series)
    error_array = np.array(error_series)
    T_set_array = np.array(T_set_series)

    v0 = compute_vol(conditions.Fa0, conditions.Fb0, conditions.Fm0, conditions.V)
    conversion = compute_conversion(conditions.Fa0, Ca, v0)
    steady_time = detect_steady_state(t_all, Ca, T)

    return SimulationResult(
        t=t_all,
        Ca=Ca,
        Cb=Cb,
        Cc=Cc,
        Cm=Cm,
        T=T,
        conversion=conversion,
        steady_time=steady_time,
        mc=mc_array,
        error=error_array,
        T_set_series=T_set_array,
    )


__all__ = [
    "run_open_loop_simulation",
    "run_pid_control_simulation",
    "cstr_odes_pid_aw",
    "build_sp_fun"
]
