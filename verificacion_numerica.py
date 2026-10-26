"""Pruebas de consistencia y sensibilidad numerica (Anexo de la tesis).

Uso:
    python verificacion_numerica.py

Imprime las filas LaTeX de los dos cuadros del Anexo y los desvios maximos
que se citan en el texto.
"""

from dataclasses import replace

import numpy as np

from app_core.parameters import NOMINAL_PARAMS as P
from app_core.plant import compute_vol, cstr_odes
from app_core.simulate import run_open_loop_simulation, run_pid_control_simulation
from app_core.types import PidConfig, ReactorConditions, SimulationConfig

# Mismas condiciones que las paginas de Streamlit.
COND = ReactorConditions(Fa0=P.Fa0, Fb0=P.Fb0, Fm0=P.Fm0, mc=P.mc, UA=P.UA,
                         V=(1 / 7.484) * 500, T0=P.T0, Ta1=P.Ta1)
PID = PidConfig(T_set=P.T_SET_NOMINAL, mc_bias=P.mc_bias, mc_min=P.mc_min,
                mc_max=P.mc_max, Kp=200.0, Ki=5.0, Kd=0.0, tau_d=0.0,
                sp_profile="linear", sp_t_ramp=2.0, rate_max=None,
                anti_windup_gain=P.Kb, dual_gains=True, Kp2=200.0, Ki2=5.0,
                Kd2=0.0, tau_d2=0.0, t_switch=2.0)

# (method, rtol, atol). La primera es la usada en todo el trabajo.
CONFIGS = [("BDF", 1e-3, 1e-6), ("BDF", 1e-8, 1e-10), ("RK45", 1e-8, 1e-10)]
CASOS = [("abierto", "Lazo abierto"), ("pid", "PID, $K_p = 200$")]


def simular(caso, method, rtol, atol, t_end, n_points):
    cfg = SimulationConfig(t_end=t_end, n_points=n_points, method=method,
                           rtol=rtol, atol=atol)
    if caso == "abierto":
        return run_open_loop_simulation(COND, cfg)
    pid = PID
    if method == "RK45":
        # Con Kd = 0 el filtro derivativo no entra en la ley de control; su
        # constante por defecto (1e-6 h) solo volveria rigido al sistema.
        pid = replace(PID, tau_d=1.0, tau_d2=1.0)
    return run_pid_control_simulation(COND, pid, cfg)


def invariantes(r):
    """Desvio maximo respecto de la solucion analitica de C_A+C_C, C_B-C_A y C_M."""
    v0 = compute_vol(COND.Fa0, COND.Fb0, COND.Fm0, COND.V)
    tau = COND.V / v0
    Ca0, Cb0, Cm0 = COND.Fa0 / v0, COND.Fb0 / v0, COND.Fm0 / v0
    g = np.exp(-r.t / tau)
    s0 = 3.45  # C_B(0) - C_A(0)
    e_ac = np.max(np.abs(r.Ca + r.Cc - Ca0 * (1 - g)))
    e_ba = np.max(np.abs(r.Cb - r.Ca - ((Cb0 - Ca0) + (s0 - (Cb0 - Ca0)) * g)))
    e_m = np.max(np.abs(r.Cm - Cm0 * (1 - g)))
    return e_ac, e_ba, e_m


def residuo_final(r):
    """|dT/dt| y |dC_A/dt| evaluados en el ultimo instante de la corrida."""
    y = [r.Ca[-1], r.Cb[-1], r.Cc[-1], r.Cm[-1], r.T[-1]]
    d = cstr_odes(r.t[-1], y, COND.Fa0, COND.Fb0, COND.Fm0, COND.V, COND.UA,
                  COND.Ta1, float(r.mc[-1]), COND.T0)
    return abs(d[4]), abs(d[0])


def exp10(x):
    return f"$10^{{{int(round(np.log10(x)))}}}$"


def main():
    print("Cuadro A: sensibilidad al integrador (t_f = 4 h, 300 puntos)\n")
    dT_max, dX_max = 0.0, 0.0
    for caso, nombre in CASOS:
        ref = None
        for method, rtol, atol in CONFIGS:
            r = simular(caso, method, rtol, atol, 4.0, 300)
            v = (float(r.T.max()), float(r.T[-1]), float(r.conversion[-1]))
            ref = ref or v
            dT_max = max(dT_max, abs(v[0] - ref[0]), abs(v[1] - ref[1]))
            dX_max = max(dX_max, abs(v[2] - ref[2]))
            print(f"{nombre} & {method} & {exp10(rtol)} / {exp10(atol)} & "
                  f"{v[0]:.3f} & {v[1]:.3f} & {v[2]:.4f} \\\\")
            nombre = ""
        print("\\midrule")
    print(f"\nDesvio maximo respecto de BDF por defecto: "
          f"dT = {dT_max:.2e} F, dX = {dX_max:.2e}\n")

    print("Cuadro B: invariantes y residuo final (BDF, tolerancias por defecto)\n")
    e_inv = 0.0
    for caso, nombre, t_end in (("abierto", "Lazo abierto", 12.0),
                                ("pid", "PID, $K_p = 200$", 200.0)):
        r = simular(caso, "BDF", 1e-3, 1e-6, t_end, int(20 * t_end) + 1)
        e = invariantes(r)
        e_inv = max(e_inv, *e)
        dT, dCa = residuo_final(r)
        print(f"{nombre} & {t_end:g} & {e[0]:.1e} & {e[1]:.1e} & {e[2]:.1e} & "
              f"{dT:.1e} & {dCa:.1e} \\\\")
    print(f"\nDesvio maximo de los invariantes: {e_inv:.1e} lb-mol/ft3")


if __name__ == "__main__":
    main()