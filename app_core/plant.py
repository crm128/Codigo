from __future__ import annotations

"""Modelo de planta del reactor TAC (CSTR) en lazo abierto.

Este modulo concentra:
- ecuaciones diferenciales de balances de materia y energia,
- utilidades de postproceso (caudal volumetrico, conversion),
- deteccion simple de tiempo de estabilizacion.
"""

from typing import Optional

import numpy as np

# Constante de gases en unidades consistentes con el modelo historico.
R = 1.987


def cstr_odes(
    t: float,
    y: np.ndarray,
    Fa0: float,
    Fb0: float,
    Fm0: float,
    V: float,
    UA: float,
    Ta1: float,
    mc: float,
    T0: float,
) -> np.ndarray:
    """Ecuaciones de estado del CSTR sin control PID.

    Parametros
    ----------
    t:
        Tiempo de integracion (no usado de forma explicita en las ecuaciones,
        pero requerido por la firma de `solve_ivp`).
    y:
        Estado de planta `[Ca, Cb, Cc, Cm, T]`.
    Fa0, Fb0, Fm0:
        Flujos de alimentacion.
    V:
        Volumen del reactor.
    UA:
        Coeficiente global por area de transferencia de calor.
    Ta1:
        Temperatura del refrigerante de entrada.
    mc:
        Flujo de refrigerante (constante en lazo abierto).
    T0:
        Temperatura de referencia de alimentacion.

    Retorna
    -------
    np.ndarray
        Vector de derivadas `[dCa/dt, dCb/dt, dCc/dt, dCm/dt, dT/dt]`.
    """
    Ca, Cb, Cc, Cm, T = y

    # Cinetica de Arrhenius (temperatura en Rankine: T + 460).
    k = 16.96e12 * np.exp(-32400 / (R * (T + 460)))
    ra = -k * Ca
    rb = -k * Ca
    rc = k * Ca
    rm = 0.0

    # Caudal volumetrico total y tiempo espacial.
    v0 = Fa0 / 0.923 + Fb0 / 3.45 + Fm0 / 1.54
    tau = V / v0

    # Concentraciones de entrada equivalentes.
    Ca0 = Fa0 / v0
    Cb0 = Fb0 / v0
    Cm0 = Fm0 / v0

    # Balances dinamicos de materia.
    dCa_dt = (Ca0 - Ca) / tau + ra
    dCb_dt = (Cb0 - Cb) / tau + rb
    dCc_dt = (0.0 - Cc) / tau + rc
    dCm_dt = (Cm0 - Cm) / tau + rm

    # Balance de energia con camisa (sin aproximacion lineal UA*(Tc-T)).
    ThetaCp = 35 + Fb0 / Fa0 * 18 + Fm0 / Fa0 * 19.5
    Q = mc * 18 * (Ta1 - (T - (T - Ta1) * np.exp(-UA / (18 * mc))))
    NCp = Ca * V * 35 + Cb * V * 18 + Cc * V * 46 + Cm * V * 19.5
    dT_dt = (Q - Fa0 * ThetaCp * (T - T0) + (-36000) * ra * V) / (NCp + 1e-8)

    return np.array([dCa_dt, dCb_dt, dCc_dt, dCm_dt, dT_dt], dtype=float)


def compute_vol(Fa0: float, Fb0: float, Fm0: float, V: float) -> float:
    """Calcula el caudal volumetrico de alimentacion `v0`."""
    _ = V
    v0 = Fa0 / 0.923 + Fb0 / 3.45 + Fm0 / 1.54
    return float(v0)


def compute_conversion(Fa0: float, Ca: np.ndarray, v0: float) -> np.ndarray:
    """Calcula conversion de A como `X = (Fa0 - Fa) / Fa0` con `Fa = Ca * v0`."""
    return (Fa0 - Ca * v0) / Fa0


def detect_steady_state(
    t: np.ndarray,
    Ca: np.ndarray,
    T: np.ndarray,
    epsilon: float = 1e-3,
) -> Optional[float]:
    """Detecta el primer instante cuasi-estacionario por criterio incremental.

    Se considera cuasi-estacionario cuando variaciones relativas consecutivas de
    `Ca` y `T` son menores que `epsilon`.
    """
    for i in range(1, len(t)):
        crit_Ca = abs(Ca[i] - Ca[i - 1]) / max(abs(Ca[i]), 1e-8)
        crit_T = abs(T[i] - T[i - 1]) / max(abs(T[i]), 1e-8)
        if crit_Ca < epsilon and crit_T < epsilon:
            return float(t[i])
    return None

def compute_energy_balance(
    Ca: float,
    T: float,
    Fa0: float,
    Fb0: float,
    Fm0: float,
    V: float,
    UA: float,
    Ta1: float,
    mc: float,
    T0: float,
) -> dict:
    """Calcula los terminos del balance de energia en estado estacionario.

    Retorna un diccionario con los flujos de calor en BTU/h y en kW.
    """
    # Constante de gases y cinetica
    R_gas = 1.987
    k = 16.96e12 * np.exp(-32400 / (R_gas * (T + 460.0)))
    ra = -k * Ca

    # Termino 1: Generacion quimica (exotermica > 0)
    # (-Delta H_r) = 36000 BTU/lbmol
    Q_gen = (-36000.0) * ra * V  # BTU/h

    # Termino 2: Intercambio con la camisa (salida < 0 cuando T > Ta1)
    Q_camisa = mc * 18.0 * (Ta1 - (T - (T - Ta1) * np.exp(-UA / (18.0 * mc))))  # BTU/h

    # Termino 3: Transporte entalpico por corrientes (salida < 0 cuando T > T0)
    ThetaCp = 35.0 + (Fb0 / Fa0) * 18.0 + (Fm0 / Fa0) * 19.5
    Q_corrientes = - Fa0 * ThetaCp * (T - T0)  # BTU/h

    # Residuo de balance (debe ser aprox 0 en estado estacionario)
    residuo = Q_gen + Q_camisa + Q_corrientes
    pct_base = max(abs(Q_gen), 1e-8)

    # Factor de conversion: 1 BTU/h = 0.000293071 kW
    BTU_TO_KW = 0.000293071

    return {
        "Q_gen_btu": float(Q_gen),
        "Q_gen_kw": float(Q_gen * BTU_TO_KW),
        "Q_gen_pct": 100.0,
        "Q_camisa_btu": float(Q_camisa),
        "Q_camisa_kw": float(Q_camisa * BTU_TO_KW),
        "Q_camisa_pct": float((Q_camisa / pct_base) * 100.0),
        "Q_corrientes_btu": float(Q_corrientes),
        "Q_corrientes_kw": float(Q_corrientes * BTU_TO_KW),
        "Q_corrientes_pct": float((Q_corrientes / pct_base) * 100.0),
        "residuo_btu": float(residuo),
        "residuo_kw": float(residuo * BTU_TO_KW),
        "residuo_pct": float((residuo / pct_base) * 100.0),
    }


def build_energy_balance_dataframe(eb: dict) -> "pd.DataFrame":
    import pandas as pd

    data = [
        {
            "Término energético": "Generación por Reacción (Q̇_gen)",
            "Ecuación asociada": "(-ΔH_r)·(-r_A)·V",
            "Flujo [BTU/h]": f"{eb['Q_gen_btu']:+,.2f}",
            "Flujo [kW]": f"{eb['Q_gen_kw']:+,.2f}",
            "% del Total": f"{eb['Q_gen_pct']:+.2f}%",
        },
        {
            "Término energético": "Retiro por Camisa (Q̇_camisa)",
            "Ecuación asociada": "Q (intercambio camisa)",
            "Flujo [BTU/h]": f"{eb['Q_camisa_btu']:+,.2f}",
            "Flujo [kW]": f"{eb['Q_camisa_kw']:+,.2f}",
            "% del Total": f"{eb['Q_camisa_pct']:+.2f}%",
        },
        {
            "Término energético": "Remoción Convectiva por Corrientes (Q̇_corrientes)",
            "Ecuación asociada": "-F_A0·ΘCp·(T - T_0)",
            "Flujo [BTU/h]": f"{eb['Q_corrientes_btu']:+,.2f}",
            "Flujo [kW]": f"{eb['Q_corrientes_kw']:+,.2f}",
            "% del Total": f"{eb['Q_corrientes_pct']:+.2f}%",
        },
        {
            "Término energético": "Suma / Residuo de Cierre",
            "Ecuación asociada": "Q̇_gen + Q̇_camisa + Q̇_corrientes",
            "Flujo [BTU/h]": f"{eb['residuo_btu']:+,.2f}",
            "Flujo [kW]": f"{eb['residuo_kw']:+,.2f}",
            "% del Total": f"{eb['residuo_pct']:+.2f}%",
        },
    ]
    return pd.DataFrame(data)