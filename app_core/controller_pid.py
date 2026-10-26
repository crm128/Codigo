from __future__ import annotations

"""Utilidades del controlador PID.

Contiene funciones puras para:
- aplicar limitacion de pendiente a la variable manipulada,
- calcular comando PID con saturacion y banderas de estado.
"""

from typing import Dict, Optional

import numpy as np


def apply_rate_limit(u: float, u_prev: float, rate_max: Optional[float], dt: float) -> float:
    """Aplica limitacion de pendiente a la accion de control.

    Parameters
    ----------
    u:
        Comando deseado actual.
    u_prev:
        Valor aplicado en el paso previo.
    rate_max:
        Pendiente maxima permitida (unidades de `u` por unidad de tiempo).
    dt:
        Paso temporal asociado al limite de pendiente.

    Returns
    -------
    float
        Comando limitado.
    """
    if rate_max is None or not np.isfinite(rate_max) or rate_max <= 0.0 or dt <= 0.0:
        return u
    delta = np.clip(u - u_prev, -rate_max * dt, rate_max * dt)
    return u_prev + delta


def pid_command(
    T: float,
    I_term: float,
    d_state: float,
    T_set: float,
    mc_bias: float,
    Kp: float,
    Kd: float,
    mc_min: float,
    mc_max: float,
) -> Dict[str, float]:
    """Calcula el comando PID antes/despues de saturacion.

    Convencion de control usada:
    `u_cmd = mc_bias - Kp*error + I_term + Kd*d_state`
    con `error = T_set - T`.

    Retorna tambien el estado de saturacion alta/baja para que el
    integrador pueda aplicar anti-windup de forma coherente.
    """
    error = T_set - T
    u_cmd = mc_bias - Kp * error + I_term + Kd * d_state
    u_sat = float(np.clip(u_cmd, mc_min, mc_max))
    saturated_high = bool(np.isclose(u_sat, mc_max, rtol=1e-6, atol=1e-8))
    saturated_low = bool(np.isclose(u_sat, mc_min, rtol=1e-6, atol=1e-8))
    return {
        "error": float(error),
        "u_cmd": float(u_cmd),
        "u_sat": u_sat,
        "saturated_high": saturated_high,
        "saturated_low": saturated_low,
    }
