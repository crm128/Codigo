from __future__ import annotations

"""Metricas de desempeno para respuestas de control."""

from typing import Dict

import numpy as np


def compute_metrics(time: np.ndarray, y: np.ndarray, ysp: np.ndarray, u: np.ndarray) -> Dict[str, float]:
    """Calcula metricas de desempeno sobre series temporales (definiciones O21).

    Entradas (misma grilla temporal):
    - time [h], y = T [degF], u = mc [lb-mol/h].
    - ysp: referencia efectivamente aplicada Tsp(t) [degF] (rampa + constante).
      Se asume ysp[0] = T0 y ysp[-1] = Tsp,f.

    Convencion: e(t) = ysp(t) - y(t). Intervalo: [time[0], time[-1]].

    Metricas:
    - IAE_seg [degF*h]: integral de |Tsp(t) - T| (seguimiento de rampa).
    - IAE_fin [degF*h]: integral de |Tsp,f - T| (error al objetivo final).
    - dT_max [degF]: max(0, max T - Tsp,f).
    - overshoot [%]: 100 * dT_max / |Tsp,f - T0| (relativo al salto).
    - TVu [lb-mol/h]: suma de |u_k - u_(k-1)| sobre la grilla.
    """
    t = np.asarray(time, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ysp = np.asarray(ysp, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()

    if t.size == 0:
        return {k: np.nan for k in ("IAE_seg", "IAE_fin", "dT_max", "overshoot", "TVu")}

    sp_final = ysp[-1]
    iae_seg = np.trapezoid(np.abs(ysp - y), t)
    iae_fin = np.trapezoid(np.abs(sp_final - y), t)

    dT_max = max(0.0, float(np.max(y) - sp_final))
    step = abs(sp_final - ysp[0])
    # Con setpoint constante no hay salto: Mp no esta definido (NaN).
    overshoot = 100.0 * dT_max / step if step > 0.0 else np.nan

    tvu = np.sum(np.abs(np.diff(u)))
    return {
        "IAE_seg": float(iae_seg),
        "IAE_fin": float(iae_fin),
        "dT_max": float(dT_max),
        "overshoot": float(overshoot),
        "TVu": float(tvu),
    }

def compute_score(
    metrics: Dict[str, float],
    weights_servo: Dict[str, float],
    weights_reg: Dict[str, float],
    alpha: float = 0.5,
) -> float:
    """Combina metricas de servo/regulacion en un puntaje unico [0, 1].

    El score normaliza cada metrica para que valores bajos (mejor desempeno)
    produzcan valores cercanos a 1. Luego combina por pesos y por `alpha`.
    """

    def _normalize(name: str, value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        if name == "overshoot":
            return 1.0 / (1.0 + max(0.0, value) / 100.0)
        if name == "settling_time":
            return 1.0 / (1.0 + max(0.0, value))
        if name == "sat_time_fraction":
            return 1.0 - min(max(value, 0.0), 1.0)
        if name == "TVu":
            return 1.0 / (1.0 + max(0.0, value))
        return 1.0 / (1.0 + max(0.0, value))

    def _weighted_score(weights: Dict[str, float]) -> float:
        if not weights:
            return 0.0
        total_w = sum(max(w, 0.0) for w in weights.values())
        if total_w <= 0.0:
            return 0.0
        acc = 0.0
        for name, weight in weights.items():
            acc += max(weight, 0.0) * _normalize(name, metrics.get(name, np.nan))
        return acc / total_w

    alpha = float(np.clip(alpha, 0.0, 1.0))
    servo_score = _weighted_score(weights_servo)
    reg_score = _weighted_score(weights_reg)
    base_score = alpha * servo_score + (1.0 - alpha) * reg_score
    return float(max(min(base_score, 1.0), 0.0))

def compute_steady_state_report(
    time: np.ndarray,
    y: np.ndarray,
    ysp: np.ndarray,
    u: np.ndarray,
    t_check: float = 4.0,
    bands: tuple = (0.5, 0.1),
) -> Dict[str, float]:
    """Cuantifica el error estacionario de una corrida y el horizonte necesario.

    - `t_s(d)`: primer instante a partir del cual |e| <= d hasta el final de la
      corrida (NaN si al final sigue fuera de la banda).
    - `tau_I`: constante de tiempo del modo lento, ajustando ln|e| = a - t/tau
      sobre la cola de la corrida (t >= t_check). Permite estimar el horizonte
      necesario para cualquier banda sin extender la simulacion.
    """
    t = np.asarray(time, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ysp = np.asarray(ysp, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    e = ysp - y

    def _settling(tol: float) -> float:
        outside = np.flatnonzero(np.abs(e) > tol)
        if outside.size == 0:
            return float(t[0])
        k = outside[-1] + 1
        return float(t[k]) if k < t.size else np.nan

    e_check = float(np.interp(t_check, t, e))
    report = {
        "t_end": float(t[-1]),
        "T_check": float(np.interp(t_check, t, y)),
        "e_check": e_check,
        "u_check": float(np.interp(t_check, t, u)),
        "T_end": float(y[-1]),
        "e_end": float(e[-1]),
        "u_end": float(u[-1]),
    }
    for d in bands:
        report[f"t_s({d:g})"] = _settling(d)

    # Ajuste exponencial de la cola (solo si el error no cambia de signo).
    tail = (t >= t_check) & (np.abs(e) > 1e-3)
    tau_I = np.nan
    if tail.sum() > 10 and np.all(np.sign(e[tail]) == np.sign(e_check)):
        slope, _ = np.polyfit(t[tail], np.log(np.abs(e[tail])), 1)
        if slope < 0.0:
            tau_I = -1.0 / slope
    report["tau_I"] = float(tau_I)
    for d in bands:
        report[f"t_est({d:g})"] = (
            float(t_check + tau_I * np.log(abs(e_check) / d))
            if np.isfinite(tau_I) and abs(e_check) > d else np.nan
        )
    return report