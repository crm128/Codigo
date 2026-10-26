from __future__ import annotations

"""Estructuras de datos compartidas por el simulador TAC.

Este modulo define dataclasses para:
- condiciones fisicas de planta,
- configuracion numerica de la simulacion,
- configuracion del controlador PID,
- contenedor de resultados de corrida.

La separacion en tipos explicitos reduce errores de integracion entre
interfaz (Streamlit) y nucleo numerico (`app_core`).
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np


@dataclass(frozen=True)
class ReactorConditions:
    """Parametros fisicos y de operacion del reactor.

    Unidades esperadas en esta implementacion:
    - Caudales molares/masicos: lb-mol/h
    - Temperatura: degF
    - Volumen: ft^3
    - UA: BTU/(h*degF)
    """

    #: Flujo de alimentacion de reactivo A.
    Fa0: float
    #: Flujo de alimentacion de reactivo B.
    Fb0: float
    #: Flujo de alimentacion de inerte M.
    Fm0: float
    #: Flujo de refrigerante en la camisa.
    mc: float
    #: Coeficiente global de transferencia por area (U*A).
    UA: float
    #: Volumen util del reactor.
    V: float
    #: Temperatura inicial del reactor.
    T0: float
    #: Temperatura de entrada del refrigerante.
    Ta1: float


@dataclass(frozen=True)
class SimulationConfig:
    """Configuracion de integracion temporal.

    Atributos
    ---------
    t_start, t_end:
        Intervalo de simulacion en horas.
    n_points:
        Cantidad de puntos de muestreo solicitados al integrador.
    method:
        Metodo de `scipy.integrate.solve_ivp` (por defecto `BDF`).
    rtol, atol: 
        Tolerancias de solve_ivp.
    mc_disturbance:
        Funcion opcional `f(t)->float` para perturbar la variable manipulada.
        Se modela como factor relativo sobre `mc` (ejemplo: +0.05 = +5%).
    """

    #: Tiempo inicial de simulacion.
    t_start: float = 0.0
    #: Tiempo final de simulacion.
    t_end: float = 4.0
    #: Numero de puntos en la grilla de salida.
    n_points: int = 300
    #: Metodo del integrador ODE.
    method: str = "BDF"
    #: Tolerancias relativa y absoluta (por defecto, las de SciPy).
    rtol: float = 1e-3
    atol: float = 1e-6
    #: Funcion de perturbacion opcional sobre `mc`.
    mc_disturbance: Optional[Callable[[float], float]] = None

    def time_grid(self) -> np.ndarray:
        """Construye una grilla temporal uniforme para `t_eval`."""
        return np.linspace(self.t_start, self.t_end, self.n_points)


@dataclass(frozen=True)
class PidConfig:
    """Parametros del controlador PID y de la logica de actuacion.

    Incluye:
    - ganancias de tramo 1 (servo/arranque),
    - ganancias de tramo 2 (regulatorio),
    - limites fisicos de la variable manipulada,
    - opciones de perfil de setpoint y anti-windup.
    """

    #: Setpoint final de temperatura.
    T_set: float
    #: Valor de sesgo/base de la accion de control.
    mc_bias: float
    #: Limite inferior de la accion de control.
    mc_min: float
    #: Limite superior de la accion de control.
    mc_max: float
    #: Ganancia proporcional del tramo 1.
    Kp: float
    #: Ganancia integral del tramo 1.
    Ki: float
    #: Ganancia derivativa del tramo 1.
    Kd: float
    #: Constante de tiempo del filtro derivativo del tramo 1.
    tau_d: float
    #: Perfil de setpoint (`constant` o `linear`).
    sp_profile: Literal["constant", "linear"] = "constant"
    #: Duracion de la rampa de setpoint en horas.
    sp_t_ramp: float = 0.0
    #: Habilita uso de dos tramos de ganancias.
    dual_gains: bool = True
    #: Ganancia proporcional del tramo 2.
    Kp2: float = 0.0
    #: Ganancia integral del tramo 2.
    Ki2: float = 0.0
    #: Ganancia derivativa del tramo 2.
    Kd2: float = 0.0
    #: Constante de tiempo del filtro derivativo del tramo 2.
    tau_d2: float = 0.0
    #: Instante de cambio de tramo.
    t_switch: float = 0.0
    #: Limite de pendiente de actuador (opcional).
    rate_max: Optional[float] = None
    #: Ganancia de retrocalculo para anti-windup (opcional).
    anti_windup_gain: Optional[float] = None
 


@dataclass
class SimulationResult:
    """Series temporales resultantes de una corrida de simulacion."""

    #: Tiempo [h].
    t: np.ndarray
    #: Concentracion de A.
    Ca: np.ndarray
    #: Concentracion de B.
    Cb: np.ndarray
    #: Concentracion de C (producto).
    Cc: np.ndarray
    #: Concentracion de inerte M.
    Cm: np.ndarray
    #: Temperatura del reactor [degF].
    T: np.ndarray
    #: Conversion de A.
    conversion: np.ndarray
    #: Tiempo estimado de cuasi-estacionario.
    steady_time: float
    #: Variable manipulada efectiva (caudal refrigerante).
    mc: Optional[np.ndarray] = None
    #: Error de control (setpoint - temperatura).
    error: Optional[np.ndarray] = None
    #: Setpoint muestreado en el tiempo.
    T_set_series: Optional[np.ndarray] = None
