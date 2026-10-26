"""Parametros nominales identificados del reactor TAC y consignas de control.

Este modulo es la unica fuente de verdad (Single Source of Truth) para las
condiciones nominales de proceso y consignas de operacion.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NominalProcessParameters:
    """Parametros fisicos y de alimentacion nominales."""
    Fa0: float = 80.0       # Flujo alimentacion A [lb-mol/h]
    Fb0: float = 1000.0     # Flujo alimentacion B [lb-mol/h]
    Fm0: float = 100.0      # Flujo inerte M [lb-mol/h]
    mc: float = 1000.0      # Flujo refrigerante base [lb-mol/h]
    T0: float = 75.0        # Temperatura inicial / alimentacion [°F]
    Ta1: float = 60.0       # Temperatura refrigerante entrada [°F]
    UA: float = 16000.0     # Coeficiente global por area [BTU/(h·°F)]
    V_ft3: float = 500.0 / 7.484  # Volumen util [ft³] (aprox 66.81 ft³)

    # Limites del actuador mc
    mc_min: float = 200.0
    mc_max: float = 4000.0
    mc_bias: float = 1000.0
    Kb: float = 200.0       # Ganancia anti-windup [lb-mol/(h·°F)]

    # Consignas de temperatura identificadas (Resolucion O16)
    T_SET_NOMINAL: float = 120.0       # Consigna unificada base de diseno [°F]

NOMINAL_PARAMS = NominalProcessParameters()