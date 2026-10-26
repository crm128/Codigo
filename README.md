# Simulador TAC - Documentacion Tecnica

## Objetivo del proyecto

Este proyecto implementa un simulador dinamico para el arranque de un reactor tanque agitado continuo (TAC/CSTR) con reaccion exotermica, y una variante con control PID de temperatura.

El codigo esta dividido en:

- `app_core`: núcleo numerico y logica de control.
- `pages`: interfaz Streamlit para ejecucion y visualizacion.
- `Planteo_del_problema.py`: portada/contexto del caso de estudio.

## Estructura y responsabilidades

- `app_core/types.py`: dataclasses de entrada/salida (`ReactorConditions`, `SimulationConfig`, `PidConfig`, `SimulationResult`).
- `app_core/plant.py`: ecuaciones diferenciales de planta en lazo abierto y utilidades de postproceso.
- `app_core/controller_pid.py`: funciones puras de control (PID y rate limit).
- `app_core/simulate.py`: orquestacion de integracion numerica en lazo abierto y cerrado.
- `app_core/metrics.py`: metricas de desempeno (`IAE_seg, IAE_fin, ΔT_max, Mp, TVu — ver definiciones en § 4.6 de la tesis).`
- `app_core/identification.py`: utilidades para preparacion de datos de identificacion por escalon.
- `pages/1_Simulacion_TAC.py`: UI para arranque sin control.
- `pages/3_Control_PID.py`: UI para simulacion PID y comparacion de sintonia.

## Flujo de datos (alto nivel)

1. El usuario carga parametros en la UI de Streamlit.
2. La UI construye objetos `ReactorConditions`, `SimulationConfig` y, en PID, `PidConfig`.
3. Se ejecuta:
   - `run_open_loop_simulation(...)` para lazo abierto, o
   - `run_pid_control_simulation(...)` para lazo cerrado.
4. El simulador devuelve `SimulationResult` con series temporales.
5. La UI genera tablas, graficos y exportaciones CSV.

## Modelo de planta

En `plant.py` y `simulate.py`:

- Estados de planta: `Ca, Cb, Cc, Cm, T`.
- Cinetica de Arrhenius con temperatura en Rankine (`T + 460`).
- Balance de energia con intercambio por camisa usando expresion exponencial en `UA/(Cp*mc)`.
- Conversion calculada desde `Fa = Ca * v0`.

## Modelo de control PID

En `controller_pid.py` y `simulate.py`:

- Error: `e = T_set - T`.
- Comando: `u_cmd = mc_bias - Kp*e + I + Kd*d_state`.
- Saturacion fisica: `[mc_min, mc_max]`.
- Anti-windup por retrocalculo y congelamiento en saturacion.
- Rate limit opcional sobre la variable manipulada.
- Setpoint constante o rampa lineal.
- Posibilidad de doble tramo de ganancias (`dual_gains`, `t_switch`).

## Representación interna del estado en lazo cerrado

`simulate.py` usa estado extendido:

- Planta: `[Ca, Cb, Cc, Cm, T, u_state]`
- PID: `[I_term, d_state]`
- Estado completo empaquetado: `[planta..., pid...]`

## Integracion numerica

- Integrador principal: `scipy.integrate.solve_ivp`.
- Metodo por defecto: `BDF`.
- Fallback local en tramos PID: `Radau` si falla el metodo solicitado.

## Metricas de desempeno

`metrics.py` calcula:

- `IAE`: integral del error absoluto.
- `overshoot`: sobreimpulso porcentual respecto al setpoint final.
- `TVu`: variacion total de la accion de control.

## Requisitos

- Python 3.10 o superior
- Dependencias listadas en `requirements.txt` (`streamlit`, `pandas`, `plotly`, `numpy`, `scipy`, `matplotlib`)

## Cómo levantar el proyecto

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run Planteo_del_problema.py
```

### Windows (CMD / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run Planteo_del_problema.py
```

La aplicación se abrirá automáticamente en el navegador en `http://localhost:8501`. Navegar entre páginas desde la barra lateral de Streamlit.

## Recomendaciones de mantenimiento

- Mantener la separacion UI / nucleo numerico.
- Evitar logica de control dentro de `pages/`.
- Documentar en docstring toda nueva funcion publica en `app_core`.
- Si se modifica el orden de estados, actualizar `_pack_pid_state` y `_split_pid_state`.
