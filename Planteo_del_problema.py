"""Pagina de portada del simulador Streamlit.

Presenta:
- contexto del problema industrial,
- objetivo del simulador,
- reaccion quimica de referencia.
"""

import streamlit as st

st.set_page_config(
    page_title="Simulador TAC",
    page_icon="👋",
)
st.write("# Simulador TAC")

st.latex(r"C_{3}H_{6}O + H_{2}O \rightarrow C_{3}H_{8}O_{2}")

st.markdown(
    """
    # Producción de Propilenglicol en un TAC (CSTR)
    
    ## Planteo del problema
    El propilenglicol se obtiene a partir de la **hidrólisis del óxido de propileno** en presencia de un catalizador ácido (ácido sulfúrico).  
    Se trata de un proceso de gran relevancia industrial: En 2024, la demanda global alcanzó aproximadamente 2.620 mil toneladas (≈ 2,62 millones de toneladas). Ademas el mercado en términos monetarios se estimó en USD 4.850,4 millones en 2024, y se proyecta que llegue a cerca de USD 6.975,8 millones para 2030.
    El propilenglicol representa alrededor del **25% de los principales derivados del óxido de propileno**, lo que lo posiciona como un compuesto de alto valor agregado en la industria química.  

    El estudio de este sistema en un **reactor continuo de tanque agitado (CSTR)** permite analizar de manera controlada el comportamiento dinámico del proceso, en particular durante la etapa de **arranque del reactor**.

    ---

    ## Objetivos de la simulación
    1. **Modelar matemáticamente** el arranque de un CSTR adiabático para la reacción de hidrólisis del óxido de propileno.  
    2. **Resolver el sistema de ecuaciones diferenciales** que gobiernan los balances de materia y energía del reactor.  
    3. **Analizar la evolución temporal** de las variables de estado clave:  
    - Concentraciones de reactivos e inertes.  
    - Temperatura del sistema.  
    4. **Explorar la estabilidad del reactor** y los posibles regímenes de operación.  

    ---

    ## Potencial y relevancia
    - **Formación académica:** La simulación del arranque de un CSTR constituye un ejemplo representativo para comprender fenómenos de **dinámica de reactores**, **transferencia de calor** y **control de procesos químicos**.  
    - **Aplicación industrial:** Los resultados pueden servir de base para la optimización de condiciones de operación y diseño de estrategias de control en plantas de producción de propilenglicol.  
    - **Extensibilidad:** La metodología utilizada es aplicable a otros sistemas de reacción en fase líquida catalizados por ácidos, con lo cual el marco de modelado puede adaptarse a distintos contextos industriales.  

"""
)
