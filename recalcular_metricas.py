"""Recalcula las metricas del Cuadro 5.4 a partir de las series exportadas.

Uso:
    python recalcular_metricas.py resultados/series_barrido.csv

El CSV debe tener las columnas: sintonia, t_h, Tsp_F, T_F, mc_lbmol_h
(es el archivo que exporta la pagina "Control PID" con el boton
"Descargar series del barrido").
"""

import sys

import pandas as pd

from app_core.metrics import compute_metrics


def main(path: str) -> None:
    df = pd.read_csv(path)
    rows = []
    for label, g in df.groupby("sintonia", sort=False):
        m = compute_metrics(g["t_h"].to_numpy(), g["T_F"].to_numpy(),
                            g["Tsp_F"].to_numpy(), g["mc_lbmol_h"].to_numpy())
        rows.append({"Sintonía": label, **m})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(out.round(2).to_string(index=False))
    out.round(2).to_csv(path.replace("series", "metricas"), index=False)

    print("\nFilas LaTeX (Cuadro 5.4):")
    for r in rows:
        print(f"{r['Sintonía']} & {r['IAE_seg']:.2f} & {r['IAE_fin']:.2f} & "
              f"{r['dT_max']:.2f} & {r['overshoot']:.2f} & {r['TVu']:.1f} \\\\")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "resultados/O21_series_barrido.csv")