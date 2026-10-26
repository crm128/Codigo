"""Regenera las figuras del barrido de Kp y las filas de tabla de barridos.

Uso:
    python graficar_barrido.py resultados/series_barrido.csv
Genera SintoniavsIAE.png, SintoniavsOvershoot.png y SintoniavsTVu.png
en la carpeta del CSV.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from app_core.metrics import compute_metrics


def main(path: str) -> None:
    df = pd.read_csv(path)
    out_dir = os.path.dirname(path) or "."
    kps, m_rows = [], []
    for label, g in df.groupby("sintonia", sort=False):
        if not label.startswith("Kp="):
            continue
        kp = float(label.split("=")[1].split()[0])
        m = compute_metrics(g["t_h"].to_numpy(), g["T_F"].to_numpy(),
                            g["Tsp_F"].to_numpy(), g["mc_lbmol_h"].to_numpy())
        kps.append(kp)
        m_rows.append(m)
    res = pd.DataFrame(m_rows, index=kps).sort_index()

    figs = [
        ("IAE_seg", "IAE$_{seg}$ (°F·h)", "SintoniavsIAE.png"),
        ("dT_max", "ΔT$_{max}$ (°F)", "SintoniavsOvershoot.png"),
        ("TVu", "TV$_u$ (lb-mol/h)", "SintoniavsTVu.png"),
    ]
    for col, ylabel, fname in figs:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(res.index, res[col], marker="o")
        ax.set_xlabel("$K_p$ (lb-mol/(h·°F))")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close(fig)

    g = df[df["sintonia"].str.startswith("Kp=200")].head(10)
    print("Filas LaTeX (Cuadro 5.3, Kp = 200):")
    for _, r in g.iterrows():
        e = r["Tsp_F"] - r["T_F"]
        print(f"{r['t_h']:.2f} & {r['T_F']:.2f} & {r['mc_lbmol_h']:.1f} & {e:.2f} \\\\")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "resultados/O21_series_barrido.csv")