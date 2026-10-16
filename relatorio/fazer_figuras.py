"""Figuras 2 e 3 do relatório (PDF, largura de uma coluna).
Lê dados_figuras/curvas_T_a48.json (de dados_figuras/curvas_T.py) e
dados_figuras/tabela.csv + ajustes.csv (copiados de cristal_lego_g/analise 2.0/resultados)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

AQUI = Path(__file__).resolve().parent
D, F = AQUI / "dados_figuras", AQUI / "figuras"
F.mkdir(exist_ok=True)
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.3,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "lines.linewidth": 0.9,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL = 3.37  # largura da coluna (in)
AZ = ["#9ecae1", "#4292c6", "#08306b"]
COR = {32: "#1b9e77", 40: "#d95f02", 48: "#7570b3", 56: "#e7298a"}
MARC = {32: "o", 40: "s", 48: "^", 56: "D"}

# ---------------------------------------------------------------- Fig. 2
cur = json.loads((D / "curvas_T_a48.json").read_text())
# teoria (teoria.txt) e medido (resultados.txt), a = 48 mm
info = {"GX": dict(tit=r"$\Gamma X$, $a=48$ mm", fB=3578, teo=(2988, 3880), N=["a48_N4", "a48_N7", "a48_N10"],
                   finf=(2969, 102), fsup=(4230, 135), dif=7156, xl=(1.6, 8.0)),
        "GM": dict(tit=r"$\Gamma M$, $a=48$ mm", fB=5056, teo=(4385, 4942), N=["a48_N4", "a48_N7", "a48_N11"],
                   finf=(4040, 93), fsup=None, dif=5056, xl=(1.6, 10.5))}
fig, axs = plt.subplots(2, 1, figsize=(COL, 3.45))
for ax, k in zip(axs, ("GX", "GM")):
    I, f = info[k], np.array(cur[k]["f"]) / 1000
    ax.axvspan(*np.array(I["teo"]) / 1000, color="0.85", lw=0, label="hiato teórico (PWE)")
    for c, m in zip(AZ, I["N"]):
        ax.plot(f, cur[k][m], color=c, label=f"$N={m.split('N')[1]}$")
    ax.axhline(0, color="0.4", lw=0.5)
    ax.axhline(-6, color="0.3", lw=0.6, ls=":", label=r"limiar $T_0-6$ dB")
    ax.axvline(I["fB"] / 1000, color="k", lw=0.7, ls="--", label=r"$f_B=c/2d$")
    for v, s in [I["finf"]] + ([I["fsup"]] if I["fsup"] else []):
        ax.axvspan((v - s) / 1000, (v + s) / 1000, color="#de2d26", alpha=0.25, lw=0)
        ax.axvline(v / 1000, color="#de2d26", lw=0.8)
    if k == "GM":
        ax.annotate(r"difração: $c/L=f_B$", xy=(I["dif"] / 1000, 7), xytext=(6.4, 7.3), fontsize=6.3,
                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.3"), va="center")
    ax.set_xlim(*I["xl"]); ax.set_ylim(-24, 10)
    ax.set_ylabel(r"$T-T_0$ (dB)")
    ax.text(0.015, 0.95, I["tit"], transform=ax.transAxes, va="top", fontsize=7.5)
axs[1].set_xlabel(r"$f$ (kHz)")
h, l = axs[0].get_legend_handles_labels()
from matplotlib.patches import Patch
h0, l0 = [h[0], h[4], h[5]], [l[0], l[4], l[5]]
h0.append(Patch(color="#de2d26", alpha=0.5)); l0.append(r"bordos medidos ($\pm1\sigma$)")
fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 0.915))
fig.legend(h0, l0, loc="upper center", ncol=2, handlelength=1.6, columnspacing=1.0,
           bbox_to_anchor=(0.55, 1.0), labelspacing=0.3)
for ax in axs:
    hh, ll = ax.get_legend_handles_labels()
    ax.legend(hh[1:4], ll[1:4], loc="lower right", ncol=3, handlelength=1.4, columnspacing=0.8)
fig.savefig(F / "fig2_transmissao.pdf")

# ---------------------------------------------------------------- Fig. 3
t = pd.read_csv(D / "tabela.csv")
aj = pd.read_csv(D / "ajustes.csv")
aj = aj[aj.ajuste == "D_vs_N"]
fig, axs = plt.subplots(1, 2, figsize=(COL, 1.8), sharey=True)
leg = []
for ax, d in zip(axs, ("ΓX", "ΓM")):
    for a in (32, 40, 48, 56):
        s = t[(t.dir == d) & (t.a_mm == a) & (t.valida == 1)]
        r = aj[(aj.dir == d) & (aj.a_mm == a)]
        if s.empty or r.empty:
            continue
        r = r.iloc[0]
        eb = ax.errorbar(s.N + (a - 44) / 60, s.profundidade, s.s_profundidade, fmt=MARC[a], ms=2.6, color=COR[a],
                    elinewidth=0.5, capsize=1.2, mew=0.4, mfc="white" if d == "ΓM" else COR[a])
        x = np.linspace(0, 11.5, 2)
        ln, = ax.plot(x, r.m * x + r.b, color=COR[a], lw=0.8)
        if d == "ΓX":
            leg.append(((eb[0], ln), rf"$a={a}$ mm"))
    ax.axhline(0, color="0.4", lw=0.5)
    ax.set_xlim(0, 11.7); ax.set_ylim(-8, 25)
    ax.set_xlabel(r"$N$ (filas)")
    ax.text(0.04, 0.96, r"$\Gamma X$" if d == "ΓX" else r"$\Gamma M$", transform=ax.transAxes, va="top")
axs[0].set_ylabel(r"$D$ (dB)")
fig.tight_layout(w_pad=0.3, rect=(0, 0, 1, 0.9))
fig.legend(*zip(*leg), loc="upper center", ncol=4, handlelength=1.5,
           columnspacing=0.9, bbox_to_anchor=(0.53, 1.0))
fig.savefig(F / "fig3_profundidade.pdf")

# ---------------------------------------------------------------- tabela do apêndice A
def pm(v, s, casas=0):
    if not np.isfinite(v):
        return "---"
    return f"${v:.{casas}f}\\pm{s:.{casas}f}$"

lin = []
for _, r in t.iterrows():
    d = r"$\Gamma X$" if r.dir == "ΓX" else r"$\Gamma M$"
    av = [] if r.valida == 1 else ["inv."]
    if r.n < r.n_reps:
        av.append(f"{int(r.n)}/{int(r.n_reps)} rep.")
    if r.N >= 4 and np.isfinite(r.f_inf) and not r.fecha:
        av.append("não fecha")
    if str(r.ref).split("_a")[-1].split("_")[0] != str(int(r.a_mm)):
        av.append("ref. outro $a$")
    lin.append(" & ".join([d, f"{int(r.a_mm)}", f"{int(r.N)}", r.hora[:10][5:], r.ref.replace("_", r"\_"),
                           f"{r.ondulacao:.1f}", pm(r.profundidade, r.s_profundidade, 1),
                           pm(r.f_inf, r.s_f_inf), pm(r.f_sup, r.s_f_sup), ", ".join(av)]) + r" \\")
(AQUI / "tabela_apendice.tex").write_text("\n".join(lin), encoding="utf-8")
print("ok")
