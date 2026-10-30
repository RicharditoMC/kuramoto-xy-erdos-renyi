"""
Figuras del informe a partir de resultados/er.json:
  figuras/er_chi.png, er_r.png, er_r_vs_N.png, er_mu_vs_D.png y er_dibujos.png
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dinamica import maximo_chi
from teoria import r_campo_medio
peak = lambda res: maximo_chi(res['mu'], res['chi'])

plt.rcParams.update({"font.size": 16, "axes.labelsize": 17, "axes.titlesize": 17, "legend.fontsize": 14,
                     "figure.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})
COLS = ["#fdae6b", "#e6550d", "#7f2704"]; C2 = "#2f5d50"
d = json.load(open("resultados/er.json"))
os.makedirs("figuras", exist_ok=True); D = 0.2; mus = 2 * D
rmf = r_campo_medio

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for res, col in zip(d["sizes"], COLS):
    ax.errorbar(res["mu"], res["chi"], yerr=res["chi_err"], fmt="o-", ms=5, color=col, capsize=2, label=f"N = {res['N']}")
mm = np.linspace(0.2, mus - 0.015, 100)
ax.plot(mm, (1 - np.pi/4) / (D * (1 - mm/mus)), color=C2, ls="--", lw=1.5, label="teoría (N→∞)")
ax.axvline(mus, color=C2, lw=1, alpha=0.5); ax.text(mus + 0.01, 24, r"$\mu^*=2D/K$", color=C2)
ax.set_xlabel(r"conectividad $\mu \simeq p$"); ax.set_ylabel(r"susceptibilidad $\chi$"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figuras/er_chi.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for res, col in zip(d["sizes"], COLS):
    ax.plot(res["mu"], res["r"], "o-", ms=5, color=col, label=f"N = {res['N']}")
mt = np.linspace(0.2, 0.7, 300)
ax.plot(mt, [rmf(m, D) for m in mt], color=C2, lw=2.2, label=r"$r=I_1/I_0(K\mu r/D)$")
ax.axvline(mus, color=C2, lw=1, alpha=0.5)
ax.set_xlabel(r"conectividad $\mu$"); ax.set_ylabel(r"parámetro de orden $\langle r\rangle$"); ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figuras/er_r.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
Ds = [r["D"] for r in d["noise"]] + [0.2]; ms = [peak(r) for r in d["noise"]] + [peak(d["sizes"][1])]
o = np.argsort(Ds); Ds = np.array(Ds)[o]; ms = np.array(ms)[o]
dd = np.linspace(0, 0.35, 50)
ax.fill_between(dd, 2*dd, 1, color="#e6550d", alpha=0.08)
ax.plot(dd, 2*dd, color=C2, lw=2.2, label=r"$\mu^*=2D/K$")
ax.plot(Ds, ms, "o", ms=10, color="#e6550d", label=r"máximo de $\chi$ (N = 200)")
ax.text(0.03, 0.72, "sincronización\nparcial", fontsize=15); ax.text(0.22, 0.12, "desordenado", fontsize=15)
ax.set_xlim(0, 0.35); ax.set_ylim(0, 0.9)
ax.set_xlabel("ruido D"); ax.set_ylabel(r"conectividad característica $\mu^*$")
ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.18))
fig.tight_layout(); fig.savefig("figuras/er_mu_vs_D.png", bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for target, mk in [(0.3, "o"), (0.5, "s"), (0.7, "D")]:
    Ns = [r["N"] for r in d["sizes"]]
    rs = [r["r"][int(np.argmin(abs(np.array(r["mu"]) - target)))] for r in d["sizes"]]
    ax.plot(Ns, rs, mk + "-", ms=8, label=f"μ = {target}")
    ax.axhline(rmf(target, D), color="gray", lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_ylim(0, 0.9); ax.set_xlabel("N"); ax.set_ylabel(r"$\langle r\rangle$")
ax.legend(frameon=False, loc="center right")
fig.tight_layout(); fig.savefig("figuras/er_r_vs_N.png", bbox_inches="tight"); plt.close(fig)

# dibujos de Erdős–Rényi
rng = np.random.default_rng(4); N = 24
a = 2*np.pi*np.arange(N)/N + np.pi/2; pos = np.c_[np.cos(a), np.sin(a)]
fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
for ax, p in zip(axs, (0.15, 0.4, 0.75)):
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < p:
                ax.plot(*zip(pos[i], pos[j]), color="#7f7f7f", lw=0.6, alpha=0.8, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=40, color=C2, zorder=2)
    ax.set_title(f"p = {p}"); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(); fig.savefig("figuras/er_dibujos.png", bbox_inches="tight"); plt.close(fig)
print("ok")
