"""
Simulaciones de Kuramoto con ruido sobre grafos de Erdős–Rényi (K = 1).

  1. D = 0.2, N = 100, 200 y 400: χ(μ) y ⟨r⟩(μ) barriendo p de 0.2 a 0.7.
  2. N = 200: el mismo barrido para D = 0.10, 0.15, 0.25 y 0.30.

Guarda todo en resultados/er.json. Tarda unos 15 minutos con 2 núcleos.
Las semillas son las mismas que dieron los números del informe.
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dinamica import barrido_denso, erdos_renyi, maximo_chi

K, DT, T_TERM, T_MED, CADA = 1.0, 0.05, 300.0, 4000.0, 10


def barrido(N, D, ps, n_grafos, semilla_grafos, semilla_ruido):
    rng = np.random.default_rng(semilla_grafos)
    As, mu, lam = [], [], []
    for g in range(n_grafos):
        for p in ps:
            A = erdos_renyi(N, p, rng)
            As.append(A)
            mu.append(A.sum() / N / (N - 1))                 # grado medio / (N − 1)
            lam.append(np.linalg.eigvalsh(A)[-1] / N)       # λ_max / N
    R1, R2 = barrido_denso(np.array(As), K / N, D, DT, T_TERM, T_MED, CADA, semilla_ruido)
    P = len(ps)
    chi = (N / D * (R2 - R1 ** 2)).reshape(n_grafos, P)      # χ de cada grafo
    return dict(N=N, D=D, p=list(map(float, ps)),
                mu=np.array(mu).reshape(n_grafos, P).mean(0).tolist(),
                lam=np.array(lam).reshape(n_grafos, P).mean(0).tolist(),
                r=R1.reshape(n_grafos, P).mean(0).tolist(),
                chi=chi.mean(0).tolist(),
                chi_err=(chi.std(0, ddof=1) / np.sqrt(n_grafos)).tolist())


if __name__ == "__main__":
    t0 = time.time()
    out = {"sizes": [], "noise": []}
    ps = np.round(np.arange(0.20, 0.701, 0.025), 4)
    for N, ng, sg, sr in ((100, 4, 200, 200), (200, 3, 300, 300), (400, 2, 5, 31)):
        res = barrido(N, 0.2, ps, ng, sg, sr)
        out["sizes"].append(res)
        print(f"D=0.2  N={N}: μ* = {maximo_chi(res['mu'], res['chi']):.3f}  "
              f"χ_max = {max(res['chi']):.2f}   ({time.time()-t0:.0f} s)", flush=True)
    for D in (0.1, 0.15, 0.25, 0.3):
        psD = np.round(np.arange(max(2 * D - 0.2, 0.04), 2 * D + 0.301, 0.025), 4)
        s = 900 + int(100 * D)
        res = barrido(200, D, psD, 2, s, s)
        out["noise"].append(res)
        print(f"N=200  D={D}: μ* = {maximo_chi(res['mu'], res['chi']):.3f}  (2D/K = {2*D/K:.2f})"
              f"   ({time.time()-t0:.0f} s)", flush=True)
    os.makedirs("resultados", exist_ok=True)
    json.dump(out, open("resultados/er.json", "w"))
    print("listo")
