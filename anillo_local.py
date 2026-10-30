"""
Contraste: anillo de alcance fijo (cada nodo unido con sus m vecinos más cercanos de cada lado),
acoplamiento fijo por arista (J = 1) y ruido T = 0.5.

En una dimensión no hay transición de fase con ruido (Mermin–Wagner): el máximo de χ existe
para cada N pero se corre hacia μ → 0 al crecer N. Para mover μ en forma continua, el vecino
m+1 entra con un peso w ∈ [0, 1).

Guarda resultados/anillo_local.json. Tarda unos 15 minutos con 2 núcleos.
"""
import json
import os
import sys
import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dinamica import maximo_chi


@njit(cache=True)
def _fuerza(th, k, w, J, c, s, F):
    # suma deslizante sobre la ventana i−k … i+k (costo O(N) por paso)
    N = th.shape[0]
    for i in range(N):
        c[i] = np.cos(th[i]); s[i] = np.sin(th[i])
    sc = 0.0; ss = 0.0
    for d in range(-k, k + 1):
        sc += c[d % N]; ss += s[d % N]
    for i in range(N):
        Sc = sc - c[i] + w * (c[(i + k + 1) % N] + c[(i - k - 1) % N])
        Ss = ss - s[i] + w * (s[(i + k + 1) % N] + s[(i - k - 1) % N])
        F[i] = J * (c[i] * Ss - s[i] * Sc)
        sc += c[(i + k + 1) % N] - c[(i - k) % N]
        ss += s[(i + k + 1) % N] - s[(i - k) % N]


@njit(parallel=True, cache=True)
def barrido_anillo(N, ks, ws, J, T, dt, t_term, t_med, cada, semilla):
    P = ks.shape[0]
    R1 = np.zeros(P); R2 = np.zeros(P)
    amp = np.sqrt(2 * T * dt)
    nt = int(t_term / dt); nm = int(t_med / dt)
    for p in prange(P):
        np.random.seed(semilla + 7919 * p)
        th = np.zeros(N); c = np.empty(N); s = np.empty(N); F = np.empty(N)
        a1 = 0.0; a2 = 0.0; n = 0
        for paso in range(nt + nm):
            _fuerza(th, ks[p], ws[p], J, c, s, F)
            for i in range(N):
                th[i] += dt * F[i] + amp * np.random.standard_normal()
            if paso >= nt and paso % cada == 0:
                cx = 0.0; sy = 0.0
                for i in range(N):
                    cx += np.cos(th[i]); sy += np.sin(th[i])
                r = np.sqrt(cx * cx + sy * sy) / N
                a1 += r; a2 += r * r; n += 1
        R1[p] = a1 / n; R2[p] = a2 / n
    return R1, R2


if __name__ == "__main__":
    T, reps = 0.5, 4
    grados = np.linspace(2.0, 20.0, 37)
    out = []
    for N in (50, 100, 200):
        ks = np.floor(grados / 2).astype(np.int64); ws = grados / 2 - ks
        R1, R2 = barrido_anillo(N, np.tile(ks, reps), np.tile(ws, reps), 1.0, T, 0.01, 300.0, 3000.0, 10, 23)
        chi = (N / T * (R2 - R1 ** 2)).reshape(reps, -1)
        mu = (grados / (N - 1)).tolist()
        res = dict(N=N, mu=mu, r=R1.reshape(reps, -1).mean(0).tolist(), chi=chi.mean(0).tolist())
        out.append(res)
        print(f"N={N}: μ* = {maximo_chi(mu, res['chi']):.4f}", flush=True)
    os.makedirs("resultados", exist_ok=True)
    json.dump(out, open("resultados/anillo_local.json", "w"))
