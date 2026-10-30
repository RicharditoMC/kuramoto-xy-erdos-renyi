"""
Verificación de la correspondencia Kuramoto con ruido ↔ modelo XY.

Sobre un mismo grafo de Erdős–Rényi se compara la distribución estacionaria de r obtenida
  (a) integrando la dinámica de Kuramoto con ruido (Euler–Maruyama, varios pasos de tiempo), y
  (b) muestreando el modelo XY con el algoritmo de Metropolis a temperatura T = D.
Si la correspondencia es correcta, las distribuciones coinciden (y la de Langevin converge a la
de Metropolis al achicar dt).

Uso:  python verificacion_metropolis.py [N] [p] [D]      (por defecto 100 0.6 0.2)
Genera figuras/verificacion_metropolis.png. Tarda unos minutos.
"""
import os
import sys
import numpy as np
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dinamica import erdos_renyi


@njit(cache=True)
def langevin_r(A, J, D, dt, t_term, t_med, cada, semilla):
    np.random.seed(semilla)
    N = A.shape[0]
    th = np.zeros(N); c = np.empty(N); s = np.empty(N)
    amp = np.sqrt(2 * D * dt)
    nt = int(t_term / dt); nm = int(t_med / dt)
    out = np.empty(nm // cada + 1); k = 0
    for paso in range(nt + nm):
        for i in range(N):
            c[i] = np.cos(th[i]); s[i] = np.sin(th[i])
        for i in range(N):
            sc = 0.0; ss = 0.0
            for j in range(N):
                sc += A[i, j] * c[j]; ss += A[i, j] * s[j]
            th[i] += dt * J * (c[i] * ss - s[i] * sc) + amp * np.random.standard_normal()
        if paso >= nt and (paso - nt) % cada == 0:
            out[k] = np.abs(np.mean(np.exp(1j * th))); k += 1
    return out[:k]


@njit(cache=True)
def metropolis_r(A, J, T, delta, barridos_term, barridos, cada, semilla):
    """Metropolis para H = −J Σ_{i<j} A_ij cos(θ_i − θ_j) a temperatura T.
    Un barrido = N intentos de mover una fase elegida al azar en θ_i + U(−δ, δ)."""
    np.random.seed(semilla)
    N = A.shape[0]
    th = np.zeros(N)
    out = np.empty(barridos // cada + 1); k = 0
    for b in range(barridos_term + barridos):
        for _ in range(N):
            i = np.random.randint(N)
            nuevo = th[i] + delta * (2 * np.random.random() - 1)
            dE = 0.0
            for j in range(N):
                if A[i, j] != 0.0:
                    dE -= J * (np.cos(nuevo - th[j]) - np.cos(th[i] - th[j]))
            if dE <= 0 or np.random.random() < np.exp(-dE / T):
                th[i] = nuevo
        if b >= barridos_term and (b - barridos_term) % cada == 0:
            out[k] = np.abs(np.mean(np.exp(1j * th))); k += 1
    return out[:k]


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    D = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    K = 1.0; J = K / N
    A = erdos_renyi(N, p, np.random.default_rng(1))
    res = {}
    for dt in (0.1, 0.05, 0.02):
        res[f"Kuramoto dt={dt}"] = langevin_r(A, J, D, dt, 200.0, 20000.0, max(1, int(0.5 / dt)), 7)
    res["XY (Metropolis)"] = metropolis_r(A, J, D, 1.0, 2000, 40000, 1, 11)
    for k, v in res.items():
        print(f"{k:18s}  ⟨r⟩ = {v.mean():.4f}   desvío = {v.std():.4f}   ({len(v)} muestras)")
    os.makedirs("figuras", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(v.min() for v in res.values()), max(v.max() for v in res.values()), 50)
    for k, v in res.items():
        ax.hist(v, bins=bins, density=True, alpha=0.4, label=k)
    ax.set_xlabel("parámetro de orden r"); ax.set_ylabel("densidad de probabilidad")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig("figuras/verificacion_metropolis.png", dpi=200)
