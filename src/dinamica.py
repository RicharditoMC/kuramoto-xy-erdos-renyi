"""
Dinámica de Kuramoto con osciladores idénticos y ruido (= modelo XY en un baño térmico).

    dθ_i = J Σ_j A_ij sin(θ_j − θ_i) dt + √(2D dt) ξ_i ,      J = K/N

Se integra con Euler–Maruyama. En el estado estacionario las fases tienen la distribución de
Boltzmann del modelo XY, P ∝ exp(−H/D), con H = −J Σ_{i<j} A_ij cos(θ_i − θ_j).
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def barrido_denso(As, J, D, dt, t_term, t_med, cada, semilla):
    """Integra la dinámica para cada matriz de adyacencia de la pila As (P × N × N).

    Arranca con todas las fases iguales, descarta t_term unidades de tiempo (termalización)
    y promedia r y r² durante t_med, midiendo cada `cada` pasos.
    Devuelve ⟨r⟩ y ⟨r²⟩ para cada grafo.
    """
    P, N, _ = As.shape
    R1 = np.zeros(P); R2 = np.zeros(P)
    amp = np.sqrt(2 * D * dt)
    nt = int(t_term / dt); nm = int(t_med / dt)
    for p in prange(P):
        np.random.seed(semilla + 7919 * p)
        A = As[p]
        th = np.zeros(N)
        c = np.empty(N); s = np.empty(N)
        a1 = 0.0; a2 = 0.0; n = 0
        for paso in range(nt + nm):
            for i in range(N):
                c[i] = np.cos(th[i]); s[i] = np.sin(th[i])
            for i in range(N):
                sc = 0.0; ss = 0.0
                for j in range(N):
                    sc += A[i, j] * c[j]; ss += A[i, j] * s[j]
                # Σ_j A_ij sin(θ_j − θ_i) = cos θ_i Σ A sin θ_j − sin θ_i Σ A cos θ_j
                th[i] += dt * J * (c[i] * ss - s[i] * sc) + amp * np.random.standard_normal()
            if paso >= nt and paso % cada == 0:
                cx = 0.0; sy = 0.0
                for i in range(N):
                    cx += np.cos(th[i]); sy += np.sin(th[i])
                r = np.sqrt(cx * cx + sy * sy) / N
                a1 += r; a2 += r * r; n += 1
        R1[p] = a1 / n; R2[p] = a2 / n
    return R1, R2


def erdos_renyi(N, p, rng):
    """Matriz de adyacencia de un grafo de Erdős–Rényi G(N, p)."""
    U = np.triu(rng.random((N, N)) < p, 1)
    return (U | U.T).astype(np.float64)


def maximo_chi(mu, chi):
    """Posición del máximo de χ: parábola sobre los 5 puntos alrededor del máximo."""
    mu = np.asarray(mu); chi = np.asarray(chi)
    i = int(np.argmax(chi))
    sl = slice(max(i - 2, 0), min(i + 3, len(mu)))
    a, b, _ = np.polyfit(mu[sl], chi[sl], 2)
    return float(-b / (2 * a))
