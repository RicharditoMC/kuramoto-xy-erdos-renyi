"""Predicciones de campo medio para grafos densos (Erdős–Rényi con p fijo)."""
import numpy as np
from scipy.special import ive
from scipy.optimize import brentq


def mu_estrella(D, K=1.0):
    """Conectividad de la transición: el desorden se vuelve inestable en K μ = 2D."""
    return 2 * D / K


def r_campo_medio(mu, D, K=1.0):
    """Solución de r = I1(Kμr/D) / I0(Kμr/D); vale 0 por debajo de μ*."""
    if mu <= mu_estrella(D, K):
        return 0.0
    f = lambda r: r - ive(1, K * mu * r / D) / ive(0, K * mu * r / D)
    return brentq(f, 1e-9, 1.0)


def chi_desordenado(mu, D, K=1.0):
    """χ = (1 − π/4) / (D (1 − μ/μ*)) para μ < μ* y N → ∞."""
    return (1 - np.pi / 4) / (D * (1 - np.asarray(mu) / mu_estrella(D, K)))
