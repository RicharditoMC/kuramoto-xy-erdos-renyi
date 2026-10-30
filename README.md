# Sincronización con ruido en grafos de Erdős–Rényi

Código del trabajo *Sincronización con ruido en grafos de Erdős–Rényi: correspondencia entre el modelo de Kuramoto estocástico y el modelo XY*.

Estudiamos osciladores de Kuramoto idénticos con ruido sobre grafos de Erdős–Rényi:

$$d\theta_i=\frac{K}{N}\sum_j A_{ij}\sin(\theta_j-\theta_i)\,dt+\sqrt{2D}\,dW_i .$$

Esta dinámica tiene como estado estacionario la distribución de Boltzmann del modelo XY, con temperatura $k_BT=D$. Variando la conectividad $\mu=\langle g\rangle/(N-1)\simeq p$ a ruido fijo, la susceptibilidad del parámetro de orden muestra una transición de fase en $\mu^*=2D/K$.

## Requisitos

Python 3.9 o posterior y los paquetes de `requirements.txt`:

```
pip install -r requirements.txt
```

Todo corre en una computadora común o en Google Colab. `numba` compila las simulaciones y las paraleliza.

## Contenido

```
src/dinamica.py              integración de la dinámica (Euler–Maruyama), grafos ER, máximo de χ
src/teoria.py                predicciones de campo medio: μ* = 2D/K, ecuación de Bessel, χ
simulacion_er.py             barridos en Erdős–Rényi            → resultados/er.json
figuras_er.py                figuras del informe                → figuras/er_*.png
verificacion_metropolis.py   Kuramoto con ruido vs. XY con Metropolis → figuras/verificacion_metropolis.png
anillo_local.py              contraste: anillo de alcance fijo  → resultados/anillo_local.json
resultados/                  resultados ya calculados (los del informe)
figuras/                     figuras ya generadas
```

## Cómo reproducir los resultados

1. **Simulaciones de Erdős–Rényi** (unos 15 minutos):

   ```
   python simulacion_er.py
   ```

   Hace dos barridos, con $K=1$, $\Delta t=0.05$, 300 unidades de tiempo de termalización y 4000 de medición:

   - $D=0.2$ con $N=100,200,400$, barriendo $p$ de 0.2 a 0.7 (4, 3 y 2 grafos por punto);
   - $N=200$ con $D=0.10,0.15,0.25,0.30$.

   Las semillas son las del informe, así que los números coinciden.

2. **Figuras** (segundos; usan `resultados/er.json`, que ya está incluido):

   ```
   python figuras_er.py
   ```

3. **Verificación con Metropolis** (unos 3 minutos):

   ```
   python verificacion_metropolis.py            # N=100, p=0.6, D=0.2
   python verificacion_metropolis.py 100 0.4 0.3
   ```

   Compara la distribución de $r$ de la dinámica de Kuramoto (tres pasos de tiempo) con la del modelo XY muestreado con Metropolis. Con los valores por defecto, $\langle r\rangle$ da 0.707, 0.713 y 0.717 para $\Delta t=0.1, 0.05, 0.02$, y 0.719 con Metropolis: Langevin converge a Metropolis al achicar el paso.

4. **Contraste con el anillo local** (opcional, unos 15 minutos):

   ```
   python anillo_local.py
   ```

   En un anillo donde cada nodo se une con unos pocos vecinos cercanos, el máximo de $\chi$ se corre hacia cero al crecer $N$. Es un cruce de tamaño finito, no una transición: en una dimensión el ruido destruye el orden.

## Resultados principales

- **Posición del máximo de $\chi$ con $D=0.2$:** $\mu^*(N)=0.435$, 0.420 y 0.409 para $N=100$, 200 y 400. Se acerca a $2D/K=0.4$.
- **$\langle r\rangle$ por encima de $\mu^*$:** sigue $r=I_1(K\mu r/D)/I_0(K\mu r/D)$ y no depende de $N$.
- **$\langle r\rangle$ por debajo de $\mu^*$:** cae como $N^{-1/2}$.
- **Dependencia con el ruido ($N=200$):** $\mu^*=0.208$, 0.316, 0.420, 0.525 y 0.621 para $D=0.10$ a 0.30, de acuerdo con $\mu^*=2D/K$.

## Referencias

- S. Gupta, A. Campa, S. Ruffo, *Statistical Physics of Synchronization*, Springer (2018).
- J. A. Acebrón et al., Rev. Mod. Phys. 77, 137 (2005).
