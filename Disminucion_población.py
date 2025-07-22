import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo
S0 = 1000           # Población inicial
alpha = 0.1         # Tasa de mortalidad (mes^-1)
t_horizon = 24      # Horizonte de tiempo (meses)
dt = 5            # Paso de tiempo (meses)

# Vectores de tiempo y población
time_steps = np.arange(0, t_horizon + dt, dt)
population_euler = np.zeros(len(time_steps))
population_euler[0] = S0

# Simulación usando integración de Euler
for i in range(1, len(time_steps)):
    dS = -alpha * population_euler[i-1]
    population_euler[i] = population_euler[i-1] + dS * dt

# Solución analítica
population_analytic = S0 * np.exp(-alpha * time_steps)

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(time_steps, population_analytic, label='Solución Analítica', linewidth=2)
plt.plot(time_steps, population_euler, 'o-', label='Integración de Euler (Δt=5)', markersize=3)
plt.title('Disminución de la población con tasa de mortalidad constante')
plt.xlabel('Tiempo (meses)')
plt.ylabel('Población')
plt.legend()
plt.grid(True)
plt.show()
