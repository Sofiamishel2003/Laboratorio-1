import numpy as np
import matplotlib.pyplot as plt
import time

# Initial parameters
r1, r2 = 0.1, 0.1
K1, K2 = 1000, 1000
alpha, beta = 0.5, 0.3
x0, y0 = 10, 10
t_max = 100
h = 10

def derivates(x, y):
  dxdt = r1 * x * (1 - (x + alpha * y) / K1)
  dydt = r2 * y * (1 - (y + beta * x) / K2)
  return dxdt, dydt

def euler_system(x0, y0, h, t_max):
  t_values = np.arange(0, t_max + h, h)
  x_values, y_values = [x0], [y0]

  for t in t_values[:-1]:
    x, y = x_values[-1], y_values[-1]
    dxdt, dydt = derivates(x, y)
    x_values.append(x + h * dxdt)
    y_values.append(y + h * dydt)

  return t_values, np.array(x_values), np.array(y_values)

def rk4_system(x0, y0, h, t_max):
  t_values = np.arange(0, t_max + h, h)
  x_values, y_values = [x0], [y0]

  for t in t_values[:-1]:
    x, y = x_values[-1], y_values[-1]

    k1x, k1y = derivates(x, y)
    k2x, k2y = derivates(x + h/2 * k1x, y + h/2 * k1y)
    k3x, k3y = derivates(x + h/2 * k2x, y + h/2 * k2y)
    k4x, k4y = derivates(x + h * k3x, y + h * k3y)

    x_next = x + (h/6)*(k1x + 2*k2x + 2*k3x + k4x)
    y_next = y + (h/6)*(k1y + 2*k2y + 2*k3y + k4y)

    x_values.append(x_next)
    y_values.append(y_next)

  return t_values, np.array(x_values), np.array(y_values)

t_euler, x_euler, y_euler = euler_system(x0, y0, h, t_max)
t_rk4, x_rk4, y_rk4 = rk4_system(x0, y0, h, t_max)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(t_euler, x_euler, label='x (Euler)', linestyle='--', color='orange')
plt.plot(t_euler, y_euler, label='y (Euler)', linestyle='--', color='green')
plt.title('Método de Euler')
plt.xlabel('Tiempo')
plt.ylabel('Poblaciones')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_rk4, x_rk4, label='x (RK4)', color='blue')
plt.plot(t_rk4, y_rk4, label='y (RK4)', color='red')
plt.title('Método de Runge-Kutta (RK4)')
plt.xlabel('Tiempo')
plt.ylabel('Poblaciones')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Error relativo
_, x_ref_full, y_ref_full = rk4_system(x0, y0, 0.01, 100)
x_ref_50 = x_ref_full[int(50 / 0.01)]

_, x_euler, _ = euler_system(x0, y0, 1.0, 100)
_, x_rk4, _ = rk4_system(x0, y0, 1.0, 100)
x_euler_50 = x_euler[50]
x_rk4_50 = x_rk4[40]

error_euler = abs(x_euler_50 - x_ref_50) / abs(x_ref_50)
error_rk4 = abs(x_rk4_50 - x_ref_50) / abs(x_ref_50)

print(f"Error relativo en t=50 (Euler, Δt=1.0): {error_euler:.6f}")
print(f"Error relativo en t=50 (RK4, Δt=1.0):   {error_rk4:.6f}")
