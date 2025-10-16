#!/usr/bin/env python3
"""
Задача 1.9.2 (§9.3).
Решение неявной осреднённой схемы (формула (1.9.2)) методом Ньютона
для одномерной квази-одномерной гемодинамической модели.

Входное давление задаётся как гармоническое колебание (аналог формулы (1.9.3)).
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. ФИЗИЧЕСКИЕ И ЧИСЛЕННЫЕ ПАРАМЕТРЫ
# -------------------------------
L = 1.0                  # длина сосуда, м
Nx = 101                 # число узлов
h = L / (Nx - 1)
rho = 1060.0             # плотность крови, кг/м^3
nu = 0.0005              # кинематическая вязкость, м^2/с
g = 9.81
phi = 0.0
S0 = 1.0e-4              # опорная площадь, м^2
p0 = 13300.0             # опорное давление, Па
K = 2e5                  # жёсткость стенки, Па
tau = 1e-4               # шаг по времени, с
Nt = 600                 # шагов по времени
sigma = 0.5              # вес осреднения (Crank–Nicolson)

# --- возбуждение давления на входе ---
A = 800.0                # амплитуда, Па
omega = 30.0 * np.pi     # частота, рад/с

# -------------------------------
# 2. НАЧАЛЬНЫЕ УСЛОВИЯ
# -------------------------------
x = np.linspace(0, L, Nx)
p = np.ones(Nx) * p0
u = np.zeros(Nx)
S = S0 * (1 + (p - p0) / K)

# -------------------------------
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -------------------------------
def S_of_p(p):
    """Линейное уравнение состояния."""
    return S0 * (1 + (p - p0) / K)

def F_system(S, u, p):
    """
    Правая часть F(y) = (F_S, F_u)
    F_S = -(u dS/dx + S du/dx)
    F_u = -(u du/dx + (1/rho) dp/dx)
    """
    dSdx = np.zeros_like(S)
    dudx = np.zeros_like(u)
    dpdx = np.zeros_like(p)

    dSdx[1:-1] = (S[2:] - S[:-2]) / (2*h)
    dudx[1:-1] = (u[2:] - u[:-2]) / (2*h)
    dpdx[1:-1] = (p[2:] - p[:-2]) / (2*h)

    F_S = -(u * dSdx + S * dudx)
    F_u = -(u * dudx + (1/rho) * dpdx)
    return F_S, F_u

def residual(S_new, u_new, S_old, u_old, p_old):
    """Невязка G(y_new) = 0 для схемы (1.9.2)."""
    p_new = p0 + K * (S_new / S0 - 1)
    F_S_old, F_u_old = F_system(S_old, u_old, p_old)
    F_S_new, F_u_new = F_system(S_new, u_new, p_new)
    G_S = S_new - S_old - tau * ((1 - sigma) * F_S_old + sigma * F_S_new)
    G_u = u_new - u_old - tau * ((1 - sigma) * F_u_old + sigma * F_u_new)
    return np.concatenate([G_S, G_u])

# -------------------------------
# 4. МЕТОД НЬЮТОНА
# -------------------------------
def newton_step(S_old, u_old, p_old, tol=1e-6, max_iter=15):
    """Решает G(y_new)=0 методом Ньютона."""
    S_new = S_old.copy()
    u_new = u_old.copy()
    N = len(S_old)

    for k in range(max_iter):
        G = residual(S_new, u_new, S_old, u_old, p_old)
        normG = np.linalg.norm(G)
        if normG < tol:
            print(f"  Newton converged in {k} iters, |G|={normG:.2e}")
            break

        # Якобиан численно (маленький Nx, можно позволить)
        eps = 1e-6
        J = np.zeros((2*N, 2*N))
        for j in range(2*N):
            dS = S_new.copy()
            dU = u_new.copy()
            if j < N:
                dS[j] += eps
            else:
                dU[j - N] += eps
            G1 = residual(dS, dU, S_old, u_old, p_old)
            J[:, j] = (G1 - G) / eps

        delta = np.linalg.solve(J, -G)
        dS = delta[:N]
        dU = delta[N:]
        S_new += dS
        u_new += dU

        if np.linalg.norm(delta) < tol:
            print(f"  Δy small, stopping at iter {k}")
            break

    return S_new, u_new

# -------------------------------
# 5. ЦИКЛ ПО ВРЕМЕНИ
# -------------------------------
p_history_mid = []
t_array = []

for n in range(Nt):
    t = n * tau

    # пульсирующее давление на входе
    p_in = p0 + A * np.sin(omega * t)
    p[0] = p_in
    S[0] = S_of_p(p_in)

    # свободный выход
    p[-1] = p[-2]
    S[-1] = S[-2]

    # шаг Ньютона
    S, u = newton_step(S, u, p)

    # обновление давления
    p = p0 + K * (S / S0 - 1)

    if n % 50 == 0:
        print(f"[t={t:.4f}s] p_in={p_in:.1f}, p_mid={p[Nx//2]:.1f}, u_mid={u[Nx//2]:.4f}")

    p_history_mid.append(p[Nx // 2])
    t_array.append(t)

# -------------------------------
# 6. ВИЗУАЛИЗАЦИЯ
# -------------------------------
plt.figure(figsize=(8,4))
plt.plot(t_array, p_history_mid)
plt.xlabel("t, c")
plt.ylabel("p(середина), Па")
plt.title("Изменение давления в середине сосуда (неявная схема, метод Ньютона)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,4))
plt.plot(x, p, label="p(x)")
plt.xlabel("x, м")
plt.ylabel("p, Па")
plt.title("Профиль давления вдоль сосуда (после Nt шагов)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
