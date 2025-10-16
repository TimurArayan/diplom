#!/usr/bin/env python3
"""
newton_full_1_9_1.py

Решение полной схемы (1.9.1) методом Ньютона для одного упругого капилляра.
- неявная σ-схема (Crank–Nicolson-like)
- силы: трение Fe = -8πν u / S, гравитация g cos(phi), место для F_mp
- искусственная вязкость (laplacian)
- граничное условие на входе: p_in(t) и характеристическая формула для u_in
- неизвестные для НЬЮТОНА: внутренние узлы (1 .. Nx-2) для S и u
"""

import numpy as np
import time

# -------------------------
# 1. ПАРАМЕТРЫ
# -------------------------
L = 1.0
Nx = 101
h = L / (Nx - 1)

rho = 1060.0
nu = 0.0005
g = 9.81
phi = 0.0

S0 = 1.0e-4
p0 = 13300.0
K = 2e5

# возбуждение (сердце)
A = 800.0
omega = 30.0 * np.pi

sigma = 0.6
alpha = 0.4

c0 = np.sqrt(K / rho)
tau = 5e-5
Nt = 4000

S_min = 1e-8
U_max = 10.0
F_clip = 200.0
p_limit = 1e6

a_u_base = alpha * h * c0
a_s_base = a_u_base * 0.1

newton_tol = 1e-6
newton_maxiter = 20
jac_eps = 1e-7


# -------------------------
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -------------------------
def S_of_p(p_arr):
    return S0 * (1.0 + (p_arr - p0) / K)


def laplacian(arr, h_local):
    lap = np.zeros_like(arr)
    lap[1:-1] = (arr[2:] - 2.0 * arr[1:-1] + arr[:-2]) / (h_local ** 2)
    lap[0] = lap[-1] = 0.0
    return lap


def characteristic_u_in(p_in, p_star, S_in, S_star, u_star, tau_local, Fe_star, Fmp_star,
                        Fe_in_est=0.0, Fmp_in_est=0.0):
    sqrt_Sin = np.sqrt(max(S_in, S_min))
    sqrt_Ss = np.sqrt(max(S_star, S_min))
    num = (p_in - p_star) * (sqrt_Sin - sqrt_Ss)
    den = rho * (sqrt_Sin + sqrt_Ss) + 1e-12
    term_val = np.sqrt(max(abs(num / den), 0.0)) if den != 0.0 else 0.0
    sg = np.sign(S_in - S_star) if S_in != S_star else 1.0
    F_avg = 0.5 * (Fe_star + Fe_in_est + Fmp_star + Fmp_in_est)
    F_avg = np.clip(F_avg, -F_clip, F_clip)
    u_in = u_star + 2.0 * sg * term_val + tau_local * F_avg
    u_in = np.clip(u_in, -U_max, U_max)
    return u_in


# -------------------------
# 3. ОСНОВНЫЕ ОПЕРАТОРЫ
# -------------------------
def compute_F_fields(S_arr, u_arr, p_arr, a_s, a_u):
    dSdx = np.zeros_like(S_arr)
    dudx = np.zeros_like(u_arr)
    dpdx = np.zeros_like(p_arr)

    dSdx[1:-1] = (S_arr[2:] - S_arr[:-2]) / (2.0 * h)
    dudx[1:-1] = (u_arr[2:] - u_arr[:-2]) / (2.0 * h)
    dpdx[1:-1] = (p_arr[2:] - p_arr[:-2]) / (2.0 * h)

    lap_s = laplacian(S_arr, h)
    lap_u = laplacian(u_arr, h)

    S_safe = np.maximum(S_arr, S_min)
    F_S = -(u_arr * dSdx + S_arr * dudx) + a_s * lap_s
    F_u = -(u_arr * dudx + (1.0 / rho) * dpdx + (8.0 * np.pi * nu * u_arr) / S_safe - g * np.cos(phi)) + a_u * lap_u

    return F_S, F_u


# -------------------------
# 4. НЕВЯЗКА ДЛЯ NEWTON
# -------------------------
def build_residual_vec(S_new, u_new, S_old, u_old, p_old, a_s, a_u):
    Nint = Nx - 2
    p_new = p0 + K * (S_new / S0 - 1.0)
    F_S_old, F_u_old = compute_F_fields(S_old, u_old, p_old, a_s, a_u)
    F_S_new, F_u_new = compute_F_fields(S_new, u_new, p_new, a_s, a_u)

    G = np.zeros(2 * Nint)
    for idx, j in enumerate(range(1, Nx - 1)):
        GS = S_new[j] - S_old[j] - tau * ((1 - sigma) * F_S_old[j] + sigma * F_S_new[j])
        GU = u_new[j] - u_old[j] - tau * ((1 - sigma) * F_u_old[j] + sigma * F_u_new[j])
        G[idx] = GS
        G[Nint + idx] = GU
    return G


# -------------------------
# 5. МЕТОД НЬЮТОНА
# -------------------------
def newton_solve_step(S_old, u_old, p_old, S_boundary, u_boundary, a_s, a_u):
    F_S_old, F_u_old = compute_F_fields(S_old, u_old, p_old, a_s, a_u)
    S_pred = np.maximum(S_old + tau * F_S_old, S_min)
    u_pred = np.nan_to_num(u_old + tau * F_u_old)

    S_new, u_new = S_pred.copy(), u_pred.copy()
    S_new[0], S_new[-1] = S_boundary
    u_new[0], u_new[-1] = u_boundary

    Nint = Nx - 2
    for k in range(newton_maxiter):
        G = build_residual_vec(S_new, u_new, S_old, u_old, p_old, a_s, a_u)
        normG = np.linalg.norm(G)
        if normG < newton_tol:
            return S_new, u_new, True, k, normG

        J = np.zeros((2 * Nint, 2 * Nint))
        base_S, base_u = S_new.copy(), u_new.copy()

        for col in range(2 * Nint):
            S_try, u_try = base_S.copy(), base_u.copy()
            if col < Nint:
                j = 1 + col
                S_try[j] += jac_eps
            else:
                j = 1 + (col - Nint)
                u_try[j] += jac_eps
            S_try[0], S_try[-1] = S_boundary
            u_try[0], u_try[-1] = u_boundary
            G_try = build_residual_vec(S_try, u_try, S_old, u_old, p_old, a_s, a_u)
            J[:, col] = (G_try - G) / jac_eps

        delta = np.linalg.solve(J, -G)
        dS, dU = np.zeros(Nx), np.zeros(Nx)
        dS[1:-1] = delta[:Nint]
        dU[1:-1] = delta[Nint:]
        S_new += dS
        u_new += dU

        S_new = np.clip(S_new, S_min, None)
        u_new = np.clip(u_new, -U_max, U_max)
        S_new[0], S_new[-1] = S_boundary
        u_new[0], u_new[-1] = u_boundary

        if np.linalg.norm(delta) < newton_tol:
            return S_new, u_new, True, k + 1, normG

    return S_new, u_new, False, newton_maxiter, np.linalg.norm(G)


# -------------------------
# 6. ОСНОВНОЙ ЦИКЛ
# -------------------------
def run_simulation():
    x = np.linspace(0, L, Nx)
    p = np.ones(Nx) * p0
    S = S_of_p(p)
    u = np.zeros(Nx)

    history_t, history_p0, history_pmid, history_pend, history_u_mid = [], [], [], [], []

    a_u, a_s = a_u_base, a_s_base
    start_time = time.time()

    for n in range(Nt):
        t = n * tau
        p_in = p0 + A * np.sin(omega * t)

        p_star, S_star, u_star = p[1], S[1], u[1]
        S_in = S_of_p(p_in)
        Fe_star = -8 * np.pi * nu * u_star / max(S_star, S_min) + g * np.cos(phi)
        u_in = characteristic_u_in(p_in, p_star, S_in, S_star, u_star, tau, Fe_star, 0.0)

        S_boundary = (S_in, S[-1])
        u_boundary = (u_in, u[-1])

        S_new, u_new, ok, iters, normG = newton_solve_step(S, u, p, S_boundary, u_boundary, a_s, a_u)
        if not ok:
            print(f"[t={t:.6f}] Newton failed, |G|={normG:.2e}")
            break

        p_new = p0 + K * (S_new / S0 - 1.0)
        S, u, p = S_new, u_new, p_new

        if n % 50 == 0:
            print(f"[t={t:.6f}] iters={iters}, |G|={normG:.2e}, p0={p[0]:.1f}, p_mid={p[Nx//2]:.1f}, u_mid={u[Nx//2]:.4f}")

        history_t.append(t)
        history_p0.append(p[0])
        history_pmid.append(p[Nx // 2])
        history_pend.append(p[-1])
        history_u_mid.append(u[Nx // 2])

    elapsed = time.time() - start_time
    print(f"Simulation finished, elapsed: {elapsed:.2f} s")

    # --- Сохранение результатов ---
    np.savez(
        "results_newton.npz",
        t=np.array(history_t),
        p0=np.array(history_p0),
        pmid=np.array(history_pmid),
        pend=np.array(history_pend),
        u_mid=np.array(history_u_mid),
    )
    print("✅ Results saved to results_newton.npz")

    return x, p, S, u


if __name__ == "__main__":
    x, p_final, S_final, u_final = run_simulation()
