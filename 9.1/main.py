import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==============================
# 9.1. СХЕМА-ХАРАКТЕРИСТИКА (связь p, u, S)
# ==============================

# --- Параметры ---
L = 1.0        # длина сосуда, м
Nx = 151       # число узлов
Nt = 4000      # шагов по времени
h = L / (Nx - 1)
tau = 0.0002   # шаг по времени, с

rho = 1060.0   # плотность
nu = 0.0005    # физ. вязкость
g = 9.81
phi = 0.0

S0 = 1.0e-4    # площадь покоя
p0 = 13300.0   # давление покоя
K = 2e5        # жёсткость сосуда

# пульсирующее давление на входе
A = 800.0         # амплитуда пульсации, Па
omega = 30.0*np.pi  # частота, рад/с

sigma = 0.6  # вес для неявности

# --- Искусственная вязкость ---
c0 = np.sqrt(K / rho)
alpha = 0.25
a_u = alpha * h * c0
a_s = a_u * 0.1

print(f"c0={c0:.2f}, a_u={a_u:.3e}, a_s={a_s:.3e}, tau={tau:.5f}, h={h:.4e}")

# --- Сетка и поля ---
x = np.linspace(0, L, Nx)
t = np.arange(0, Nt) * tau

p = np.ones(Nx) * p0
S = S0 * (1.0 + (p - p0)/K)
u = np.zeros(Nx)

# для визуализации
log_interval = 200
p_frames = []
frames_to_save = 250
skip = max(1, Nt // frames_to_save)

# безопасный минимум
S_min_allowed = 1e-7

# --- Вспомогательные функции ---
def S_of_p(p_arr):
    return S0 * (1.0 + (p_arr - p0)/K)

def laplacian(arr, h_local):
    lap = np.zeros_like(arr)
    lap[1:-1] = (arr[2:] - 2*arr[1:-1] + arr[:-2]) / (h_local*h_local)
    lap[0]  = lap[1]
    lap[-1] = lap[-2]
    return lap

# ==============================
# ОСНОВНОЙ ЦИКЛ
# ==============================
for n in range(Nt - 1):

    # --- ГРАНИЦА ВХОДА (x=0): пульсирующее давление и характеристическое условие ---
    p_in = p0 + A * np.sin(omega * t[n])
    p_star = p[1]
    S_star = S[1]
    u_star = u[1]

    S_in = S_of_p(p_in)
    # формула из (1.9.1), упрощённая (без F_e и F_mp)
    term = np.sqrt(np.abs((p_in - p_star) * (np.sqrt(S_in) - np.sqrt(S_star))) /
                   (rho * (np.sqrt(S_in) + np.sqrt(S_star)) + 1e-12))
    u_in = u_star + 2.0 * np.sign(S_in - S_star) * term

    # применяем граничные значения
    p[0] = p_in
    u[0] = u_in

    # --- ГРАНИЦА ВЫХОДА (x=L): свободный выход ---
    p[-1] = p[-2]
    u[-1] = u[-2]

    # --- Производные ---
    dSdx = np.zeros_like(S)
    dudx = np.zeros_like(u)
    dpdx = np.zeros_like(p)

    dSdx[1:-1] = (S[2:] - S[:-2]) / (2*h)
    dudx[1:-1] = (u[2:] - u[:-2]) / (2*h)
    dpdx[1:-1] = (p[2:] - p[:-2]) / (2*h)

    dSdx[0] = (S[1] - S[0]) / h
    dudx[0] = (u[1] - u[0]) / h
    dpdx[0] = (p[1] - p[0]) / h

    dSdx[-1] = (S[-1] - S[-2]) / h
    dudx[-1] = (u[-1] - u[-2]) / h
    dpdx[-1] = (p[-1] - p[-2]) / h

    # --- Лапласианы ---
    lap_u = laplacian(u, h)
    lap_s = laplacian(S, h)

    # --- Правая часть (старый слой) ---
    S_safe = np.maximum(S, S_min_allowed)
    F_S = -(u*dSdx + S*dudx) + a_s*lap_s
    F_u = -(u*dudx + (1.0/rho)*dpdx + 8*np.pi*nu*u/S_safe - g*np.cos(phi)) + a_u*lap_u

    # --- Предсказатель ---
    S_pred = S + tau * F_S
    u_pred = u + tau * F_u
    S_pred = np.maximum(S_pred, S_min_allowed)
    p_pred = p0 + K * (S_pred/S0 - 1.0)

    # --- Производные нового слоя ---
    dSdx_new = np.zeros_like(S)
    dudx_new = np.zeros_like(u)
    dpdx_new = np.zeros_like(p)

    dSdx_new[1:-1] = (S_pred[2:] - S_pred[:-2]) / (2*h)
    dudx_new[1:-1] = (u_pred[2:] - u_pred[:-2]) / (2*h)
    dpdx_new[1:-1] = (p_pred[2:] - p_pred[:-2]) / (2*h)

    lap_u_new = laplacian(u_pred, h)
    lap_s_new = laplacian(S_pred, h)

    # --- Правая часть нового слоя ---
    S_pred_safe = np.maximum(S_pred, S_min_allowed)
    F_S_new = -(u_pred*dSdx_new + S_pred*dudx_new) + a_s*lap_s_new
    F_u_new = -(u_pred*dudx_new + (1.0/rho)*dpdx_new +
                8*np.pi*nu*u_pred/S_pred_safe - g*np.cos(phi)) + a_u*lap_u_new

    # --- Итоговое обновление ---
    S_new = S + tau*((1-sigma)*F_S + sigma*F_S_new)
    u_new = u + tau*((1-sigma)*F_u + sigma*F_u_new)
    S_new = np.maximum(S_new, S_min_allowed)
    p_new = p0 + K*(S_new/S0 - 1.0)

    S, u, p = S_new, u_new, p_new

    # --- Лог ---
    if n % log_interval == 0:
        print(f"[t={n*tau:.4f}s] p0={p[0]:.1f}, p_mid={p[Nx//2]:.1f}, p_end={p[-1]:.1f}")

    # --- Сохранение кадров ---
    if n % skip == 0:
        p_frames.append(p.copy())

# ==============================
# ВИЗУАЛИЗАЦИЯ
# ==============================
fig, ax = plt.subplots(figsize=(9,4))
line, = ax.plot(x, p_frames[0], color='blue')
ax.set_xlim(0, L)
p_all = np.concatenate(p_frames)
ax.set_ylim(np.min(p_all)-200, np.max(p_all)+200)
ax.set_xlabel("x, м")
ax.set_ylabel("p, Па")
ax.set_title("Анимация давления p(x,t)")

def update(frame):
    line.set_ydata(p_frames[frame])
    ax.set_title(f"t = {frame * skip * tau:.4f} c")
    return line,

ani = FuncAnimation(fig, update, frames=len(p_frames), blit=True, interval=30)
plt.show()
