import numpy as np
import matplotlib.pyplot as plt

sdfsdf
L, Nx = 1.0, 101
h = L / (Nx - 1)
rho, K = 1060.0, 2e5
S0, p0 = 1, 13300.0
A, omega = 800.0, 30*np.pi
tau, Nt = 1e-4, 1000
sigma = 0.5

x = np.linspace(0, L, Nx)
S = np.ones(Nx) * S0
u = np.zeros(Nx)
p = np.ones(Nx) * p0

def S_of_p(p): return S0 * (1 + (p - p0)/K)


for n in range(Nt):
    t = n * tau
    
    p_in = p0 + A*np.sin(omega*t)
    p[0] = p_in
    S[0] = S_of_p(p_in)
    p[-1] = p[-2]
    S[-1] = S[-2]
    u[-1] = u[-2]

    dSdx = np.gradient(S, h)
    dudx = np.gradient(u, h)
    dpdx = np.gradient(p, h)

    F_S = -(u * dSdx + S * dudx)
    F_u = -(u * dudx + (1/rho) * dpdx)

    S_pred = S + tau * F_S
    u_pred = u + tau * F_u
    p_pred = p0 + K * (S_pred/S0 - 1)

    dSdx_new = np.gradient(S_pred, h)
    dudx_new = np.gradient(u_pred, h)
    dpdx_new = np.gradient(p_pred, h)
    F_S_new = -(u_pred * dSdx_new + S_pred * dudx_new)
    F_u_new = -(u_pred * dudx_new + (1/rho) * dpdx_new)

    S = S + tau * ((1 - sigma)*F_S + sigma*F_S_new)
    u = u + tau * ((1 - sigma)*F_u + sigma*F_u_new)
    p = p0 + K * (S/S0 - 1)

    
    print(f"[t={t:.4f}s] p_in={p[0]:.1f}, p_mid={p[Nx//2]:.1f}, p_end={p[-1]:.1f}")

plt.plot(x, p, label="p(x)")
plt.xlabel("x, м"); plt.ylabel("p, Па")
plt.title("Схема (1.9.1) — σ-схема (предсказатель-корректор)")
plt.grid(); plt.legend(); plt.show()
