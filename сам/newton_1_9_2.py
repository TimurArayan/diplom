import numpy as np
import matplotlib.pyplot as plt


L, Nx = 1.0, 101
h = L / (Nx - 1)
rho, K = 1060.0, 2e5
S0, p0 = 1e-4, 13300.0
A, omega = 800.0, 30*np.pi
tau, Nt = 1e-4, 1000
sigma = 0.5

x = np.linspace(0, L, Nx)
S = np.ones(Nx) * S0
u = np.zeros(Nx)
p = np.ones(Nx) * p0

def S_of_p(p):
    return S0 * (1 + (p - p0)/K)

def F_system(S, u, p):
    dSdx = np.gradient(S, h)
    dudx = np.gradient(u, h)
    dpdx = np.gradient(p, h)
    F_S = -(u*dSdx + S*dudx)
    F_u = -(u*dudx + (1/rho)*dpdx)
    return F_S, F_u

def residual(S_new, u_new, S_old, u_old, p_old):
    p_new = p0 + K * (S_new/S0 - 1)
    F_S_old, F_u_old = F_system(S_old, u_old, p_old)
    F_S_new, F_u_new = F_system(S_new, u_new, p_new)
    G_S = S_new - S_old - tau*((1 - sigma)*F_S_old + sigma*F_S_new)
    G_u = u_new - u_old - tau*((1 - sigma)*F_u_old + sigma*F_u_new)
    return np.concatenate([G_S, G_u])

def newton_step(S_old, u_old, p_old, tol=1e-6, max_iter=10):
    S_new = S_old.copy()
    u_new = u_old.copy()
    N = len(S_old)
    for k in range(max_iter):
        G = residual(S_new, u_new, S_old, u_old, p_old)
        normG = np.linalg.norm(G)
        if normG < tol:
            break
        eps = 1e-6
        J = np.zeros((2*N, 2*N))
        for j in range(2*N):
            dS = S_new.copy()
            dU = u_new.copy()
            if j < N:
                dS[j] += eps
            else:
                dU[j-N] += eps
            G1 = residual(dS, dU, S_old, u_old, p_old)
            J[:, j] = (G1 - G) / eps
        delta = np.linalg.solve(J, -G)
        dS, dU = delta[:N], delta[N:]
        S_new += dS
        u_new += dU
        if np.linalg.norm(delta) < tol:
            break
    return S_new, u_new

for n in range(Nt):
    t = n * tau
    p_in = p0 + A*np.sin(omega*t)
    p[0] = p_in
    S[0] = S_of_p(p_in)
    p[-1] = p[-2]
    S[-1] = S[-2]
    u[-1] = u[-2]
    S, u = newton_step(S, u, p)
    p = p0 + K*(S/S0 - 1)
    
    print(f"[t={t:.4f}s] p_in={p[0]:.1f}, p_mid={p[Nx//2]:.1f}, p_end={p[-1]:.1f}")

plt.plot(x, p, label="p(x)")
plt.xlabel("x, м"); plt.ylabel("p, Па")
plt.title("Схема (1.9.2) — неявная, метод Ньютона")
plt.grid(); plt.legend(); plt.show()
