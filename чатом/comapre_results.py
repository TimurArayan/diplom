import numpy as np
import matplotlib.pyplot as plt

# === Загрузка данных ===
sigma = np.load("results_sigma.npz")
newton = np.load("results_newton.npz")

t_sigma, p_sigma = sigma["t"], sigma["pmid"]
t_newton, p_newton = newton["t"], newton["pmid"]

# === Ограничиваем общую временную область ===
t_common_max = min(t_sigma.max(), t_newton.max())
mask_sigma = t_sigma <= t_common_max
mask_newton = t_newton <= t_common_max

print(f"Сравнивается диапазон времени: 0 – {t_common_max:.3f} c")

# === Построение графика ===
plt.figure(figsize=(9, 4))
plt.plot(t_sigma[mask_sigma], p_sigma[mask_sigma], label="σ-схема (pmid)", lw=2)
plt.plot(t_newton[mask_newton], p_newton[mask_newton], '--', label="Ньютон (pmid)", lw=2)
plt.xlabel("t, s")
plt.ylabel("p(mid), Па")
plt.title("Сравнение давления в середине капилляра")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
