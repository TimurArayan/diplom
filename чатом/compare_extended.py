#!/usr/bin/env python3
"""
compare_extended.py

Расширенное сравнение σ-схемы и метода Ньютона для задачи (1.9.1):
 - сравнение давления на входе p0(t)
 - сравнение давления в середине капилляра p_mid(t)
 - разность решений Δp_mid(t)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- пути к файлам ---
sigma_file = "results_sigma.npz"
newton_file = "results_newton.npz"

if not (os.path.exists(sigma_file) and os.path.exists(newton_file)):
    raise FileNotFoundError("Файлы results_sigma.npz и results_newton.npz должны быть в текущей папке.")

# --- загрузка данных ---
sigma = np.load(sigma_file)
newton = np.load(newton_file)

t_sigma = sigma["t"]
p0_sigma = sigma["p0"]
pmid_sigma = sigma["pmid"]

t_newton = newton["t"]
p0_newton = newton["p0"]
pmid_newton = newton["pmid"]

# --- обрезаем до общего диапазона времени ---
t_max_common = min(t_sigma.max(), t_newton.max())
mask_sigma = t_sigma <= t_max_common
mask_newton = t_newton <= t_max_common

t_sigma_c = t_sigma[mask_sigma]
p0_sigma_c = p0_sigma[mask_sigma]
pmid_sigma_c = pmid_sigma[mask_sigma]

t_newton_c = t_newton[mask_newton]
p0_newton_c = p0_newton[mask_newton]
pmid_newton_c = pmid_newton[mask_newton]

# --- интерполяция σ-схемы на сетку Ньютона для разности ---
pmid_sigma_interp = np.interp(t_newton_c, t_sigma_c, pmid_sigma_c)
dp_mid = pmid_newton_c - pmid_sigma_interp

# --- 1. Давление на входе ---
plt.figure(figsize=(9,4))
plt.plot(t_sigma_c, p0_sigma_c, label="σ-схема (p0)", lw=1.5)
plt.plot(t_newton_c, p0_newton_c, '--', label="Ньютон (p0)", lw=1.5)
plt.xlabel("t, с")
plt.ylabel("p₀(t), Па")
plt.title("Сравнение давления на входе капилляра")
plt.legend()
plt.grid(True)

# --- 2. Давление в середине ---
plt.figure(figsize=(9,4))
plt.plot(t_sigma_c, pmid_sigma_c, label="σ-схема (pmid)", lw=1.5)
plt.plot(t_newton_c, pmid_newton_c, '--', label="Ньютон (pmid)", lw=1.5)
plt.xlabel("t, с")
plt.ylabel("p(mid), Па")
plt.title("Сравнение давления в середине капилляра")
plt.legend()
plt.grid(True)

# --- 3. Разность решений Δp_mid ---
plt.figure(figsize=(9,4))
plt.plot(t_newton_c, dp_mid, color="purple", lw=1.8)
plt.xlabel("t, с")
plt.ylabel("Δp(mid) = p_newton - p_sigma, Па")
plt.title("Разность решений в середине капилляра")
plt.grid(True)
plt.axhline(0, color="gray", linestyle="--", lw=1)

plt.tight_layout()
plt.show()
#!/usr/bin/env python3
"""
compare_extended.py

Расширенное сравнение σ-схемы и метода Ньютона для задачи (1.9.1):
 - сравнение давления на входе p0(t)
 - сравнение давления в середине капилляра p_mid(t)
 - разность решений Δp_mid(t)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- пути к файлам ---
sigma_file = "results_sigma.npz"
newton_file = "results_newton.npz"

if not (os.path.exists(sigma_file) and os.path.exists(newton_file)):
    raise FileNotFoundError("Файлы results_sigma.npz и results_newton.npz должны быть в текущей папке.")

# --- загрузка данных ---
sigma = np.load(sigma_file)
newton = np.load(newton_file)

t_sigma = sigma["t"]
p0_sigma = sigma["p0"]
pmid_sigma = sigma["pmid"]

t_newton = newton["t"]
p0_newton = newton["p0"]
pmid_newton = newton["pmid"]

# --- обрезаем до общего диапазона времени ---
t_max_common = min(t_sigma.max(), t_newton.max())
mask_sigma = t_sigma <= t_max_common
mask_newton = t_newton <= t_max_common

t_sigma_c = t_sigma[mask_sigma]
p0_sigma_c = p0_sigma[mask_sigma]
pmid_sigma_c = pmid_sigma[mask_sigma]

t_newton_c = t_newton[mask_newton]
p0_newton_c = p0_newton[mask_newton]
pmid_newton_c = pmid_newton[mask_newton]

# --- интерполяция σ-схемы на сетку Ньютона для разности ---
pmid_sigma_interp = np.interp(t_newton_c, t_sigma_c, pmid_sigma_c)
dp_mid = pmid_newton_c - pmid_sigma_interp

# --- 1. Давление на входе ---
plt.figure(figsize=(9,4))
plt.plot(t_sigma_c, p0_sigma_c, label="σ-схема (p0)", lw=1.5)
plt.plot(t_newton_c, p0_newton_c, '--', label="Ньютон (p0)", lw=1.5)
plt.xlabel("t, с")
plt.ylabel("p₀(t), Па")
plt.title("Сравнение давления на входе капилляра")
plt.legend()
plt.grid(True)

# --- 2. Давление в середине ---
plt.figure(figsize=(9,4))
plt.plot(t_sigma_c, pmid_sigma_c, label="σ-схема (pmid)", lw=1.5)
plt.plot(t_newton_c, pmid_newton_c, '--', label="Ньютон (pmid)", lw=1.5)
plt.xlabel("t, с")
plt.ylabel("p(mid), Па")
plt.title("Сравнение давления в середине капилляра")
plt.legend()
plt.grid(True)

# --- 3. Разность решений Δp_mid ---
plt.figure(figsize=(9,4))
plt.plot(t_newton_c, dp_mid, color="purple", lw=1.8)
plt.xlabel("t, с")
plt.ylabel("Δp(mid) = p_newton - p_sigma, Па")
plt.title("Разность решений в середине капилляра")
plt.grid(True)
plt.axhline(0, color="gray", linestyle="--", lw=1)

plt.tight_layout()
plt.show()
