import numpy as np
import matplotlib.pyplot as plt

# === Загрузка результатов системы (9.4) ===
data = np.load("results_graph_newton.npz")
print("Содержимое:", list(data.keys()))

t = data["t"]
p0 = data["p0"]

# Собираем все pmid по ключам pmid0, pmid1, pmid2 ...
pmid_keys = sorted([k for k in data.keys() if k.startswith("pmid")])
pmid_all = [data[k] for k in pmid_keys]
pmid_all = np.vstack(pmid_all).T  # (Nt, Nbranches)

# --- Давление в серединах всех сосудов ---
plt.figure(figsize=(9,4))
for i in range(pmid_all.shape[1]):
    plt.plot(t, pmid_all[:, i], label=f"Сосуд {i}")
plt.xlabel("t, с")
plt.ylabel("p(mid), Па")
plt.title("Давление в серединах сосудов (Ньютон, система 9.4)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Давление на входе ---
plt.figure(figsize=(9,4))
plt.plot(t, p0, color='k', label='p0 (вход)')
plt.xlabel("t, с")
plt.ylabel("Давление, Па")
plt.title("Давление на входе системы (Ньютон, 9.4)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
