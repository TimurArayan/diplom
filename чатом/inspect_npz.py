import numpy as np

for name in ["results_sigma.npz", "results_newton.npz"]:
    print(f"\n=== {name} ===")
    data = np.load(name)
    for key in data.keys():
        arr = data[key]
        print(f"{key:5s}: shape={arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}, mean={arr.mean():.3f}")
