import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# データ読み込み
# -----------------------------
data = np.loadtxt("output_1_16_1.txt", delimiter=",", comments="#")

# 列を分解
alphas = data[:, 0]
cs     = data[:, 1]
betas  = data[:, 2]
vars_  = data[:, 12]   # 13列目：分散（0始まりなので index 12）

# -----------------------------
# alpha ごとに整理
# data_dict[alpha][c] = (beta_array, variance_array)
# -----------------------------
data_dict = defaultdict(lambda: defaultdict(list))

for a, c, b, v in zip(alphas, cs, betas, vars_):
    data_dict[a][c].append((b, v))

# -----------------------------
# 描画
# -----------------------------
for alpha, c_dict in data_dict.items():
    plt.figure()

    for c, values in c_dict.items():
        values = np.array(values)
        b_vals = values[:, 0]
        v_vals = values[:, 1]

        # beta 昇順に並び替え（重要）
        idx = np.argsort(b_vals)
        b_vals = b_vals[idx]
        v_vals = v_vals[idx]

        plt.plot(b_vals, v_vals, label=f"c={c:g}")

    plt.xlabel(r"$\beta$")
    plt.ylabel("Variance")
    plt.title(rf"$\alpha={alpha:g}$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
