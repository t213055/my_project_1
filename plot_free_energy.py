import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# データ読み込み
# -----------------------------
data = np.loadtxt("output_1_15_2.txt", delimiter=",", comments="#")

# 列を分解
alphas = data[:, 0]
cs     = data[:, 1]
betas  = data[:, 2]
freeEs = data[:, 13]   # 14列目（0始まりなので index 13）

# -----------------------------
# alpha ごとに整理
# data_dict[alpha][c] = (beta_array, free_energy_array)
# -----------------------------
data_dict = defaultdict(lambda: defaultdict(list))

for a, c, b, f in zip(alphas, cs, betas, freeEs):
    data_dict[a][c].append((b, f))

# -----------------------------
# 描画
# -----------------------------
fs = 20
for alpha, c_dict in data_dict.items():
    plt.figure()

    for c, values in c_dict.items():
        values = np.array(values)
        b_vals = values[:, 0]
        f_vals = values[:, 1]

        # beta 昇順に並び替え（重要）
        idx = np.argsort(b_vals)
        b_vals = b_vals[idx]
        f_vals = f_vals[idx]

        plt.plot(b_vals, f_vals, label=f"c={c:g}")

    plt.xlabel(r"$\beta$", fontsize=fs)
    plt.ylabel("Free Energy", fontsize=fs)
    plt.title(rf"$\alpha={alpha:g}$", fontsize=fs)
    plt.legend(fontsize=fs)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
