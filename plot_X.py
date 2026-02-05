import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# データ読み込み
# -----------------------------
#data = np.loadtxt("output_reproduction_exp.txt", delimiter=",")
data = np.loadtxt("output_2026_2_4.txt", delimiter=",", comments="#")
# 列を分解
alphas = data[:, 0]
cs     = data[:, 1]
betas  = data[:, 2]
chis   = data[:, 3]

# -----------------------------
# alpha ごとに整理
# data_dict[alpha][c] = (beta_array, chi_array)
# -----------------------------
data_dict = defaultdict(lambda: defaultdict(list))

for a, c, b, chi in zip(alphas, cs, betas, chis):
    data_dict[a][c].append((b, chi))


# -----------------------------
# 描画
# -----------------------------
fs = 20
for alpha, c_dict in data_dict.items():
    plt.figure()

    for c, values in c_dict.items():
        values = np.array(values)
        b_vals = values[:, 0]
        chi_vals = values[:, 1]

        # beta 昇順に並び替え（重要）
        idx = np.argsort(b_vals)
        b_vals = b_vals[idx]
        chi_vals = chi_vals[idx]

        plt.plot(b_vals, chi_vals, marker='', label=f"c={c:g}")

    plt.xlabel(r"$\beta$", fontsize=fs)
    plt.ylabel(r"$\chi$", fontsize=fs)
    plt.yscale("log")
    plt.title(rf"$\alpha={alpha:g}$", fontsize=fs)
    plt.legend(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
