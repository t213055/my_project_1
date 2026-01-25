import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# データ読み込み
# -----------------------------
data = np.loadtxt("output_1_25_1.txt", delimiter=",", comments="#")

# 基本列
alphas = data[:, 0]
cs     = data[:, 1]
betas  = data[:, 2]

# χ に対応する列（5列目〜12列目 → index 4〜11）
chi_cols = list(range(4, 12))

# -----------------------------
# 整理
# data_dict[alpha][c] = {
#     "beta": [...],
#     "chi": {k: [...]}
# }
# -----------------------------
data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for row in data:
    alpha = row[0]
    c     = row[1]
    beta  = row[2]

    data_dict[alpha][c]["beta"].append(beta)
    for k in chi_cols:
        data_dict[alpha][c][f"chi_{k}"].append(row[k])

col = ["q_v", "q_h", "hat_q_v", "hat_q_h", "r_v", "r_h", "hat_r_v", "hat_r_h"]
# -----------------------------
# 描画
# -----------------------------
fs = 20
for alpha, c_dict in data_dict.items():
    for c, val_dict in c_dict.items():
        plt.figure()

        beta_vals = np.array(val_dict["beta"])
        idx = np.argsort(beta_vals)
        beta_vals = beta_vals[idx]

        for k in chi_cols:
            chi_vals = np.array(val_dict[f"chi_{k}"])[idx]
            plt.plot(beta_vals, chi_vals, label=col[k-4])

        plt.xlabel(r"$\beta$", fontsize=fs)
        plt.ylabel(r"$\chi$", fontsize=fs)
        plt.yscale("log")
        plt.title(rf"$\alpha={alpha:g},\ c={c:g}$", fontsize=fs)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
