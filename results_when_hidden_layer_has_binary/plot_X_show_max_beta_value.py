"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# データ読み込み
# -----------------------------
data = np.loadtxt("beta_increase.txt", delimiter=",", comments="#")

alphas = data[:, 0]
cs     = data[:, 1]
betas  = data[:, 2]
chis   = data[:, 3]

# -----------------------------
# alpha ごとに整理
# -----------------------------
data_dict = defaultdict(lambda: defaultdict(list))
for a, c, b, chi in zip(alphas, cs, betas, chis):
    data_dict[a][c].append((b, chi))

# -----------------------------
# 描画（縦並び）
# -----------------------------
fs = 20
alpha_keys = sorted(data_dict.keys())
n_alpha = len(alpha_keys)

fig, axes = plt.subplots(
    n_alpha, 1,
    figsize=(7, 4*n_alpha),
    sharex=True,
    sharey=True
)

# alpha が1つのときの保険
if n_alpha == 1:
    axes = [axes]

for ax, alpha in zip(axes, alpha_keys):
    c_dict = data_dict[alpha]

    for c, values in c_dict.items():
        values = np.array(values)
        b_vals = values[:, 0]
        chi_vals = values[:, 1]

        idx = np.argsort(b_vals)
        b_vals = b_vals[idx]
        chi_vals = chi_vals[idx]

        ax.plot(b_vals, chi_vals, label=f"c={c:g}")

    ax.set_yscale("log")
    ax.set_title(rf"$\alpha={alpha:g}$", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.grid(True)
    ax.legend(fontsize=fs)

# 軸ラベルは共通で1つだけ
axes[-1].set_xlabel(r"$\beta$", fontsize=fs)
axes[0].set_ylabel(r"$\chi$", fontsize=fs)

plt.tight_layout()
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# データ読み込み
# -----------------------------
data = np.loadtxt("beta_increase_stepsize0.001.txt", delimiter=",", comments="#")

alphas = data[:, 0]
cs     = data[:, 1]
betas  = data[:, 2]
chis   = data[:, 3]

# -----------------------------
# alpha ごとに整理
# -----------------------------
data_dict = defaultdict(lambda: defaultdict(list))
for a, c, b, chi in zip(alphas, cs, betas, chis):
    data_dict[a][c].append((b, chi))

# -----------------------------
# 描画（縦並び）
# -----------------------------
fs = 20
alpha_keys = sorted(data_dict.keys())
n_alpha = len(alpha_keys)

fig, axes = plt.subplots(
    n_alpha, 1,
    figsize=(7, 4*n_alpha),
    sharex=True,
    sharey=True
)

if n_alpha == 1:
    axes = [axes]

for ax, alpha in zip(axes, alpha_keys):
    c_dict = data_dict[alpha]

    for c, values in c_dict.items():
        values = np.array(values)
        b_vals = values[:, 0]
        chi_vals = values[:, 1]

        # beta 昇順ソート
        idx = np.argsort(b_vals)
        b_vals = b_vals[idx]
        chi_vals = chi_vals[idx]

        # 通常プロット（色を取得）
        line, = ax.plot(b_vals, chi_vals, label=f"c={c:g}")
        color = line.get_color()

        # χ 最大点
        imax = np.argmax(chi_vals)
        b_peak = b_vals[imax]
        chi_peak = chi_vals[imax]

        # 最大点を強調
        ax.plot(b_peak, chi_peak, "o", color=color, markersize=8)

        # 横軸上に目印（短い縦線）
        ax.axvline(
            b_peak,
            ymin=0.0, ymax=0.05,
            color=color,
            linestyle="-",
            linewidth=2
        )

        # β の数値表示（横軸付近）
        ax.text(
        b_peak,
        chi_peak * 1.15,     # logスケールなので倍率で上にずらす
        rf"{b_peak:.3g}",
        color=color,
        ha="center",
        va="bottom",
        fontsize=fs*0.7
        )


    ax.set_yscale("log")
    ax.set_title(rf"$\alpha={alpha:g}$", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.grid(True)
    ax.legend(fontsize=fs)

axes[-1].set_xlabel(r"$\beta$", fontsize=fs)
axes[0].set_ylabel(r"$\chi$", fontsize=fs)

plt.tight_layout()
plt.show()
