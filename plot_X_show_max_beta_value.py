"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# データ読み込み
# -----------------------------
data = np.loadtxt("results_when_hidden_layer_has_binary/beta_increase_stepsize0.001.txt", delimiter=",", comments="#")

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
"""
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# データ読み込み
# -----------------------------
data = np.loadtxt("results_when_hidden_layer_has_binary/beta_increase_stepsize0.001.txt", delimiter=",", comments="#")
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
    figsize=(8, 5 * n_alpha), # 少し縦を広げました
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
        line, = ax.plot(b_vals, chi_vals, label=f"c={c:g}", alpha=0.8)
        color = line.get_color()

        # --- χ 最大点 (Peak) ---
        imax = np.argmax(chi_vals)
        b_peak = b_vals[imax]
        chi_peak = chi_vals[imax]

        ax.plot(b_peak, chi_peak, "o", color=color, markersize=8)
        ax.text(
            b_peak, chi_peak * 1.2, 
            rf"{b_peak:.7g}", color=color, ha="center", va="bottom", fontsize=fs*0.6, weight='bold'
        )

        # --- χ 下落後の極小点 (Drop Bottom) ---
        # 最大値のインデックス以降のデータから最小値を探す
        if imax < len(chi_vals) - 1:
            # imax以降のスライスに対してargmin
            imin_relative = np.argmin(chi_vals[imax:])
            imin = imax + imin_relative
            
            b_min = b_vals[imin]
            chi_min = chi_vals[imin]

            # 極小点を強調（塗りつぶしなしの丸印などで区別しても良いですが、ここでは同じスタイルに）
            ax.plot(b_min, chi_min, "o", color=color, markersize=8, markerfacecolor='white', markeredgewidth=2)
            
            # x軸上に目印
            ax.axvline(b_min, ymin=0.0, ymax=0.05, color=color, linestyle="--", linewidth=1.5)

            # β の数値表示（点は下側に表示して重なりを避ける）
            ax.text(
                b_min, chi_min * 0.7, 
                rf"{b_min:.7g}", color=color, ha="center", va="top", fontsize=fs*0.6
            )

    ax.set_yscale("log")
    ax.set_title(rf"$\alpha={alpha:g}$", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=fs*0.8, loc='upper right')

axes[-1].set_xlabel(r"$\beta$", fontsize=fs)
axes[0].set_ylabel(r"$\chi$", fontsize=fs)

plt.tight_layout()
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =============================
# 凡例の配置設定
# =============================
# loc: 凡例ボックスの基準となる角（'upper left', 'best', 'center right' など）
# bbox: グラフ枠に対する相対座標 (x, y)。枠外に出す場合は (1.02, 1.0) のように指定。
#       枠内に収める（重なりを許容しつつ自動で良い場所を探す）場合は bbox = None に設定。
LEGEND_LOC = 'upper left'
LEGEND_BBOX = (1.02, 1.0)  

# -----------------------------
# データ読み込み
# -----------------------------
#data = np.loadtxt("results_when_hidden_layer_has_binary/beta_increase_stepsize0.001.txt", delimiter=",", comments="#")
data = np.loadtxt("beta_increase_stepsize0.001_eps=0.001_α=1.0_c=-5_beta25.0.txt", delimiter=",", comments="#")
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
    figsize=(8, 5 * n_alpha),
    sharex=True,
    sharey=True,
    layout="constrained" # 枠外の凡例が見切れないように自動調整するオプション
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
        line, = ax.plot(b_vals, chi_vals, label=f"c={c:g}", alpha=0.8)
        color = line.get_color()

        # --- χ 最大点 (Peak) ---
        imax = np.argmax(chi_vals)
        b_peak = b_vals[imax]
        chi_peak = chi_vals[imax]

        ax.plot(b_peak, chi_peak, "o", color=color, markersize=8)
        ax.text(
            b_peak, chi_peak * 1.2, 
            rf"{b_peak:.7g}", color=color, ha="center", va="bottom", fontsize=fs*0.6, weight='bold'
        )

        # --- χ 下落後の極小点 (Drop Bottom) ---
        if imax < len(chi_vals) - 1:
            imin_relative = np.argmin(chi_vals[imax:])
            imin = imax + imin_relative
            
            b_min = b_vals[imin]
            chi_min = chi_vals[imin]

            ax.plot(b_min, chi_min, "o", color=color, markersize=8, markerfacecolor='white', markeredgewidth=2)
            ax.axvline(b_min, ymin=0.0, ymax=0.05, color=color, linestyle="--", linewidth=1.5)

            ax.text(
                b_min, chi_min * 0.7, 
                rf"{b_min:.7g}", color=color, ha="center", va="top", fontsize=fs*0.6
            )

    ax.set_yscale("log")
    ax.set_title(rf"$\alpha={alpha:g}$", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # --- 凡例の描画 ---
    if LEGEND_BBOX is not None:
        ax.legend(fontsize=fs*0.8, loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX)
    else:
        # LEGEND_BBOX が None の場合はグラフ内に描画（loc='best'にすると自動で隙間を探します）
        ax.legend(fontsize=fs*0.8, loc='best')

axes[-1].set_xlabel(r"$\beta$", fontsize=fs)
axes[0].set_ylabel(r"$\chi$", fontsize=fs)

plt.show()