import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# =============================
# ユーザ設定パラメータ
# =============================
# 1. 読み込むファイルのリスト
FILE_PATHS = [
    #"LC_alpha=1.0/beta_increase_stepsize0.001_eps=0.001_α=1.0_c=-2_beta25.0.txt",
    #"LC_alpha=1.0/beta_increase_stepsize0.001_eps=0.001_α=1.0_c=-5_beta25.0.txt",
    "LC_alpha=1.0/beta_increase_stepsize0.001_eps=0.001_α=1.0_c=0.001_beta25.0.txt"
]

# 2. フォントサイズ（可変）
FONT_SIZE = 20

# 3. 凡例の配置設定（可変）
# loc: 凡例ボックスの基準となる角（'upper left', 'best', 'center right' など）
# bbox: グラフ枠に対する相対座標 (x, y)。枠外に出す場合は (1.02, 1.0) のように指定。
#       枠内に収める場合は None に設定。
LEGEND_LOC = 'lower right'
LEGEND_BBOX = None#(1.02, 1.0)  

# 4. 読み込むβの範囲 (デフォルト: [0.0, 4.0])
# ※この範囲外のデータは無視され、最大値・極小値の計算にも含まれません
BETA_RANGE = [0.0, 4.0]

# 5. 極小点の表示を切り替える設定（cの値をキーにして True/False を指定）
# False を指定した c については、下落後の極小点のプロットを行いません。
# 辞書に記載のない c の値はデフォルトで True 扱いになります。
SHOW_MINIMA_DICT = {
    -2: True,
    -5: True,
    0.001: False
}

# 6. ピーク時テキストのY方向オフセット（文字の重なり回避用）
# cの値をキーにして、テキストを点から上方向にどれくらいずらすかの倍率を指定します。
# デフォルトは 1.2。重なる場合は片方を 1.6 や 2.0 などに変更して調整してください。
PEAK_TEXT_OFFSET_DICT = {
    #-2: 3.6,
    #-5: 1.6,     # -2 の値と重なる場合は、こちらを上にずらして回避
    0.001: 0.1
}

# 7. 画像の保存先
SAVE_DIR = "LC_alpha=1.0"
SAVE_FILENAME = "combined_plot_conference.png"

# =============================
# データ読み込みと統合
# =============================
data_dict = defaultdict(lambda: defaultdict(list))

for file_path in FILE_PATHS:
    try:
        data = np.loadtxt(file_path, delimiter=",", comments="#")
        
        # データが1行しかない場合のエラー回避
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        alphas = data[:, 0]
        cs     = data[:, 1]
        betas  = data[:, 2]
        chis   = data[:, 3]

        # -----------------------------
        # 指定したBETA_RANGEでデータをフィルタリング
        # -----------------------------
        if BETA_RANGE is not None:
            mask = (betas >= BETA_RANGE[0]) & (betas <= BETA_RANGE[1])
            alphas = alphas[mask]
            cs     = cs[mask]
            betas  = betas[mask]
            chis   = chis[mask]

        for a, c, b, chi in zip(alphas, cs, betas, chis):
            data_dict[a][c].append((b, chi))
            
    except Exception as e:
        print(f"警告: ファイルの読み込みに失敗しました ({file_path})\n詳細: {e}")

# =============================
# 描画（縦並び）
# =============================
alpha_keys = sorted(data_dict.keys())
n_alpha = len(alpha_keys)

if n_alpha == 0:
    print("有効なデータが読み込めませんでした。プログラムを終了します。")
    exit()

fig, axes = plt.subplots(
    n_alpha, 1,
    figsize=(8, 5 * n_alpha),
    sharex=True,
    sharey=True,
    layout="constrained" # 枠外の凡例が見切れないように自動調整
)

if n_alpha == 1:
    axes = [axes]

for ax, alpha in zip(axes, alpha_keys):
    c_dict = data_dict[alpha]
    
    # cの昇順などでプロットしたい場合は sorted(c_dict.items()) を使用
    for c, values in sorted(c_dict.items()):
        values = np.array(values)
        
        # フィルタリングの結果、データが空になってしまった場合はスキップ
        if len(values) == 0:
            continue
            
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
        if len(chi_vals) > 0:
            imax = np.argmax(chi_vals)
            b_peak = b_vals[imax]
            chi_peak = chi_vals[imax]
            
            # 設定したオフセット倍率を取得
            offset = PEAK_TEXT_OFFSET_DICT.get(c, 1.2)

            ax.plot(b_peak, chi_peak, "o", color=color, markersize=8)
            ax.text(
                b_peak, chi_peak * offset, 
                rf"{b_peak:.7g}", color=color, ha="center", va="bottom", 
                fontsize=FONT_SIZE*1.2, weight='bold'
            )

            # --- χ 下落後の極小点 (Drop Bottom) ---
            # ユーザ設定で極小点表示が True の場合のみ実行
            show_minima = SHOW_MINIMA_DICT.get(c, True)
            
            if show_minima and (imax < len(chi_vals) - 1):
                imin_relative = np.argmin(chi_vals[imax:])
                imin = imax + imin_relative
                
                b_min = b_vals[imin]
                chi_min = chi_vals[imin]

                ax.plot(b_min, chi_min, "o", color=color, markersize=8, markerfacecolor='white', markeredgewidth=2)
                
                # 不要な縦線は削除されました
                
                ax.text(
                    b_min, chi_min * 0.7, 
                    rf"{b_min:.7g}", color=color, ha="center", va="top", fontsize=FONT_SIZE*0.6
                )

    ax.set_yscale("log")
    #ax.set_title(rf"$\alpha={alpha:g}$", fontsize=FONT_SIZE)
    #ax.set_title(rf"$\alpha={alpha:g}$での層相関", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.grid(True, which="both", linestyle="--", alpha=1.0)
    
    # 軸の表示範囲を設定
    if BETA_RANGE is not None:
        ax.set_xlim(BETA_RANGE)
    
    # --- 凡例の描画 ---
    """
    if LEGEND_BBOX is not None:
        ax.legend(fontsize=FONT_SIZE*0.8, loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX)
    else:
        ax.legend(fontsize=FONT_SIZE*0.8, loc=LEGEND_LOC)
    """

axes[-1].set_xlabel(r"$\beta$", fontsize=FONT_SIZE)
axes[0].set_ylabel(r"$\chi$", fontsize=FONT_SIZE)

# =============================
# 保存と表示
# =============================
# フォルダが存在しない場合は作成
os.makedirs(SAVE_DIR, exist_ok=True)

save_path = os.path.join(SAVE_DIR, SAVE_FILENAME)
plt.savefig(save_path, dpi=300) # dpi=300で高画質保存
print(f"画像を保存しました: {save_path}")

plt.show()