#plot_chi_all_alpha.pyの出力を受け取り、各αでの層相関を計算するスクリプト
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 設定・デザイン (ここで編集可能)
# ==========================================
INPUT_FILE = "【LC_all-alpha_bias0.1】.txt"
OUTPUT_IMAGE = "LC_plot.png"

# プロット対象とする alpha のリスト
ALPHA_LIST = [0.5, 1.0, 2.0]

# 描画する β (beta) の範囲を指定 (制限しない場合は None に設定してください)
START_BETA = 0.8
END_BETA = None

# --- フォント・線の太さ デザイン設定 ---
LINE_WIDTH = 5           # グラフの線の太さ
FS = 30                  # 基本のフォントサイズ
FONT_SIZE_LABEL = FS     # 軸ラベルのフォントサイズ
FONT_SIZE_TICK = FS      # 軸のメモリのフォントサイズ
FONT_SIZE_LEGEND = FS    # 凡例のフォントサイズ
FONT_SIZE_TEXT = FS      # 最大値テキストのフォントサイズ

# ★ Y軸の目盛りを表示するかどうか (True: 表示, False: 非表示)
SHOW_Y_TICKS = False

# 凡例の配置位置 (例: 'upper right', 'lower right', 'upper left', 'best' など)
LEGEND_LOC = 'upper left'

# --- ピーク値テキストの配置調整 ---
# 各αごとのテキスト位置のズレを個別に設定できます。文字被りを防ぐのに便利です。
# 形式: { alpha: (X軸方向のズレ, Y軸方向のズレ(倍率)) }
# ※ Y軸は対数スケールのため、ズレは「倍率」で指定します（例: 0.5ならピークの半分の高さ、2.0なら2倍の高さ）。
TEXT_OFFSETS = {
    0.5: (0.01, 1),  # α=0.5 のテキスト位置 (Xを+0.01、Yをピークの0.5倍の位置に)
    1.0: (0.01, 1),  # α=1.0 のテキスト位置
    2.0: (0.01, 1)   # α=2.0 のテキスト位置
}

def main():
    # 1. ファイルの存在確認
    if not os.path.exists(INPUT_FILE):
        print(f"エラー: データファイルが見つかりません: {INPUT_FILE}")
        return
        
    print(f"Reading data from {INPUT_FILE}...")

    # 2. データの読み込み (ヘッダー行をスキップ)
    # alpha, beta, q_v, q_h, q_v_hat, q_h_hat, chi_vh (列インデックス: 0, 1, 2, 3, 4, 5, 6)
    data = np.loadtxt(INPUT_FILE, delimiter=',', skiprows=1)
    
    # 列データの抽出
    alphas = data[:, 0]
    betas = data[:, 1]
    chis = data[:, 6]

    # 3. グラフの準備
    plt.figure(figsize=(9, 6))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    line_styles = ['-', '-', '-']
    
    # 描画された全βの最小値・最大値を記録するための変数（X軸目盛り計算用）
    plotted_min_beta = float('inf')
    plotted_max_beta = float('-inf')
    
    # 4. 指定された alpha ごとにデータを抽出してプロット
    for i, alpha_target in enumerate(ALPHA_LIST):
        # 該当する alpha の行だけをマスク（抽出）
        mask = (alphas == alpha_target)
        
        # --- 指定されたβの範囲でデータをフィルタリング ---
        if START_BETA is not None:
            mask &= (betas >= START_BETA)
        if END_BETA is not None:
            mask &= (betas <= END_BETA)
        # --------------------------------------------------------
            
        if not np.any(mask):
            print(f"警告: α={alpha_target} (指定されたβの範囲内) のデータは見つかりませんでした。")
            continue
            
        b_vals = betas[mask]
        chi_vals = chis[mask]
        
        # 描画範囲の更新
        plotted_min_beta = min(plotted_min_beta, b_vals.min())
        plotted_max_beta = max(plotted_max_beta, b_vals.max())
        
        # グラフのプロット (LINE_WIDTH を適用)
        c = colors[i % len(colors)]
        ls = line_styles[i % len(line_styles)]
        label_text = f"α = {alpha_target}"
        plt.plot(b_vals, chi_vals, color=c, linestyle=ls, linewidth=LINE_WIDTH, label=label_text)
        
        # ----------------------------------------------------
        # 表示範囲内での最大値（ピーク）の特定とテキスト表示
        # ----------------------------------------------------
        max_idx = np.argmax(chi_vals)
        max_beta = b_vals[max_idx]
        max_chi = chi_vals[max_idx]
        
        # グラフ内にテキストを配置
        text_str = f"{max_beta:.3f}"
        
        # 点を描画して強調
        plt.plot(max_beta, max_chi, marker='o', markersize=6, color=c)
        
        # オフセットを TEXT_OFFSETS 辞書から取得（未設定の場合はデフォルト値を適用）
        x_offset, y_ratio = TEXT_OFFSETS.get(alpha_target, (0.01, 0.8))
        
        # テキストを配置
        plt.text(max_beta + x_offset, max_chi * y_ratio, text_str, 
                 fontsize=FONT_SIZE_TEXT, color=c, 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # 5. グラフの装飾 (タイトル削除済み)
    # ★ 軸ラベル (X軸の"β", Y軸の"χ") を描画しないようにコメントアウト
    # plt.xlabel("β", fontsize=FONT_SIZE_LABEL)
    # plt.ylabel("χ", fontsize=FONT_SIZE_LABEL)
    
    # --- X軸の目盛りを 0.10 の自然数倍のみに設定 ---
    if plotted_min_beta != float('inf') and plotted_max_beta != float('-inf'):
        # 自然数倍（0.1, 0.2, 0.3...）とするため、インデックスは1から開始するように制御
        min_idx = max(1, int(np.ceil(plotted_min_beta / 0.1 - 1e-9)))
        max_idx = int(np.floor(plotted_max_beta / 0.1 + 1e-9))
        
        # 丸め誤差を防ぐために round で小数を処理
        x_ticks = [round(idx * 0.1, 1) for idx in range(min_idx, max_idx + 1)]
        plt.xticks(x_ticks, fontsize=FONT_SIZE_TICK)
    
    # y軸は非常に値が小さく、また大きくなるため対数スケールに設定
    plt.yscale('log')
    
    # --- ★ Y軸の目盛り（数値テキスト）を完全に消去 ---
    if SHOW_Y_TICKS:
        plt.yticks(fontsize=FONT_SIZE_TICK)
    else:
        # 主目盛りと副目盛りの両方のフォーマッタを空に設定することで、
        # 対数スケール特有の「3 x 10^-2」のような文字も強制的に非表示にする
        plt.gca().yaxis.set_major_formatter(plt.NullFormatter())
        plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc=LEGEND_LOC, fontsize=FONT_SIZE_LEGEND)
    plt.tight_layout()

    # 6. 保存と表示
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"\nプロット完了。画像を保存しました: {os.path.abspath(OUTPUT_IMAGE)}")
    
    # グラフを画面にも表示
    plt.show()

if __name__ == "__main__":
    main()