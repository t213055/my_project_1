import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 設定・デザイン (ここで編集可能)
# ==========================================
INPUT_FILE = "Ising_LC_output.txt"
OUTPUT_IMAGE = "LC_plot.png"

# プロット対象とする alpha のリスト
ALPHA_LIST = [0.5, 1.0, 2.0]

# 描画する β (beta) の範囲を指定 (制限しない場合は None に設定してください)
START_BETA = 0.8
END_BETA = None

# --- フォント・デザイン設定 ---
FONT_SIZE_LABEL = 20     # 軸ラベルのフォントサイズ
FONT_SIZE_TICK = 20      # 軸のメモリのフォントサイズ
FONT_SIZE_LEGEND = 20    # 凡例のフォントサイズ
FONT_SIZE_TEXT = 20      # 最大値テキストのフォントサイズ

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
    plt.figure(figsize=(10, 6))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    line_styles = ['-', '-', '-']
    
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
        
        # グラフのプロット
        c = colors[i % len(colors)]
        ls = line_styles[i % len(line_styles)]
        label_text = f"α = {alpha_target}"
        plt.plot(b_vals, chi_vals, color=c, linestyle=ls, linewidth=2, label=label_text)
        
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
    plt.xlabel("β", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("χ", fontsize=FONT_SIZE_LABEL)
    
    plt.xticks(fontsize=FONT_SIZE_TICK)
    # y軸は非常に値が小さく、また大きくなるため対数スケールに設定
    plt.yscale('log')
    plt.yticks(fontsize=FONT_SIZE_TICK)
    
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