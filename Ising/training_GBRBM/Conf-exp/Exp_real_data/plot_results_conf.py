import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # 追加: 目盛りの設定用モジュール
import os

# ==========================================
# 設定
# ==========================================
#INPUT_FILE = "./diabetes/【alpha_2.0】.txt"
INPUT_FILE = "./wine/【alpha_2.0】.txt"
OUTPUT_IMAGE = "log_likelihood_plot.png"

# 描画するエポックの範囲を指定 (最初から最後まで描画する場合は None に設定してください)
START_EPOCH = None
END_EPOCH = None

# --- フォント・レイアウト設定 ---
TICK_FONT_SIZE = 25      # 軸の目盛りのフォントサイズ
LEGEND_FONT_SIZE = 25    # 凡例のフォントサイズ

# 凡例の配置位置を指定
# 'best': グラフの線と重ならない最適な位置を自動で探して配置します
# その他手動設定: 'upper right', 'lower right', 'upper left', 'lower left' など
LEGEND_LOC = 'best'

# 描画する対象の列名（グラフ）をリストで指定してください。
# すべての "Mean" 列を描画したい場合は None に設定してください。
PLOT_COLUMNS = ["beta_max/4","beta_max","4beta_max"] #α=0.5
  
def main():
    # 1. ファイルの存在確認
    if not os.path.exists(INPUT_FILE):
        print(f"エラー: データファイルが見つかりません: {INPUT_FILE}")
        return
        
    print(f"Reading data from {INPUT_FILE}...")

    # 2. ヘッダー行を読み込んで列構成を解析
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
    headers = header_line.split(',')

    if "Epoch" not in headers:
        print("エラー: 'Epoch' 列が見つかりません。ヘッダーを確認してください。")
        return
        
    epoch_idx = headers.index("Epoch")

    # 3. プロット対象の列インデックスと列名を抽出
    if PLOT_COLUMNS is None:
        mean_indices = [i for i, h in enumerate(headers) if "Mean" in h]
    else:
        mean_indices = []
        for col_name in PLOT_COLUMNS:
            if col_name in headers:
                mean_indices.append(headers.index(col_name))
            else:
                print(f"警告: 指定された列 '{col_name}' はデータ内に見つかりませんでした。")

    mean_columns = [headers[i] for i in mean_indices]

    if not mean_columns:
        print("エラー: 描画対象の列が一つも見つかりません。")
        return

    print(f"描画対象の列: {mean_columns}")

    # 4. 数値データの読み込み (ヘッダーをスキップ)
    data = np.loadtxt(INPUT_FILE, delimiter=',', skiprows=1, ndmin=2)
    epochs = data[:, epoch_idx]

    # --- 指定されたエポック範囲でデータをフィルタリング ---
    if START_EPOCH is not None:
        mask = epochs >= START_EPOCH
        data = data[mask]
        epochs = epochs[mask]
        
    if END_EPOCH is not None:
        mask = epochs <= END_EPOCH
        data = data[mask]
        epochs = epochs[mask]
    # --------------------------------------------------------------

    # 5. グラフの描画
    plt.figure(figsize=(10, 6))
    
    for i, (idx, col) in enumerate(zip(mean_indices, mean_columns)):
        # 該当列のデータを抽出
        mean_values = data[:, idx]
        
        # --- ラベルと色、線種の判定 ---
        if col in ["beta_max/4"]:
            label_text = r"$\beta_{\mathrm{max}}/4$"
            c = 'blue'
            ls = '--'  # 破線
        elif col in ["beta_max"]:
            label_text = r"$\beta_{\mathrm{max}}$"
            c = 'green'
            ls = '-'   # 実線 (目立たせる)
        elif col in ["4beta_max"]:
            label_text = r"$4\beta_{\mathrm{max}}$"
            c = 'red'
            ls = '--'  # 破線
        else:
            # 想定外の列名の場合はそのまま出力して黒の破線にする
            label_text = col
            c = 'black'
            ls = '--'
        
        # プロット
        plt.plot(epochs, mean_values, color=c, linestyle=ls, linewidth=2.5, label=label_text)

    # グラフの装飾
    # 追加: X軸の目盛りを強制的に整数（自然数）にする
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    
    plt.grid(True, linestyle='--')
    # 凡例の配置を LEGEND_LOC で指定
    plt.legend(loc=LEGEND_LOC, fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()

    # 6. 保存と表示
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"\nプロット完了。画像を保存しました: {os.path.abspath(OUTPUT_IMAGE)}")
    
    # グラフを画面にも表示
    plt.show()

if __name__ == "__main__":
    main()