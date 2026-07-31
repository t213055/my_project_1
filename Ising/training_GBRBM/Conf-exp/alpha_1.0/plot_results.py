import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 設定
# ==========================================
INPUT_FILE = "output.txt"
OUTPUT_IMAGE = "log_likelihood_plot.png"

# 描画するエポックの範囲を指定 (最初から最後まで描画する場合は None に設定してください)
START_EPOCH = None
END_EPOCH = None

# 描画する対象の列名（グラフ）をリストで指定してください。
# すべての "Mean" 列を描画したい場合は None に設定してください。
# 例: ["chi_Mean_0.25", "chi_Mean_1.0"]
PLOT_COLUMNS = ["chi_Mean_0.25", "chi_Mean_1.0", "chi_Mean_4.0"] #α=1.0

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
        # Noneの場合は、従来通り名前に "Mean" が含まれる列をすべて抽出
        mean_indices = [i for i, h in enumerate(headers) if "Mean" in h]
    else:
        # 指定された列名が存在するかチェックし、インデックスを取得
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
    # ndmin=2 を指定し、データが1行しかない場合でも2次元配列として扱う
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
    
    # 描画スタイルの設定（拡張性を持たせるため、リストをループして使用）
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # デフォルトのカラーパレット
    line_styles = ['-', '--', '-.', ':']
    
    for i, (idx, col) in enumerate(zip(mean_indices, mean_columns)):
        # 該当列のデータを抽出
        mean_values = data[:, idx]
        
        # 凡例ラベルの整形（例: "chi_Mean_0.25" -> "β = 0.25 * βmax"）
        beta_value = col.replace("chi_Mean_", "").replace("Mean_", "")
        label_text = f"β = {beta_value} * βmax"
        
        # 色と線種を順番に割り当てる
        c = colors[i % len(colors)]
        ls = line_styles[i % len(line_styles)]
        
        # プロット
        plt.plot(epochs, mean_values, color=c, linestyle=ls, linewidth=2, label=label_text)

    # グラフの装飾
    # タイトルに描画範囲を反映
    title_suffix = ""
    if START_EPOCH is not None and END_EPOCH is not None:
        title_suffix = f" (Epochs {START_EPOCH}-{END_EPOCH})"
    plt.title(f"Log-Likelihood Progress for Different β{title_suffix}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Log-Likelihood (Mean)")
    plt.grid(True, linestyle='--')
    plt.legend(loc='lower right')
    plt.tight_layout()

    # 6. 保存と表示
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"\nプロット完了。画像を保存しました: {os.path.abspath(OUTPUT_IMAGE)}")
    
    # グラフを画面にも表示
    plt.show()

if __name__ == "__main__":
    main()