import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # 目盛りの設定用モジュール
import os

# ==========================================
# 設定: 出力する画像のエポック範囲とファイル名
# ==========================================
# ここでそれぞれの画像のエポック数を調整できます。
# 最初から描画する場合は "start" を None に設定してください。
PLOT_CONFIGS = [
    {"start": None, "end": 500, "filename": "log_likelihood_plot_0_500.png"},
    {"start": None, "end": 10,  "filename": "log_likelihood_plot_0_10.png"},
    {"start": 350,  "end": 500, "filename": "log_likelihood_plot_350_500.png"},
]

INPUT_FILE = "Exp-results.txt" 

# --- フォント・レイアウト設定 ---
TICK_FONT_SIZE = 30      # 軸の目盛りのフォントサイズ
LEGEND_FONT_SIZE = 30    # 凡例のフォントサイズ

# Y軸の目盛りを表示するかどうか (True: 表示, False: 非表示)
SHOW_Y_TICKS = False

# 凡例の配置位置を指定
# 'best': グラフの線と重ならない最適な位置を自動で探して配置します
LEGEND_LOC = 'best' 

# 描画する対象の列名（グラフ）をリストで指定してください。
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
    full_data = np.loadtxt(INPUT_FILE, delimiter=',', skiprows=1, ndmin=2) 
    full_epochs = full_data[:, epoch_idx] 

    # ==========================================
    # PLOT_CONFIGS の設定に沿って画像を連続で生成
    # ==========================================
    for config in PLOT_CONFIGS:
        start_epoch = config["start"]
        end_epoch = config["end"]
        out_filename = config["filename"]
        
        print(f"\n--- 生成中: {out_filename} (範囲: {start_epoch}〜{end_epoch}) ---")

        # --- 指定されたエポック範囲でデータをフィルタリング ---
        mask = np.ones(len(full_epochs), dtype=bool)
        if start_epoch is not None: 
            mask &= (full_epochs >= start_epoch)
        if end_epoch is not None: 
            mask &= (full_epochs <= end_epoch)
            
        data = full_data[mask] 
        epochs = full_epochs[mask] 

        if len(epochs) == 0:
            print("警告: 該当するエポック範囲にデータが存在しません。スキップします。")
            continue

        # 5. グラフの描画
        plt.figure(figsize=(8, 6)) 
        
        for i, (idx, col) in enumerate(zip(mean_indices, mean_columns)): 
            mean_values = data[:, idx] 
            
            # --- ラベルと色、線種、重なり順(zorder)の判定 ---
            if col in ["chi_Mean_0.235", "chi_Mean_0.25", "chi_Mean_0.2775"]: 
                label_text = r"$\beta_{\mathrm{max}}/4$" 
                c = 'blue' 
                ls = ':'  # 破線
                z_val = 3 # 最前面
            elif col in ["chi_Mean_0.94", "chi_Mean_1.0", "chi_Mean_1.11"]: 
                label_text = r"$\beta_{\mathrm{max}}$" 
                c = 'green' 
                ls = '-'   # 実線
                z_val = 2 # 中間
            elif col in ["chi_Mean_3.76", "chi_Mean_4.0", "chi_Mean_4.44"]: 
                label_text = r"$4\beta_{\mathrm{max}}$" 
                c = 'red' 
                ls = '--'  # 破線
                z_val = 1 # 最背面
            else: 
                label_text = col 
                c = 'black' 
                ls = '--' 
                z_val = 0
            
            # プロット
            plt.plot(epochs, mean_values, color=c, linestyle=ls, linewidth=5, label=label_text, zorder=z_val) 

        # ==========================================
        # グラフの装飾 (目盛りのカスタム設定)
        # ==========================================
        # --- Y軸: 最大5個の目盛りに制限 ---
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=4)) 
        
        # --- X軸: 10の倍数で最大6個、開始・終了を必ず含む ---
        actual_start = int(epochs.min()) 
        actual_end = int(epochs.max()) 
        
        # 確実な10の倍数に丸める
        start_tick = (actual_start // 10) * 10 
        end_tick = int(np.ceil(actual_end / 10.0)) * 10 
        span = end_tick - start_tick 
        
        # 最大6個(5区間)の目盛りを作成するため、10の倍数のステップ幅を計算
        if span > 0:
            step = int(np.ceil((span / 5.0) / 10.0) * 10) 
            if step == 0: 
                step = 10 
        else:
            step = 10
            
        x_ticks = list(range(start_tick, end_tick, step)) 
        
        # 終了エポックを必ず含める
        if not x_ticks or x_ticks[-1] != end_tick: 
            x_ticks.append(end_tick) 
            
        # もし目盛りが6個を超えてしまった場合の保険処理
        while len(x_ticks) > 6: 
            step += 10 
            x_ticks = list(range(start_tick, end_tick, step)) 
            if not x_ticks or x_ticks[-1] != end_tick: 
                x_ticks.append(end_tick) 

        # 計算したリストをX軸の目盛りに強制設定
        plt.xticks(x_ticks, fontsize=TICK_FONT_SIZE) 
        # グラフの左右の余白を消してスッキリさせる
        plt.xlim(start_tick, end_tick)  
        
        # Y軸の表示/非表示の切り替え
        if SHOW_Y_TICKS:
            plt.yticks(fontsize=TICK_FONT_SIZE)
        else:
            plt.yticks([])
            
        plt.legend(loc=LEGEND_LOC, fontsize=LEGEND_FONT_SIZE) 
        plt.tight_layout() 

        # 6. 保存と表示
        plt.savefig(out_filename, dpi=150) 
        print(f"画像を保存しました: {os.path.abspath(out_filename)}") 
        
        # 連続出力のため画面表示をスキップし、描画をクリアする
        plt.close()

if __name__ == "__main__": 
    main()