import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ==========================================
# 拡張設定（ここを変更するだけで自動追従します）
# ==========================================
# 1. 実験条件の設定
S_NH = 10                  # 描画対象の生徒の隠れ層数 (s_nh)
B_TYPES = ["max", "min"]   # 比較するβのタイプ
B_RATIOS = [0.25, 1.0, 4.0, 8.0] # 比較するリレーション倍率

# 2. 表示範囲の指定設定 (4パターン対応)
X_RANGE_MODE = 'range'     # 'all', 'from_start', 'to_end', 'range' の中から選択 
X_START = 3500              # from_start または range モードの開始エポック
X_END = 10000               # to_end または range モードの終了エポック
Y_MARGIN = 0.05

# 3. 凡例の配置設定 (可変)
# loc: 凡例ボックスの基準位置 ('upper left', 'center right', 'best' など)
# bbox: グラフ枠に対する相対座標 (x, y)。枠外に出す場合は (1.02, 0.5) のように指定。
#       グラフ枠内に収める場合は None に設定。
LEGEND_LOC = 'lower right'
LEGEND_BBOX = None#(1.02, 0.5)  

# 4. 標準偏差（誤差のシャドウ）の表示切り替え
SHOW_STD = False           # True: 表示する, False: 表示しない

# 5. パス設定 (スクリプトの配置場所から見た相対パス)
DATA_DIR = "../results"
SAVE_DIR = "plots"

# 6. デザイン・配色設定 (実線のまま濃淡で区別)
COLORS = {
    0.25: {'max': 'mediumblue',  'min': 'cornflowerblue'}, # 青系
    1.0:  {'max': 'forestgreen', 'min': 'limegreen'},      # 緑系
    4.0:  {'max': 'firebrick',   'min': 'lightcoral'},     # 赤系
    8.0:  {'max': 'indigo',      'min': 'mediumorchid'}    # 紫系
}

# 7. サイズ変更しやすいフォント設定
FS = {
    'suptitle': 20,    
    'title': 20,       
    'label': 20,       
    'tick': 20,        
    'legend': 20       
}

# ==========================================
# 横軸（エポック数）の自動生成ロジック
# ==========================================
def get_x_axis_from_data_len(n_points):
    if n_points <= 101:
        return np.arange(0, n_points)
    else:
        x_base = list(range(0, 101))
        x_rest = [110 + i * 10 for i in range(n_points - 101)]
        return np.array(x_base + x_rest)

# ==========================================
# 表示範囲・メモリ自動調整のユーティリティ関数
# ==========================================
def apply_x_limits(ax):
    if X_RANGE_MODE == 'to_end':
        ax.set_xlim(0, X_END)
    elif X_RANGE_MODE == 'from_start':
        ax.set_xlim(left=X_START)
    elif X_RANGE_MODE == 'range':
        ax.set_xlim(X_START, X_END)
    elif X_RANGE_MODE == 'all':
        pass 

def get_y_min_max_in_visible_range(x, mean, std=None):
    if X_RANGE_MODE == 'to_end':
        mask = x <= X_END
    elif X_RANGE_MODE == 'from_start':
        mask = x >= X_START
    elif X_RANGE_MODE == 'range':
        mask = (x >= X_START) & (x <= X_END)
    else:
        mask = np.ones_like(x, dtype=bool)
        
    if not np.any(mask):
        return np.inf, -np.inf
        
    valid_mean = mean[mask]
    if std is not None:
        valid_std = std[mask]
        return np.min(valid_mean - valid_std), np.max(valid_mean + valid_std)
    else:
        return np.min(valid_mean), np.max(valid_mean)

def apply_auto_y_limits(ax, y_min, y_max):
    if y_min != np.inf and y_max != -np.inf:
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0
        ax.set_ylim(y_min - y_range * Y_MARGIN, y_max + y_range * Y_MARGIN)

# ==========================================
# データのデータ一括読み込み関数
# ==========================================
def load_all_results():
    data_dict = {}
    for b_type in B_TYPES:
        for b_ratio in B_RATIOS:
            pattern = os.path.join(DATA_DIR, f"ll_snh{S_NH}_{b_type}_ratio{b_ratio:.3f}_*.npy")
            files = glob.glob(pattern)
            
            if not files:
                print(f"Warning: Data not found for s_nh={S_NH}, {b_type}, ratio={b_ratio:.3f}. Skipping.")
                continue
            
            latest_file = sorted(files)[-1]
            data = np.load(latest_file)
            
            n_points = data.shape[1]
            x_axis = get_x_axis_from_data_len(n_points)
            mean_ll = np.mean(data, axis=0)
            std_ll = np.std(data, axis=0)
            data_dict[(b_type, b_ratio)] = (x_axis, mean_ll, std_ll)
            
    return data_dict

# ==========================================
# グラフ: 全8通り同時比較 (1x1)
# ==========================================
def plot_all_8_combined(data_dict):
    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    plt.title(f'All 8 Experimental Configurations Combined ($n_h={S_NH}$)', fontsize=FS['suptitle'])
    plt.xlabel('Epochs', fontsize=FS['label'])
    plt.ylabel('Log-Likelihood', fontsize=FS['label'])
    plt.tick_params(labelsize=FS['tick'])
    plt.grid(True, linestyle='--', alpha=0.7)

    ax_y_min, ax_y_max = np.inf, -np.inf

    for b_type in B_TYPES:
        for b_ratio in B_RATIOS:
            if (b_type, b_ratio) in data_dict:
                x, mean, std = data_dict[(b_type, b_ratio)]
                c_val = COLORS[b_ratio][b_type]
                
                # 凡例ラベルの書式変更 (例: "0.25 βmax")
                # グラフ内のフォントと揃えるため、数学フォント(LaTeX)を使用しています
                label = f'{b_ratio} $\\beta_{{{b_type}}}$'
                
                # 通常プロット
                plt.plot(x, mean, label=label, color=c_val, linestyle='-', linewidth=2)
                
                # 標準偏差（シャドウ）の表示切り替え
                if SHOW_STD:
                    plt.fill_between(x, mean - std, mean + std, color=c_val, alpha=0.15)
                
                # 表示中の可視範囲に応じたY軸の自動スケーリング計算
                y_min, y_max = get_y_min_max_in_visible_range(x, mean, std if SHOW_STD else None)
                ax_y_min, ax_y_max = min(ax_y_min, y_min), max(ax_y_max, y_max)

    apply_x_limits(ax)
    apply_auto_y_limits(ax, ax_y_min, ax_y_max)
    
    # 可変対応した凡例の描画処理
    if LEGEND_BBOX is not None:
        plt.legend(fontsize=FS['legend'], loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX)
    else:
        plt.legend(fontsize=FS['legend'], loc=LEGEND_LOC)
    
    plt.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR, "all_8_combined.png"), dpi=300)
    plt.show()

# ==========================================
# メイン実行処理
# ==========================================
if __name__ == "__main__":
    print(f"Loading data from '{DATA_DIR}'...")
    loaded_data = load_all_results()
    
    if not loaded_data:
        print("Error: No data was loaded. Please check your DATA_DIR path and S_NH setting.")
    else:
        print(f"Successfully loaded {len(loaded_data)} configurations.")
        print(f"Current Range Mode: {X_RANGE_MODE}")
        
        print("\nGenerating All-in-one combined plot...")
        plot_all_8_combined(loaded_data)
        
        print(f"\nPlot has been successfully saved inside the '{SAVE_DIR}' directory!")