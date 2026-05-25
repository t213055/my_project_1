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

# 2. ★表示範囲の指定設定 (3パターン対応)
X_RANGE_MODE = 'all' #'all', 'from_start', 'to_end'の中から選択 
X_START = 9000
X_END = 200
Y_MARGIN = 0.05

# 3. パス設定 (スクリプトの配置場所から見た相対パス)
DATA_DIR = "../results"
SAVE_DIR = "plots"

# 4. ★デザイン・配色設定 (実線のまま濃淡で区別)
# max は濃い色(dark)、min は薄い/明るい色(light)に設定
COLORS = {
    0.25: {'max': 'mediumblue',  'min': 'cornflowerblue'}, # 青系
    1.0:  {'max': 'forestgreen', 'min': 'limegreen'},      # 緑系
    4.0:  {'max': 'firebrick',   'min': 'lightcoral'},     # 赤系
    8.0:  {'max': 'indigo',      'min': 'mediumorchid'}    # 紫系
}

# 5. サイズ変更しやすいフォント設定
FS = {
    'suptitle': 16,    
    'title': 14,       
    'label': 12,       
    'tick': 10,        
    'legend': 10       
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
    elif X_RANGE_MODE == 'all':
        pass 

def get_y_min_max_in_visible_range(x, mean, std=None):
    if X_RANGE_MODE == 'to_end':
        mask = x <= X_END
    elif X_RANGE_MODE == 'from_start':
        mask = x >= X_START
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
# グラフ①: 各 ratio ごとの max vs min 比較 (2x2)
# ==========================================
def plot_max_vs_min_per_ratio(data_dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Student $n_h={S_NH}$ : $\\beta_{{max}}$ vs $\\beta_{{min}}$ Comparison', fontsize=FS['suptitle'])
    axes = axes.flatten()

    for idx, b_ratio in enumerate(B_RATIOS):
        ax = axes[idx]
        ax.set_title(f'Beta Ratio = {b_ratio}', fontsize=FS['title'])
        ax.set_xlabel('Epochs', fontsize=FS['label'])
        ax.set_ylabel('Log-Likelihood', fontsize=FS['label'])
        ax.tick_params(labelsize=FS['tick'])
        ax.grid(True, linestyle='--', alpha=0.7)

        ax_y_min, ax_y_max = np.inf, -np.inf

        # maxのプロット (実線・濃い色)
        if ("max", b_ratio) in data_dict:
            x, mean, std = data_dict[("max", b_ratio)]
            c_max = COLORS[b_ratio]['max']
            ax.plot(x, mean, label=r'$\beta_{max}$', color=c_max, linestyle='-', linewidth=2)
            ax.fill_between(x, mean - std, mean + std, color=c_max, alpha=0.15)
            
            y_min, y_max = get_y_min_max_in_visible_range(x, mean, std)
            ax_y_min, ax_y_max = min(ax_y_min, y_min), max(ax_y_max, y_max)

        # minのプロット (実線・薄い色)
        if ("min", b_ratio) in data_dict:
            x, mean, std = data_dict[("min", b_ratio)]
            c_min = COLORS[b_ratio]['min']
            ax.plot(x, mean, label=r'$\beta_{min}$', color=c_min, linestyle='-', linewidth=2)
            ax.fill_between(x, mean - std, mean + std, color=c_min, alpha=0.15)
            
            y_min, y_max = get_y_min_max_in_visible_range(x, mean, std)
            ax_y_min, ax_y_max = min(ax_y_min, y_min), max(ax_y_max, y_max)

        apply_x_limits(ax)
        apply_auto_y_limits(ax, ax_y_min, ax_y_max)
        ax.legend(fontsize=FS['legend'], loc='lower right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(os.path.join(SAVE_DIR, "1_max_vs_min_per_ratio.png"), dpi=300)
    plt.show()

# ==========================================
# グラフ②: 各 Type 内での ratio 比較 (1x2)
# ==========================================
def plot_ratios_per_type(data_dict):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Student $n_h={S_NH}$ : Beta Ratios Comparison', fontsize=FS['suptitle'])

    for idx, b_type in enumerate(B_TYPES):
        ax = axes[idx]
        type_title = r'$\beta_{max}$ Strategy' if b_type == "max" else r'$\beta_{min}$ Strategy'
        ax.set_title(type_title, fontsize=FS['title'])
        ax.set_xlabel('Epochs', fontsize=FS['label'])
        ax.set_ylabel('Log-Likelihood', fontsize=FS['label'])
        ax.tick_params(labelsize=FS['tick'])
        ax.grid(True, linestyle='--', alpha=0.7)

        ax_y_min, ax_y_max = np.inf, -np.inf

        for b_ratio in B_RATIOS:
            if (b_type, b_ratio) in data_dict:
                x, mean, std = data_dict[(b_type, b_ratio)]
                c_val = COLORS[b_ratio][b_type] # maxなら濃い色、minなら薄い色を使用
                ax.plot(x, mean, label=f'Ratio: {b_ratio}', color=c_val, linewidth=2)
                ax.fill_between(x, mean - std, mean + std, color=c_val, alpha=0.15)
                
                y_min, y_max = get_y_min_max_in_visible_range(x, mean, std)
                ax_y_min, ax_y_max = min(ax_y_min, y_min), max(ax_y_max, y_max)

        apply_x_limits(ax)
        apply_auto_y_limits(ax, ax_y_min, ax_y_max)
        ax.legend(fontsize=FS['legend'], loc='lower right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(SAVE_DIR, "2_ratios_per_type.png"), dpi=300)
    plt.show()

# ==========================================
# グラフ③: 全8通り同時比較 (1x1)
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
                x, mean, _ = data_dict[(b_type, b_ratio)]
                
                # すべて実線にし、色（濃淡）で区別する
                c_val = COLORS[b_ratio][b_type]
                type_str = r'$\beta_{max}$' if b_type == "max" else r'$\beta_{min}$'
                label = f'{type_str}, Ratio: {b_ratio}'
                
                plt.plot(x, mean, label=label, color=c_val, linestyle='-', linewidth=2)
                
                y_min, y_max = get_y_min_max_in_visible_range(x, mean, std=None)
                ax_y_min, ax_y_max = min(ax_y_min, y_min), max(ax_y_max, y_max)

    apply_x_limits(ax)
    apply_auto_y_limits(ax, ax_y_min, ax_y_max)
    
    # 8本あるので、凡例を外側に出すか2列にして見やすくする
    plt.legend(fontsize=FS['legend'], loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "3_all_8_combined.png"), dpi=300)
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
        
        print("\n[1/3] Generating Max vs Min per Ratio plot...")
        plot_max_vs_min_per_ratio(loaded_data)
        
        print("\n[2/3] Generating Ratios per Type plot...")
        plot_ratios_per_type(loaded_data)
        
        print("\n[3/3] Generating All-in-one combined plot...")
        plot_all_8_combined(loaded_data)
        
        print(f"\nAll plots have been successfully saved inside the '{SAVE_DIR}' directory!")