import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def plot_combined_results(k=5):
    """
    k: 表示する最大エポック数 (デフォルトは5)
    """
    # ------------------------------------------
    # 1. 描画用の設定
    # ------------------------------------------
    epochs = 5000
    # 元の全エポック軸を作成
    full_x_axis = np.array(list(range(0, 101)) + list(range(110, epochs + 1, 10)))
    
    # ★ 修正ポイント: 0エポック目からkエポック目までのインデックスを探す
    # full_x_axis の中で値が k 以下のものだけを抽出するマスクを作成
    mask = full_x_axis <= k
    x_axis = full_x_axis[mask]
    
    # グラフの構成設定
    s_nh_list = [5, 10, 20]
    alpha_list = [0.5, 1.0, 2.0]
    b_ratios = [0.25, 1.0, 4.0]
    
    colors = {0.25: 'blue', 1.0: 'green', 4.0: 'red'}
    labels = {
        0.25: r'$\beta_{max} / 4$', 
        1.0: r'$\beta_{max}$ (Optimal)', 
        4.0: r'$4 \times \beta_{max}$'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Early Log-Likelihood Growth (Epoch 0 to {k})', fontsize=16)

    # ------------------------------------------
    # 2. データの読み込みと描画
    # ------------------------------------------
    for i, (s_nh, alpha) in enumerate(zip(s_nh_list, alpha_list)):
        ax = axes[i]
        
        ax.set_title(r'Student $\alpha = {}$ ($n_h = {}$)'.format(alpha, s_nh), fontsize=14)
        ax.set_xlabel('Epochs', fontsize=12)
        if i == 0:
            ax.set_ylabel('Log-Likelihood', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        for b_ratio in b_ratios:
            pattern = f"results/ll_snh{s_nh}_beta{b_ratio:.2f}_*.npy"
            files = glob.glob(pattern)
            
            if not files:
                print(f"Warning: Data not found for n_h={s_nh}, beta={b_ratio:.2f}. Skipping.")
                continue
            
            latest_file = sorted(files)[-1]
            data = np.load(latest_file)
            
            # 平均と標準偏差を計算
            mean_ll_full = np.mean(data, axis=0)
            std_ll_full = np.std(data, axis=0)
            
            # ★ 修正ポイント: マスクを適用して序盤のデータだけ取り出す
            mean_ll = mean_ll_full[mask]
            std_ll = std_ll_full[mask]
            
            # プロット
            ax.plot(x_axis, mean_ll, label=labels[b_ratio], color=colors[b_ratio], linewidth=2, marker='o', markersize=4)
            ax.fill_between(x_axis, mean_ll - std_ll, mean_ll + std_ll, color=colors[b_ratio], alpha=0.2)

        # 序盤は変化が激しいため、凡例が線を隠さないように配置
        ax.legend(loc='lower right', fontsize=10)

    # ------------------------------------------
    # 3. 画像の保存と表示
    # ------------------------------------------
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # 保存名に k の値を含めるように修正
    save_path = f"../results/early_growth_k{k}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Early growth plot (k={k}) saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # ここで k の値を自由に変えられます
    plot_combined_results(k=10)