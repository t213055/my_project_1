import numpy as np
import matplotlib.pyplot as plt
import glob

def plot_combined_results_epoch_100():
    # ------------------------------------------
    # 1. 描画用の設定
    # ------------------------------------------
    epochs = 2000
    # 元の記録タイミング
    full_x_axis = list(range(0, 101)) + list(range(110, epochs + 1, 10))
    
    # ★ 変更点: インデックス100（ちょうど100エポック目）以降のみを抽出
    start_idx = 115
    x_axis = full_x_axis[start_idx:]
    
    s_nh_list = [5, 10, 20]
    alpha_list = [0.5, 1.0, 2.0]
    b_ratios = [0.25, 1.0, 4.0]
    
    colors = {0.25: 'blue', 1.0: 'green', 4.0: 'red'}
    labels = {
        0.25: r'$\beta_{max} / 4$ (Low)', 
        1.0: r'$\beta_{max}$ (Optimal)', 
        4.0: r'$4 \times \beta_{max}$ (High)'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # タイトルも100エポック以降であることを明記
    fig.suptitle('Log-Likelihood Growth (Epoch 100-1000): Teacher-Student Strategy', fontsize=16)

    # ------------------------------------------
    # 2. データの読み込みと描画
    # ------------------------------------------
    for i, (s_nh, alpha) in enumerate(zip(s_nh_list, alpha_list)):
        ax = axes[i]
        
        ax.set_title(f'Student α = {alpha} (n_h = {s_nh})', fontsize=14)
        ax.set_xlabel('Epochs', fontsize=12)
        if i == 0:
            ax.set_ylabel('Log-Likelihood', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        for b_ratio in b_ratios:
            # resultsディレクトリ内のファイルを参照
            pattern = f"results/ll_snh{s_nh}_beta{b_ratio:.2f}_*.npy"
            files = glob.glob(pattern)
            
            if not files:
                print(f"Warning: Data not found for n_h={s_nh}, beta={b_ratio:.2f}. Skipping.")
                continue
            
            latest_file = sorted(files)[-1]
            data = np.load(latest_file)
            
            # 平均と標準偏差を計算後、100エポック以降（start_idx以降）をスライス
            mean_ll = np.mean(data, axis=0)[start_idx:]
            std_ll = np.std(data, axis=0)[start_idx:]
            
            ax.plot(x_axis, mean_ll, label=labels[b_ratio], color=colors[b_ratio], linewidth=2)
            ax.fill_between(x_axis, mean_ll - std_ll, mean_ll + std_ll, color=colors[b_ratio], alpha=0.2)

        ax.legend(loc='lower right', fontsize=11)

    # ------------------------------------------
    # 3. 画像の保存と表示
    # ------------------------------------------
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # 保存名も変更
    save_path = "results/combined_loglikelihood_plot_epoch_100_to_1000.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_combined_results_epoch_100()