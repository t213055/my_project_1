import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def plot_combined_results():
    # ------------------------------------------
    # 1. 描画用の設定
    # ------------------------------------------
    epochs = 2000
    # X軸のスケールを前回の学習スクリプトに合わせる
    x_axis = list(range(0, 101)) + list(range(110, epochs + 1, 10))
    
    # グラフの構成設定
    s_nh_list = [5, 10, 20]
    alpha_list = [0.5, 1.0, 2.0]
    b_ratios = [0.25, 1.0, 4.0]
    
    # 色とラベルの設定
    colors = {0.25: 'blue', 1.0: 'green', 4.0: 'red'}
    labels = {
        0.25: r'$\beta_{max} / 4$ (Low)', 
        1.0: r'$\beta_{max}$ (Optimal)', 
        4.0: r'$4 \times \beta_{max}$ (High)'
    }

    # 1行3列のグラフを作成
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Log-Likelihood Growth: Teacher-Student Initialization Strategy', fontsize=16)

    # ------------------------------------------
    # 2. データの読み込みと描画
    # ------------------------------------------
    for i, (s_nh, alpha) in enumerate(zip(s_nh_list, alpha_list)):
        ax = axes[i]
        
        # サブプロットのタイトルと軸ラベル
        ax.set_title(r'Student $\alpha = {}$ ($n_h = {}$)'.format(alpha, s_nh), fontsize=14)
        ax.set_xlabel('Epochs', fontsize=12)
        if i == 0:
            ax.set_ylabel('Log-Likelihood', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        for b_ratio in b_ratios:
            # globを使用して、条件に合致する最新のファイルを検索
            pattern = f"results/ll_snh{s_nh}_beta{b_ratio:.2f}_*.npy"
            #pattern = f"2000epoch_α=2.0_β=0.25_1.0/ll_snh{s_nh}_beta{b_ratio:.2f}_*.npy"
            files = glob.glob(pattern)
            
            if not files:
                print(f"Warning: Data not found for n_h={s_nh}, beta={b_ratio:.2f}. Skipping.")
                continue
            
            # 複数ファイルがある場合は、名前順（タイムスタンプ順）で最新のものを取得
            latest_file = sorted(files)[-1]
            data = np.load(latest_file)
            
            # 平均と標準偏差を計算 (axis=0 は試行の次元)
            mean_ll = np.mean(data, axis=0)
            std_ll = np.std(data, axis=0)
            
            # プロット (平均値の線と、標準偏差の影)
            ax.plot(x_axis, mean_ll, label=labels[b_ratio], color=colors[b_ratio], linewidth=2)
            ax.fill_between(x_axis, mean_ll - std_ll, mean_ll + std_ll, color=colors[b_ratio], alpha=0.2)

        # 凡例の配置 (右下に配置して、初期の成長曲線を隠さないようにする)
        ax.legend(loc='lower right', fontsize=11)

    # ------------------------------------------
    # 3. 画像の保存と表示
    # ------------------------------------------
    plt.tight_layout()
    # サブプロットのタイトルと全体のタイトルが被らないように調整
    plt.subplots_adjust(top=0.88)
    
    save_path = "results/combined_loglikelihood_plot.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_combined_results()