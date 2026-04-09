import numpy as np
import matplotlib.pyplot as plt
from gbrbm import GBRBM, BinaryUnit, ContrastiveDivergence, xp

def generate_gmm_toy(n_samples=2000, n_peaks=2, n_features=2):
    """
    N(サンプル数)と峰の数を指定してGMMデータを生成する
    """
    n_per_class = n_samples // n_peaks
    data = []
    
    # 峰の中心座標の候補 (必要に応じて増減可能)
    centers = [[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [0.0, 0.0]]
    
    for i in range(n_peaks):
        center = centers[i % len(centers)]
        c = xp.random.normal(loc=center, scale=0.5, size=(n_per_class, n_features))
        data.append(c)
        
    data = xp.vstack(data).astype(xp.float32)
    return data[xp.random.permutation(len(data))]

def main():
    # ==========================================
    # 実験設定のパラメータ
    # ==========================================
    N = 2000              # データセットのサイズ
    n_peaks = 2           # 混合ガウス分布の峰の数
    n_v = 2               # 可視層のサイズ (データの次元数)
    
    epochs = 200          # エポック数
    lr = 0.01             # 学習率
    batch_size = 20       # バッチサイズ
    
    n_trials = 100        # ★追加: 各パターンの試行回数 (調整可能)
    
    alphas = [0.5, 1.0, 2.0]        # α = m / n
    
    # αごとのβ_maxの値を辞書で定義
    beta_max_dict = {
        0.5: 1.84,
        1.0: 1.78,
        2.0: 1.78
    }
    
    # グラフの線のスタイル設定
    line_styles = ['--', '-', ':'] 
    colors = ['r', 'g', 'b'] # αごとに色を固定
    
    # ==========================================
    # 1. データ準備
    # ==========================================
    print(f"Generating Dataset: N={N}, Peaks={n_peaks}, Features={n_v}")
    v_train = generate_gmm_toy(n_samples=N, n_peaks=n_peaks, n_features=n_v)
    
    # 結果保存用の辞書 { (alpha, beta): (epochs_list, mean_ll_list) }
    results = {}

    # 記録するエポックのリストを事前に作成 (0, 10, 20, ..., 200)
    recorded_epochs = [0] + [e + 1 for e in range(epochs) if (e + 1) % 10 == 0]

    # ==========================================
    # 2. 実験ループ (データ収集)
    # ==========================================
    for i, alpha in enumerate(alphas):
        # α = m / n より、隠れ変数の数 m を計算
        n_h = int(n_v * alpha)
        if n_h < 1: n_h = 1 
        
        beta_max = beta_max_dict[alpha]
        betas = [beta_max / 4, beta_max, 4 * beta_max]
        
        for j, beta in enumerate(betas):
            print(f"\n--- Starting Experiment: α={alpha} (m={n_h}), β={beta:.4f} (Base: β_max={beta_max}) ---")
            
            # 各試行の対数尤度を格納するリスト (shape: n_trials x len(recorded_epochs))
            all_trials_lls = []
            
            # 指定された回数(n_trials)だけ実験を繰り返す
            for trial in range(n_trials):
                # コンソールに上書きで進捗を表示
                print(f"  Running Trial {trial + 1}/{n_trials} ...", end="\r")
                
                # モデルの初期化 (毎試行新しく初期化することで重みサンプリングのランダム性を評価)
                model = GBRBM(n_v=n_v, n_h=n_h, 
                              unit_type=BinaryUnit(), 
                              sampler=ContrastiveDivergence(k=1),
                              weight_std=beta)
                
                trial_log_likelihoods = []
                
                # エポック0 (学習前) の対数尤度を記録
                ll_val_initial = model.compute_log_likelihood(v_train)
                trial_log_likelihoods.append(float(ll_val_initial.get() if hasattr(ll_val_initial, 'get') else ll_val_initial))

                # 学習ループ
                for epoch in range(epochs):
                    perm = xp.random.permutation(len(v_train))
                    for batch_idx in range(0, len(v_train), batch_size):
                        model.update(v_train[perm[batch_idx:batch_idx+batch_size]], lr)
                    
                    # 10エポックごとに記録
                    if (epoch + 1) % 10 == 0:
                        ll_val = model.compute_log_likelihood(v_train)
                        trial_log_likelihoods.append(float(ll_val.get() if hasattr(ll_val, 'get') else ll_val))

                all_trials_lls.append(trial_log_likelihoods)
            
            print() # 進捗表示後の改行
            
            # n_trials回分の結果の平均を計算 (axis=0で列方向の平均を取る)
            mean_lls = np.mean(all_trials_lls, axis=0)
            
            # 結果を保存
            results[(alpha, beta)] = (recorded_epochs, mean_lls)

    # ==========================================
    # 3. グラフの描画 (並べて表示)
    # ==========================================
    fig, axes = plt.subplots(len(alphas), 1, figsize=(10, 5 * len(alphas)), sharex=True)
    
    # データセット情報の文字列作成 (平均化の試行回数も追記)
    dataset_info = f"Dataset: N={N}, Peaks={n_peaks}, n_v={n_v} | Averaged over {n_trials} trials\nHyperparams: Epochs={epochs}, LR={lr}, Batch={batch_size}"
    
    # 収集した実験結果をサブプロットごとに描画
    for i, alpha in enumerate(alphas):
        ax = axes[i] 
        color = colors[i] 
        
        current_beta_max = beta_max_dict[alpha]
        betas = [current_beta_max / 4, current_beta_max, 4 * current_beta_max]
        
        # 1つのαに対して、3つのβのパターンを描画
        for j, beta in enumerate(betas):
            eps, lls = results[(alpha, beta)]
            
            # 線種とラベル
            if j == 0:
                beta_label = "β_max / 4"
                ls_idx = 0
            elif j == 1:
                beta_label = "β_max"
                ls_idx = 1
            else:
                beta_label = "4 * β_max"
                ls_idx = 2
                
            label = f"{beta_label} ({beta:.4f})"
            
            # ax.plot()を使ってサブプロットに描画
            ax.plot(eps, lls, label=label, color=color, linestyle=line_styles[ls_idx], marker='o', markersize=4)

        # サブプロットごとの設定
        ax.set_title(f"Mean Log Likelihood Progress (α={alpha}, m={int(n_v*alpha)}, Base β_max={current_beta_max})", fontsize=12)
        ax.set_ylabel('Mean Log Likelihood')
        ax.grid(True)
        # 凡例をグラフの外側（右）に配置
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)

    # 共通の設定
    axes[-1].set_xlabel('Epochs', fontsize=12)

    # 全体のタイトルとデータセット情報の表示
    fig.suptitle(f'Effect of Weight Initialization (β) and Layer Ratio (α) on GBRBM Log-Likelihood\n{dataset_info}', fontsize=14)
    
    # 全体のレイアウトを調整
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    # 描画
    plt.show()

if __name__ == "__main__":
    main()