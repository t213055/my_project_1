import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    csv_filename = "gbrbm_experiment_results_2000epochs.csv"
    
    # データを格納するための辞書
    # 構造: data[alpha][beta] = {epoch: mean_ll}
    data = defaultdict(lambda: defaultdict(dict))
    
    # ==========================================
    # 1. CSVファイルの読み込みとデータ抽出
    # ==========================================
    try:
        with open(csv_filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                alpha = float(row['alpha'])
                beta = float(row['beta'])
                epoch = int(row['epoch'])
                mean_ll = float(row['mean_log_likelihood'])
                
                # 指定通り、100エポック目以降のみを抽出対象とする
                if epoch >= 100:
                    data[alpha][beta][epoch] = mean_ll
                    
    except FileNotFoundError:
        print(f"Error: '{csv_filename}' が見つかりません。先に学習プログラムを実行してCSVを生成してください。")
        return
        
    alphas = sorted(data.keys())
    
    # αごとのβ_maxの値を辞書で定義 (学習時と同じ設定)
    beta_max_dict = {
        0.5: 1.84,
        1.0: 1.78,
        2.0: 1.78
    }
    
    # ==========================================
    # 2. グラフの描画設定
    # ==========================================
    # αの数だけ縦にサブプロットを作成
    fig, axes = plt.subplots(len(alphas), 1, figsize=(10, 4 * len(alphas)), sharex=True)
    if len(alphas) == 1:
        axes = [axes]
        
    for i, alpha in enumerate(alphas):
        ax = axes[i]
        alpha_data = data[alpha]
        beta_max = beta_max_dict[alpha]
        
        # 浮動小数点の微小な誤差を考慮して、ターゲットに最も近いbetaのデータを取得する関数
        def get_closest_beta_data(target_beta):
            closest_beta = min(alpha_data.keys(), key=lambda k: abs(k - target_beta))
            epochs = sorted(alpha_data[closest_beta].keys())
            lls = [alpha_data[closest_beta][ep] for ep in epochs]
            return np.array(epochs), np.array(lls)
            
        # 各βにおけるエポックと対数尤度の配列を取得
        epochs, ll_base = get_closest_beta_data(beta_max)
        _, ll_quarter = get_closest_beta_data(beta_max / 4.0)
        _, ll_quad = get_closest_beta_data(beta_max * 4.0)
        
        # ==========================================
        # 3. 差分の計算
        # ==========================================
        diff_quarter = ll_base - ll_quarter
        diff_quad = ll_base - ll_quad
        
        # ==========================================
        # 4. サブプロットへの描画とスケール・範囲調整
        # ==========================================
        ax.plot(epochs, diff_quarter, label=r'LL(β_max) - LL(β_max / 4)', marker='o', linestyle='-', color='b')
        ax.plot(epochs, diff_quad, label=r'LL(β_max) - LL(4 * β_max)', marker='s', linestyle='--', color='r')
        
        ax.set_title(f'Log-Likelihood Difference (α={alpha}, Base β_max={beta_max})', fontsize=12)
        ax.set_ylabel('Difference')
        
        # ★修正ポイント1: 負の値も扱える symlog (対称対数) スケールに変更
        # linthreshは0付近の線形スケール領域を定義します。値が小さいほど細かく対数化されます。
        ax.set_yscale('symlog', linthresh=0.1) 
        
        # ★修正ポイント2: 差の最小値から最大値まで確実に収まるようにY軸の範囲を動的に設定
        min_val = min(np.min(diff_quarter), np.min(diff_quad))
        max_val = max(np.max(diff_quarter), np.max(diff_quad))
        
        # 上下に少しだけ余白（マージン）を持たせる
        margin = max(abs(max_val), abs(min_val)) * 0.5 
        if margin == 0: margin = 1.0 # 完全に0だった場合のエラー回避
        ax.set_ylim(min_val - margin, max_val + margin)
        
        ax.grid(True, which="both", ls="--", alpha=0.7)
        ax.legend(loc='upper right')
        
    # 共通の設定
    axes[-1].set_xlabel('Epoch (>= 100)', fontsize=12)
    fig.suptitle('Long-term Impact of Weight Initialization on Log-Likelihood Difference', fontsize=14)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()