import numpy as np
import os
import time
from datetime import datetime
import gbrbm

# ==========================================
# Backend: CPU/GPU の切り替え抽象化
# ==========================================
try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np

def run_experiments():
    # ------------------------------------------
    # 1. 実験設定
    # ------------------------------------------
    n_v = 10
    n_trials = 10 #50
    epochs = 2000 #
    batch_size = 100
    lr = 0.01  # 学習率 (標準的な値)

    # 記録するエポックのリストを作成 (0~100は毎回、以後は10ごと)
    log_epochs = list(range(0, 101)) + list(range(110, epochs + 1, 10))
    n_log_points = len(log_epochs)

    # マトリックス設定 (Teacher nh, Student nh, beta_max)
    experiment_configs = [
        {"t_nh": 8,  "s_nh": 5,  "beta_max": 1.84},
        {"t_nh": 15, "s_nh": 10, "beta_max": 1.78},
        #{"t_nh": 30, "s_nh": 20, "beta_max": 1.78},
    ]
    beta_ratios = [0.25, 1.0, 4.0]

    # 保存用ディレクトリの作成
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ------------------------------------------
    # 2. メインの実験ループ
    # ------------------------------------------
    for config in experiment_configs:
        t_nh = config["t_nh"]
        s_nh = config["s_nh"]
        beta_max = config["beta_max"]
        
        # データのロード
        data_path = f"data/teacher_nv10_nh{t_nh}_s5000.npy"
        print(f"\n[{data_path}] Loading data...")
        if not os.path.exists(data_path):
            print(f"Error: {data_path} not found. Skipping.")
            continue
            
        raw_data = np.load(data_path)
        train_data = xp.array(raw_data, dtype=xp.float32)
        n_samples = train_data.shape[0]

        for b_ratio in beta_ratios:
            beta_init = beta_max * b_ratio
            print(f"\n>>> Starting Experiment: Student n_h={s_nh}, beta_init={beta_init:.4f} ({b_ratio} * beta_max)")
            
            # 結果保存用の配列 (試行数, 記録ポイント数)
            ll_results = np.zeros((n_trials, n_log_points), dtype=np.float32)

            for trial in range(n_trials):
                start_time = time.time()
                
                # 生徒モデルの初期化 (バイアスと分散は gbrbm.py 側で 0.001, 1.0 に設定済)
                model = gbrbm.GBRBM(
                    n_v=n_v, 
                    n_h=s_nh, 
                    unit_type=gbrbm.BinaryUnit(), 
                    sampler=gbrbm.ContrastiveDivergence(k=1), 
                    weight_std=beta_init
                )
                
                log_idx = 0
                
                # Epoch 0 (学習前) の対数尤度を計算
                if 0 in log_epochs:
                    ll_0 = model.compute_log_likelihood(train_data)
                    ll_results[trial, log_idx] = float(ll_0)
                    log_idx += 1

                # 学習ループ
                for epoch in range(1, epochs + 1):
                    # データをシャッフル
                    indices = xp.random.permutation(n_samples)
                    shuffled_data = train_data[indices]
                    
                    # ミニバッチ学習
                    for i in range(0, n_samples, batch_size):
                        batch = shuffled_data[i : i + batch_size]
                        model.update(batch, lr)
                    
                    # 指定エポックで対数尤度を計算
                    if epoch in log_epochs:
                        ll = model.compute_log_likelihood(train_data)
                        ll_results[trial, log_idx] = float(ll)
                        log_idx += 1
                
                # GPUメモリの解放 (CuPy使用時のメモリリーク対策)
                if hasattr(xp, 'get_default_memory_pool'):
                    xp.get_default_memory_pool().free_all_blocks()

                elapsed_time = time.time() - start_time
                print(f"  Trial {trial+1:02d}/{n_trials} completed in {elapsed_time:.2f}s | Final LL: {ll:.2f}")

            # 50試行が終わったら、この設定の結果をファイルに保存
            save_name = f"results/ll_snh{s_nh}_beta{b_ratio:.2f}_{timestamp}.npy"
            np.save(save_name, ll_results)
            print(f"Saved results to {save_name}")

    print("\nAll experiments finished successfully!")

if __name__ == "__main__":
    run_experiments()