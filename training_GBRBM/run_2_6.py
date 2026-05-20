import numpy as np
import os
import time
from datetime import datetime
import gbrbm

# ==========================================
# Backend: CPU/GPU の切り替え抽象化
# ==========================================
import cupy as cp

def run_experiments():
    # ------------------------------------------
    # 1. 実験設定
    # ------------------------------------------
    n_v = 10
    n_trials = 100 # 実際には 50 などの適切な値を設定
    epochs = 10000   # 実際には 5000 などの適切な値を設定
    batch_size = 100
    lr = 0.01

    #最初の100エポックは毎回記録、以降は10エポックごとに記録
    log_epochs = list(range(0, 101)) + list(range(110, epochs + 1, 10))
    n_log_points = len(log_epochs)

    # マトリックス設定 (beta_max または beta_min のいずれかを含む)
    experiment_configs = [
        #{"t_nh": 8,  "s_nh": 5,  "beta_max": 2.599}, # α=0.5
        #{"t_nh": 8,  "s_nh": 5,  "beta_min": 2.673}, # α=0.5
        #{"t_nh": 15, "s_nh": 10, "beta_max": 2.703}, # α=1.0
        #{"t_nh": 15, "s_nh": 10, "beta_min": 2.704}, # α=1.0
        #{"t_nh": 30, "s_nh": 20, "beta_max": 2.966}, # α=2.0
        {"t_nh": 30, "s_nh": 20, "beta_min": 3.56},  # α=2.0
    ]
    
    beta_ratios = [10.0]

    os.makedirs("results_2", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ------------------------------------------
    # 2. メインの実験ループ
    # ------------------------------------------
    for config in experiment_configs:
        t_nh = config["t_nh"]
        s_nh = config["s_nh"]
        
        # ★ 修正ポイント: beta_max か beta_min かを判定
        if "beta_max" in config:
            base_beta = config["beta_max"]
            beta_type = "max"
        else:
            base_beta = config["beta_min"]
            beta_type = "min"
        
        data_path = f"data_2/teacher_nv10_nh{t_nh}_s5000.npy"
        if not os.path.exists(data_path):
            print(f"Error: {data_path} not found. Skipping.")
            continue
            
        #データをCPUからロードする
        raw_data = np.load(data_path)
        #ロードしたデータをGPU上にtrain_data配列として定義
        train_data = cp.array(raw_data, dtype=cp.float32)
        n_samples = train_data.shape[0]

        for b_ratio in beta_ratios:
            # 判定した base_beta を使用
            beta_init = base_beta * b_ratio
            print(f"\n>>> Exp: Student n_h={s_nh}, Type={beta_type}, beta_init={beta_init:.4f}")
            
            #記録用の箱をGPU上に作る
            ll_results = cp.zeros((n_trials, n_log_points), dtype=cp.float32)

            for trial in range(n_trials):
                start_time = time.time()
                
                model = gbrbm.GBRBM(
                    n_v=n_v, 
                    n_h=s_nh, 
                    unit_type=gbrbm.BinaryUnit(), 
                    sampler=gbrbm.ContrastiveDivergence(k=1), 
                    weight_std=beta_init
                )
                
                log_idx = 0
                #0epoch目の対数尤度関数を計算し、ll_resultsに記録
                if 0 in log_epochs:
                    ll_0 = model.compute_log_likelihood(train_data)
                    ll_results[trial, log_idx] = ll_0
                    log_idx += 1

                for epoch in range(1, epochs + 1):
                    indices = cp.random.permutation(n_samples)
                    shuffled_data = train_data[indices]
                    
                    for i in range(0, n_samples, batch_size):
                        batch = shuffled_data[i : i + batch_size]
                        model.update(batch, lr)
                    
                    if epoch in log_epochs:
                        ll = model.compute_log_likelihood(train_data)
                        ll_results[trial, log_idx] = ll
                        log_idx += 1
                
                if hasattr(cp, 'get_default_memory_pool'):
                    cp.get_default_memory_pool().free_all_blocks()

                elapsed_time = time.time() - start_time
                print(f"  Trial {trial+1:02d}/{n_trials} completed in {elapsed_time:.2f}s | LL: {ll_results[trial, log_idx - 1]:.2f}")

            # ★ 修正ポイント: 保存ファイル名に beta_type (max/min) を含める
            save_name = f"results_2/ll_snh{s_nh}_{beta_type}_ratio{b_ratio:.3f}_{timestamp}.npy"
            np.save(save_name, cp.asnumpy(ll_results))
            print(f"Saved: {save_name}")

    print("\nAll 8 experiments (2 configs * 4 ratios) finished!")

if __name__ == "__main__":
    run_experiments()