import numpy as np
import os
import gbrbm  # GBRBMのモジュール（NumPy版）をインポート

# ==========================================
# 0. ハイパーパラメータと実験全体の設定
# ==========================================
# --- 全体の実行設定 ---
N_DATASETS = 10          # 作成するデータセットの数 (N)

# --- データ生成（教師モデル）の設定 ---
T_NV = 10               # 可視変数の数
T_NH = 8                # 隠れ変数の数
N_SAMPLES = 5000        # サンプル数
BURN_IN = 1000          # バーンイン期間
SIGMA_DIST = 0.5        # 分散のばらつき
TEACHER_WEIGHT_STD = 3.00 # 教師モデルの重み（相関を強くしてパターンを作るため大きめ）

# --- 学習（生徒モデル）の設定 ---
S_NH = 5                # 生徒モデルの隠れ変数の数 (Ising)
LEARNING_RATE = 0.001   # 学習率
EPOCHS = 1000           # エポック数
BATCH_SIZE = 100        # バッチサイズ
K_CD = 1                # Contrastive Divergence のステップ数
N_TRIAL = 10             # 1つのデータセットに対する各条件の試行回数

# 比較する weight_std のリスト
WEIGHT_STD_BASE = 1.11
WEIGHT_STD_LIST = [WEIGHT_STD_BASE / 4.0, WEIGHT_STD_BASE, WEIGHT_STD_BASE * 4.0]

def generate_dataset(dataset_idx):
    """
    データセットを生成し、保存したのち、その生データを返す関数
    """
    os.makedirs("data", exist_ok=True)
    print(f"\n>>> [Dataset {dataset_idx}/{N_DATASETS}] Generating Teacher Data...")
    
    # 1. 教師モデルの初期化 (calc_exact_ll=Falseを指定してメモリ爆発を防ぐ)
    model = gbrbm.GBRBM(T_NV, T_NH, gbrbm.IsingUnit(), gbrbm.ContrastiveDivergence(), weight_std=TEACHER_WEIGHT_STD, calc_exact_ll=False)
    
    # 教師の個性を設定 (バイアス)
    model.b = np.random.normal(0, 0.5, T_NV).astype(np.float32)
    model.c = np.random.normal(0, 0.5, T_NH).astype(np.float32)
    
    # 可視変数の分散のばらつき設定
    vars_sampled = np.random.normal(1.0, SIGMA_DIST, T_NV).astype(np.float32)
    vars_sampled = np.maximum(vars_sampled, 0.1)
    model.gamma = np.log(np.exp(vars_sampled) - 1.0)
    
    # 2. ギブスサンプリングによるデータ生成
    _, v_current = model.sample_v_given_h(np.zeros((N_SAMPLES, T_NH), dtype=np.float32))
    
    for step in range(BURN_IN):
        _, h = model.sample_h_given_v(v_current)
        _, v_current = model.sample_v_given_h(h)

    # 3. データの保存
    filename = f"data/teacher_dataset_{dataset_idx}.npy"
    np.save(filename, v_current)
    print(f">>> [Dataset {dataset_idx}] Saved: {filename}")
    
    return v_current

def main():
    # 全データセットの結果を蓄積するための辞書
    # 構造: all_datasets_results[w_std] = [データセット1の平均配列, データセット2の平均配列, ...]
    all_datasets_results = {w_std: [] for w_std in WEIGHT_STD_LIST}
    
    # N回データセットを作成して学習するメインループ
    for d in range(1, N_DATASETS + 1):
        
        # 1. データセットの生成
        X_train = generate_dataset(d)
        n_samples, n_v = X_train.shape
        
        # このデータセット d における各 weight_std の結果を一時保存する辞書
        results_d = {}

        # 2. 生成したデータセットに対して学習を実行
        for w_std in WEIGHT_STD_LIST:
            print(f"\n--- Dataset {d} | weight_std = {w_std:.4f} ---")
            
            all_ll_history = np.zeros((N_TRIAL, EPOCHS))

            for trial in range(N_TRIAL):
                # モデルの初期化 (生徒モデルの隠れ変数 S_NH を使用)
                model = gbrbm.GBRBM(
                    n_v=n_v, 
                    n_h=S_NH, 
                    unit_type=gbrbm.IsingUnit(), 
                    sampler=gbrbm.ContrastiveDivergence(k=K_CD), 
                    weight_std=w_std
                )
                
                # エポックごとの学習
                for epoch in range(EPOCHS):
                    indices = np.random.permutation(n_samples)
                    X_train_shuffled = X_train[indices]
                    
                    for i in range(0, n_samples, BATCH_SIZE):
                        batch = X_train_shuffled[i : i + BATCH_SIZE]
                        model.update(batch, LEARNING_RATE)
                        
                    ll = model.compute_log_likelihood(X_train).item()
                    all_ll_history[trial, epoch] = ll
                    
                    # 進行状況の表示 (出力過多を防ぐため100エポックごと)
                    if (epoch + 1) % 100 == 0 or epoch == 0:
                        print(f"  Trial {trial+1}/{N_TRIAL} | Epoch {epoch + 1:3d}/{EPOCHS} | LL: {ll:.4f}")

            # N_TRIAL回の平均を計算して保存
            mean_ll = np.mean(all_ll_history, axis=0)
            results_d[w_std] = mean_ll
            all_datasets_results[w_std].append(mean_ll)

        # 3. データセット d の結果を "output_d.txt" に出力
        output_filename = f"output_{d}.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            header_elements = ["Epoch"] + [f"chi_Mean_{w_std}" for w_std in WEIGHT_STD_LIST]
            f.write(",".join(header_elements) + "\n")
            
            for epoch in range(EPOCHS):
                row_elements = [str(epoch + 1)]
                for w_std in WEIGHT_STD_LIST:
                    row_elements.append(f"{results_d[w_std][epoch]:.6f}")
                f.write(",".join(row_elements) + "\n")
                
        print(f"\n>>> [Dataset {d}] Data exported to: {output_filename}")

    # 4. 全てのデータセットの平均を計算して "Exp-results.txt" を作成
    print("\n=============================================")
    print(" All datasets processed. Calculating grand mean...")
    print("=============================================")
    
    with open("Exp-results.txt", "w", encoding="utf-8") as f:
        header_elements = ["Epoch"] + [f"chi_Mean_{w_std}" for w_std in WEIGHT_STD_LIST]
        f.write(",".join(header_elements) + "\n")
        
        for epoch in range(EPOCHS):
            row_elements = [str(epoch + 1)]
            for w_std in WEIGHT_STD_LIST:
                # all_datasets_results[w_std] は (N_DATASETS, EPOCHS) のデータを持つので、
                # その特定エポックにおける全データセットの平均値を計算する
                grand_mean = np.mean([res[epoch] for res in all_datasets_results[w_std]])
                row_elements.append(f"{grand_mean:.6f}")
            f.write(",".join(row_elements) + "\n")
            
    print(">>> Grand mean data exported to: Exp-results.txt\n")

if __name__ == "__main__":
    main()