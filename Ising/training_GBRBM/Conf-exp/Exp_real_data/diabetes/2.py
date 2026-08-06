import sys
import os

# 現在のファイル (auto_exp.py) のディレクトリパスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2つ上のディレクトリ (training_GBRBM) のパスを作成して追加
target_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(target_dir)

# これで training_GBRBM ディレクトリ内の gbrbm.py がインポート可能になります
import gbrbm
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. ハイパーパラメータと実験全体の設定
# ==========================================
# --- 学習（生徒モデル）の設定 ---
S_NH = 10                # 生徒モデルの隠れ変数の数 (Ising)
LEARNING_RATE = 0.001   # 学習率
EPOCHS = 2000           # エポック数
BATCH_SIZE = 32        # バッチサイズ
K_CD = 1                # Contrastive Divergence のステップ数
N_TRIAL = 30            # 1つの条件の試行回数

# 比較する weight_std のリスト
WEIGHT_STD_BASE = 1.00
WEIGHT_STD_LIST = [WEIGHT_STD_BASE / 4.0, WEIGHT_STD_BASE, WEIGHT_STD_BASE * 4.0]

def main():
    # 1. 糖尿病データセットのロードと前処理
    print(">>> Loading and preprocessing Diabetes dataset...")
    diabetes = load_diabetes()
    X_train_raw = diabetes.data
    
    # GBRBMの入力として適正化するため、標準化（平均0、分散1）を実施
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    
    n_samples, n_v = X_train.shape
    print(f">>> Dataset shape: {n_samples} samples, {n_v} features")
    
    # 結果を一時保存する辞書
    results = {}

    # 2. 各 weight_std 条件に対して学習を実行
    for w_std in WEIGHT_STD_LIST:
        print(f"\n--- weight_std = {w_std:.4f} ---")
        
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
                    model.update_adam(batch, LEARNING_RATE)
                    
                # 対数尤度関数の計算
                ll = model.compute_log_likelihood(X_train).item()
                all_ll_history[trial, epoch] = ll
                
                # 進行状況の表示 (出力過多を防ぐため100エポックごと)
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    print(f"  Trial {trial+1}/{N_TRIAL} | Epoch {epoch + 1:3d}/{EPOCHS} | LL: {ll:.4f}")

        # N_TRIAL回の平均を計算して保存
        mean_ll = np.mean(all_ll_history, axis=0)
        results[w_std] = mean_ll

    # 3. 結果を "Exp-results.txt" に出力
    output_filename = "Exp-results_adam_1.0.txt"
    print("\n=============================================")
    print(f" All trials processed. Exporting data to {output_filename}...")
    print("=============================================")
    
    with open(output_filename, "w", encoding="utf-8") as f:
        # ご指定のヘッダーを記述
        header = "Epoch,beta_max/4,beta_max,4beta_max"
        f.write(header + "\n")
        
        for epoch in range(EPOCHS):
            row_elements = [str(epoch + 1)]
            for w_std in WEIGHT_STD_LIST:
                row_elements.append(f"{results[w_std][epoch]:.6f}")
            f.write(",".join(row_elements) + "\n")
            
    print(">>> Export complete.\n")

if __name__ == "__main__":
    main()