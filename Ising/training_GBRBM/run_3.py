import numpy as np
import matplotlib.pyplot as plt
import os
import gbrbm  # GBRBMのモジュール（NumPy版）をインポート

# ==========================================
# 0. ハイパーパラメータの設定
# ==========================================
DATA_PATH = "data/teacher_nv10_nh30_s5000_2.npy"
N_HIDDEN = 20           # 隠れ変数の数 (Ising)
LEARNING_RATE = 0.001   # 学習率
EPOCHS = 200             # エポック数
BATCH_SIZE = 100        # バッチサイズ
K_CD = 1                # Contrastive Divergence のステップ数
N_TRIAL = 20            # 各条件の試行回数

# 比較する weight_std の基準値
WEIGHT_STD_BASE = 0.94
# 今回実験する weight_std のリスト（必要に応じて追加・コメントアウト）
WEIGHT_STD_LIST = [WEIGHT_STD_BASE / 4.0, WEIGHT_STD_BASE, WEIGHT_STD_BASE * 4.0]

# ==========================================
# 出力モードの設定
# 1: グラフにプロットして保存
# 2: 結果をCSV形式で"output.txt"に出力（別プログラムで可視化用）
# ==========================================
OUTPUT_MODE = 2

def main():
    # 1. データの読み込み
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"データファイルが見つかりません: {DATA_PATH}")
    
    print(f"Loading data from {DATA_PATH}...")
    X_train = np.load(DATA_PATH)  # [5000, 10] のスケーリングされていない生データ
    n_samples, n_v = X_train.shape
    print(f"Data shape: {X_train.shape}")

    # 各 weight_std ごとの結果（平均と標準偏差）を保存する辞書
    results = {}

    # 2. 各 weight_std 条件について実験を実行
    for w_std in WEIGHT_STD_LIST:
        print(f"\n=============================================")
        print(f" Starting experiments for weight_std = {w_std}")
        print(f"=============================================")
        
        # この条件での全トライアルの履歴を保存する配列
        all_ll_history = np.zeros((N_TRIAL, EPOCHS))

        for trial in range(N_TRIAL):
            print(f"\n--- weight_std={w_std} | Trial {trial + 1}/{N_TRIAL} ---")
            
            # モデルの初期化 (IsingUnit を使用、現在のループの weight_std を適用)
            model = gbrbm.GBRBM(
                n_v=n_v, 
                n_h=N_HIDDEN, 
                unit_type=gbrbm.IsingUnit(), 
                sampler=gbrbm.ContrastiveDivergence(k=K_CD), 
                weight_std=w_std
            )
            
            # 1エポックごとの学習
            for epoch in range(EPOCHS):
                # データのシャッフル
                indices = np.random.permutation(n_samples)
                X_train_shuffled = X_train[indices]
                
                # ミニバッチ学習
                for i in range(0, n_samples, BATCH_SIZE):
                    batch = X_train_shuffled[i : i + BATCH_SIZE]
                    model.update(batch, LEARNING_RATE)
                    
                # エポック終了時に対数尤度を計算
                ll = model.compute_log_likelihood(X_train).item()
                all_ll_history[trial, epoch] = ll
                
                # 進捗を適度に表示
                #if (epoch + 1) % 10 == 0 or epoch == 0:
                    #print(f"  Epoch {epoch + 1:2d}/{EPOCHS} | Log-Likelihood: {ll:.4f}")

        # N_TRIAL回終了後、平均と標準偏差を計算して保存
        results[w_std] = {
            'mean': np.mean(all_ll_history, axis=0),
            #'std': np.std(all_ll_history, axis=0)
        }

    print("\nAll training completed. Processing output...")

    # 3. 出力処理 (OUTPUT_MODE に応じて分岐)
    epochs_x = np.arange(1, EPOCHS + 1)

    if OUTPUT_MODE == 1:
        # ==========================================
        # ① グラフにプロットして保存
        # ==========================================
        line_styles = ['-', '--', ':']   # 実線、破線、点線
        colors = ['blue', 'green', 'red'] # 青、緑、赤
        
        plt.figure(figsize=(9, 6))
        
        # 条件ごとにプロット
        for i, w_std in enumerate(WEIGHT_STD_LIST):
            mean_ll = results[w_std]['mean']
            #std_ll = results[w_std]['std']
            
            label_text = f'weight_std = {w_std}'
            
            # 平均値のプロット
            plt.plot(epochs_x, mean_ll, color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], linewidth=2.5, label=label_text)
            
            # 標準偏差の塗りつぶし領域を描画
            #plt.fill_between(epochs_x, mean_ll - std_ll, mean_ll + std_ll, color=colors[i % len(colors)], alpha=0.15)

        plt.title(f"GBRBM (Ising) Log-Likelihood Comparison\n(Hidden={N_HIDDEN}, LR={LEARNING_RATE}, {N_TRIAL} trials each)")
        plt.xlabel("Epoch")
        plt.ylabel("Log-Likelihood")
        plt.grid(True, linestyle='--')
        
        plt.legend(loc='lower right')
        plt.tight_layout()

        # 画像として保存
        save_filename = "log_likelihood.png"
        plt.savefig(save_filename, dpi=150)
        print(f"Plot saved to: {os.path.abspath(save_filename)}")
        # plt.show()

    elif OUTPUT_MODE == 2:
        # ==========================================
        # ② txtファイルにCSV形式で出力
        # ==========================================
        save_filename = "output_2.txt"
        
        with open(save_filename, "w", encoding="utf-8") as f:
            # ヘッダーの作成
            header_elements = ["Epoch"]
            for w_std in WEIGHT_STD_LIST:
                header_elements.append(f"chi_Mean_{w_std}")
                #header_elements.append(f"Std_{w_std}")
            f.write(",".join(header_elements) + "\n")
            
            # データの書き込み
            for epoch in range(EPOCHS):
                row_elements = [str(epoch + 1)]
                for w_std in WEIGHT_STD_LIST:
                    mean_val = results[w_std]['mean'][epoch]
                    #std_val = results[w_std]['std'][epoch]
                    # 小数点以下6桁まで出力
                    row_elements.append(f"{mean_val:.6f}")
                    #row_elements.append(f"{std_val:.6f}")
                f.write(",".join(row_elements) + "\n")
                
        print(f"Data exported to: {os.path.abspath(save_filename)}")

    else:
        print("Error: 不明な OUTPUT_MODE が設定されています。")

if __name__ == "__main__":
    main()