import numpy as np
import matplotlib.pyplot as plt
import os
import gbrbm  # GBRBMのモジュール（NumPy版）をインポート

# ==========================================
# 0. ハイパーパラメータの設定
# ==========================================
DATA_PATH = "data/teacher_nv10_nh8_s5000.npy"
N_HIDDEN = 15           # 隠れ変数の数 (Ising)
LEARNING_RATE = 0.001   # 学習率
EPOCHS = 50             # エポック数
BATCH_SIZE = 100        # バッチサイズ
K_CD = 1                # Contrastive Divergence のステップ数
N_TRIAL = 20            # 各条件の試行回数

# 比較する weight_std の基準値
WEIGHT_STD_BASE = 1.0
# 今回実験する3つの weight_std のリスト
WEIGHT_STD_LIST = [WEIGHT_STD_BASE / 4.0, WEIGHT_STD_BASE]#Y, WEIGHT_STD_BASE * 4.0]

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
            'std': np.std(all_ll_history, axis=0)
        }

    print("\nAll training completed. Plotting comparison results...")

    # 3. プロットと保存
    epochs_x = np.arange(1, EPOCHS + 1)
    
    # グラフのスタイル設定用
    line_styles = ['-', '--', ':']   # 実線、破線、点線
    colors = ['blue', 'green', 'red'] # 青、緑、赤
    
    plt.figure(figsize=(9, 6))
    
    # 3つの条件をプロット
    for i, w_std in enumerate(WEIGHT_STD_LIST):
        mean_ll = results[w_std]['mean']
        std_ll = results[w_std]['std']
        
        label_text = f'weight_std = {w_std}'
        
        # 平均値のプロット（色と線種を変更）
        plt.plot(epochs_x, mean_ll, color=colors[i], linestyle=line_styles[i], linewidth=2.5, label=label_text)
        
        # 標準偏差の塗りつぶし領域を描画（alphaで透明度を調整）
        plt.fill_between(epochs_x, mean_ll - std_ll, mean_ll + std_ll, color=colors[i], alpha=0.15)

    plt.title(f"GBRBM (Ising) Log-Likelihood Comparison\n(Hidden={N_HIDDEN}, LR={LEARNING_RATE}, {N_TRIAL} trials each)")
    plt.xlabel("Epoch")
    plt.ylabel("Log-Likelihood")
    plt.grid(True, linestyle='--')
    
    # 凡例をわかりやすい位置に表示
    plt.legend(loc='lower right')
    plt.tight_layout()

    # 画像として保存
    save_filename = "log_likelihood_comparison.png"
    plt.savefig(save_filename, dpi=150)
    print(f"Plot saved to: {os.path.abspath(save_filename)}")
    
    # 画面にも表示（不要ならコメントアウト）
    # plt.show()

if __name__ == "__main__":
    main()