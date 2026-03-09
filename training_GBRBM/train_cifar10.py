import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
# 自作のgbrbm.pyから必要なクラスをインポート
from gbrbm import GBRBM, BinaryUnit, ContrastiveDivergence, xp

def load_cifar10_batch(folder, batch_id=1):
    """指定したバッチファイルを読み込み、正規化したグレースケールデータを返す"""
    file_path = os.path.join(folder, f"data_batch_{batch_id}")
    with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    
    raw_data = entry['data'].astype(xp.float32)
    
    # グレースケール化 (RGB平均)
    v_data_gray = raw_data.reshape(-1, 3, 1024).mean(axis=1)
    
    # 標準化 (平均0, 標準偏差1)
    v_mean = v_data_gray.mean(axis=0)
    v_std = v_data_gray.std(axis=0) + 1e-5
    v_normalized = (v_data_gray - v_mean) / v_std
    
    return v_normalized

def visualize_reconstruction(model, v_data, n_images=5):
    """
    元の画像と再構成画像を並べて表示する
    n_images: 表示する枚数
    """
    # 最初の数枚をピックアップ
    v_batch = v_data[:n_images]
    
    # 1. 隠れ層をサンプリング (v -> h)
    h_prob, _ = model.sample_h_given_v(v_batch)
    
    # 2. 可視層を再構成 (h -> v_reconst)
    # 決定的な特徴を見るため、サンプリングではなく平均（v_mean）を使います
    v_reconst, _ = model.sample_v_given_h(h_prob)
    
    # NumPyに変換 (表示用)
    v_orig = xp.asnumpy(v_batch) if hasattr(v_batch, 'get') else np.array(v_batch)
    v_re = xp.asnumpy(v_reconst) if hasattr(v_reconst, 'get') else np.array(v_reconst)
    
    plt.figure(figsize=(n_images * 2, 4))
    for i in range(n_images):
        # 元画像
        plt.subplot(2, n_images, i + 1)
        plt.imshow(v_orig[i].reshape(32, 32), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # 再構成画像
        plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(v_re[i].reshape(32, 32), cmap='gray')
        plt.title("Reconst")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # --- 1. 設定 ---
    data_folder = "cifar-10-batches-py"
    n_h = 2048          # 隠れ層のユニット数
    lr = 0.001         # 学習率
    batch_size = 64    # ミニバッチサイズ
    epochs = 50        # 学習回数
    
    # --- 2. データの準備 ---
    # とりあえず batch_1 (10,000枚) を使用
    v_train = load_cifar10_batch(data_folder, batch_id=1)
    n_samples, n_v = v_train.shape
    
    # --- 3. モデルの初期化 ---
    # ユニットタイプやサンプラーをここで選択できるのがメリット
    model = GBRBM(
        n_v=n_v, 
        n_h=n_h, 
        unit_type=BinaryUnit(), 
        sampler=ContrastiveDivergence(k=1)
    )
    
    print(f"Training started: n_v={n_v}, n_h={n_h}, device={xp.__name__}")

    # --- 4. 学習ループ ---
    mse_history = []
    
    for epoch in range(epochs):
        # シャッフル
        perm = xp.random.permutation(n_samples)
        v_shuffled = v_train[perm]
        
        epoch_mse = 0
        for i in range(0, n_samples, batch_size):
            v_batch = v_shuffled[i : i + batch_size]
            
            # パラメータ更新
            model.update(v_batch, lr)
            
            # 再構成誤差の計算 (評価用)
            _, h_prob = model.sample_h_given_v(v_batch)
            v_reconst, _ = model.sample_v_given_h(h_prob)
            epoch_mse += xp.mean((v_batch - v_reconst)**2)
            
        avg_mse = float(epoch_mse / (n_samples / batch_size))
        mse_history.append(avg_mse)
        print(f"Epoch {epoch+1}/{epochs} - MSE: {avg_mse:.4f}")

    # --- 5. 結果の保存と可視化 ---
    # 学習後の重みを可視化すると、エッジ検出器のような模様が見えるはずです
    
    plt.plot(mse_history)
    plt.title("Reconstruction Error")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()
    
    visualize_reconstruction(model, v_train)

if __name__ == "__main__":
    main()