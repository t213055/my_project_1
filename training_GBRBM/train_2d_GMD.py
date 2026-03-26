import numpy as np
import matplotlib.pyplot as plt
from gbrbm import GBRBM, BinaryUnit, ContrastiveDivergence, xp

def generate_gmm_toy(n_samples=2000):
    # 2つのクラスタを作成
    n_per_class = n_samples // 2
    # クラスタ1: 右上 (2, 2)
    c1 = xp.random.normal(loc=2.0, scale=0.5, size=(n_per_class, 2))
    # クラスタ2: 左下 (-2, -2)
    c2 = xp.random.normal(loc=-2.0, scale=0.5, size=(n_per_class, 2))
    data = xp.vstack([c1, c2]).astype(xp.float32)
    return data[xp.random.permutation(n_samples)]

def main():
    # 1. データ準備
    v_train = generate_gmm_toy()
    
    # 2. モデル初期化 (2次元入力なので n_v=2)
    # 隠れ層は 8〜16 程度で十分です
    model = GBRBM(n_v=2, n_h=16, 
                  unit_type=BinaryUnit(), 
                  sampler=ContrastiveDivergence(k=1))
    
    # 3. 学習ループ
    lr = 0.01
    epochs = 50
    batch_size = 20
    
    for epoch in range(epochs):
        perm = xp.random.permutation(len(v_train))
        for i in range(0, len(v_train), batch_size):
            model.update(v_train[perm[i:i+batch_size]], lr)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} done")

    # 4. サンプリングによる検証
    print("Generating samples via Gibbs sampling...")
    
    # スタート地点: データセットからランダムに1000個選ぶ (アプローチA)
    # ランダムノイズから始める場合は xp.random.normal(0, 1, (1000, 2)) などにします
    v_fantasy = v_train[:1000].copy() 
    
    # ギブスサンプリングを複数回（例：50ステップ）繰り返す
    n_gibbs_steps = 50
    for _ in range(n_gibbs_steps):
        _, h_sample = model.sample_h_given_v(v_fantasy)
        _, v_fantasy = model.sample_v_given_h(h_sample)
    
    # 最終的なサンプルを取得
    v_samples = v_fantasy
    
    # 可視化
    # GPU(CuPy)ならCPU(NumPy)に変換、CPUならそのまま
    v_t = v_train.get() if hasattr(v_train, 'get') else v_train
    v_s = v_samples.get() if hasattr(v_samples, 'get') else v_samples
    
    # 変換後のデータでプロット (.get()は外す)
    plt.scatter(v_t[:, 0], v_t[:, 1], alpha=0.3, label="True Data")
    plt.scatter(v_s[:, 0], v_s[:, 1], alpha=0.3, label="RBM Samples")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()