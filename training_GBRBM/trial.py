import numpy as np
import itertools
import matplotlib.pyplot as plt
from gbrbm import GBRBM, BinaryUnit, ContrastiveDivergence, xp

def generate_gmm_toy(n_samples=2000):
    # 2つのクラスタを作成
    n_per_class = n_samples // 2  # //：整数除算
    # クラスタ1: 右上 (2, 2)
    c1 = xp.random.normal(loc=2.0, scale=0.5, size=(n_per_class, 2))
    # クラスタ2: 左下 (-2, -2)
    c2 = xp.random.normal(loc=-2.0, scale=0.5, size=(n_per_class, 2))
    data = xp.vstack([c1, c2]).astype(xp.float32)
    return data[xp.random.permutation(n_samples)]

def LogLikelihood(v_train, model, n_h):
    pos = -(v_train**2 @ (0.5/model.get_var())) + v_train @ model.b + np.log(1 + np.exp(model.c.T + v_train @ model.W)).sum(axis = 1)
    LL_pos = pos.mean(axis = 0)
    #print("LL_pos :", LL_pos)

    #(2,)は行列ではなく、ベクトルであり、@では内積を計算する仕組みになっているため、.Tの転置が無効
    neg = 0.5*np.log(2*np.pi*model.get_var()).sum(axis = 0) + (0.5*model.b**2) @ model.get_var()
    #print("neg :", neg)
    #print("neg.shape :", neg.shape)

    #隠れ変数の和の計算
    H_all = np.array(list(itertools.product([0, 1], repeat=n_h)), dtype=np.float32)
    neg1 = model.W.T @ (model.get_var()*model.b) + model.c
    neg1 = H_all @ neg1
    neg2 = (H_all @ (model.get_var()[0,]*model.W[0,].T))**2 + (H_all @ (model.get_var()[1,]*model.W[1,].T))**2
    neg3 = np.log((neg1 + neg2).sum())
    LL_neg = neg + neg3

    #print("neg1 :", neg1)
    #print("neg2 :", neg2)
    #print("neg3 :", neg3)
    #print("LL_neg :", LL_neg)

    LL = LL_pos - LL_neg
    return LL


def main():
    # 1. データ準備
    v_train = generate_gmm_toy()
    #v_train = xp.array([[2, 0], [0, -2]], dtype=xp.float32)
    print("v_train :\n", v_train)
    print("v_train.shape :", v_train.shape)

    # 2. モデル初期化 (2次元入力なので n_v=2)
    # 隠れ層は 8〜16 程度で十分です
    n_v = 2
    n_h = 4
    model = GBRBM(n_v=n_v, n_h=n_h, 
                  unit_type=BinaryUnit(), 
                  sampler=ContrastiveDivergence(k=1))

    #model.W = np.array([[0.5, -0.5, 0], [0, 0.5, 1.0]])
    print("b :", model.b, "shape :", model.b.shape)
    print("c :", model.c, "shape :", model.c.shape)
    print("W :", model.W, "shape :", model.W.shape)
    print("sfp(gamma) :", model.get_var(), "shape :", model.gamma.shape)

    #print("loglikelihood :", LogLiklihood(v_train, model, n_h))

    import matplotlib.pyplot as plt

    # 3. 学習ループ
    lr = 0.01
    epochs = 1000
    batch_size = 20

    # --- 【追加】記録用のリストを用意 ---
    recorded_epochs = []
    log_likelihoods = []

    for epoch in range(epochs):
        perm = xp.random.permutation(len(v_train))
        for i in range(0, len(v_train), batch_size):
            model.update(v_train[perm[i:i+batch_size]], lr)
        
        # 10エポックごとに記録と表示
        if (epoch + 1) % 10 == 0:
            ll_value = LogLikelihood(v_train, model, n_h)
            print(f"Epoch {epoch+1} - Log Likelihood: {ll_value}")
            
            # エポック数と対数尤度をリストに保存
            recorded_epochs.append(epoch + 1)
            
            # GPU(CuPy)のデータならCPU(NumPy)に変換して保存
            ll_cpu = ll_value.get() if hasattr(ll_value, 'get') else ll_value
            log_likelihoods.append(ll_cpu)

    # --- 【追加】4. 対数尤度のグラフ描画 ---
    plt.figure(figsize=(10, 5))
    plt.plot(recorded_epochs[:50], log_likelihoods[:50], marker='o', linestyle='-', color='b', label='Log Likelihood')
    plt.xlabel('Epoch')
    plt.ylabel('Log Likelihood')
    plt.title('GBRBM Training Progress')
    plt.grid(True)
    plt.legend()
    plt.show()
"""
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
"""
if __name__ == "__main__":
    main()