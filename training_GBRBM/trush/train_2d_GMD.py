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
    n_h = 3
    model = GBRBM(n_v=2, n_h=n_h, 
                  unit_type=BinaryUnit(), 
                  sampler=ContrastiveDivergence(k=1))
    print("b :", model.b); print("c :", model.c); print("W :\n", model.W)
    
    # 3. 学習ループ
    lr = 0.01
    epochs = 100
    batch_size = 20
    
    # グラフ記録用のリスト
    recorded_epochs = []
    log_likelihoods = []
    
    print("Training started...")
    
    # ==========================================
    # メソッド呼び出しへの変更 (エポック0)
    # ==========================================
    ll_val_initial = model.compute_log_likelihood(v_train)
    ll_cpu_initial = float(ll_val_initial.get() if hasattr(ll_val_initial, 'get') else ll_val_initial)
    print(f"Epoch 0 done | Log Likelihood: {ll_cpu_initial:.4f}")
    recorded_epochs.append(0)
    log_likelihoods.append(ll_cpu_initial)

    for epoch in range(epochs):
        perm = xp.random.permutation(len(v_train))
        for i in range(0, len(v_train), batch_size):
            model.update(v_train[perm[i:i+batch_size]], lr)
        
        # 10エポックごとに対数尤度を計算して表示・記録
        if (epoch + 1) % 10 == 0:
            # メソッド呼び出しへの変更 (学習ループ内)
            ll_val = model.compute_log_likelihood(v_train)
            
            # GPUの場合はCPU(NumPy)の数値に変換
            ll_cpu = float(ll_val.get() if hasattr(ll_val, 'get') else ll_val)
            
            print(f"Epoch {epoch+1} done | Log Likelihood: {ll_cpu:.4f}")
            
            recorded_epochs.append(epoch + 1)
            log_likelihoods.append(ll_cpu)

    # --- 対数尤度のグラフ描画 ---
    plt.figure(figsize=(8, 4))
    plt.plot(recorded_epochs, log_likelihoods, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood Progress')
    plt.grid(True)
    plt.show()

    # 4. サンプリングによる検証
    print("Generating samples via Gibbs sampling...")
    v_fantasy = v_train[:1000].copy() 
    
    n_gibbs_steps = 50
    for _ in range(n_gibbs_steps):
        _, h_sample = model.sample_h_given_v(v_fantasy)
        _, v_fantasy = model.sample_v_given_h(h_sample)
    
    v_samples = v_fantasy
    
    v_t = v_train.get() if hasattr(v_train, 'get') else v_train
    v_s = v_samples.get() if hasattr(v_samples, 'get') else v_samples
    
    plt.figure(figsize=(6, 6))
    plt.scatter(v_t[:, 0], v_t[:, 1], alpha=0.3, label="True Data")
    plt.scatter(v_s[:, 0], v_s[:, 1], alpha=0.3, label="RBM Samples")
    plt.title('Data vs Generated Samples')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

"""
import numpy as np
import matplotlib.pyplot as plt
import itertools # itertoolsのインポートを追加
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

# --- LogLikelihood関数 ---
def LogLikelihood(v_train, model, n_h):
    # すべての np を xp に変更 (GPU対応のため)
    pos = -(v_train**2 @ (0.5/model.get_var())) + v_train @ model.b + xp.log(1 + xp.exp(model.c.T + v_train @ model.W)).sum(axis=1)
    LL_pos = pos.mean(axis=0)
    # print("LL_pos :", LL_pos) # ※出力が多すぎる場合はコメントアウトしてください

    # (2,)は行列ではなく、ベクトルであり、@では内積を計算する仕組みになっているため、.Tの転置が無効
    neg1 = 0.5*xp.log(2*xp.pi*model.get_var()).sum(axis=0) + (0.5*model.b**2) @ model.get_var()
    # print("neg1 :", neg1)
    
    # 隠れ変数の和の計算 (xp.arrayに変更)
    H_all = xp.array(list(itertools.product([0, 1], repeat=n_h)), dtype=xp.float32)
    neg2_1 = model.W.T @ (model.get_var()*model.b) + model.c
    neg2_1 = (H_all @ neg2_1) 
    
    neg2_2 = ((model.W @ H_all.T)**2 * 0.5*model.get_var()[:, None]).sum(axis=0) 
    
    # 【重要】オーバーフロー(inf)を防ぐための LogSumExp トリック
    exponents = neg2_1 + neg2_2
    max_val = xp.max(exponents) # 最大値を取得
    neg2 = max_val + xp.log(xp.exp(exponents - max_val).sum())
    
    # print("neg2 :", neg2)
    LL_neg = neg1 + neg2
    # print("LL_neg :", LL_neg)

    LL = LL_pos - LL_neg
    return LL

def main():
    # 1. データ準備
    v_train = generate_gmm_toy()
    
    # 2. モデル初期化 (2次元入力なので n_v=2)
    n_h = 3
    model = GBRBM(n_v=2, n_h=n_h, 
                  unit_type=BinaryUnit(), 
                  sampler=ContrastiveDivergence(k=1))
    print("b :", model.b); print("c :", model.c); print("W :\n", model.W)
    
    # 3. 学習ループ
    lr = 0.01
    epochs = 100
    batch_size = 20
    
    # グラフ記録用のリスト
    recorded_epochs = []
    log_likelihoods = []
    
    print("Training started...")
    
    # ==========================================
    # 追加部分: エポック0（学習前）の対数尤度を計算・記録
    # ==========================================
    ll_val_initial = LogLikelihood(v_train, model, n_h)
    ll_cpu_initial = float(ll_val_initial.get() if hasattr(ll_val_initial, 'get') else ll_val_initial)
    print(f"Epoch 0 done | Log Likelihood: {ll_cpu_initial:.4f}")
    recorded_epochs.append(0)
    log_likelihoods.append(ll_cpu_initial)
    # ==========================================

    for epoch in range(epochs):
        perm = xp.random.permutation(len(v_train))
        for i in range(0, len(v_train), batch_size):
            model.update(v_train[perm[i:i+batch_size]], lr)
        
        # 10エポックごとに対数尤度を計算して表示・記録
        if (epoch + 1) % 10 == 0:
            ll_val = LogLikelihood(v_train, model, n_h)
            
            # GPUの場合はCPU(NumPy)の数値に変換
            ll_cpu = float(ll_val.get() if hasattr(ll_val, 'get') else ll_val)
            
            print(f"Epoch {epoch+1} done | Log Likelihood: {ll_cpu:.4f}")
            
            recorded_epochs.append(epoch + 1)
            log_likelihoods.append(ll_cpu)

    # --- 対数尤度のグラフ描画 ---
    plt.figure(figsize=(8, 4))
    plt.plot(recorded_epochs, log_likelihoods, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Log Likelihood')
    plt.title('Log Likelihood Progress')
    plt.grid(True)
    plt.show()

    # 4. サンプリングによる検証
    print("Generating samples via Gibbs sampling...")
    v_fantasy = v_train[:1000].copy() 
    
    n_gibbs_steps = 50
    for _ in range(n_gibbs_steps):
        _, h_sample = model.sample_h_given_v(v_fantasy)
        _, v_fantasy = model.sample_v_given_h(h_sample)
    
    v_samples = v_fantasy
    
    v_t = v_train.get() if hasattr(v_train, 'get') else v_train
    v_s = v_samples.get() if hasattr(v_samples, 'get') else v_samples
    
    plt.figure(figsize=(6, 6))
    plt.scatter(v_t[:, 0], v_t[:, 1], alpha=0.3, label="True Data")
    plt.scatter(v_s[:, 0], v_s[:, 1], alpha=0.3, label="RBM Samples")
    plt.title('Data vs Generated Samples')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
"""