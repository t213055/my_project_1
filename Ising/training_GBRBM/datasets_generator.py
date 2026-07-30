import numpy as np
import gbrbm
import os

def generate_and_save_teacher_data():
    # 実験設定
    n_v = 10
    n_h_list = [30]
    n_samples = 5000
    burn_in = 1000
    thinning = 100
    sigma_dist = 0.5  # 分散のばらつき
    
    # 保存用ディレクトリ
    os.makedirs("data", exist_ok=True)

    for n_h in n_h_list:
        print(f"Generating Teacher Data: n_v={n_v}, n_h={n_h}...")
        
        # 1. 教師モデルの初期化
        # 重み std=1.0, サンプラーは適当で良い（手動で回すため）
        model = gbrbm.GBRBM(n_v, n_h, gbrbm.IsingUnit(), gbrbm.ContrastiveDivergence(), weight_std=3.00); print("mode.l.W:", model.W)
        # 教師モデルのweight_stdだが、理論値βmaxよりも十分大きくする。α=0.5でもβmax=1.11のため、3.0もあれば十分大きい。
        #これにより、教師モデルは強い相関を持ち、データにパターンが現れる。教師モデルのweight_stdが小さいと、データはほぼランダムになり、学習が困難になる。

        # 教師の個性を設定 (バイアス)
        model.b = np.random.normal(0, 0.5, n_v).astype(np.float32); print("model.b:", model.b)
        model.c = np.random.normal(0, 0.5, n_h).astype(np.float32); print("model.c:", model.c)
        
        # 可視変数の分散のばらつき設定 (N(1.0, 0.5) からサンプリングし、正値を保証)
        vars_sampled = np.random.normal(1.0, sigma_dist, n_v).astype(np.float32)
        vars_sampled = np.maximum(vars_sampled, 0.1)  # 最小値を 0.1 に制限
        model.gamma = np.log(np.exp(vars_sampled) - 1.0) # gammaに変換
        
        # 2. ギブスサンプリングによるデータ生成
        # 最初から 5000行(n_samples) の行列を作って初期値にする！
        # これにより、5000個のサンプルを一斉に計算させます。
        _, v_current = model.sample_v_given_h(np.zeros((n_samples, n_h), dtype=np.float32))
        
        print("Starting parallel Gibbs sampling...")
        # 5000個のデータを一気に burn_in 回だけサンプリングする
        for step in range(burn_in):
            _, h = model.sample_h_given_v(v_current)
            _, v_current = model.sample_v_given_h(h)
            
            # 進捗表示
            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}/{burn_in} completed.")

        # これだけで5000個の独立したサンプルが完成！
        raw_data = v_current
        
        # 3. [0, 1] スケーリング
        #v_min = raw_data.min(axis=0)
        #v_max = raw_data.max(axis=0)
        #scaled_data = (raw_data - v_min) / (v_max - v_min + 1e-8)
        
        # 4. 保存 (NumPy形式に変換して保存)
        filename = f"data/teacher_nv10_nh{n_h}_s5000_2.npy"
            
        np.save(filename, raw_data)
        print(f"Saved: {filename}\n")

if __name__ == "__main__":
    generate_and_save_teacher_data()