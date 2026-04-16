import numpy as np
import gbrbm
import os

# Backend check
try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np

def generate_and_save_teacher_data():
    # 実験設定
    n_v = 10
    n_h_list = [8, 15, 30]
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
        model = gbrbm.GBRBM(n_v, n_h, gbrbm.BinaryUnit(), gbrbm.ContrastiveDivergence(), weight_std=1.0)
        
        # 教師の個性を設定 (バイアス)
        model.b = xp.random.normal(0, 0.5, n_v).astype(xp.float32)
        model.c = xp.random.normal(0, 0.5, n_h).astype(xp.float32)
        
        # 分散のばらつき設定 (N(1.0, 0.5) からサンプリングし、正値を保証)
        vars_sampled = xp.random.normal(1.0, sigma_dist, n_v).astype(xp.float32)
        vars_sampled = xp.maximum(vars_sampled, 0.1)  # 最小値を 0.1 に制限
        model.gamma = xp.log(xp.exp(vars_sampled) - 1.0) # gammaに変換
        
        # 2. ギブスサンプリングによるデータ生成
        data_list = []
        # 初期値
        _, v_current = model.sample_v_given_h(xp.zeros((1, n_h)))
        
        for i in range(n_samples):
            # 最初は burn_in、次からは thinning 回数回す
            steps = burn_in if i == 0 else thinning
            for _ in range(steps):
                _, h = model.sample_h_given_v(v_current)
                _, v_current = model.sample_v_given_h(h)
            
            data_list.append(v_current.copy())
            if (i + 1) % 1000 == 0:
                print(f"  Sample {i+1}/{n_samples} generated.")

        # 結合
        raw_data = xp.vstack(data_list)
        
        # 3. [0, 1] スケーリング
        v_min = raw_data.min(axis=0)
        v_max = raw_data.max(axis=0)
        scaled_data = (raw_data - v_min) / (v_max - v_min + 1e-8)
        
        # 4. 保存 (NumPy形式に変換して保存)
        filename = f"data/teacher_nv10_nh{n_h}_s5000.npy"
        
        # CuPy(GPU)を使っている場合のみ asnumpy で変換
        if hasattr(xp, 'asnumpy'):
            save_data = xp.asnumpy(scaled_data)
        else:
            save_data = scaled_data
            
        np.save(filename, save_data)
        print(f"Saved: {filename}\n")

if __name__ == "__main__":
    generate_and_save_teacher_data()