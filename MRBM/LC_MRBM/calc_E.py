import numpy as np
from numpy.polynomial.hermite import hermgauss

def compute_replica_integral(t, alpha, beta, q_v_hat, b, num_nodes=50):
    """
    指定されたパラメータと分割数tに基づき、積分方程式を計算する関数
    
    Parameters:
        t (int): 可視変数の分割数パラメータ
        alpha, beta, q_v_hat, b (float): モデルの各種ハイパーパラメータ
        num_nodes (int): ガウス・エルミート積分の分点数（通常30〜100程度で十分な精度が出ます）
    
    Returns:
        float: 積分結果
    """
    
    # 1. 定数 A の計算
    A = 0.5 * ((alpha * beta**2) / (1 + alpha) - q_v_hat)
    
    # 2. ガウス・エルミート求積法のノード(x)と重み(w)を取得
    x, w = hermgauss(num_nodes) ; #print("x:", x, "\nw:", w) # shape: (num_nodes,)
    
    # zへの変換 (標準正規分布の測度に合わせるため sqrt(2) を掛ける)
    z = np.sqrt(2) * x   ; print("shape", z.shape) # shape: (num_nodes,)
    
    # B(z) の計算
    B_z = b + z * np.sqrt(q_v_hat)  # shape: (num_nodes,)
    
    # 3. 可視変数 v_k の配列を生成
    k_start = (t // 2) + 1
    k = np.arange(k_start, t + 1) ; print("k:", k)
    v_k = (2 * k - t) / t ; print("v_k:", v_k) ; shape: (len(k),)
    
    # 4. ブロードキャスト（行列計算）用の次元調整
    # zを縦ベクトルに、v_kを横ベクトルとして扱うことで、zとv_kの全組み合わせを一度に計算します
    z_col = z[:, np.newaxis]       # shape: (num_nodes, 1)
    B_z_col = B_z[:, np.newaxis]   # shape: (num_nodes, 1)
    v_k_row = v_k[np.newaxis, :] ; print("v_k_row:", v_k_row) # shape: (1, len(k))
    
    # 5. シグマ（和）の中身の計算
    exp_Av2 = np.exp(A * v_k_row**2)
    cosh_Bv = np.cosh(B_z_col * v_k_row)
    sinh_Bv = np.sinh(B_z_col * v_k_row)
    
    # 分子 N(z) の計算 (横方向 = k方向に和をとる)
    # v_k_row**2 や z_col*v_k_row は対応する要素同士の掛け算になります
    N_z = np.sum(
        exp_Av2 * (v_k_row**2 * cosh_Bv + z_col * v_k_row * sinh_Bv),
        axis=1
    )
    
    # 分母 D(z) の計算
    # tが偶数の場合は 0.5、奇数の場合は 0 を足す
    delta_term = 0.5 if (t % 2 == 0) else 0.0
    D_z = delta_term + np.sum(exp_Av2 * cosh_Bv, axis=1)
    
    # 6. ガウス・エルミート積分 (重み w を掛けて和をとり、1/sqrt(pi) で割る)
    integrand = N_z / D_z
    integral_result = (1.0 / np.sqrt(np.pi)) * np.sum(w * integrand)
    
    return integral_result

# -----------------
# 実行例（テスト）
# -----------------
if __name__ == "__main__":
    # 仮のパラメータ設定（実際の研究の初期値等に書き換えてください）
    t_val = 6        # {-1, 1} の2値モデル
    alpha_val = 0.5
    beta_val = 1.0
    q_v_hat_val = 0.2
    b_val = 0.1
    
    result = compute_replica_integral(t_val, alpha_val, beta_val, q_v_hat_val, b_val)
    print(f"t={t_val:2d} のときの積分結果: {result:.6f}")
    
    # tを変更してテスト
    """for test_t in [2, 3, 4, 10]:
        res = compute_replica_integral(test_t, alpha_val, beta_val, q_v_hat_val, b_val)
        print(f"t={test_t:2d} のときの積分結果: {res:.6f}")"""