import numpy as np
from scipy.integrate import quad

def compute_replica_integral_quad(t, alpha, beta, q_v_hat, b):
    """
    scipy.integrate.quad を用いて積分方程式を計算する関数
    
    Parameters:
        t (int): 可視変数の分割数パラメータ
        alpha, beta, q_v_hat, b (float): モデルの各種ハイパーパラメータ
    
    Returns:
        float: 積分結果
    """
    
    # --- zに依存しない定数と配列の事前計算（quad呼び出し時の負荷を下げるため） ---
    # 1. 定数 A の計算
    A = 0.5 * ((alpha * beta**2) / (1 + alpha) - q_v_hat)
    
    # 2. 可視変数 v_k の配列を生成
    k_start = (t // 2) + 1
    k = np.arange(k_start, t + 1)
    v_k = (2 * k - t) / t  # shape: (len(k),)
    
    # 3. 積分内で繰り返し使う配列の事前計算
    v_k_sq = v_k**2
    exp_Av2 = np.exp(A * v_k_sq)
    
    # 4. tの偶奇によるデルタ項
    delta_term = 0.5 if (t % 2 == 0) else 0.0
    
    # 測度の正規化定数 1 / sqrt(2*pi)
    norm_const = 1.0 / np.sqrt(2 * np.pi)

    # --- quadに渡す被積分関数の定義 ---
    def integrand(z):
        # B(z) の計算
        B_z = b + z * np.sqrt(q_v_hat)
        
        # zに依存する双曲線関数の計算
        cosh_Bv = np.cosh(B_z * v_k)
        sinh_Bv = np.sinh(B_z * v_k)
        
        # 分子 N(z) の計算
        N_z = np.sum(exp_Av2 * (v_k_sq * cosh_Bv + z * v_k * sinh_Bv))
        
        # 分母 D(z) の計算
        D_z = delta_term + np.sum(exp_Av2 * cosh_Bv)
        
        # 標準正規分布の重み exp(-z^2/2) を掛ける
        weight = np.exp(-0.5 * z**2)
        
        return norm_const * weight * (N_z / D_z)

    # --- quadによる数値積分（区間は -∞ から ∞） ---
    # quadは (積分結果, 推定誤差) のタプルを返すため、[0]で結果のみを取得
    integral_result, error_estimate = quad(integrand, -np.inf, np.inf)
    
    return integral_result

# -----------------
# 実行例（テスト）
# -----------------
if __name__ == "__main__":
    t_val = 2
    alpha_val = 0.5
    beta_val = 1.0
    q_v_hat_val = 0.2
    b_val = 0.1
    
    result = compute_replica_integral_quad(t_val, alpha_val, beta_val, q_v_hat_val, b_val)
    print(f"t={t_val} のときの積分結果: {result:.6f}")
    
    # tを変更してテスト
    for test_t in [1, 2, 3, 4, 10]:
        res = compute_replica_integral_quad(test_t, alpha_val, beta_val, q_v_hat_val, b_val)
        print(f"t={test_t:2d} のときの積分結果: {res:.6f}")