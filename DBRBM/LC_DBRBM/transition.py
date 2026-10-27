import numpy as np
from scipy.optimize import root_scalar

def find_transition_point(t, alpha):
    # 可視変数の配列を生成
    k_start = (t // 2) + 1
    k = np.arange(t, k_start - 1, -1)
    v_k = (2 * k - t) / t
    
    # 対称性を利用するため、v>0 の要素と v=0 の要素を分ける
    v_pos = v_k[v_k > 0]
    delta_term = 1.0 if (t % 2 == 0) else 0.0
    
    # 解きたい方程式: f(beta) = 0 になるよう移項した形
    def equation(beta):
        A = (alpha * beta**2) / (2 * (1 + alpha))
        
        # 分子と分母の計算（対称性を利用して計算量を削減）
        num = 2 * np.sum((v_pos**2) * np.exp(A * v_pos**2))
        den = delta_term + 2 * np.sum(np.exp(A * v_pos**2))
        
        expected_v2 = num / den
        target_val = np.sqrt(alpha) + 1.0 / np.sqrt(alpha)
        
        return beta**2 * expected_v2 - target_val

    # beta が 0.1 ~ 10.0 の範囲で解を探索
    sol = root_scalar(equation, bracket=[0.1, 10.0], method='brentq')
    return sol.root

# テスト実行
print(f"t=2, alpha=2.5 の転移点 beta_c: {find_transition_point(2, 2.5):.5f}")