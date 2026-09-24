#特定のαにおける層相関の計算を行うスクリプト
import numpy as np
import matplotlib as plt
import csv
import sys
from scipy.integrate import quad
from numpy.polynomial.hermite import hermgauss

#モデルのパラメータ
eps = 1e-16
b = eps
c = eps
alpha = 2.0

#温度のスタート, ゴール, ステップサイズ
beta_init = 0.0 + 1e-16
beta_limit = 1.2
beta_step = 0.001

#収束判定、ループ上限回数、緩和法の強さ
tol_sp = 1e-10

beta = beta_init

#秩序パラメータ計算時の重み行列
T_alpha = (1+alpha)**-1 * np.array([[0, alpha], [1, 0]])

#秩序パラメータと補助変数の初期値設定
q_init = np.array(np.ones(2))*1e-16
q_hat_init = beta**2 * T_alpha @ q_init
print("q_init :", q_init, "q_hat_init :", q_hat_init)

def integrand_q_v(z, q_hat_val):

def integrand_q_h(z, q_hat_val):
    f_z = np.tanh(c + z * np.sqrt(q_hat_val[1]))

def SP(t, alpha, beta, q, q_hat):
    """
    Parameters:
        t (int): 可視変数の分割数パラメータ
        alpha, beta, q_v_hat, q_h_hat, b, c (float): モデルの各種ハイパーパラメータ
        num_nodes (int): ガウス・エルミート積分の分点数（通常30〜100程度で十分な精度が出ます）
    Returns:
        float: 積分結果
    """
    ### q[1]の計算 ###
    
    ## 1. 定数 A 
    A = 0.5 * ((alpha * beta**2) / (1 + alpha) - q_hat[0])

    ## 2. ガウスエルミート求積法の準備
    #ノード(x)と重み(w)を取得
    x, w = hermgauss(50)
    
    #標準正規分布の測度に合わせる
    z = np.sqrt(2) * x
    
    # B(z)の計算
    B_z = b + z * np.sqrt(q_hat[0])

    ## 3. 可視変数 v_k の配列を生成
    k_start = (t // 2) + 1
    k = np.arange(k_start, t + 1)
    v_k = (2 * k - t) / t

    ## 4. ブロードキャスト（行列計算）用の次元調整
    z_col = z[:, np.newaxis]       # shape: (num_nodes, 1)
    B_z_col = B_z[:, np.newaxis]   # shape: (num_nodes, 1)
    v_k_row = v_k[np.newaxis, :]   # shape: (1, len(k))

    ## 5. シグマ（和）の中身の計算
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
    delta_term = 0.5 if (t % 2 == 0) else 0.0
    D_z = delta_term + np.sum(exp_Av2 * cosh_Bv, axis=1)

    # 6. ガウス・エルミート積分 (重み w を掛けて和をとり、1/sqrt(pi) で割る)
    integrand = N_z / D_z
    q[0] = (1.0 / np.sqrt(np.pi)) * np.sum(w * integrand)
    
    return q[0]

# -----------------
# 実行例（テスト）
# -----------------
if __name__ == "__main__":
    t_val = 2
    alpha_val = 1.0
    beta_val = 0.1
    b_val = 1e-16
    c_val = 1e-16
    
    #前回のスピングラス秩序を受け取る
    if q_hat != None:
        q_v_hat_val = q[0]
        q_h_hat_val = q[1]
    elif q_hat == None:
        q_v_hat_val = 1e-16
        q_h_hat_val = 1e-16
    
    q_v = SP(t_val, alpha_val, beta_val, q_v_hat_val, q_h_hat_val, b_val, c_val)
    print(f"t={t_val}, q_v = {q_v}")