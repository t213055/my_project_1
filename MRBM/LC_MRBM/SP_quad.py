#特定のαにおける層相関の計算を行うスクリプト
import numpy as np
#import matplotlib as plt
#import csv
#import sys
from scipy.integrate import quad
#from numpy.polynomial.hermite import hermgauss

#モデルのパラメータ
eps = 1e-16
b = eps
c = eps
alpha = 1.0
t = 4

#温度のスタート, ゴール, ステップサイズ
beta_init = 0.0 + 1e-16
beta_limit = 1.2
beta_step = 0.01

#収束判定、ループ上限回数、緩和法の強さ
tol_sp = 1e-10

#秩序パラメータ計算時の重み行列
T_alpha = (1+alpha)**-1 * np.array([[0, alpha], [1, 0]])

def gaussian_pdf(z):
    return (np.sqrt(2 * np.pi))**-1 * np.exp(-0.5 * z**2)

def visible_variable_array(t):
    k_start = (t // 2) + 1
    # k を t から k_start まで -1 刻み（降順）で生成
    k = np.arange(t, k_start - 1, -1)
    v_k = (2 * k - t) / t ; #print("v_k:", v_k)
    return v_k

def SP(beta, q, q_hat, T_alpha): #alphaが変わる度にT_alphaも変わるため引数とする
    
    v_k = visible_variable_array(t)  #tの値に応じて可視変数の配列作成
    delta_term = 0.5 if (t % 2 == 0) else 0.0 #tが偶数 → 1/2 奇数 → 0

    def integrand_q_v(z):
        B_z = b + z * np.sqrt(q_hat[0])
        N_z = np.sum(np.exp(A * v_k**2) * (v_k**2 * np.cosh(B_z * v_k) + z * v_k * np.sinh(B_z * v_k)))
        D_z = delta_term + np.sum(np.exp(A * v_k**2) * np.cosh(B_z * v_k))
        f_z = N_z / D_z
        return gaussian_pdf(z) * f_z

    def integrand_q_h(z):
        f_z = np.tanh(c + z * np.sqrt(q_hat[1]))
        return gaussian_pdf(z) * f_z**2

    while True:
        #収束判定のため、前回のqとq_hatを保存
        q_old = q.copy()
        q_hat_old = q_hat.copy()
        
        # q[0]の計算準備
        A = 0.5 * ((alpha * beta**2) / (1 + alpha) - q_hat[0])
        
        # q, q_hatの計算
        q[0], _ = quad(integrand_q_v, -12, 12)
        q[1], _ = quad(integrand_q_h, -12, 12)
        q_hat = beta**2 * T_alpha @ q
    
        if (np.all(np.abs(q - q_old) <= tol_sp) and
            np.all(np.abs(q_hat - q_hat_old) <= tol_sp)):
            return q, q_hat

beta = beta_init

#初期値設定（秩序パラメータと補助変数）
q_init = np.ones(2) * 1e-16
q_hat_init = beta**2 * T_alpha @ q_init

#初期化
q = q_init.copy()
q_hat = q_hat_init.copy()

while beta <= beta_limit:
    q, q_hat = SP(beta, q, q_hat, T_alpha)
    print(f"beta: {beta:.3f}, q: {q}, q_hat: {q_hat}")
    beta += beta_step