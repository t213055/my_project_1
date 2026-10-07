#特定のαにおける層相関の計算を行うスクリプト
import numpy as np
import matplotlib.pyplot as plt
#import csv
#import sys
from scipy.integrate import quad
from numpy.polynomial.hermite import hermgauss

#モデルのパラメータ
eps = 1e-10
b = eps
c = eps
alpha = 0.5
t = 2

#温度のスタート, ゴール, ステップサイズ
beta_init = 0.0 + 1e-16
beta_limit = 2.0
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
    
    def integrand_q_v(z):
        B_z = b + z * np.sqrt(q_hat[0])
        N_z = np.sum(np.exp(A * v_k**2) * (v_k**2 * np.cosh(B_z * v_k) + z * v_k * np.sinh(B_z * v_k)))
        D_z = 0.5 * delta_term + np.sum(np.exp(A * v_k**2) * np.cosh(B_z * v_k))
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
            return q, q_hat, A #Aをmoment計算用に返す

def Effective_Susceptibility_Matrices(q_hat):
    global v_k
    v_k_row = v_k[np.newaxis, :]
    #betaにおける可視層/隠れ層のモーメント
    #AはSP()で計算 → main文内でグローバル変数として扱う

    #実行感受率行列の計算（ガウスエルミート求積法）
    deg = 20
    x_i, w_i = hermgauss(deg)

    z_i = np.sqrt(2) * x_i
    weights = w_i / np.sqrt(np.pi)

    #z_iに依存する変数を一括計算
    B_i = b + z_i * np.sqrt(q_hat[0])
    B_i = B_i[:, np.newaxis] # shape: (deg, 1)
    
    #-------------------
    #np.sumで和をとる方向がごっちゃになっているかも。B_iの和はまだ計算しない。計算するのは、v_k_rowについての和のみ
    #-------------------
    #↓↓
    #モーメントの分母：Denominator, 分子：Numerator （_1は1次, _2は2次）
    D = delta_term + 2 * np.sum(np.exp(A * v_k_row**2) * np.cosh(B_i * v_k_row), axis=1)
    #N_1 = delta_term + 2 * np.sum(v_k_row * np.exp(A * v_k_row**2) * np.sinh(B_i * v_k_row), axis=1)
    #N_2 = delta_term + 2 * np.sum(v_k_row**2 * np.exp(A * v_k_row**2) * np.cosh(B_i * v_k_row), axis=1)
    N_1 = 2 * np.sum(v_k_row * np.exp(A * v_k_row**2) * np.sinh(B_i * v_k_row), axis=1)
    N_2 = 2 * np.sum(v_k_row**2 * np.exp(A * v_k_row**2) * np.cosh(B_i * v_k_row), axis=1)
    ## --モーメント-- ##
    Ev1 = N_1 / D
    Ev2 = N_2 / D
    Eh1 = np.tanh(c + z_i * np.sqrt(q_hat[1]))
    #print("Ev1:", Ev1.shape, "Ev2:", Ev2.shape, "Eh1:", Eh1.shape)
    
    #実行感受率行列の要素の計算
    V00 = np.sum((Ev2 - Ev1**2) * weights)
    V11 = np.sum((1 - Eh1**2) * weights)
    V = ([[V00, 0], [0, V11]])#; print("V", V)

    U00 = np.sum((Ev2 * Ev1 - Ev1**3) * weights)
    U11 = np.sum((Eh1 - Eh1**3) * weights)
    U = ([[U00, 0], [0, U11]])#; print("U", U)

    W00 = np.sum((Ev2**2 - 4 * Ev2 * Ev1**2 + 3 * Ev1**4) * weights)
    W11 = np.sum((1 - 4 * Eh1**2 + 3 * Eh1 **4) * weights)
    W = ([[W00, 0], [0, W11]])#; print("W", W)
    return V, U, W

def Q_HQ_solver_chi(V, U, W, beta, T_alpha, alpha):
    M = np.identity(2) - beta**2 * T_alpha @ W
    B = 2 * beta**2 * T_alpha @ U
    HQ = np.linalg.solve(M, B)
    chi = (1 + alpha)**-1 * np.array([[1, 0], [0, alpha]]) @ (V - U @ HQ)
    return chi

beta = beta_init
v_k = visible_variable_array(t); print("v_k:\n", v_k)
delta_term = 1.0 if (t % 2 == 0) else 0.0 #tが偶数 → 1/2 奇数 → 0

#初期値設定（秩序パラメータと補助変数）
q_init = np.ones(2) * 1e-16
q_hat_init = beta**2 * T_alpha @ q_init

#初期化
q = q_init.copy()
q_hat = q_hat_init.copy()

beta_list = []
chi_list = []

while beta <= beta_limit:
    #鞍点の計算 内部で収束するまでループする
    q, q_hat, A = SP(beta, q, q_hat, T_alpha)
    
    #実行感受率行列を計算
    V, U, W = Effective_Susceptibility_Matrices(q_hat)

    #感受率行列の計算（Q, HQをソルバーで計算 → χを計算）
    chi = Q_HQ_solver_chi(V, U, W, beta, T_alpha, alpha)
    #print(f"beta: {beta:.3f}", q: {q}, q_hat: {q_hat}", f"A: {A:.6e}", f"chi: {chi[0, 1]:.10e}")
    beta_list.append(beta)
    chi_list.append(np.abs(chi[0, 1]))
    print(f"beta: {beta:.3f}", f"chi: {chi[0, 1]:.10e}")

    beta += beta_step

#グラフの描画
plt.figure(figsize=(8, 6))
plt.plot(beta_list, chi_list, marker='o', markersize=3, linestyle='-', color='b')

# 軸ラベルとタイトル（数式にはLaTeX記法を使用）
plt.xlabel(r'$\beta$', fontsize=14)
plt.ylabel(r'$\chi_{vh}$', fontsize=14)
plt.title(f'Layer Correlation vs Inverse Temperature (t={t}, alpha={alpha})', fontsize=14)

# グリッドの表示とレイアウト調整
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# グラフを表示
plt.show()