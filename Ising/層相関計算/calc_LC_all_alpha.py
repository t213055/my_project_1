#すべてのαでの層相関を計算するスクリプト
import numpy as np
import csv
import sys
from scipy.integrate import quad
from numpy.polynomial.hermite import hermgauss

# ==========================================
# 1. モデルの共通パラメータ
# ==========================================
eps = 0.1
b = eps
c = eps
gamma = np.log(np.exp(1) - 1)

# 計算する alpha のリスト
alpha_list = [0.5, 1.0, 2.0]

# 温度のスタート, ゴール, ステップサイズ
beta_init = 0.0 + 1e-16
beta_limit = 1.2
beta_step = 0.01

# 収束判定
tol_sp = 1e-10
tol_lc = 1e-3

# ==========================================
# 2. 関数定義
# ==========================================
def softplus(x):
    return np.log(1 + np.exp(x))

def gaussian_pdf(z):
    return (np.sqrt(2*np.pi))**-1 * np.exp(-0.5*z**2)

def integrand_q_h(z, q_hat_val):
    f_z = np.tanh(c + z * np.sqrt(q_hat_val[1]))
    return gaussian_pdf(z) * f_z**2

def saddle_point(beta, q, q_hat, A, T_alpha): 
    iter_count = 0
    # 代入法で反復計算(収束判定によりループを抜ける)
    while True:
        iter_count += 1
        q_old = q.copy()
        q_hat_old = q_hat.copy()

        if A <= 0:
            print("variance is less than zero")
            sys.exit()

        # 秩序パラメータの更新
        q[0] = (b**2 + q_hat[0]) / (A**2)
        q[1], _ = quad(integrand_q_h, -12, 12, args=(q_hat,))

        # 補助変数の更新
        q_hat = beta**2 * T_alpha @ q

        # 収束判定
        if (np.all(np.abs(q - q_old) <= tol_sp) and
            np.all(np.abs(q_hat - q_hat_old) <= tol_sp)):
            return q, q_hat, iter_count

def calc_coeff(q_hat_val, A_val):
    def get_moments(z):
        B = b + z * np.sqrt(q_hat_val[0])
        V_moment_1 = B / A_val
        V_moment_2 = (1.0 / A_val) + V_moment_1**2
        H_moment_1 = np.tanh(c + z * np.sqrt(q_hat_val[1]))
        return V_moment_1, V_moment_2, H_moment_1

    def integrand_S(z):
        Ev1, Ev2, Eh1 = get_moments(z)
        return Ev2 - Ev1**2
        
    def integrand_T(z):
        Ev1, Ev2, Eh1 = get_moments(z)
        return Ev2*Ev1 - Ev1**3
        
    def integrand_U(z):
        Ev1, Ev2, Eh1 = get_moments(z)
        return Eh1 - Eh1**3
        
    def integrand_V(z):
        Ev1, Ev2, Eh1 = get_moments(z)
        return 1 - Eh1**2
        
    def integrand_W(z):
        Ev1, Ev2, Eh1 = get_moments(z)
        return Ev2*Ev1 - Ev1**3
        
    def integrand_X(z):
        Ev1, Ev2, Eh1 = get_moments(z)
        return Ev2**2 - 4*Ev2*Ev1**2 + 3*Ev1**4
        
    def integrand_Y(z):
        Ev1, Ev2, Eh1 = get_moments(z)
        return 3*Eh1**4 - 4*Eh1**2 + 1
        
    def integrand_Z(z):
        Ev1, Ev2, Eh1 = get_moments(z)
        return Eh1 - Eh1**3

    # ガウスエルミート求積法で計算
    deg = 20
    x_i, w_i = hermgauss(deg)
    z_i = np.sqrt(2) * x_i
    weights = w_i / np.sqrt(np.pi)

    S = np.sum(integrand_S(z_i) * weights)
    T = np.sum(integrand_T(z_i) * weights)
    U = np.sum(integrand_U(z_i) * weights)
    V = np.sum(integrand_V(z_i) * weights)
    W = np.sum(integrand_W(z_i) * weights)
    X = np.sum(integrand_X(z_i) * weights)
    Y = np.sum(integrand_Y(z_i) * weights)
    Z = np.sum(integrand_Z(z_i) * weights)

    return S, T, U, V, W, X, Y, Z

def Q_HQ_solver_chi(X, Y, W, Z, T, beta, T_alpha, alpha):
    M = np.identity(2) - beta**2 * T_alpha @ np.array([[X, 0],[0, Y]])
    B = beta**2 * T_alpha @ np.array([[2*W, 0],[0, 2*Z]])
    HQ = np.linalg.solve(M, B)
    chi_vh = -(1/(1+alpha)) * HQ[0,1] * T
    return abs(chi_vh)


# ==========================================
# 3. メイン処理 (alphaごとのループ計算)
# ==========================================
filename = "Ising_LC_output.txt"

# ファイルを新規作成（上書きモード）し、ヘッダーを書き込む
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    # pandasなどで読み込みやすいよう、alpha列を先頭に追加
    writer.writerow(["alpha", "beta", "q_v", "q_h", "q_v_hat", "q_h_hat", "chi_vh"])

    for alpha in alpha_list:
        print(f"\n=============================================")
        print(f" Start calculation for alpha = {alpha}")
        print(f"=============================================")
        
        beta = beta_init
        T_alpha = (1+alpha)**-1 * np.array([[0, alpha], [1, 0]])

        # 秩序パラメータの初期化
        q_init = np.array(np.ones(2)) * 1e-16
        q_hat_init = beta**2 * T_alpha @ q_init

        q = q_init.copy()
        q_hat = q_hat_init.copy()

        # βの計算ループ
        while beta < beta_limit:
            A = softplus(gamma)**-1 - alpha*beta**2/(1+alpha) + q_hat[0]
            
            # 秩序パラメータの計算
            q, q_hat, iter_count = saddle_point(beta, q, q_hat, A, T_alpha)
            
            # 層相関の計算
            S, T, U, V, W, X, Y, Z = calc_coeff(q_hat, A)
            chi_vh = Q_HQ_solver_chi(X, Y, W, Z, T, beta, T_alpha, alpha)
            
            # ファイルへの書き込み
            writer.writerow([
                f"{alpha}",
                f"{beta:.4f}",
                f"{q[0]:.10e}",
                f"{q[1]:.10e}",
                f"{q_hat[0]:.10e}",
                f"{q_hat[1]:.10e}",
                f"{chi_vh:.5e}"
            ])

            # βを更新
            beta += beta_step
            
        print(f" Completed alpha = {alpha}")

print(f"\nすべての計算が完了しました。結果は {filename} に保存されました。")