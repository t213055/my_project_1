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
gamma = np.log(np.exp(1) - 1)
alpha = 2.0

#温度のスタート, ゴール, ステップサイズ
beta_init = 0.0 + 1e-16
beta_limit = 1.2
beta_step = 0.001

#収束判定、ループ上限回数、緩和法の強さ
tol_sp = 1e-10
#tol_lc = 1e-3
#max_iter = 100000
#damping = 0.2

#温度の進行方向
direction = "FW"

#温度の初期化
if direction == "FW":
    beta = beta_init
elif direction == "BW":
    beta = beta_limit

#秩序パラメータ計算時の重み行列
T_alpha = (1+alpha)**-1 * np.array([[0, alpha], [1, 0]])

#秩序パラメータと補助変数の初期値設定

if direction == "FW":
    init_pattern = "F" #"F(Fixed)" or "R(Random)"
    if init_pattern == "F":
        q_init = np.array(np.ones(2))*1e-16
        q_hat_init = beta**2 * T_alpha @ q_init

    #分散>=0の条件を満たすようq_hat_init[0], r_hat_initを乱数で初期化
    elif init_pattern == "R":
        q_init = np.random.normal(0, 1, 2)
        q_hat_init = beta**2 * T_alpha @ q_init
        q_hat_init[0] = alpha*beta**2/(1+alpha) - 1 + abs(np.random.normal(0, 1))

#βを逆方向から動かすパターン
elif direction == "BW":
    q_init = np.array(np.ones(2))*1e-16
    q_hat_init = beta**2 * T_alpha @ q_init

    q_init[0] = 2.7290883770e+01
    q_init[1] = 9.4582258324e-01
    q_hat_init[0] = 7.5665806659e+00
    q_hat_init[1] = 2.1832707016e+02
print("q_init :", q_init, "q_hat_init :", q_hat_init)

def softplus(x):
    return np.log(1 + np.exp(x))

def gaussian_pdf(z):
    return (np.sqrt(2*np.pi))**-1 * np.exp(-0.5*z**2)

def integrand_q_h(z):
    f_z = np.tanh(c + z*np.sqrt(q_hat[1]))
    return gaussian_pdf(z) * f_z**2

def saddle_point(beta, q, q_hat): #秩序パラメータ、補助変数を返す

    iter = 0
    #代入法で反復計算(収束判定によりループを抜ける)
    while True:
        iter += 1
        #比較用に前回の値を保存
        q_old = q; q_hat_old = q_hat

        #秩序パラメータと補助変数に関する条件設定 満たさないようならば異常終了
        if A <= 0:
            print("variance is less than zero")
            sys.exit()

        #秩序パラメータの更新
        q[0] = (b**2 + q_hat[0])/(A**2)
        q[1], _ = quad(integrand_q_h, -12, 12)

        #補助変数の更新
        q_hat = beta**2 * T_alpha @ q

        #収束判定
        if (np.all(np.abs(q - q_old) <= tol_sp) and
            np.all(np.abs(q_hat - q_hat_old) <= tol_sp)):
            return q, q_hat, iter
            

#収束した秩序パラメータを受け取り、多項式を作成し、ガウス積分を実行し、多項式を数値として返す関数
def calc_coeff():
        #print(f"{beta:.2f}",2*alpha*beta**2*(1+alpha)**-1, f"{A:.3e}")

        def get_moments(z):
            B = b + z*np.sqrt(q_hat[0])
            V_moment_1 = B / A
            V_moment_2 = (1.0 / A) + V_moment_1**2
            H_moment_1 = np.tanh(c + z * np.sqrt(q_hat[1]))
            return V_moment_1, V_moment_2, H_moment_1

        #モーメント多項式 S, T, U, V, W, X, Y, Zを定義
        def integrand_S(z):
            Ev1, Ev2, Eh1 = get_moments(z)
            S_val = Ev2 - Ev1**2
            return S_val #* gaussian_pdf(z)

        def integrand_T(z):
            Ev1, Ev2, Eh1 = get_moments(z)
            T_val = Ev2*Ev1 - Ev1**3
            return T_val #* gaussian_pdf(z)
        
        def integrand_U(z):
            Ev1, Ev2, Eh1 = get_moments(z)
            U_val = Eh1 - Eh1**3
            return U_val #* gaussian_pdf(z)
        
        def integrand_V(z):
            Ev1, Ev2, Eh1 = get_moments(z)
            V_val = 1 - Eh1**2
            return V_val #* gaussian_pdf(z)
        
        def integrand_W(z):
            Ev1, Ev2, Eh1 = get_moments(z)
            W_val = Ev2*Ev1 - Ev1**3
            return W_val #* gaussian_pdf(z)
        
        def integrand_X(z):
            Ev1, Ev2, Eh1 = get_moments(z)
            X_val = Ev2**2 - 4*Ev2*Ev1**2 + 3*Ev1**4
            return X_val #* gaussian_pdf(z)
        
        def integrand_Y(z):
            Ev1, Ev2, Eh1 = get_moments(z)
            Y_val = 3*Eh1**4 - 4*Eh1**2 + 1
            return Y_val #* gaussian_pdf(z)
        
        def integrand_Z(z):
            Ev1, Ev2, Eh1 = get_moments(z)
            Z_val = Eh1 - Eh1**3
            return Z_val #* gaussian_pdf(z)
        
        """
        #quadで計算する場合(integrandの中の * gaussian_pdf(z)が必須)
        S, _ = quad(integrand_S, -12, 12)
        T, _ = quad(integrand_T, -12, 12, limit=200)
        U, _ = quad(integrand_U, -12, 12)
        V, _ = quad(integrand_V, -12, 12)
        W, _ = quad(integrand_W, -12, 12, limit=200)
        X, _ = quad(integrand_X, -12, 12, limit=200)
        Y, _ = quad(integrand_Y, -12, 12)
        Z, _ = quad(integrand_Z, -12, 12)
        """
        
        #ガウスエルミート求積法で計算する場合（integrandの中の * gaussian_pdf(z)を除く必要あり）
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
    
#計算したモーメントを基にHQを計算し層相関の値を導出
def Q_HQ_substitution_chi():
    
    #スピングラス秩序の補助変数の感受率だから、初期値は乱数で問題ないと思う
    Q = np.empty((2,2))
    HQ = np.random.normal(loc = 0, scale = 0.1, size = 4).reshape(2,2) #;print("beta :", beta, "\nQ\n", Q, "\nHQ\n", HQ)
    
    while(True):
        HQ_old = np.copy(HQ)#; print("HQ_old\n", HQ_old)

        Q[0,0] = 2 * W +    HQ[0,0] * X
        Q[0,1] =            HQ[0,1] * X
        Q[1,0] =            HQ[1,0] * Y
        Q[1,1] = 2 * Z +    HQ[1,1] * Y
        #print("Q\n", Q)
        
        #HQの更新
        HQ = beta**2 * T_alpha @ Q#;print("HQ\n", HQ)

        #代入法で発散するかどうかを増幅率の観点から調査する
        eigenvalues = np.linalg.eigvals(HQ) ; #print(f"固有値: {eigenvalues}")
        abs_eigenvalues = np.abs(eigenvalues)
        max_amp = np.max(abs_eigenvalues) ; #print(f"最大増幅率 (絶対値): {max_amp:.4f}")
        if max_amp >= 1.0:
            print(beta, "【警告】最大増幅率が 1 を超えています。代入法では確実に発散（NaN）します！")
            sys.exit()
        else:
            chi_vh = -(1/1+alpha) * HQ[0,1] * T
            #chi_hv = -(alpha/1+alpha) * HQ[1,0] * U
        
        #HQの値の収束確認
        if (np.all(np.abs(HQ-HQ_old) <= tol_lc)):
            return abs(chi_vh)


def Q_HQ_solver_chi():

    #スピングラス秩序の補助変数の感受率だから、初期値は乱数で問題ないと思う
    #print("X", X); print("Y", Y); print("beta**2 :", beta**2)
    M = np.identity(2) - beta**2 * T_alpha @ np.array([[X, 0],[0, Y]])
    #print("M :", M)
    #print("T_alpha :", T_alpha)
    B = beta**2 * T_alpha @ np.array([[2*W, 0],[0, 2*Z]])
    #print("B :", B)
    HQ = np.linalg.solve(M, B)
    #print("HQ :", HQ)
    chi_vh = -(1/1+alpha) * HQ[0,1] * T
    #print("chi_vh :", abs(chi_vh))
    return abs(chi_vh)

#秩序パラメータを初期化
q = q_init
q_hat = q_hat_init
#print("initialized parameters :", q, q_hat)

#出力の設定
filename = "Ising_output.txt"
with open(filename, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["beta", "q_v", "q_h", "q_v_hat", "q_h_hat", "iter", "chi_vh"])

    #計算部
    if direction == "FW":
        while beta < beta_limit:

            #秩序パラメータの計算
            A = softplus(gamma)**-1 - alpha*beta**2/(1+alpha) + q_hat[0]
            q, q_hat, iter = saddle_point(beta, q, q_hat)

            #層相関の計算
            #モーメントの要素を定義（更新）
            S, T, U, V, W, X, Y, Z = calc_coeff()
            """
            print( #各モーメント多項式の値を表示
                "β:", f"{beta:.4f}",
                "S:", f"{S:.3e}",
                "T:", f"{T:.3e}",
                "U:", f"{U:.3e}",
                "V:", f"{V:.3e}",
                "W:", f"{W:.3e}",
                "X:", f"{X:.3e}",
                "Y:", f"{Y:.3e}",
                "Z:", f"{Z:.3e}") """
            
            chi_vh = Q_HQ_solver_chi()
            #chi_vh = Q_HQ_substitution_chi()
            
            
            #ファイルへの書き込み "書き込み先ファイルはoutput.txt"
            writer.writerow([
                f"{beta:.4f}",
                f"{q[0]:.10e}",
                f"{q[1]:.10e}",
                f"{q_hat[0]:.10e}",
                f"{q_hat[1]:.10e}",
                f"{iter:.1f}",
                f"{chi_vh:.5e}"])
            

            #βを更新
            beta += beta_step

            
            
           
"""            
    elif direction == "BW":
        while beta > beta_init:
            A = softplus(gamma)**-1 - alpha*beta**2/(1+alpha) + q_hat[0]
            q, q_hat, iter = saddle_point(beta, q, q_hat)
            
            #ファイルへの書き込み "書き込み先ファイルはoutput.txt"
            writer.writerow([
                f"{beta:.4f}",
                f"{q[0]:.10e}",
                f"{q[1]:.10e}",
                f"{q_hat[0]:.10e}",
                f"{q_hat[1]:.10e}",
                f"{iter:.1f}"])

            #βを更新
            beta -= beta_step
"""