import numpy as np
import matplotlib as plt
import csv
from scipy.integrate import quad

#モデルのパラメータ
b = 0.001
c = 0.001
gamma = np.log(np.exp(1) - 1)
alpha = 1.0

#温度のスタート, ゴール, ステップサイズ
beta_init = 0.0 + 1e-16
beta_limit = 10.0
beta_step = 0.01

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
        r_init = np.ones(1)*1e-16
        q_hat_init = beta**2 * T_alpha @ q_init
        r_hat_init = alpha*beta**2*(2*(1+alpha))**-1
    elif init_pattern == "R":
        #分散>=0の条件を満たすようq_hat_init[0], r_hat_initを乱数で初期化
        q_init = np.random.normal(0, 1, 2)
        r_init = abs(np.random.normal(0, 1, 1))
        q_hat_init = beta**2 * T_alpha @ q_init
        r_hat_init = alpha*beta**2*(2*(1+alpha))**-1
        q_hat_init[0] = 2*r_hat_init - 1 + abs(np.random.normal(0, 1))
elif direction == "BW":
    q_init = np.array(np.ones(2))*1e-16
    q_hat_init = beta**2 * T_alpha @ q_init

    q_init[0] = 2.7518165438e-02
    q_init[1] = 6.8620972396e-01
    r_init = 8.0203367796e-04
    q_hat_init[0] = 3.4310486198e+01
    q_hat_init[1] = 1.3759082719e+00
    r_hat_init = 2.5000000000e-33
print("q_init :", q_init, "r_init :", r_init, "q_hat_init :", q_hat_init, "r_hat_init", r_hat_init)

#収束判定、ループ上限回数、緩和法の強さ
tol = 1e-10
#max_iter = 100000
#damping = 0.2

def softplus(x):
    return np.log(1 + np.exp(x))

def gaussian_pdf(z):
    return (np.sqrt(2*np.pi))**-1 * np.exp(-0.5*z**2)

def integrand_q_h(z):
    f_z = np.tanh(c + np.sqrt(q_hat[1]))
    return gaussian_pdf(z) * f_z**2

def saddle_point(beta, q, r, q_hat, r_hat): #秩序パラメータ、補助変数を返す

    iter = 0
    #代入法で反復計算(収束判定によりループを抜ける)
    while True:
        iter += 1
        #比較用に前回の値を保存
        q_old = q; r_old = r; q_hat_old = q_hat; r_hat_old = r_hat

        #秩序パラメータと補助変数に関する条件設定 満たさないようならば異常終了
        if (softplus(gamma)**-1 + q_hat[0] - 2*r_hat) <= 0:
            print("variance is less than zero")
            sys.exit()

        #秩序パラメータの更新
        q[0] = (b**2 + q_hat[0])/(softplus(gamma)**-1 + q_hat[0] - 2*r_hat)**2
        q[1], _ = quad(integrand_q_h, -12, 12)
        r = (softplus(gamma)**-1 - b**2 - 2*r_hat)/(softplus(gamma)**-1 + q_hat[0] - 2*r_hat)**2

        #補助変数の更新
        q_hat = beta**2 * T_alpha @ q

        #収束判定
        if (np.all(np.abs(q - q_old) <= tol) and
            np.all(np.abs(q_hat - q_hat_old) <= tol) and
            abs(r - r_old) <= tol and
            abs(r_hat - r_hat_old) <= tol):
            return q, r, q_hat, r_hat, iter
            #print(beta, q[0], q[1], r, q_hat[0], q_hat[1], r_hat, iter)
            #break

#秩序パラメータを初期化
q = q_init
r = r_init
q_hat = q_hat_init
r_hat = r_hat_init
print(q, r, q_hat, r_hat)

#出力の設定
filename = "Ising_output.txt"
with open(filename, mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["beta", "q_v", "q_h", "r_v", "q_v_hat", "q_h_hat", "r_v_hat", "iter"])

    #betaを変えながら鞍点を計算
    if direction == "FW":
        while beta < beta_limit:
            q, r, q_hat, r_hat, iter = saddle_point(beta, q, r, q_hat, r_hat)
            
            #ファイルへの書き込み "書き込み先ファイルはoutput.txt"
            writer.writerow([
                f"{beta:.4f}",
                f"{q[0]:.10e}",
                f"{q[1]:.10e}",
                f"{r:.10e}",
                f"{q_hat[0]:.10e}",
                f"{q_hat[1]:.10e}",
                f"{r_hat:.10e}",
                f"{iter:.1f}"])

            #βを更新
            
            beta += beta_step
            
    elif direction == "BW":
        while beta > beta_init:
            q, r, q_hat, r_hat, iter = saddle_point(beta, q, r, q_hat, r_hat)
            
            #ファイルへの書き込み "書き込み先ファイルはoutput.txt"
            writer.writerow([
                f"{beta:.4f}",
                f"{q[0]:.10e}",
                f"{q[1]:.10e}",
                f"{r:.10e}",
                f"{q_hat[0]:.10e}",
                f"{q_hat[1]:.10e}",
                f"{r_hat:.10e}",
                f"{iter:.1f}"])

            #βを更新
            beta -= beta_step