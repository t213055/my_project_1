#.\venv\Scripts\activate
#で仮想環境をアクティベート
import numpy as np
from numpy import exp, sqrt, tanh

from scipy.special import ndtr       # 標準正規分布の積分
from scipy.optimize import root      # 非線形方程式の解法
from scipy.stats import norm
from scipy.integrate import quad
from datetime import datetime

#sys.exit()でプログラム全体を終了させる
import sys
#while文が無限ループになるのを指定時間で強制終了
import time
TIMEOUT_SECONDS = 120

#再開用のパッケージ
import os
import csv

#警告を無視するため
#import warnings
#warnings.filterwarnings(
#    "ignore",
#    category=RuntimeWarning
#)

def load_last_state(csv_file):
    if not os.path.exists(csv_file):
        return None
    
    valid_rows = []

    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            #空行 "#"で始まる行、をスキップ
            if not row:
                continue
            if row[0].strip().startswith("#"):
                continue
            valid_rows.append(row)

    if not valid_rows:
        return None

    #valid_rowsの最後の要素を取り出す
    last = valid_rows[-1]

    last_alpha = float(last[0])
    last_c = float(last[1])
    last_beta = float(last[2])

    q = np.array([[float(last[4])],[float(last[5])]])
    hq = np.array([[float(last[6])],[float(last[7])]])
    r = np.array([[float(last[8])],[float(last[9])]])
    hr = np.array([[float(last[10])],[float(last[11])]])

    return last_alpha, last_c, last_beta, q, hq, r, hr

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def saddle_point(q, hq, r, hr, start_time, tol):
    iter = 0
    
    while True:
        q_old, hq_old, r_old, hr_old = (x.copy() for x in (q, hq, r, hr))

        # q, r, hq, hrの値を更新
        q[0] = (b**2 + hq[0])/(hq[0] + s**(-2) - 2*hr[0])**2
        r[0] = (2*hq[0] + s**(-2) - 2*hr[0] + b**2) / (((hq[0] + s**(-2) - 2*hr[0])**2))

        integrand_qh = lambda z: sigmoid(c - 0.5*hq[1] + hr[1] + z*sqrt(hq[1]))**2 * norm.pdf(z)
        q[1], error = quad( integrand_qh, -10, 10) #積分範囲を小さくする
        
        integrand_rh = lambda z: sigmoid(c - 0.5*hq[1] + hr[1] + z*sqrt(hq[1])) * norm.pdf(z)
        r[1], error = quad( integrand_rh, -10, 10) #積分範囲を小さくする

        hq = beta**2 * T_alpha @ q
        hr = 0.5 * beta**2 * T_alpha @ r
        
        #更新回数を加算
        iter += 1
        var = hq[0] + s**(-2) - 2*hr[0]
        #分散が負の場合、異常終了
        if var <= 0:
            var_error = (
                "#variance is negative value: "
                f"var={var}, alpha={alpha}, c={c}, beta={beta}"
            )
            with open(csv_file, "a") as f:
                f.write(var_error + "\n")
            #sys.exit(var_error)
            

        #収束判定
        if (np.max(np.abs(q - q_old)) < tol and np.max(np.abs(hq - hq_old)) < tol and np.max(np.abs(r - r_old)) < tol and np.max(np.abs(hr - hr_old)) < tol):
            #print(np.linalg.norm(q - q_old)); print(np.linalg.norm(hq - hq_old)); print(np.linalg.norm(r - r_old)); print(np.linalg.norm(hr - hr_old))
            break

        #時間経過で異常終了
        if time.time() - start_time >= TIMEOUT_SECONDS:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timeout_error = (
                f"#Timeout at {now_str}, beta={beta}, iter={iter}, {np.max(np.abs(q - q_old))}, {np.max(np.abs(hq - hq_old))}, {np.max(np.abs(r - r_old))}, {np.max(np.abs(hr - hr_old))}, "
            )
            with open(csv_file, "a") as f:
                f.write(timeout_error + "\n")
            sys.exit(timeout_error)

    return q, hq, r, hr, var[0], iter

def free_energy(q, r, hq, hr):#divergent 発散, roundoff error is detected 丸め誤差が検出され、許容誤差を達成できない, algorithm does not converge 丸め誤差が支配
    F_1_1 = ((alpha*beta**2)/(2*(1+alpha)))*(q[0]*q[1]) - ((alpha*beta**2)/(2*(1+alpha)))*(r[0]*r[1])
    F_1_2 = -(1/(2*(1+alpha)))*(q[0]*hq[0]) - (alpha/(2*(1+alpha)))*(q[1]*hq[1])
    F_1_3 = (1/(1+alpha))*(r[0]*hr[0]) + (alpha/(1+alpha))*(r[1]*hr[1])
    F_1 = F_1_1 - F_1_2 - F_1_3
    F_1 = F_1.item()
    return F_1

def Ev(r):
    mu = lambda z: (b + z*sqrt(hq[0]))/(hq[0] + s**(-2) - 2*hr[0])
    var = 1 / (hq[0] + s**(-2) - 2*hr[0])
    
    if r == 1:
        return lambda z:mu(z)
    elif r == 2:
        return lambda z: (mu(z)**2 + var)
    elif r == 3:
        return lambda z: (mu(z)**3 + 3*mu(z)*var)
    elif r == 4:
        return lambda z: (mu(z)**4 + 6*mu(z)**2 * var + 3*var**2)

def Eh():
    return lambda z: sigmoid(c - 0.5*hq[1] + hr[1] + z*sqrt(hq[1]))

def all_moment():
    a = Ev(1); b = Ev(2); c = Ev(3); d = Ev(4); e = Eh()
    return a, b, c, d, e 

#各モーメント行列
def A():
    #使用するモーメントの呼び出し
    Ev1, Ev2, Ev3, Ev4, Eh1 = all_moment()
    v = lambda z: (Ev2(z) - Ev1(z)**2) * norm.pdf(z)
    h = lambda z: (Eh1(z) - Eh1(z)**2) * norm.pdf(z)
    #積分を実行
    v,error = quad(v, -10, 10)
    h,error = quad(h, -10, 10)
    #行列を作成
    A = np.array([[v,0],[0,h]])
    return A

def B():
    #使用するモーメントの呼び出し
    Ev1, Ev2, Ev3, Ev4, Eh1 = all_moment()
    v = lambda z: (Ev3(z) - Ev1(z)*Ev2(z)) * norm.pdf(z)
    h = lambda z: (Eh1(z) - Eh1(z)**2) * norm.pdf(z)
    #積分を実行
    v,error = quad(v, -10, 10)
    h,error = quad(h, -10, 10)
    #行列を作成
    B = np.array([[v,0],[0,h]])
    return B

def C():
    #使用するモーメントの呼び出し
    Ev1, Ev2, Ev3, Ev4, Eh1 = all_moment()
    v = lambda z: (2*(Ev1(z)*Ev2(z) - Ev1(z)**3)) * norm.pdf(z)
    #積分を実行
    v,error = quad(v, -10, 10)
    #行列を作成
    C = np.array([[v,0],[0,0]])
    return C

def D():
    #使用するモーメントの呼び出し
    Ev1, Ev2, Ev3, Ev4, Eh1 = all_moment()
    v = lambda z: Ev1(z)*(Ev2(z) - Ev1(z)**2) * norm.pdf(z)
    h = lambda z: Eh1(z)**2 * (1 - Eh1(z)) * norm.pdf(z)
    #積分を実行
    v,error = quad(v, -10, 10)
    h,error = quad(h, -10, 10)
    #行列を作成
    D = np.array([[v,0],[0,h]])
    return D

def E():
    #使用するモーメントの呼び出し
    Ev1, Ev2, Ev3, Ev4, Eh1 = all_moment()
    v = lambda z: Ev1(z)*(Ev3(z) - Ev1(z)*Ev2(z)) * norm.pdf(z)
    h = lambda z: Eh1(z)**2 * (1 - Eh1(z)) * norm.pdf(z)
    #積分を実行
    v,error = quad(v, -10, 10)
    h,error = quad(h, -10, 10)
    #行列を作成
    E = np.array([[v,0],[0,h]])
    return E

def F():
    #使用するモーメントの呼び出し
    Ev1, Ev2, Ev3, Ev4, Eh1 = all_moment()
    v = lambda z: (4*Ev1(z)**2*Ev2(z) - Ev2(z)**2 - 3*Ev1(z)**4) * norm.pdf(z)
    h = lambda z: (4*Eh1(z)**3 - Eh1(z)**2 - 3*Eh1(z)**4) * norm.pdf(z)
    #積分を実行
    v,error = quad(v, -10, 10)
    h,error = quad(h, -10, 10)
    #行列を作成
    F = np.array([[v,0],[0,h]])
    return F

def G():
    #使用するモーメントの呼び出し
    Ev1, Ev2, Ev3, Ev4, Eh1 = all_moment()
    v = lambda z: (Ev3(z) - Ev1(z)*Ev2(z)) * norm.pdf(z)
    h = lambda z: (Eh1(z)*(1 - Eh1(z))) * norm.pdf(z)
    #積分を実行
    v,error = quad(v, -10, 10)
    h,error = quad(h, -10, 10)
    #行列を作成
    G = np.array([[v,0],[0,h]])
    return G

def H():
    #使用するモーメントの呼び出し
    Ev1, Ev2, Ev3, Ev4, Eh1 = all_moment()
    v = lambda z: (Ev4(z) - Ev2(z)**2) * norm.pdf(z)
    h = lambda z: (Eh1(z)*(1 - Eh1(z))) * norm.pdf(z)
    #積分を実行
    v,error = quad(v, -10, 10)
    h,error = quad(h, -10, 10)
    #行列を作成
    H = np.array([[v,0],[0,h]])
    return H

def I():
    #使用するモーメントの呼び出し
    Ev1, Ev2, Ev3, Ev4, Eh1 = all_moment()
    v = lambda z: (2*Ev1(z)*Ev3(z) - 2*Ev1(z)**2*Ev2(z)) * norm.pdf(z)
    h = lambda z: 2*Eh1(z)**2*(1 - Eh1(z)) * norm.pdf(z)
    #積分を実行
    v,error = quad(v, -10, 10)
    h,error = quad(h, -10, 10)
    #行列を作成
    I = np.array([[v,0],[0,h]])
    return I

def all_matrix():
    a = A(); b = B(); c = C(); d = D(); e = E()
    f = F(); g = G(); h = H(); i = I()
    return a, b, c, d, e, f, g, h, i

def HQ_HR(tol = 1e-8):
    Q = np.ones((2,2)); R = np.ones((2,2))
    HQ = np.random.normal(loc = 0, scale = 0.1, size = 4).reshape(2,2)#感受率行列の初期値を乱数で生成 ##loc:平均, scale:標準偏差
    HR = np.random.normal(loc = 0, scale = 0.1, size = 4).reshape(2,2)
    _, _, _, D, E, F, G, H, I = all_matrix()
    while True:
        Q_old = Q.copy(); R_old = R.copy()
        HQ_old = HQ.copy(); HR_old = HR.copy()
        Q = 2*D + 2*E @ HR - F @ HQ
        R = G + H @ HR - 0.5*I @ HQ
        HQ = beta**2 * T_alpha @ Q
        HR = 0.5 * beta**2 * T_alpha @ R
        if any(not np.isfinite(M).all() for M in [Q, R, HQ, HR]):
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            overflow_error = (
                f"#overflow is occuered in matmul at {now_str}"
            )
            with open(csv_file, "a") as f:
                f.write(overflow_error + "\n")
            sys.exit("#NaN or inf detected")

        if (np.max(np.abs(Q - Q_old)) < tol and np.max(np.abs(HQ - HQ_old)) < tol and np.max(np.abs(R - R_old)) < tol and np.max(np.abs(HR - HR_old)) < tol):
            break
        
    return HQ, HR

def layer_correlation():
    HQ, HR = HQ_HR()
    A, B, C, _, _, _, _, _, _ = all_matrix()
    X = hat_T_alpha @ A + hat_T_alpha @ B @ HR - 0.5*hat_T_alpha @ C @ HQ
    return X
 
#main
eps = 1e-3
b = eps
s = 1
beta_ini = 4.0000000000000000
beta_step = 0.001000000000000
#beta_step = 0.100000000000000

 
#出力ファイルを指定 or 出力ファイルから最終状態を読み込む
csv_file = "beta_decrease_stepsize0.001.txt"
last_state = load_last_state(csv_file)
if last_state is not None:
    last_alpha, last_c, last_beta, q_last, hq_last, r_last, hr_last = last_state
else:
    last_alpha = last_c = last_beta = None

#開始時刻を取得
start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("starts at ",start_time)
print("last_state :", last_state)

#実験開始
for alpha in [0.5, 1.0, 2.0]:
    T_alpha = (1/(1+alpha)) * np.array([[0,alpha],[1,0]])
    hat_T_alpha = (1/(1+alpha)) * np.array([[1,0],[0,alpha]])

    for c in [eps, -2, -5]:
        if last_alpha is None:
            beta = beta_ini
            #q = np.zeros((2, 1)); hq = np.zeros((2, 1))
            #r = np.zeros((2, 1)); hr = np.zeros((2, 1))
            if alpha == 0.5:
                if c == eps:
                    q = np.array([6.078147,0.751751]).reshape(2,1)
                    hq = np.array([4.009336,64.833565]).reshape(2,1)
                    r = np.array([7.309405,0.786967]).reshape(2,1)
                    hr = np.array([2.098580,38.983496]).reshape(2,1)
                elif c == -2:
                    q = np.array([5.955128,0.681609]).reshape(2,1)
                    hq = np.array([3.635250,63.521367]).reshape(2,1)
                    r = np.array([7.235035,0.722614]).reshape(2,1)
                    hr = np.array([1.926972,38.586853]).reshape(2,1)
                elif c == -5:
                    q = np.array([5.479261,0.562013]).reshape(2,1)
                    hq = np.array([2.997404,58.445454]).reshape(2,1)
                    r = np.array([6.831297,0.610834]).reshape(2,1)
                    hr = np.array([1.628889,36.433586]).reshape(2,1)
            elif alpha == 1.0:
                if c == eps:
                    q = np.array([10.724629,0.687582]).reshape(2,1)
                    hq = np.array([5.500655,85.797029]).reshape(2,1)
                    r = np.array([12.120945,0.723061]).reshape(2,1)
                    hr = np.array([2.892243,48.483781]).reshape(2,1)
                elif c == -2:
                    q = np.array([10.527670,0.619655]).reshape(2,1)
                    hq = np.array([4.957243,84.221358]).reshape(2,1)
                    r = np.array([11.984960,0.658880]).reshape(2,1)
                    hr = np.array([2.635519,47.939839]).reshape(2,1)
                elif c == -5:
                    q = np.array([9.684066,0.507988]).reshape(2,1)
                    hq = np.array([4.063902,77.472524]).reshape(2,1)
                    r = np.array([11.227745,0.552012]).reshape(2,1)
                    hr = np.array([2.208050,44.910981]).reshape(2,1)
            elif alpha == 2.0:
                if c == eps:
                    q = np.array([18.017263,0.632659]).reshape(2,1)
                    hq = np.array([6.748359,96.092068]).reshape(2,1)
                    r = np.array([19.651237,0.669033]).reshape(2,1)
                    hr = np.array([3.568177,52.403299]).reshape(2,1)
                elif c == -2:
                    q = np.array([17.666913,0.563668]).reshape(2,1)
                    hq = np.array([6.012461,94.223537]).reshape(2,1)
                    r = np.array([19.381084,0.602727]).reshape(2,1)
                    hr = np.array([3.214544,51.682892]).reshape(2,1)
                elif c == -5:
                    q = np.array([16.030121,0.451863]).reshape(2,1)
                    hq = np.array([4.819868,85.493979]).reshape(2,1)
                    r = np.array([17.853810,0.494206]).reshape(2,1)
                    hr = np.array([2.635764,47.610160]).reshape(2,1)


        elif alpha < last_alpha or (alpha == last_alpha and c > last_c):
            continue

        elif alpha == last_alpha and c == last_c:
            beta = last_beta
            q, hq, r, hr = q_last.copy(), hq_last.copy(), r_last.copy(), hr_last.copy()

        else:
            beta = beta_ini
            q = np.zeros((2, 1)); hq = np.zeros((2, 1))
            r = np.zeros((2, 1)); hr = np.zeros((2, 1))
    
# -----------------------------------------------------------------

        #while beta <= 4.0 + 1e-16:
            #beta += beta_step

        while beta >= 0.0 + 1e-16:
            beta -= beta_step
            beta = max(beta, 0.0) #βが負にならないようにクリップ
            
            #鞍点を計算
            start_time = time.time()
            q, hq, r, hr, var, iter_saddle = saddle_point(q, hq, r, hr, start_time, tol = 1e-6)
            
            #自由エネルギーを計算
            fn = free_energy(q, hq, r, hr)

            #層相関を計算
            X = layer_correlation()
            line = (
                f"{alpha:.1f},{c:.4f},{beta:.6f},{abs(X[0,1]):.5e},"
                f"{q[0,0]:.6f},{q[1,0]:.6f},"
                f"{hq[0,0]:.6f},{hq[1,0]:.6f},"
                f"{r[0,0]:.6f},{r[1,0]:.6f},"
                f"{hr[0,0]:.6f},{hr[1,0]:.6f},"
                f"{var:.5f},{fn:.5f},{iter_saddle}"
            )
            #print(line)
            
            with open(csv_file, "a") as f:
                f.write(line + "\n")