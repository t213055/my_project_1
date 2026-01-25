#.\venv\Scripts\activate
#で仮想環境をアクティベート
import numpy as np
from numpy import exp, sqrt, tanh

from scipy.special import ndtr       # 標準正規分布の積分
from scipy.optimize import root      # 非線形方程式の解法
from scipy.stats import norm
from scipy.integrate import quad

#sys.exit()でプログラム全体を終了させる
import sys
#while文が無限ループになるのを指定時間で強制終了
import time
TIMEOUT_SECONDS = 300

#警告を無視するため
#import warnings
#warnings.filterwarnings(
#    "ignore",
#    category=RuntimeWarning
#)

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
        #分散が正かを判定
        if var <= 0:
            print("#variance is negative value", end=':')
            print(var)
            #sys.exit()

        #収束判定
        if (np.max(np.abs(q - q_old)) < tol and np.max(np.abs(hq - hq_old)) < tol and np.max(np.abs(r - r_old)) < tol and np.max(np.abs(hr - hr_old)) < tol):
            #print(np.linalg.norm(q - q_old)); print(np.linalg.norm(hq - hq_old)); print(np.linalg.norm(r - r_old)); print(np.linalg.norm(hr - hr_old))
            break

        #時間経過でやり直し or exit
        if time.time() - start_time >= TIMEOUT_SECONDS:
            print(f"Timeout at beta={beta}, iter={iter}, {np.max(np.abs(q - q_old))}, {np.max(np.abs(hq - hq_old))}, {np.max(np.abs(r - r_old))}, {np.max(np.abs(hr - hr_old))}, ")
            sys.exit()

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
            sys.exit("NaN or inf detected")

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
beta = 0.0
beta_step = 0.0001

q = 2 * np.ones((2, 1)); hq = 2 * np.ones((2, 1))
r = 2 * np.ones((2, 1)); hr = 2 * np.ones((2, 1))


for alpha in [0.5, 1.0, 2.0, 2.5]:
    T_alpha = (1/(1+alpha)) * np.array([[0,alpha],[1,0]])
    hat_T_alpha = (1/(1+alpha)) * np.array([[1,0],[0,alpha]])

    for c in [eps, -2, -5]:
        beta = 0.0

        while beta <= 6.0 + 1e-16:
            times_saddle = 0

            #次のbetaへ
            beta += beta_step

            #鞍点を計算
            start_time = time.time()
            q, hq, r, hr, var, times_saddle = saddle_point(q, hq, r, hr, start_time, tol = 1e-6)
            
            #自由エネルギーを計算
            fn = free_energy(q, hq, r, hr)

            #層相関を計算
            X = layer_correlation()
            print(f"{alpha:.1f},{c:.4f},{beta:.6f},{abs(X[0,1]):.5e},{q[0,0]:.6f},{q[1,0]:.6f},{hq[0,0]:.6f},{hq[1,0]:.6f},{r[0,0]:.6f},{r[1,0]:.6f},{hr[0,0]:.6f},{hr[1,0]:.6f},{var:.5f},{fn:.5f}")