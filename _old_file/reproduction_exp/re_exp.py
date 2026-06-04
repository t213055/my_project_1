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
TIMEOUT_SECONDS = 30
#警告を無視するため
import warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

#鞍点を計算する関数
def saddle_point(q, hq, tol):
    times = 0
    while True:
        q_old = q.copy(); hq_old = hq.copy()
        A_v = lambda z: np.tanh(b + z*sqrt(hq[0]))**2 * norm.pdf(z)
        A_h = lambda z: sigmoid(c + (0.5*beta**2)/(1 + alpha) - 0.5*hq[1] + z*sqrt(hq[1]))**2 * norm.pdf(z)
        q[0], error = quad(A_v, -15, 15)
        q[1], error = quad(A_h, -15, 15)

        hq = beta**2 * T_alpha @ q
        times += 1
        
        #収束判定
        if (np.linalg.norm(q - q_old) < tol and np.linalg.norm(hq - hq_old) < tol):
            #print(np.linalg.norm(q - q_old)); print(np.linalg.norm(hq - hq_old))
            break
         
    return q, hq, times

def E(l, r):
    if l=='v':
        if r==1:
            integrand = lambda z: np.tanh(b + z*sqrt(hq[0]))
            return integrand
        elif r==2:
            return 1
    
    elif l=='h':
        integrand = lambda z: sigmoid(c + z*sqrt(hq[1]) + 0.5*(beta**2/(1+alpha) - hq[1]))
        return integrand

def V():
    V = np.zeros((2, 2))
    Ev1 = E('v', 1)
    Eh1 = E('h', 1)
    V_v = lambda z: (E('v', 2) - Ev1(z)**2) * norm.pdf(z)
    V_h = lambda z: (Eh1(z) - Eh1(z)**2) * norm.pdf(z)
    V[0, 0], error = quad(V_v, -np.inf, np.inf)
    V[1, 1], error = quad(V_h, -np.inf, np.inf)
    return V

def U():
    Ev1 = E('v', 1)
    Eh1 = E('h', 1)
    U = np.zeros((2, 2))
    U_v = lambda z: Ev1(z) * (E('v', 2) - Ev1(z)**2) * norm.pdf(z)
    U_h = lambda z: Eh1(z) * (Eh1(z) - Eh1(z)**2) * norm.pdf(z)
    U[0, 0], error = quad(U_v, -np.inf, np.inf)
    U[1, 1], error = quad(U_h, -np.inf, np.inf)
    return U

def W():
    Ev1 = E('v', 1)
    Eh1 = E('h', 1)
    W = np.zeros((2, 2))
    W_v = lambda z: (E('v', 2)**2 - 4*E('v', 2)*Ev1(z)**2 + 3*Ev1(z)**4) * norm.pdf(z)
    W_h = lambda z: (Eh1(z)**2 - 4*Eh1(z)*Eh1(z)**2 + 3*Eh1(z)**4) * norm.pdf(z)
    W[0, 0], error = quad(W_v, -np.inf, np.inf)
    W[1, 1], error = quad(W_h, -np.inf, np.inf)
    return W


def HQ(tol = 1e-8):
    times = 0
    Q = np.ones((2, 2)); HQ = np.ones((2, 2))
    while True:
        Q_old = Q.copy(); HQ_old = HQ.copy()
        Q = 2*U() + W() @ HQ
        HQ = beta**2 * T_alpha @ Q
        times += 1

        #収束判定
        if (np.linalg.norm(Q - Q_old) < tol and np.linalg.norm(HQ - HQ_old) < tol):
            #print(np.linalg.norm(Q - Q_old)); print(np.linalg.norm(HQ - HQ_old))
            #print(times)
            return HQ


#層相関を計算する関数
def layer_correlation():
    return hat_T_alpha @ V() - hat_T_alpha @ U() @ HQ()


#main
eps = 1e-3
b = eps
beta = 0.0

update_times = 0

##q, hqの初期化
q = np.ones((2, 1)); hq = np.ones((2, 1))

for alpha in [0.5]:#, 1.0, 2.0]:
    T_alpha = (1/(1+alpha)) * np.array([[0,alpha],[1,0]])
    hat_T_alpha = (1/(1+alpha)) * np.array([[1,0],[0,alpha]])

    for c in [eps]:#, -2, -5]:
        beta = 0

        while beta <= 10 + 1e-12:
            times_saddle = 0
            #betaの刻み幅を調整
            beta += 0.1
            
            #鞍点を計算
            q, hq, times_saddle = saddle_point(q, hq, tol = 1e-6)

            #層相関を計算
            X = layer_correlation()
            print(f"{alpha:.1f},{c:.3f},{beta:.1f},{abs(X[0,1]):.5e}")