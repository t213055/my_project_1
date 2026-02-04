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
TIMEOUT_SECONDS = np.inf

def layer_correlation(b,c,s,chi_h,alpha,beta):
    #補助変数と秩序パラメータの初期化
    hqv = hqh = hrv = hrh = 1
    qv = qh = rv = rh = 1
    #ガウス測度を用いた積分
    def integrand_qh(z):
        if chi_h == 'I':
            return np.tanh(c + z*sqrt(hqh))**2 * norm.pdf(z)
        else:
            x = c - 0.5*hqh + hrh + z*sqrt(hqh) #係数修正後
            x = np.clip(x, -50, 50)
            return (1 / (1 + np.exp(x)))**2 * norm.pdf(z)

    def integrand_rh(z): #chi_h == "B" のときのみ使用する
        x = c - 0.5*hqh + hrh + z*sqrt(hqh) #係数修正後
        x = np.clip(x, -50, 50)
        return (1 / (1 + np.exp(x))) * norm.pdf(z)

    T_alpha = (1/(1+alpha)) * np.array([[0,alpha],[1,0]])
    hat_T_alpha = (1/(1+alpha)) * np.array([[1,0],[0,alpha]])
    Identity = np.array([[1,0],[0,1]])

    tmp_qv = tmp_qh = tmp_rv = tmp_rh = tmp_hqv = tmp_hqh = tmp_hrv = tmp_hrh = 0
    #補助変数の計算
    start_time = time.time()#update_times = 0
    timed_out_1 = "-"
    while True:
        if time.time() - start_time >= TIMEOUT_SECONDS:
            timed_out_1 = "!"
            break
        #抜けるための行列定義
        TMP_order_auxiliary = np.array([tmp_qv, tmp_qh, tmp_rv, tmp_rh, tmp_hqv, tmp_hqh, tmp_hrv, tmp_hrh])

        #更新前の値を保存(抜けるときのif文用)
        tmp_qv = qv; tmp_qh = qh; tmp_rv = rv; tmp_rh = rh; tmp_hqv = hqv; tmp_hqh = hqh; tmp_hrv = hrv; tmp_hrh = hrh
        
        #秩序パラメータの初期化、更新
        #qv = (b**2 + hqv)/(hqv + s**(-2) - hrv)**2
        #rv = (2*hqv + s**(-2) - hrv + b**2) / (2*((hqv + s**(-2) - hrv)**2))
        qv = (b**2 + hqv)/(hqv + s**(-2) - 2*hrv)**2
        rv = (2*hqv + s**(-2) - 2*hrv + b**2) / (((hqv + s**(-2) - 2*hrv)**2))
        if chi_h == "I":
            qh, error = quad( integrand_qh, -np.inf, np.inf)
            rh = 0.5 #Isingの場合はそもそもrhなんて存在しない
        else:
            qh, error = quad( integrand_qh, -np.inf, np.inf)
            rh, error = quad( integrand_rh, -np.inf, np.inf)
        
        #抜けるための行列定義
        order_auxiliary = np.array([qv, qh, rv, rh, hqv, hqh, hrv, hrh])

        #行列表示
        HQ = np.array([hqv, hqh]).reshape(2,1)
        HR = np.array([hrv, hrh]).reshape(2,1)
        Q = np.array([qv, qh]).reshape(2,1)
        R = np.array([rv, rh]).reshape(2,1)

        #秩序パラメータを用いて補助変数を更新
        HQ = beta**2 * T_alpha @ Q
        HR = 0.5 * beta**2 * T_alpha @ R
        #行列計算の結果を補助変数に返す
        hqv = HQ[0,0]; hqh = HQ[1,0]
        hrv = HR[0,0]; hrh = HR[1,0]
        
        #抜けるための処理
        #if (np.round(TMP_order_auxiliary,15) == np.round(order_auxiliary,15)).all():
        if (TMP_order_auxiliary == order_auxiliary).all(): #厳密性を高める。どれだけ時間がかかっても。
            print("^", end = "")#print(Q_R)
            break
        
        #分散が負の値になるとプログラム全体を終了（分散が正である前提でがガウス積分が収束する）
        if hqv + s**(-2) - hrv <= 0:
            print("variance is negative value")
            sys.exit()
        
        #update_times += 1
    #print("update_times =", update_times)

    #鞍点の座標を表示
    print("\nSaddle point coordinate:"); print(" qv : ", qv); print(" qh : ", qh); print(" rv : ", rv); print(" rh : ", rh)
    print("hqv : ", hqv); print("hqh : ", hqh); print("hrv : ", hrv); print("hrh : ", hrh)

""" AAA

    #####################################################
    
    #モーメント計算時、lで層(l = v, h)、mで次数(m = 1, 2, 3, 4)を指定
    def moment(l,m=0):
        ###平均 mu : ((b + z*sqrt(hqv))/(hqv + s**(-2) - hrv))
        ###分散 var : ((hqv + s**(-2) - hrv)**(-1))
        if l == "v":
            mu = lambda z: ((b + z*sqrt(hqv))/(hqv + s**(-2) - hrv))
            var = 1/((hqv + s**(-2) - hrv)) ; #print("var = ", var)
            if m == 1: #奇関数
                E = lambda z: mu(z)
                return E
            elif m == 2:
                E = lambda z: mu(z)**2 + var**2
                return E
            elif m == 3: #奇関数
                E = lambda z: mu(z)**3 + 3*mu(z)*var**2
                return E
            elif m == 4:
                E = lambda z: mu(z)**4 + 6*mu(z)**2 * var**2 + 3*var**4
                return E

        elif l == "h":
            if chi_h == "B":  #c + z*sqrt(hqh) - 0.5*(hqh - hrh)
                E = lambda z: 1 / (1 + np.exp(np.clip(-(c + z*np.sqrt(hqh) - 0.5*(hqh - hrh)), -100, 100))) #クリッピングしても積分結果は変わらないから問題ないと思う
                return E

            elif chi_h == "I":
                if m % 2 == 0:
                    E = lambda z: 1
                    return E
                else: #Isingだから隠れ層のバイアス=0で、モーメントはzの奇関数
                    E = lambda z: (np.tanh(c + z*sqrt(hqh)))# * norm.pdf(z)
                    return E


    #zの関数として扱える
    ### l = v
    Ev1 = moment("v", 1) #奇関数
    Ev2 = moment("v", 2)
    Ev3 = moment("v", 3) #奇関数
    Ev4 = moment("v", 4)
    #モーメント単位でガウス積分結果を表示
    #IE = lambda z: Ev1(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEv1 = ", result)
    #IE = lambda z: Ev2(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEv2 = ", result)
    #IE = lambda z: Ev3(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEv3 = ", result)
    #IE = lambda z: Ev4(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEv4 = ", result)

    if chi_h == "B": #すべて同じ値になる
        ### l = h, chi_h = "B"
        Eh1 = Eh2 = Eh3 = Eh4 = moment("h")
        #モーメントごとの積分結果を表示
        #IE = lambda z: Eh1(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEhB = ", result)

    elif chi_h == "I": #奇数、偶数で値が変わる
        ### l = h, chi_h = "I"
        Eh1 = Eh3 = moment("h", 1) #b,c = 0では奇関数
        Eh2 = Eh4 = moment("h", 2) #結果は1だからガウス積分しても変わらない
        #モーメント単位で積分結果を表示
        #IE = lambda z: Eh1(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEhI1 = ", result)
        #IE = lambda z: Eh2(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEhI2 = ", result)

    #指示子 omega
    def omega(l):
        if l == "h" and chi_h == "I": return 0
        else: return 1

    error_arr = []
    #モーメント行列の定義　各層のモーメントが単一の確率変数(v or h)に依存するため、全て対角行列でOK
    A_v = lambda z: (Ev2(z) - Ev1(z)**2) * norm.pdf(z)
    A_h = lambda z: (Eh2(z) - Eh1(z)**2) * norm.pdf(z)
    A_v,error = quad(A_v, -np.inf, np.inf); error_arr.append(error)
    A_h,error = quad(A_h, -np.inf, np.inf); error_arr.append(error)
    A = np.array([[A_v,0],[0,A_h]])
    #print("\nabout :A"); print(A)

    B_v = lambda z: 0.5 * (omega("v") * (Ev3(z) - Ev1(z) * Ev2(z)) - (Ev3(z) - Ev1(z) * Ev2(z)) + 2*Ev1(z) * (Ev2(z) - Ev1(z)**2)) * norm.pdf(z)
    B_h = lambda z: 0.5 * (omega("h") * (Eh3(z) - Eh1(z) * Eh2(z)) - (Eh3(z) - Eh1(z) * Eh2(z)) + 2*Eh1(z) * (Eh2(z) - Eh1(z)**2)) * norm.pdf(z)
    B_v,error = quad(B_v, -np.inf, np.inf); error_arr.append(error)
    B_h,error = quad(B_h, -np.inf, np.inf); error_arr.append(error)
    B = np.array([[B_v,0],[0,B_h]])
    #print("\nabout :B"); print(B)

    C_v = lambda z: 0.5 * omega("v") * (Ev3(z) - Ev1(z) * Ev2(z)) * norm.pdf(z)
    C_h = lambda z: 0.5 * omega("v") * (Eh3(z) - Eh1(z) * Eh2(z)) * norm.pdf(z)
    C_v,error = quad(C_v, -np.inf, np.inf); error_arr.append(error)
    C_h,error = quad(C_h, -np.inf, np.inf); error_arr.append(error)
    C = np.array([[C_v,0],[0,C_h]])
    #print("\nabout :C"); print(C)

    D_v = lambda z: 2 * Ev1(z) * (Ev2(z) - Ev1(z)**2) * norm.pdf(z)
    D_h = lambda z: 2 * Eh1(z) * (Eh2(z) - Eh1(z)**2) * norm.pdf(z)
    D_v,error = quad(D_v, -np.inf, np.inf); error_arr.append(error)
    D_h,error = quad(D_h, -np.inf, np.inf); error_arr.append(error)
    D = np.array([[D_v,0],[0,D_h]])
    #print("\nabout :D"); print(D)

    E_v = lambda z: (omega("v") * Ev1(z) * (Ev3(z) - Ev1(z) * Ev2(z)) - (Ev2(z) -Ev1(z)**2)**2 - Ev1(z) * (Ev3(z) - Ev1(z) * Ev2(z)) + 2*Ev1(z)**2 * (Ev2(z) - Ev1(z)**2)) * norm.pdf(z)
    E_h = lambda z: (omega("h") * Eh1(z) * (Eh3(z) - Eh1(z) * Eh2(z)) - (Eh2(z) -Eh1(z)**2)**2 - Eh1(z) * (Eh3(z) - Eh1(z) * Eh2(z)) + 2*Eh1(z)**2 * (Eh2(z) - Eh1(z)**2)) * norm.pdf(z)
    E_v,error = quad(E_v, -np.inf, np.inf); error_arr.append(error)
    E_h,error = quad(E_h, -np.inf, np.inf); error_arr.append(error)
    E = np.array([[E_v,0],[0,E_h]])
    #print("\nabout :E"); print(E)

    F_v = lambda z: omega("v") * Ev1(z) * (Ev3(z) - Ev1(z) * Ev2(z)) * norm.pdf(z)
    F_h = lambda z: omega("h") * Eh1(z) * (Eh3(z) - Eh1(z) * Eh2(z)) * norm.pdf(z)
    F_v,error = quad(F_v, -np.inf, np.inf); error_arr.append(error)
    F_h,error = quad(F_h, -np.inf, np.inf); error_arr.append(error)
    F = np.array([[F_v,0],[0,F_h]])
    #print("\nabout :F"); print(F)

    G_v = lambda z: (Ev3(z) - Ev1(z) * Ev2(z)) * norm.pdf(z)
    G_h = lambda z: (Eh3(z) - Eh1(z) * Eh2(z)) * norm.pdf(z)
    G_v,error = quad(G_v, -np.inf, np.inf); error_arr.append(error)
    G_h,error = quad(G_h, -np.inf, np.inf); error_arr.append(error)
    G = np.array([[G_v,0],[0,G_h]])
    #print("\nabout :G"); print(G)

    H_v = lambda z: 0.5 * (omega("v")*(Ev4(z) - Ev2(z)**2) - (Ev4(z) - Ev2(z)**2) + 2*Ev1(z)*(Ev3(z) - Ev1(z)*Ev2(z))) * norm.pdf(z)
    H_h = lambda z: 0.5 * (omega("h")*(Eh4(z) - Eh2(z)**2) - (Eh4(z) - Eh2(z)**2) + 2*Eh1(z)*(Eh3(z) - Eh1(z)*Eh2(z))) * norm.pdf(z)
    H_v,error = quad(H_v, -np.inf, np.inf); error_arr.append(error)
    H_h,error = quad(H_h, -np.inf, np.inf); error_arr.append(error)
    H = np.array([[H_v,0],[0,H_h]])
    #print("\nabout :H"); print(H)

    I_v = lambda z: 0.5 * omega("v")*(Ev4(z) - Ev2(z)**2) * norm.pdf(z)
    I_h = lambda z: 0.5 * omega("h")*(Eh4(z) - Eh2(z)**2) * norm.pdf(z)
    I_v,error = quad(I_v, -np.inf, np.inf); error_arr.append(error)
    I_h,error = quad(I_h, -np.inf, np.inf); error_arr.append(error)
    I = np.array([[I_v,0],[0,I_h]])
    #print("\nabout :I"); print(I)
    #print("error_arr", error_arr)

    #感受率行列の計算
    ##初期値を乱数で決定
    hQ = np.random.normal(loc = 0, scale = 1, size = 4).reshape(2,2)
    hR = np.random.normal(loc = 0, scale = 1, size = 4).reshape(2,2)
    ## ループを抜けるためのtmpを作成
    tmphQ = np.array([[0,0],[0,0]]); tmphR = np.array([[0,0],[0,0]])
    ## ループ回数記録用変数
    update_times = 0

    start_time = time.time()
    timed_out_2 = "-"
    while True:
        if time.time() - start_time >= TIMEOUT_SECONDS:
            timed_out_2 = "!"
            break

        Q = D - E @ hQ + F @ hR#; print(Q) #すべての要素が発散に向かう
        R = G - H @ hQ + I @ hR#; print(R) #1行目の要素が発散に向かう　2行目は0を保つ
        hQ = beta**2 * T_alpha @ Q#; hQ = np.round(hQ,15)#; print(hQ) #小数点15位以下を丸めむ
        hR = 0.5 * beta**2 * T_alpha @ R#; hR = np.round(hR,15)#; print(hR) #小数点15位以下を丸めむ 2行目が発散に向かう　1行目は0のまま
        
        ## hQ, hRの値が固定されたらループを抜ける
        if (hQ == tmphQ).all() and (hR == tmphR).all():
            #print("\nupdate_times =",update_times)
            break
        
        ## hQ, hR内でnanが現れたらループを抜ける
        if np.isnan(hQ).any() or np.isnan(hR).any(): #発散する(nan)範囲が広いことが問題なのか？
            break

        #更新前との差分を表示
        if update_times % 100000 == 0:
            print("tmphQ - hQ =\n",tmphQ - hQ); print("tmphR - hR =\n",tmphR - hR)

        #前回の計算結果を保持
        tmphQ = hQ; tmphR = hR
        update_times += 1 #; print(i, end=' ')
    #print("\nhQ = \n", hQ)
    #print("\nhR = \n", hR)

    #層相関を計算
    chi = hat_T_alpha @ A - hat_T_alpha @ B @ hQ + hat_T_alpha @ C @ hR
    chi = np.round(chi, 15)
    chi_vh = chi[0, 1]; chi_hv = chi[1, 0]
    #print("\nchi = \n", chi)

    #理論的には非対角要素が同じ値になるはず（15桁までで比較）
    if chi_vh != chi_hv:
        eq = "!"
    elif chi_vh == chi_hv:
        eq = "-"

    #条件と結果を再表示
    print(timed_out_1, timed_out_2, eq, end = ' ') #補助変数のタイムアウト、感受率のタイムアウト、非対角要素の不一致（nanのときはそもそも比較不能）
    print("beta =", beta, end = ':     ')
    print("chi_vh =", round(abs(chi_vh), 10))
    
    return np.array([beta,abs(chi_vh)])
  

eps = 10**(-3); b = eps; s = 1
#隠れ変数
for chi_h in ["B", "I"]:
    if chi_h == "B":
        print("chi_h = ", chi_h)
        #可視層と隠れ層のサイズ比
        for alpha in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:#12通り
            print("\n", "alpha = ", alpha)
            #隠れ層のバイアス（Binaryのときのみ）
            for c in [eps, -1, -2, -3, -4, -5, -6]:#7通り
                print("","c = ", c)
                if #beta_chiに値があれば、リセット
                for beta in np.arange(0.1, 10.0, 0.1):#100通り
                    beta = round(beta, 2)
                    #print("#### b = ", b, ", c = ", c, ", alpha = ", alpha, ", beta = ", beta, ", chi_h = ", chi_h, "####")
                    layer_correlation(b,c,s,chi_h,alpha,beta)
"""


""" AAA
    elif chi_h == "I":
        c = eps; print("\n", "chi_h = ", chi_h)
        for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            print("\n", "alpha = ", alpha)
            for beta in np.arange(0.8, 1.8, 0.01):
                #print("####b = ", b, ", c = ", c, ", alpha = ", alpha, ", beta = ", beta, ", chi_h = ", chi_h, "####")
                beta = round(beta, 3)
                layer_correlation(b,c,s,chi_h,alpha,beta)
""" 


""" AAA
eps = 10**(-3); b = eps; s = 1
for chi_h in ["B", "I"]:
    if chi_h == "I":
        c = eps; print("\n", "chi_h = ", chi_h)
        for alpha in [0.5]:#, 1.0, 1.5, 2.0, 2.5, 3.0]:
            print("\n", "alpha = ", alpha)
            vecs = []
            for beta in np.arange(1.26, 1.43, 0.01):
                beta_chi = layer_correlation(b,c,s,chi_h,alpha,beta)
                #vecs.append(beta_chi)
            #beta_chi_array= np.array(vecs)
            #print(beta_chi_array)
"""