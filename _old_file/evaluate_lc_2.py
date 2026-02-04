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


def layer_correlation(b,c,s,chi_h,alpha,beta,SAD): #SAD：鞍点

    T_alpha = (1/(1+alpha)) * np.array([[0,alpha],[1,0]])
    hat_T_alpha = (1/(1+alpha)) * np.array([[1,0],[0,alpha]])
    Identity = np.array([[1,0],[0,1]])

    #補助変数と秩序パラメータの初期化
    q = SAD[0:2, :]  #q[0,0] = qv, q[1,0] = qh
    r = SAD[2:4, :]  #r[0,0] = rv, r[1,0] = rh
    hq = SAD[4:6, :] #hq[0,0] = hqv, hq[1,0] = hqh
    hr = SAD[6:8, :] #hr[0,0] = hrv,  hr[1,0] = hrh

    #ガウス測度を用いた積分
    def integrand_qh(z):
        if chi_h == 'I':
            return np.tanh(c + z*sqrt(hr[1]))**2 * norm.pdf(z)
        else:
            x = c - 0.5*hq[1] + hr[1] + z*sqrt(hq[1]) #係数修正後
            x = np.clip(x, -50, 50)
            return (1 / (1 + np.exp(x)))**2 * norm.pdf(z)

    def integrand_rh(z): #chi_h == "B" のときのみ使用する
        x = c - 0.5*hq[1] + hr[1] + z*sqrt(hq[1]) #係数修正後
        x = np.clip(x, -50, 50)
        return (1 / (1 + np.exp(x))) * norm.pdf(z)

    def free_energy(q, r, hq, hr):#divergent 発散, roundoff error is detected 丸め誤差が検出され、許容誤差を達成できない, algorithm does not converge 丸め誤差が支配
        F_1_1 = ((alpha*beta**2)/(2*(1+alpha)))*(q[0]*q[1]) - ((alpha*beta**2)/(2*(1+alpha)))*(r[0]*r[1])
        F_1_2 = -(1/(2*(1+alpha)))*(q[0]*hq[0]) - (alpha/(2*(1+alpha)))*(q[1]*hq[1])
        F_1_3 = (1/(1+alpha))*(r[0]*hr[0]) + (alpha/(1+alpha))*(r[1]*hr[1])
        F_1 = F_1_1 - F_1_2 - F_1_3
        F_1 = F_1.item()
        return F_1
        
        gaussian = lambda z: (((b + z*sqrt(hq[0]))**2)/(hq[0] + s**(-2) - 2*hr[0]) - np.log(hq[0] + s**(-2) - 2*hr[0]) + np.log(2*np.pi)) * norm.pdf(z)
        gaussian_int_Z, error = quad(gaussian, -8, 8) #積分範囲を小さくする
        F_2 = (1/(2*(1+alpha))) * gaussian_int_Z
        
        sum_h = lambda z: (np.log(1 + np.exp(c + 0.5*hq[1] + hr[1] + z*sqrt(hq[1])))) * norm.pdf(z)
        sum_h_int_Z, error = quad(sum_h, -10, 10) #積分範囲を小さくする
        F_3 = (1/(1+alpha)) * sum_h_int_Z
        return F_1 + F_2 + F_3


    #タイムアウト用変数
    start_time = time.time()
    update_times_saddle_point = 0
    timed_out_1 = "-"

    #計算開始
    while True:
        if time.time() - start_time >= TIMEOUT_SECONDS:
            timed_out_1 = "!"
            break

        #更新前の q, r, hq, hr の値を保存
        TMP = np.vstack([q, r, hq, hr])
        #更新前の自由エネルギーを保存
        FN_TMP = free_energy(q, r, hq, hr)

        #秩序パラメータの初期化、更新
        q[0] = (b**2 + hq[0])/(hq[0] + s**(-2) - 2*hr[0])**2
        r[0] = (2*hq[0] + s**(-2) - 2*hr[0] + b**2) / (((hq[0] + s**(-2) - 2*hr[0])**2))
        if chi_h == "B":
            q[1], error = quad( integrand_qh, -50, 50) #積分範囲を小さくする
            r[1], error = quad( integrand_rh, -50, 50) #積分範囲を小さくする
        #else:
        #    q[1], error = quad( integrand_qh, -np.inf, np.inf)
        #    r[1] = 0.5 #Isingの場合はそもそもrhなんて存在しない
        
        #秩序パラメータを用いて補助変数を更新
        hq = beta**2 * T_alpha @ q
        hr = 0.5 * beta**2 * T_alpha @ r
    
        #print(free_energy(q, r, hq, hr))
        FN = free_energy(q, r, hq, hr)
        diff1 = FN_TMP - FN 
        diff2 = TMP - np.vstack([q, r, hq, hr])
        if (abs(diff1) < 1e-8) and (abs(diff2) < 1e-6).all(): #自由エネルギーの差分　鞍点の差分　を収束条件に
            #print("更新前 :", FN_TMP, "更新後 :", FN)
            #print("diff1 :",diff1)
            break
        #elif (abs(diff2) < 1e-6).all():
        #    print("diff2 :\n",diff2)
        #    break
        
        update_times_saddle_point += 1

    #分散 > 0 をチェック
    if hq[0] + s**(-2) - 2*hr[0] <= 0:
        print("variance is negative value", end=': ')
        print(hq[0] + s**(-2) - 2*hr[0])
        sys.exit()
    #鞍点の値を表示
    #print("\nSaddle point coordinate:"); print("q:\n", q); print("r:\n", r), print("hq:\n", hq), print("hr:\n", hr)
    #print("update_times(saddle point) =", update_times_saddle_point)


    #####################################################
    #####################################################


    #モーメント計算時、lで層(l = v, h)、mで次数(m = 1, 2, 3, 4)を指定
    def E(l,m=0):
        ###平均 mu : ((b + z*sqrt(hq[0]))/(hq[0] + s**(-2) - 2*hr[0]))
        ###分散 var : ((hq[0] + s**(-2) - 2*hr[0])**(-1))
        if l == "v":
            mu = lambda z: ((b + z*sqrt(hq[0]))/(hq[0] + s**(-2) - 2*hr[0]))
            var = 1/((hq[0] + s**(-2) - 2*hr[0]))#; print("var = ", var)
            if m == 1: #奇関数
                return lambda z: mu(z)
            elif m == 2:
                return lambda z: mu(z)**2 + var
            elif m == 3: #奇関数
                return lambda z: mu(z)**3 + 3*mu(z)*var
            elif m == 4:
                return lambda z: mu(z)**4 + 6*mu(z)**2 * var + 3*var**2

        elif l == "h":
            if chi_h == "B":  #c + z*sqrt(hq[1]) - 0.5*(hq[1] - hr[1])
                return lambda z: 1 / (1 + np.exp(np.clip(-(c + z*np.sqrt(hq[1]) - 0.5*(hq[1] - 2*hr[1])), -100, 100))) #クリッピングしても積分結果は変わらないから問題ないと思う

            """
            elif chi_h == "I":
                if m % 2 == 0:
                    E = lambda z: 1
                    return E
                else: #Isingだから隠れ層のバイアス=0で、モーメントはzの奇関数
                    E = lambda z: (np.tanh(c + z*sqrt(hq[1])))# * norm.pdf(z)
                    return E
            """

    #zの関数として扱える
    ### l = v
    Ev1 = E("v", 1) #奇関数
    Ev2 = E("v", 2)
    Ev3 = E("v", 3) #奇関数
    Ev4 = E("v", 4)
    #モーメント単位でガウス積分結果を表示
    #IE = lambda z: Ev1(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEv1 = ", result)
    #IE = lambda z: Ev2(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEv2 = ", result)
    #IE = lambda z: Ev3(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEv3 = ", result)
    #IE = lambda z: Ev4(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEv4 = ", result)

    if chi_h == "B": #すべて同じ値になる
        ### l = h, chi_h = "B"
        Eh1 = Eh2 = Eh3 = Eh4 = E("h")
        #モーメントごとの積分結果を表示
        #IE = lambda z: Eh1(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEhB = ", result)
    """
    elif chi_h == "I": #奇数、偶数で値が変わる
        ### l = h, chi_h = "I"
        Eh1 = Eh3 = E("h", 1) #b,c = 0では奇関数
        Eh2 = Eh4 = E("h", 2) #結果は1だからガウス積分しても変わらない
        #モーメント単位で積分結果を表示
        #IE = lambda z: Eh1(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEhI1 = ", result)
        #IE = lambda z: Eh2(z) * norm.pdf(z); result, error = quad(IE, -np.inf, np.inf); print("IEhI2 = ", result)
    """

    error_arr = []
    #モーメント行列の定義 各層のモーメントが単一の確率変数(v or h)に依存するため、全て対角行列でOK
    A_v = lambda z: (Ev2(z) - Ev1(z)**2) * norm.pdf(z)
    A_h = lambda z: (Eh2(z) - Eh1(z)**2) * norm.pdf(z)
    A_v,error = quad(A_v, -np.inf, np.inf); error_arr.append(error)
    A_h,error = quad(A_h, -np.inf, np.inf); error_arr.append(error)
    A = np.array([[A_v,0],[0,A_h]])
    #print("\nabout :A"); print(A)

    B_v = lambda z: (Ev3(z) - Ev1(z)*Ev2(z)) * norm.pdf(z)
    B_h = lambda z: (Ev2(z) - Ev1(z)*Ev1(z)) * norm.pdf(z)
    B_v,error = quad(B_v, -np.inf, np.inf); error_arr.append(error)
    B_h,error = quad(B_h, -np.inf, np.inf); error_arr.append(error)
    B = np.array([[B_v,0],[0,B_h]])

    C_v = lambda z: (Ev3(z) - Ev1(z)*Ev2(z) - Ev3(z) + 3*Ev1(z)*Ev2(z) - 2*Ev1(z)**3) * norm.pdf(z)
    C_h = lambda z: (Eh2(z) - Eh1(z)*Eh1(z) - Eh3(z) + 3*Eh1(z)*Eh2(z) - 2*Eh1(z)**3) * norm.pdf(z)
    C_v,error = quad(C_v, -np.inf, np.inf); error_arr.append(error)
    C_h,error = quad(C_h, -np.inf, np.inf); error_arr.append(error)
    C = np.array([[C_v,0],[0,C_h]])

    D_v = lambda z: (Ev1(z)*(Ev2(z) - Ev1(z)**2)) * norm.pdf(z)
    D_h = lambda z: (Eh1(z)*(Eh2(z) - Eh1(z)**2)) * norm.pdf(z)
    D_v,error = quad(D_v, -np.inf, np.inf); error_arr.append(error)
    D_h,error = quad(D_h, -np.inf, np.inf); error_arr.append(error)
    D = np.array([[D_v,0],[0,D_h]])

    E_v = lambda z: (Ev1(z)*(Ev3(z) - Ev1(z)*Ev2(z))) * norm.pdf(z)
    E_h = lambda z: (Eh1(z)*(Eh2(z) - Eh1(z)*Eh1(z))) * norm.pdf(z)
    E_v,error = quad(E_v, -np.inf, np.inf); error_arr.append(error)
    E_h,error = quad(E_h, -np.inf, np.inf); error_arr.append(error)
    E = np.array([[E_v,0],[0,E_h]])

    F_v = lambda z: (Ev1(z)*Ev3(z) - Ev1(z)**2*Ev2(z) - Ev2(z)**2 + 5*Ev1(z)**2*Ev2(z) - Ev1(z)*Ev3(z) - 3*Ev1(z)**4) * norm.pdf(z)
    F_h = lambda z: (Eh1(z)*Eh2(z) - Eh1(z)**2*Eh1(z) - Eh2(z)**2 + 5*Eh1(z)**2*Eh2(z) - Eh1(z)*Eh3(z) - 3*Eh1(z)**4) * norm.pdf(z)
    F_v,error = quad(F_v, -np.inf, np.inf); error_arr.append(error)
    F_h,error = quad(F_h, -np.inf, np.inf); error_arr.append(error)
    F = np.array([[F_v,0],[0,F_h]])

    G_v = lambda z: (Ev3(z) - Ev1(z)*Ev2(z)) * norm.pdf(z)
    G_h = lambda z: (Eh3(z) - Eh1(z)*Eh2(z)) * norm.pdf(z)
    G_v,error = quad(G_v, -np.inf, np.inf); error_arr.append(error)
    G_h,error = quad(G_h, -np.inf, np.inf); error_arr.append(error)
    G = np.array([[G_v,0],[0,G_h]])

    H_v = lambda z: (Ev4(z) - Ev2(z)*Ev2(z)) * norm.pdf(z)
    H_h = lambda z: (Eh3(z) - Eh1(z)*Eh2(z)) * norm.pdf(z)
    H_v,error = quad(H_v, -np.inf, np.inf); error_arr.append(error)
    H_h,error = quad(H_h, -np.inf, np.inf); error_arr.append(error)
    H = np.array([[H_v,0],[0,H_h]])

    I_v = lambda z: (Ev4(z) - Ev2(z)*Ev2(z) - Ev4(z) + 2*Ev1(z)*Ev3(z) - 2*Ev1(z)**2*Ev2(z) + Ev2(z)**2) * norm.pdf(z)
    I_h = lambda z: (Eh3(z) - Eh1(z)*Eh2(z) - Eh4(z) + 2*Eh1(z)*Eh3(z) - 2*Eh1(z)**2*Eh2(z) + Eh2(z)**2) * norm.pdf(z)
    I_v,error = quad(I_v, -np.inf, np.inf); error_arr.append(error)
    I_h,error = quad(I_h, -np.inf, np.inf); error_arr.append(error)
    I = np.array([[I_v,0],[0,I_h]])

    #感受率行列の計算
    HQ = np.random.normal(loc = 0, scale = 0.1, size = 4).reshape(2,2)#感受率行列の初期値を乱数で生成 ##loc:平均, scale:標準偏差
    HR = np.random.normal(loc = 0, scale = 0.1, size = 4).reshape(2,2)
    ## ループを抜けるためのtmpを作成
    tmpHQ = np.array([[0,0],[0,0]]); tmpHR = np.array([[0,0],[0,0]])
    ## ループ回数記録用変数
    update_times_matrix = 0

    start_time = time.time()
    timed_out_2 = "-"
    while True:
        if time.time() - start_time >= TIMEOUT_SECONDS:
            timed_out_2 = "!"
            break

        Q = 2*D - 2*E @ HR - F @ HQ#; print(Q) #すべての要素が発散に向かう
        R = G + H @ HR - 0.5*I @ HQ#; print(R) #1行目の要素が発散に向かう　2行目は0を保つ
        HQ = beta**2 * T_alpha @ Q#; HQ = np.round(HQ,15)#; print(HQ) #小数点15位以下を丸めむ
        HR = 0.5 * beta**2 * T_alpha @ R#; HR = np.round(HR,15)#; print(HR) #小数点15位以下を丸めむ 2行目が発散に向かう　1行目は0のまま
        
        ## HQ, HR内でnanが現れたらループを抜ける
        if not np.isfinite(HQ).all() or not np.isfinite(HR).all(): #発散する(nan)範囲が広いことが問題なのか？
            #print("Q\n",Q); print("R\n",R)
            #print("HQ\n",HQ); print("HR\n",HR)
            #sys.exit()
            break
        
        ## HQ, HRの値が固定されたら（差分 < 10**-5 で）ループを抜ける
        if ((tmpHQ - HQ) < 1e-5).all() and ((tmpHR - HR) < 1e-5).all():
            break
        
        #更新前との差分を表示
        #if update_times_matrix % 100000 == 99999:
        #    print("tmpHQ - HQ =\n",tmpHQ - HQ); print("tmpHR - HR =\n",tmpHR - HR)

        #HQ, HRの計算結果を保持
        tmpHQ = HQ; tmpHR = HR
        update_times_matrix += 1
    #print("\nHQ = \n", HQ)
    #print("\nHR = \n", HR)

    #層相関を計算
    chi = hat_T_alpha @ A - hat_T_alpha @ B @ HR - 0.5 * hat_T_alpha @ C @ HQ
    chi_vh = chi[0, 1]; chi_hv = chi[1, 0]

    #条件と結果を再表示
    print(f"{chi_h},{alpha:.1f},{c:.3f},{beta:.1f},{abs(chi_vh):.5e},{timed_out_1},{update_times_saddle_point},{timed_out_2},{update_times_matrix}")
    return np.vstack([q, r, hq, hr])

eps = 10**(-3); b = eps; s = 1
#隠れ変数
for chi_h in ["B", "I"]:
    if chi_h == "B":
        #可視層と隠れ層のサイズ比
        #for alpha in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:#12通り
        for alpha in [0.5]:#, 1.0, 2.0]:
            #隠れ層のバイアス（Binaryのときのみ）
            for c in [eps]:#, -2, -5]:#7通り
                saddle = 2 * np.ones((8,1)) #鞍点の初期値を生成（
                #saddle = np.random.normal(loc = 0, scale = 0.1, size =8).reshape(8,1) #分散が負にならない条件を付けて乱数で生成するべきか？） 小さい値からスタートすると分散が小さくなりすぎて問題が発生する？
                for beta in np.arange(1.9, 4.0, 0.1):#100通り
                    beta = round(beta, 2)
                    saddle = layer_correlation(b,c,s,chi_h,alpha,beta,saddle)
                
                #beta = 0.1
                #while beta <= 10 + 1e-12:
                #    if update_times_saddle_point <= 30:
                #        beta += 0.1
                #    elif :
                #        beta += 0.01
                #    elif :
                #        beta += 0.001


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