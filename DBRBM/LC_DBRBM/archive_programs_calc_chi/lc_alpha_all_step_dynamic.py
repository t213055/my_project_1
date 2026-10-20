import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from numpy.polynomial.hermite import hermgauss

# ==========================================
# グラフ描画・調整用パラメータ
# ==========================================
FONT_SIZE = 14
MARKER_SIZE = 4
LINE_WIDTH = 1.5
LEGEND_LOC = 'upper left'

COLORS = {0.5: 'r', 1.0: 'g', 1.5: 'm', 2.0: 'b', 2.5: 'c', 3.0: 'k'}
MARKERS = {0.5: 'o', 1.0: 's', 1.5: 'D', 2.0: '^', 2.5: 'v', 3.0: '*'}

TEXT_OFFSET_X = 0.03
TEXT_OFFSET_Y = 0.0

# ==========================================
# モデルのパラメータ
# ==========================================
eps = 0.001
b = eps
c = eps
t = 2

# 温度のスタート, ゴール
beta_init = 0.0 + 1e-16
beta_limit = 2.0

# 収束判定
tol_sp = 1e-10

# ==========================================
# 関数定義
# ==========================================
def gaussian_pdf(z):
    return (np.sqrt(2 * np.pi))**-1 * np.exp(-0.5 * z**2)

def visible_variable_array(t):
    k_start = (t // 2) + 1
    k = np.arange(t, k_start - 1, -1)
    v_k = (2 * k - t) / t
    return v_k

def SP(beta, q, q_hat, T_alpha, alpha, v_k):
    
    def integrand_q_v(z):
        B_z = b + z * np.sqrt(q_hat[0])
        N_z = np.sum(v_k * np.exp(A * v_k**2) * np.sinh(B_z * v_k))
        D_z = 0.5 * delta_term + np.sum(np.exp(A * v_k**2) * np.cosh(B_z * v_k))
        f_z = N_z / D_z
        return gaussian_pdf(z) * f_z**2

    def integrand_q_h(z):
        f_z = np.tanh(c + z * np.sqrt(q_hat[1]))
        return gaussian_pdf(z) * f_z**2

    iteration = 0
    while True:
        q_old = q.copy()
        q_hat_old = q_hat.copy()
        
        A = 0.5 * ((alpha * beta**2) / (1 + alpha) - q_hat[0])
        
        q[0], _ = quad(integrand_q_v, -12, 12)
        q[1], _ = quad(integrand_q_h, -12, 12)
        q_hat = beta**2 * T_alpha @ q
        iteration += 1
        
        if (np.all(np.abs(q - q_old) <= tol_sp) and np.all(np.abs(q_hat - q_hat_old) <= tol_sp)):
            # iteration を戻り値に追加
            return q, q_hat, A, iteration

def Effective_Susceptibility_Matrices(q_hat, A, v_k):
    v_k_row = v_k[np.newaxis, :]
    
    deg = 50
    x_i, w_i = hermgauss(deg)
    z_i = np.sqrt(2) * x_i
    weights = w_i / np.sqrt(np.pi)

    B_i = b + z_i * np.sqrt(q_hat[0])
    B_i = B_i[:, np.newaxis] 
    
    D = delta_term + 2 * np.sum(np.exp(A * v_k_row**2) * np.cosh(B_i * v_k_row), axis=1)
    N_1 =  delta_term + 2 * np.sum(v_k_row * np.exp(A * v_k_row**2) * np.sinh(B_i * v_k_row), axis=1)
    N_2 =  delta_term + 2 * np.sum(v_k_row**2 * np.exp(A * v_k_row**2) * np.cosh(B_i * v_k_row), axis=1)
    
    Ev1 = N_1 / D
    Ev2 = N_2 / D
    Eh1 = np.tanh(c + z_i * np.sqrt(q_hat[1]))
    
    V00 = np.sum((Ev2 - Ev1**2) * weights)
    V11 = np.sum((1 - Eh1**2) * weights)
    V = np.array([[V00, 0], [0, V11]])

    U00 = np.sum((Ev2 * Ev1 - Ev1**3) * weights)
    U11 = np.sum((Eh1 - Eh1**3) * weights)
    U = np.array([[U00, 0], [0, U11]])

    W00 = np.sum((Ev2**2 - 4 * Ev2 * Ev1**2 + 3 * Ev1**4) * weights)
    W11 = np.sum((1 - 4 * Eh1**2 + 3 * Eh1 **4) * weights)
    W = np.array([[W00, 0], [0, W11]])
    
    return V, U, W

def Q_HQ_solver_chi(V, U, W, beta, T_alpha, alpha):
    M = np.identity(2) - beta**2 * T_alpha @ W
    B = 2 * beta**2 * T_alpha @ U
    HQ = np.linalg.solve(M, B)
    chi = (1 + alpha)**-1 * np.array([[1, 0], [0, alpha]]) @ (V - U @ HQ)
    return chi

# ==========================================
# メイン処理 (ループ計算)
# ==========================================
delta_term = 1.0 if (t % 2 == 0) else 0.0
v_k = visible_variable_array(t)

results = {}

for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    print(f"\n--- Starting calculation for alpha = {alpha} ---")
    T_alpha = (1+alpha)**-1 * np.array([[0, alpha], [1, 0]])
    beta = beta_init
    
    q_init = np.ones(2) * 1e-16
    q_hat_init = beta**2 * T_alpha @ q_init
    
    q = q_init.copy()
    q_hat = q_hat_init.copy()
    
    beta_list = []
    chi_list = []
    
    while beta <= beta_limit:
        # SP関数から iteration を受け取る
        q, q_hat, A, iteration = SP(beta, q, q_hat, T_alpha, alpha, v_k)
        
        # iteration に応じて beta_step を動的に変更
        if iteration >= 2000:
            beta_step = 0.0001
        elif iteration >= 1000:
            beta_step = 0.0005
        elif iteration >= 500:
            beta_step = 0.001
        elif iteration >= 250:
            beta_step = 0.005
        elif iteration >= 100:
            beta_step = 0.01
        elif iteration >= 50:
            beta_step = 0.02
        else:
            beta_step = 0.03

        V, U, W = Effective_Susceptibility_Matrices(q_hat, A, v_k)
        chi = Q_HQ_solver_chi(V, U, W, beta, T_alpha, alpha)
        
        beta_list.append(beta)
        chi_list.append(np.abs(chi[0, 1]))
        
        # 動的なステップ幅と反復回数を確認できるように出力
        print(f"beta: {beta:.4f} | iteration: {iteration:4d} | next step: {beta_step}")
        
        beta += beta_step
        
    results[alpha] = (beta_list, chi_list)

# ==========================================
# グラフの描画
# ==========================================
plt.figure(figsize=(8, 6))

for alpha, (betas, chis) in results.items():
    plt.plot(betas, chis, marker=MARKERS[alpha], markersize=MARKER_SIZE, 
             linestyle='-', linewidth=LINE_WIDTH, color=COLORS[alpha], 
             label=rf'$\alpha={alpha}$')
    
    max_idx = np.argmax(chis)
    max_beta = betas[max_idx]
    max_chi = chis[max_idx]
    
    plt.plot(max_beta, max_chi, marker='*', markersize=MARKER_SIZE+6, color=COLORS[alpha])
    
    plt.text(max_beta + TEXT_OFFSET_X, max_chi + TEXT_OFFSET_Y, 
             rf'$\beta={max_beta:.4f}$', 
             color=COLORS[alpha], fontsize=FONT_SIZE-1, fontweight='bold')

plt.xlabel(r'$\beta$', fontsize=FONT_SIZE)
plt.ylabel(r'$\chi_{vh}$', fontsize=FONT_SIZE)
plt.title(f'eps = {eps}   |   t = {t}', fontsize=FONT_SIZE)
plt.legend(loc=LEGEND_LOC, fontsize=FONT_SIZE)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()