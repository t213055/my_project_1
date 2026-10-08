import numpy as np
import datetime
import csv
from calc_lc import visible_variable_array, SP, Effective_Susceptibility_Matrices, Q_HQ_solver_chi
from setting import eps, b, c, t, beta_init, beta_limit, tol_sp, alpha_list

# ==========================================
# メイン処理
# ==========================================
delta_term = 1.0 if (t % 2 == 0) else 0.0
v_k = visible_variable_array(t)

# ------------------------------------------
# CSVファイル名と書き込み準備
# ------------------------------------------
# ファイル名要素A: 現在日時(12桁)
time_str = datetime.datetime.now().strftime('%Y%m%d%H%M')

# ファイル名要素B: espの値 (例: 0.001 -> 1e-3 にフォーマット)
s_e = f"{eps:e}" 
parts = s_e.split('e')
mantissa = parts[0].rstrip('0').rstrip('.')
exponent = str(int(parts[1]))
esp_str = f"esp={mantissa}e{exponent}"

# ファイル名の結合
filename = f"{time_str}_{esp_str}_t={t}.txt"

print(f"{filename} へ計算結果を書き込みます...")

# CSV(txt)の書き込み準備
with open(filename, 'w', newline='') as f_csv:
    writer = csv.writer(f_csv)
    # ヘッダーを出力
    writer.writerow(["alpha", "beta", "q[0]", "q[1]", "q_hat[0]", "q_hat[1]", "abs(chi[0, 1])", "iteration"])

    for alpha in alpha_list:
        print(f"\n--- Starting calculation for alpha = {alpha} ---")
        T_alpha = (1+alpha)**-1 * np.array([[0, alpha], [1, 0]])
        beta = beta_init
        
        q_init = np.ones(2) * 1e-16
        q_hat_init = beta**2 * T_alpha @ q_init
        
        q = q_init.copy()
        q_hat = q_hat_init.copy()
        
        while beta <= beta_limit:
            q, q_hat, A, iteration = SP(beta, q, q_hat, T_alpha, alpha, v_k, b, c, delta_term, tol_sp)
            
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

            V, U, W = Effective_Susceptibility_Matrices(q_hat, A, v_k, b, c, delta_term)
            chi = Q_HQ_solver_chi(V, U, W, beta, T_alpha, alpha)
            
            chi_abs = np.abs(chi[0, 1])
            
            # csvファイル(txt)に1行記録
            writer.writerow([alpha, beta, q[0], q[1], q_hat[0], q_hat[1], chi_abs, iteration])
            
            print(f"beta: {beta:.4f} | iteration: {iteration:4d} | next step: {beta_step}")
            
            beta += beta_step

print(f"\nすべての計算が完了し、データを {filename} に保存しました。")
