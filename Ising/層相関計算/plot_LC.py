#calc_LC.pyの出力を受け取り、層相関を計算するスクリプト
import numpy as np
import matplotlib.pyplot as plt

filename = 'Ising_output.txt'

# names=True にすると1行目をヘッダーとして読み込みます
data = np.genfromtxt(filename, delimiter=',', names=True)

# ==========================================
# chi_vhが最大となるbetaを見つける処理
# ==========================================
# np.argmax は、配列の中で最も大きい値が入っている「場所(インデックス)」を返します
max_idx = np.argmax(data['chi_vh'])
max_beta = data['beta'][max_idx]
max_chi = data['chi_vh'][max_idx]

# 実行画面（コンソール）に結果を表示
print(f"★ chi_vhが最大となる点 ★")
print(f"  beta: {max_beta}")
print(f"  chi_vh: {max_chi}")

# ==========================================
# グラフの描画
# ==========================================
plt.figure(figsize=(8, 6))

# 通常のデータプロット（青色）
plt.plot(data['beta'], data['chi_vh'], marker='.', linestyle='-', color='b', label=r'$\chi_{vh}$')

# 最大値の箇所に赤い縦線を引き、星マークを付ける
plt.axvline(x=max_beta, color='red', linestyle='--', alpha=0.6, label=f'Peak at $\\beta={max_beta}$')
plt.plot(max_beta, max_chi, marker='*', color='red', markersize=12)

# 0列目(beta)と7列目のプロット（赤色）を表示する場合は以下をアンコメント
# plt.plot(data['beta'], data['chi_hv'], marker='^', linestyle='--', color='r', label=r'$\chi_{hv}$')

plt.xlabel(r'$\beta$')
plt.ylabel('Values')
plt.grid(True)
plt.legend()

# 画像ファイルとして保存
plt.savefig('chi_plot.png', dpi=300, bbox_inches='tight')
plt.show()