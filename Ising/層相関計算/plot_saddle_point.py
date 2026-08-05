import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. データの読み込み
# ==========================================
filename = "Ising_LC_output.txt"

# np.loadtxtを使うと、CSVデータを一気にNumPy配列として読み込めます
# skiprows=1 は、1行目のヘッダー（"beta,q_v..."など）を読み飛ばす指示です
data = np.loadtxt(filename, delimiter=",", skiprows=1)

# 列ごとにデータを分割（インデックスは0から始まります）
alpha   = data[:, 0]  # 1列目: alpha
beta    = data[:, 1]  # 2列目: 横軸
q_v     = data[:, 2]  # 3列目: 可視層 スピングラス秩序
q_h     = data[:, 3]  # 4列目: 隠れ層 スピングラス秩序
q_hat_v = data[:, 4]  # 5列目: 可視層 スピングラス補助
q_hat_h = data[:, 5]  # 6列目: 隠れ層 スピングラス補助
# 8列目(data[:, 7])の反復回数は、今回はグラフに使わないので無視します

# ==========================================
# 2. グラフの描画設定
# ==========================================
# 縦に2つのグラフを並べる設定 (figsizeで横・縦のサイズを指定)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 線のスタイル： '.-' にすることで、データ点にドットが打たれジャンプが見やすくなります
line_style = '.-'

# --- 上のグラフ：物理的な秩序パラメータ ---
ax1.plot(beta, q_v, line_style, label="q_v (Visible Spin-glass)", color="blue")
ax1.plot(beta, q_h, line_style, label="q_h (Hidden Spin-glass)", color="red")
ax1.set_ylabel("Order Parameters")
ax1.set_title("Phase Transition of GBRBM")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# --- 下のグラフ：補助変数 ---
ax2.plot(beta, q_hat_v, line_style, label="q_hat_v", color="cyan")
ax2.plot(beta, q_hat_h, line_style, label="q_hat_h", color="orange")
ax2.set_xlabel("Inverse Temperature (beta)")
ax2.set_ylabel("Auxiliary Variables")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# グラフ同士の間隔を自動調整
plt.tight_layout()

# ==========================================
# 3. 画像として保存
# ==========================================
output_img = "Ising_output_graph.png"
# dpi=300 を指定すると、論文やレポートに使える高画質な画像になります
plt.savefig(output_img, dpi=300)

print(f"グラフを {output_img} として保存しました！")

# 画面にもグラフを表示したい場合は以下のコメントを外してください
# plt.show()