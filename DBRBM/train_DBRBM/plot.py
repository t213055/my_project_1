import pandas as pd
import matplotlib.pyplot as plt

# 1行目のCONFIG設定行をスキップして、2行目をヘッダーとしてCSVを読み込む
df = pd.read_csv('results_toy_1.csv', skiprows=1)

# x軸のデータ（1列目: 'epoch'）を取得
x = df['epoch']

# グラフを描画するための設定 (3つのalphaについてループで処理)
alpha_values = [0.5, 1.0, 2.0]

# グラフの描画設定
linewidth = 5
labelsize = 20

for i, alpha in enumerate(alpha_values):
    # 各alphaに対応する列のインデックスを計算
    # i=0 (alpha=0.5): 列1, 2, 3 (0始まりのインデックスを使用)
    # i=1 (alpha=1.0): 列4, 5, 6
    # i=2 (alpha=2.0): 列7, 8, 9
    col_start = i * 3 + 1
    col_end = col_start + 3
    
    # 描画対象の列名を取得
    y_cols = df.columns[col_start:col_end]
    
    # 新しいキャンバス（画像）を作成
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 対象の3列分、グラフに線を重ねて描画
    for col in y_cols:
        ax.plot(x, df[col], label=col, linewidth=linewidth)
        
    # グラフの体裁を整える
    ax.set_title(f'alpha={alpha}', fontsize=14, fontname="Meiryo")
    # ax.set_xlabel('Epoch', fontsize=12)
    # ax.set_ylabel('Log-Likelihood', fontsize=12)
    ax.legend(title='Beta condition')
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    # ax.grid(True, linestyle='--', alpha=0.7)
    
    # グラフを保存して表示
    filename = f'graph_alpha_{alpha}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"{filename} を保存しました。")
    
    plt.show(block=False)

# すべてのグラフを描画した後、ウィンドウが閉じられるまでプログラムを待機させる
plt.show()