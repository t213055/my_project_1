import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ==========================================
# グラフ描画・調整用パラメータ (スクリプト冒頭で調整可能)
# ==========================================
TITLE_FONT_SIZE = 16       # グラフタイトルのフォントサイズ
LEGEND_FONT_SIZE = 12      # 凡例のフォントサイズ
LEGEND_LOC = 'best'        # 凡例の位置 ('upper right', 'lower right', 'best' など)
LINE_WIDTH = 2.0           # 線の太さ
MARKER_SIZE = 2            # マーカーのサイズ (細かいデータが多いため少し小さめを推奨)

# 各alphaや変数のプロット色設定
COLORS = ['r', 'g', 'b', 'm', 'c', 'k', 'y', 'orange']

# ==========================================
# ファイル選択関数
# ==========================================
def select_file():
    # カレントディレクトリの .txt ファイルを取得
    txt_files = glob.glob('*.txt')
    if not txt_files:
        print("エラー: カレントディレクトリに .txt ファイルが見つかりません。")
        return None

    print("=== 入力ファイルを選択してください ===")
    for i, file in enumerate(txt_files):
        print(f"{i + 1}: {file}")
    
    while True:
        try:
            choice = int(input("ファイル番号を入力: "))
            if 1 <= choice <= len(txt_files):
                return txt_files[choice - 1]
            else:
                print("無効な番号です。")
        except ValueError:
            print("数字を入力してください。")

# ==========================================
# ファイル名から A(日時), B(esp), C(t) を抽出
# ==========================================
def parse_filename(filename):
    # 例: 202601011505_esp=1e-3_t=2.txt
    basename = os.path.basename(filename)
    # 拡張子を除去
    name_without_ext = os.path.splitext(basename)[0]
    
    # 3つのパーツに分ける想定
    parts = name_without_ext.split('_')
    if len(parts) >= 3:
        A = parts[0]
        B = parts[1]
        C = parts[2]
    else:
        # 想定外のフォーマットの場合
        A = "UnknownTime"
        B = "esp=Unknown"
        C = "t=Unknown"
    
    return A, B, C

# ==========================================
# モード1: 層相関のグラフ (chi_A.png)
# ==========================================
def plot_chi(df, A, B, C):
    plt.figure(figsize=(8, 6))
    
    alphas = sorted(df['alpha'].unique())
    for i, alpha in enumerate(alphas):
        df_alpha = df[df['alpha'] == alpha]
        
        betas = df_alpha['beta'].values
        chis = df_alpha['abs(chi[0, 1])'].values
        
        plt.plot(betas, chis, marker='o', markersize=MARKER_SIZE, 
                 linestyle='-', linewidth=LINE_WIDTH, color=COLORS[i % len(COLORS)], 
                 label=rf'$\alpha={alpha}$')
        
        # 最大値のプロット
        max_idx = np.argmax(chis)
        max_beta = betas[max_idx]
        max_chi = chis[max_idx]
        plt.plot(max_beta, max_chi, marker='*', markersize=MARKER_SIZE+8, color=COLORS[i % len(COLORS)])
        plt.text(max_beta + 0.02, max_chi, rf'$\beta={max_beta:.4f}$', 
                 color=COLORS[i % len(COLORS)], fontsize=10, fontweight='bold')

    plt.xlabel(r'$\beta$', fontsize=14)
    plt.ylabel(r'$|\chi_{vh}|$', fontsize=14)
    plt.title(f"{B}   |   {C}", fontsize=TITLE_FONT_SIZE)
    plt.legend(loc=LEGEND_LOC, fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    out_name = f"chi_{A}.svg"
    plt.savefig(out_name, format="svg")
    print(f"\nグラフを保存しました: {out_name}")
    plt.show()

# ==========================================
# モード2: 鞍点の推移のグラフ (sp_A.svg)
# ==========================================
def plot_saddle_points(df, A, B, C):
    alphas = sorted(df['alpha'].unique())
    num_alphas = len(alphas)
    
    # 画像①・②の要件: 行方向(q, q_hat) = 2, 列方向 = alphaの種類数
    fig, axes = plt.subplots(nrows=2, ncols=num_alphas, figsize=(5 * num_alphas, 8), sharex=True)
    
    # alphaが1種類しかない場合のaxesの次元調整
    if num_alphas == 1:
        axes = axes.reshape(2, 1)

    # グラフ全体の上部にタイトルを表示
    fig.suptitle(f"{B}   |   {C}", fontsize=TITLE_FONT_SIZE, fontweight='bold')

    for j, alpha in enumerate(alphas):
        df_alpha = df[df['alpha'] == alpha]
        betas = df_alpha['beta'].values
        
        q0 = df_alpha['q[0]'].values
        q1 = df_alpha['q[1]'].values
        qhat0 = df_alpha['q_hat[0]'].values
        qhat1 = df_alpha['q_hat[1]'].values
        chis = df_alpha['abs(chi[0, 1])'].values
        
        # chiが最大となるbetaの値を特定
        max_idx = np.argmax(chis)
        beta_max = betas[max_idx]

        # -----------------------------------
        # 上段 (qのグラフ)
        # -----------------------------------
        ax_q = axes[0, j]
        ax_q.plot(betas, q0, marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color='b', label='q[0] (Visible)')
        ax_q.plot(betas, q1, marker='s', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color='r', label='q[1] (Hidden)')
        
        # 秩序パラメータが立ち上がる場所（= chi最大点）に縦線を引く (画像②の要件)
        ax_q.axvline(x=beta_max, color='k', linestyle='--', alpha=0.6, label=rf'$\chi$ max ($\beta={beta_max:.4f}$)')
        
        ax_q.set_title(rf'$\alpha = {alpha}$', fontsize=14)
        ax_q.set_ylabel('q', fontsize=14)
        ax_q.grid(True, linestyle='--', alpha=0.7)
        if j == 0: # 最初の列のみ凡例を表示してすっきりさせる
            ax_q.legend(loc=LEGEND_LOC, fontsize=LEGEND_FONT_SIZE)

        # -----------------------------------
        # 下段 (q_hatのグラフ)
        # -----------------------------------
        ax_qhat = axes[1, j]
        ax_qhat.plot(betas, qhat0, marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color='b', label='q_hat[0]')
        ax_qhat.plot(betas, qhat1, marker='s', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color='r', label='q_hat[1]')
        
        # 縦線を引く
        ax_qhat.axvline(x=beta_max, color='k', linestyle='--', alpha=0.6)
        
        ax_qhat.set_xlabel(r'$\beta$', fontsize=14)
        ax_qhat.set_ylabel(r'$\hat{q}$', fontsize=14)
        ax_qhat.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    # メインタイトルとサブプロットが被らないように調整
    fig.subplots_adjust(top=0.9)
    
    out_name = f"sp_{A}.svg"
    plt.savefig(out_name, format="svg")
    print(f"\nグラフを保存しました: {out_name}")
    plt.show()

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    target_file = select_file()
    
    if target_file:
        print(f"\n読み込み中: {target_file}")
        # CSVファイルの読み込み
        df = pd.read_csv(target_file)
        
        # ファイル名からA, B, Cを取得
        A, B, C = parse_filename(target_file)
        
        print("\n=== 出力するグラフを選択してください ===")
        print("1: 層相関のグラフ (chi_A.svg)")
        print("2: 鞍点の推移のグラフ (sp_A.svg)")
        
        while True:
            mode = input("選択 (1 または 2): ").strip()
            if mode == '1':
                plot_chi(df, A, B, C)
                break
            elif mode == '2':
                plot_saddle_points(df, A, B, C)
                break
            else:
                print("1 または 2 を入力してください。")