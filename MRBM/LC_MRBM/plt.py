import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from adjustText import adjust_text

# ==========================================
# グラフ描画・調整用パラメータ (スクリプト冒頭で調整可能)
# ==========================================
# タイトル表示フラグ (True: つける, False: つけない)
SHOW_TITLE = False

# 表示する alpha のリスト (ここで設定した alpha のみがグラフ化される)
DISPLAY_ALPHAS = [0.5, 1.0, 2.0]

# 描画範囲の設定 (ここで設定した範囲のみをグラフ化)
BETA_START = 1.56
BETA_END = 1.7

# フォントサイズの設定
TITLE_FONT_SIZE = 16       # グラフタイトルのフォントサイズ
LEGEND_FONT_SIZE = 20      # 凡例のフォントサイズ
AXIS_LABEL_FONT_SIZE = 20  # 軸ラベル(betaやchi)のフォントサイズ
TICK_FONT_SIZE = 20        # 軸のメモリ数字のフォントサイズ
BETA_MAX_FONT_SIZE = 20    # グラフ上の beta_max テキストのフォントサイズ

# その他の設定
LEGEND_LOC = 'best'        # 凡例の位置 ('upper right', 'lower right', 'best' など)
LINE_WIDTH = 2.0           # 線の太さ
MARKER_SIZE = 2            # マーカーのサイズ

# 各alphaや変数のプロット色設定
COLORS = ['r', 'g', 'b', 'm', 'c', 'k', 'y', 'orange']

# ==========================================
# ファイル選択関数
# ==========================================
def select_file():
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
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    parts = name_without_ext.split('_')
    if len(parts) >= 3:
        A = parts[0]
        B = parts[1]
        C = parts[2]
    else:
        A = "UnknownTime"
        B = "esp=Unknown"
        C = "t=Unknown"
    
    return A, B, C

# ==========================================
# モード1: 層相関のグラフ (chi_A.svg)
# ==========================================
def plot_chi(df, A, B, C):
    plt.figure(figsize=(8, 6))
    
    # DISPLAY_ALPHAS に含まれ、かつ df に存在する alpha のみを抽出
    alphas = sorted([a for a in df['alpha'].unique() if a in DISPLAY_ALPHAS])
    texts = []
    
    for i, alpha in enumerate(alphas):
        df_alpha = df[df['alpha'] == alpha]
        
        # 設定された BETA_START ~ BETA_END の範囲のデータのみを抽出
        df_filtered = df_alpha[(df_alpha['beta'] >= BETA_START) & (df_alpha['beta'] <= BETA_END)]
        
        if df_filtered.empty:
            continue
            
        betas = df_filtered['beta'].values
        chis = df_filtered['abs(chi[0, 1])'].values
        
        plt.plot(betas, chis, marker='o', markersize=MARKER_SIZE, 
                 linestyle='-', linewidth=LINE_WIDTH, color=COLORS[i % len(COLORS)], 
                 label=rf'$\alpha={alpha}$')
        
        # その描画範囲内での最大値のプロット
        max_idx = np.argmax(chis)
        max_beta = betas[max_idx]
        max_chi = chis[max_idx]
        
        plt.plot(max_beta, max_chi, marker='*', markersize=MARKER_SIZE+8, color=COLORS[i % len(COLORS)])
        
        # adjustText で後から調整するため、テキストオブジェクトをリストに保存
        texts.append(plt.text(max_beta, max_chi, rf'$\beta={max_beta:.2f}$', 
                              color=COLORS[i % len(COLORS)], fontsize=BETA_MAX_FONT_SIZE, fontweight='bold'))

    # テキストの重なりを自動調整し、必要に応じて引き出し線を引く
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

    # フォントサイズを一括適用
    plt.xlabel(r'$\beta$', fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel(r'$|\chi_{vh}|$', fontsize=AXIS_LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICK_FONT_SIZE)
    plt.yticks(fontsize=TICK_FONT_SIZE)
    
    if SHOW_TITLE:
        plt.title(f"{B}   |   {C}", fontsize=TITLE_FONT_SIZE)
        
    plt.xlim([BETA_START, BETA_END])
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
    # DISPLAY_ALPHAS に含まれ、かつ df に存在する alpha のみを抽出
    alphas = sorted([a for a in df['alpha'].unique() if a in DISPLAY_ALPHAS])
    num_alphas = len(alphas)
    
    if num_alphas == 0:
        print("指定された DISPLAY_ALPHAS に該当するデータがありません。")
        return

    fig, axes = plt.subplots(nrows=2, ncols=num_alphas, figsize=(5 * num_alphas, 8), sharex=True)
    
    if num_alphas == 1:
        axes = axes.reshape(2, 1)

    if SHOW_TITLE:
        fig.suptitle(f"{B}   |   {C}", fontsize=TITLE_FONT_SIZE, fontweight='bold')

    for j, alpha in enumerate(alphas):
        df_alpha = df[df['alpha'] == alpha]
        
        # 描画範囲のフィルタリング
        df_filtered = df_alpha[(df_alpha['beta'] >= BETA_START) & (df_alpha['beta'] <= BETA_END)]
        
        if df_filtered.empty:
            continue
            
        betas = df_filtered['beta'].values
        
        q0 = df_filtered['q[0]'].values
        q1 = df_filtered['q[1]'].values
        qhat0 = df_filtered['q_hat[0]'].values
        qhat1 = df_filtered['q_hat[1]'].values
        chis = df_filtered['abs(chi[0, 1])'].values
        
        max_idx = np.argmax(chis)
        beta_max = betas[max_idx]

        # -----------------------------------
        # 上段 (qのグラフ)
        # -----------------------------------
        ax_q = axes[0, j]
        ax_q.plot(betas, q0, marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color='b', label='q[0] (Visible)')
        ax_q.plot(betas, q1, marker='s', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color='r', label='q[1] (Hidden)')
        
        # 小数第2位までの表示に変更
        ax_q.axvline(x=beta_max, color='k', linestyle='--', alpha=0.6, label=rf'$\chi$ max ($\beta={beta_max:.2f}$)')
        
        ax_q.set_ylabel('q', fontsize=AXIS_LABEL_FONT_SIZE)
        ax_q.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        ax_q.grid(True, linestyle='--', alpha=0.7)
        if j == 0: 
            ax_q.legend(loc=LEGEND_LOC, fontsize=LEGEND_FONT_SIZE)

        # -----------------------------------
        # 下段 (q_hatのグラフ)
        # -----------------------------------
        ax_qhat = axes[1, j]
        ax_qhat.plot(betas, qhat0, marker='o', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color='b', label='q_hat[0]')
        ax_qhat.plot(betas, qhat1, marker='s', markersize=MARKER_SIZE, linewidth=LINE_WIDTH, color='r', label='q_hat[1]')
        
        ax_qhat.axvline(x=beta_max, color='k', linestyle='--', alpha=0.6)
        
        ax_qhat.set_xlabel(r'$\beta$', fontsize=AXIS_LABEL_FONT_SIZE)
        ax_qhat.set_ylabel(r'$\hat{q}$', fontsize=AXIS_LABEL_FONT_SIZE)
        ax_qhat.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        ax_qhat.grid(True, linestyle='--', alpha=0.7)
        ax_qhat.set_xlim([BETA_START, BETA_END])

    plt.tight_layout()
    if SHOW_TITLE:
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
        df = pd.read_csv(target_file)
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