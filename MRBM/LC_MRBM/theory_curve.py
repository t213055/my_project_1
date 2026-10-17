import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import pandas as pd
from adjustText import adjust_text

# ==========================================
# グラフ描画・調整用パラメータ (スクリプト冒頭で調整可能)
# ==========================================
SHOW_TITLE = False         # タイトルを表示するか (True: 表示, False: 非表示)

# 表示する alpha の設定
DISPLAY_ALPHAS = [0.5, 1.0, 2.0]

# グラフの横軸の描画範囲設定 (対数スケール: 10^x)
LOG_ALPHA_START = -1.0     # デフォルト: 10^-1 = 0.1
LOG_ALPHA_END = 1.0        # デフォルト: 10^1 = 10

# フォントサイズの設定
FONT_SIZE_TITLE = 20       # タイトルのフォントサイズ
FONT_SIZE_AXIS = 20        # 軸ラベルのフォントサイズ
FONT_SIZE_TICKS = 20       # 軸目盛のフォントサイズ
FONT_SIZE_TEXT = 20        # グラフ中のテキスト（alphaの値）のフォントサイズ
FONT_SIZE_LEGEND = 20      # 凡例のフォントサイズ

# ==========================================
# ファイル選択関数
# ==========================================
def select_file():
    txt_files = glob.glob('*.txt')
    if not txt_files:
        print("エラー: カレントディレクトリに .txt ファイルが見つかりません。")
        return None, None

    print("=== 入力ファイルを選択してください ===")
    for i, file in enumerate(txt_files):
        print(f"{i + 1}: {file}")
    
    while True:
        try:
            choice = int(input("ファイル番号を入力: "))
            if 1 <= choice <= len(txt_files):
                selected_file = txt_files[choice - 1]
                # ファイル名から t の値を抽出する処理
                # "esp=1e-3_t=2.txt" などの形式を想定
                try:
                    # ファイル名の "_t=" の後から ".txt" の前までを抽出
                    t_str = selected_file.split('_t=')[-1].replace('.txt', '')
                    t_val = int(t_str)
                    return selected_file, t_val
                except (ValueError, IndexError):
                    print("ファイル名から t の値を自動抽出できませんでした。デフォルトの t=1 で実行します。")
                    return selected_file, 1
            else:
                print("無効な番号です。")
        except ValueError:
            print("数字を入力してください。")

# ==========================================
# 転移点の計算関数 (オーバーフロー対策版)
# ==========================================
def find_transition_point(t, alpha):
    k_start = (t // 2) + 1
    k = np.arange(t, k_start - 1, -1)
    v_k = (2 * k - t) / t
    
    v_pos = v_k[v_k > 0]
    delta_term = 1.0 if (t % 2 == 0) else 0.0
    
    def equation(beta):
        A = (alpha * beta**2) / (2 * (1 + alpha))
        
        max_v2 = np.max(v_pos**2) if len(v_pos) > 0 else 0
        max_exp_val = A * max_v2
        
        num = 2 * np.sum((v_pos**2) * np.exp(A * v_pos**2 - max_exp_val))
        den = delta_term * np.exp(-max_exp_val) + 2 * np.sum(np.exp(A * v_pos**2 - max_exp_val))
        
        expected_v2 = num / den
        target_val = np.sqrt(alpha) + 1.0 / np.sqrt(alpha)
        
        return beta**2 * expected_v2 - target_val

    try:
        sol = root_scalar(equation, bracket=[0.01, 100.0], method='brentq')
        return sol.root
    except ValueError:
        return np.nan

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    # ファイルの選択と t の取得
    data_file, T_FIXED = select_file() # 本番環境用
    # data_file, T_FIXED = "esp=1e-3_t=2.txt", 2 # テスト用

    if data_file:
        print(f"\n読み込み中: {data_file} (t={T_FIXED})")

        # --- 理論カーブのデータ作成 ---
        num_points = 200
        log_alphas = np.linspace(LOG_ALPHA_START, LOG_ALPHA_END, num_points)
        alphas = 10**log_alphas
        betas = [find_transition_point(T_FIXED, a) for a in alphas]

        # --- 数値実験データの読み込み ---
        numerical_log_alphas = []
        numerical_betas = []
        
        try:
            df = pd.read_csv(data_file)
            
            # 各alphaについて、層相関 abs(chi[0, 1]) が最大となる beta を抽出
            for alpha in sorted(df['alpha'].unique()):
                # 表示設定されている alpha のみを処理
                if alpha in DISPLAY_ALPHAS:
                    df_alpha = df[df['alpha'] == alpha]
                    
                    max_idx = df_alpha['abs(chi[0, 1])'].idxmax()
                    max_beta = df_alpha.loc[max_idx, 'beta']
                    
                    numerical_log_alphas.append(np.log10(alpha))
                    numerical_betas.append(max_beta)
        except FileNotFoundError:
            print(f"Error: {data_file} が見つかりません。")

        # --- 描画 ---
        plt.figure(figsize=(10, 8))

        # 理論カーブ
        plt.plot(log_alphas, betas, linestyle='-', color='blue', linewidth=2.5, 
                 label=f'Transition Point', zorder=1)

        # 数値実験のプロット
        texts = []
        if numerical_log_alphas:
            plt.scatter(numerical_log_alphas, numerical_betas, marker='*', s=300, color='red', 
                        edgecolor='black', label=rf'$\beta_{{\text{{max}}}}$', zorder=2)
            
            # プロット点の近くに元のαの値をテキスト表示するための準備
            for la, b in zip(numerical_log_alphas, numerical_betas):
                original_alpha = 10**la
                texts.append(plt.text(la, b, rf'$\alpha={original_alpha:.1f}$', 
                                      fontsize=FONT_SIZE_TEXT))

        # adjustTextによるテキストの自動調整（重なり防止）
        if texts:
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=1.0))

        # 軸ラベルとタイトル
        plt.xlabel(r'$\log_{10}(\alpha)$', fontsize=FONT_SIZE_AXIS)
        plt.ylabel(r'$\beta$', fontsize=FONT_SIZE_AXIS)

        # 軸目盛のフォントサイズ
        plt.xticks(fontsize=FONT_SIZE_TICKS)
        plt.yticks(fontsize=FONT_SIZE_TICKS)
        
        # 描画範囲の適用
        plt.xlim([LOG_ALPHA_START, LOG_ALPHA_END])

        if SHOW_TITLE:
            plt.title(f'Phase Transition vs Maximum Layer Correlation (Fixed t={T_FIXED})', 
                      fontsize=FONT_SIZE_TITLE)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper center', fontsize=FONT_SIZE_LEGEND)
        plt.tight_layout()

        out_filename = f"transition_vs_correlation_t{T_FIXED}_log.png"
        plt.savefig(out_filename, dpi=300)
        print(f"グラフを保存しました: {out_filename}")
        # plt.show()