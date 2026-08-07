import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys
import os

# ==========================================
# [設定] グラフの描画スタイル: ここで色や線の太さ、線のスタイルを調整できます
# ==========================================
HIST_COLOR = 'black'       # ヒストグラムの色（例: 'black', 'gray', '#000000'）       
HIST_LINEWIDTH = 10       # ヒストグラムの線の太さ       
HIST_LINESTYLE = '-'       # ヒストグラムの線種（'-': 実線, '--': 破線, ':': 点線）       
HIST_BINS = 10             # ヒストグラムのビン数（棒の数）             

KDE_COLOR = '#008fd5'      
KDE_LINEWIDTH = 20        

# ==========================================
# 1. Path setting and importing gbrbm
# ==========================================
sys.path.append(os.path.abspath(".."))

try:
    from gbrbm import GBRBM, IsingUnit, ContrastiveDivergence
except ImportError:
    print("エラー: 1つ上のディレクトリ ('..') に gbrbm.py が見つかりません。")
    sys.exit(1)

# ==========================================
# 2. Graph plotting functions
# ==========================================
def plot_data_only(data_train, filename):
    plt.figure(figsize=(7, 5))
    plt.gca().set_facecolor('white')
    
    plt.hist(data_train, bins=HIST_BINS, density=True, histtype='step', 
             color=HIST_COLOR, edgecolor=HIST_COLOR, 
             linestyle=HIST_LINESTYLE, linewidth=HIST_LINEWIDTH)
             
    plt.xticks([]) 
    plt.yticks([])
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_distribution(data_train, data_sampled, filename):
    plt.figure(figsize=(7, 5))
    plt.gca().set_facecolor('white')
    
    plt.hist(data_train, bins=HIST_BINS, density=True, histtype='step', 
             color=HIST_COLOR, edgecolor=HIST_COLOR, 
             linestyle=HIST_LINESTYLE, linewidth=HIST_LINEWIDTH)
    
    try:
        kde = gaussian_kde(data_sampled.T[0])
        x_range = np.linspace(data_train.min() - 2, data_train.max() + 2, 500)
        plt.plot(x_range, kde(x_range), color=KDE_COLOR, linewidth=KDE_LINEWIDTH)
    except np.linalg.LinAlgError:
        plt.hist(data_sampled, bins=HIST_BINS, density=True, color=KDE_COLOR, alpha=0.5)

    plt.xticks([]) 
    plt.yticks([])
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ==========================================
# 3. Training loop function
# ==========================================
def train_model(model, data, epochs=200, lr=0.01):
    batch_size = 64
    n_samples = data.shape[0]
    for epoch in range(epochs):
        np.random.shuffle(data)
        for i in range(0, n_samples, batch_size):
            batch = data[i:i+batch_size]
            model.update_adam(batch, lr)

# ==========================================
# 4. Main function
# ==========================================
def main():
    np.random.seed(42)
    
    n_samples = 2000
    X_train = np.concatenate([
        np.random.normal(loc=-2.0, scale=1.0, size=int(n_samples * 0.4)),
        np.random.normal(loc=3.0, scale=1.5, size=int(n_samples * 0.6))
    ]).reshape(-1, 1)

    print("=== 実験開始 ===")
    
    print("\n[0] 学習データのみのグラフを保存中...")
    plot_data_only(X_train, "training_data.png")
    
    print("\n[1] 良い初期値のモデルを準備・学習中...")
    model_good = GBRBM(n_v=1, n_h=10, unit_type=IsingUnit(), 
                       sampler=ContrastiveDivergence(k=1), 
                       weight_std=0.1, calc_exact_ll=False)
    model_good.b = np.mean(X_train, axis=0)
    train_model(model_good, X_train, epochs=300, lr=0.01)
    
    samples_good = model_good.reconstruct(X_train, k=10)
    plot_distribution(X_train, samples_good, "result_good_init.png")

    print("\n[2] 悪い初期値のモデルを準備・学習中...")
    model_bad = GBRBM(n_v=1, n_h=10, unit_type=IsingUnit(), 
                      sampler=ContrastiveDivergence(k=1), 
                      weight_std=100.0, calc_exact_ll=False)
    model_bad.b = np.array([40.0])
    train_model(model_bad, X_train, epochs=750, lr=0.01)
    
    samples_bad = model_bad.reconstruct(X_train, k=10)
    plot_distribution(X_train, samples_bad, "result_bad_init.png")

    print("\n=== 実験完了 ===")

if __name__ == '__main__':
    main()
    pass