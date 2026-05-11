#中心極限定理 central limit theorem
import numpy as np
import matplotlib.pyplot as plt

# 試行回数
num_trials = 10000

# 標本サイズ
sample_sizes = [1, 5, 30, 100]

for n in sample_sizes:
    # 元の分布：指数分布（かなり非対称）
    samples = np.random.exponential(scale=1.0, size=(num_trials, n))

    # 各試行ごとの標本平均
    sample_means = np.mean(samples, axis=1)

    # ヒストグラム表示
    plt.figure(figsize=(6,4))
    plt.hist(sample_means, bins=50, density=True)
    plt.title(f"Sample Size n = {n}")
    plt.xlabel("Sample Mean")
    plt.ylabel("Density")
    plt.show()