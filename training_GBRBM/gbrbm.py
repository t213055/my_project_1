import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Backend: CPU/GPU の切り替え抽象化
# ==========================================
# 実行環境に応じて np か cp (cupy) を返すようにする
try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np

# ==========================================
# 2. Unit Strategy: 変数の種類 (Binary/Ising)
# ==========================================
class UnitType:
    @staticmethod
    def activation(x): raise NotImplementedError
    @staticmethod
    def to_energy_term(h): raise NotImplementedError

class BinaryUnit(UnitType):
    """ {0, 1} """
    @staticmethod
    ## 隠れ変数がbinaryの場合の、隠れ変数の条件付確率　sigmoid関数
    def activation(x):
        return 1.0 / (1.0 + xp.exp(-xp.clip(x, -50, 50)))
    
    @staticmethod
    ## 条件付確率に基づき、サンプリングを行う　閾値を0.5に設定かな？
    def sample(prob):
        return (xp.random.uniform(size=prob.shape) < prob).astype(xp.float32)
        # xp.ramdom.uniform(...): 0~1までの乱数を生成
        # < prob: 生成した乱数とprobを比較し、結果に応じてTrue, Flaseを生成
        # 

class IsingUnit(UnitType):
    """ {-1, 1} """
    @staticmethod
    def activation(x):
        return xp.tanh(x)
    
    @staticmethod
    def sample(expected):
        # prob(h=1) = (tanh(x) + 1) / 2
        prob_one = (expected + 1.0) / 2.0
        return 2.0 * (xp.random.uniform(size=prob_one.shape) < prob_one) - 1.0

# ==========================================
# 3. Sampling Strategy: アルゴリズム
# ==========================================
class Sampler:
    def __init__(self, k=1):
        self.k = k

class ContrastiveDivergence(Sampler):
    def run(self, model, v_start):
        v = v_start
        for _ in range(self.k):
            _, h = model.sample_h_given_v(v)
            _, v = model.sample_v_given_h(h)
        return v

class AIS(Sampler):
    """ 焼きなまし重点サンプリング (後日実装用) """
    def run(self, model, v_start):
        # 自由エネルギーの計算など、複雑なロジックをここに分離
        pass

# ==========================================
# 4. GBRBM Core: 行列計算本体
# ==========================================
class GBRBM:
    def __init__(self, n_v, n_h, unit_type: UnitType, sampler: Sampler):
        self.unit = unit_type
        self.sampler = sampler
        
        # パラメータ (xpを使用することでGPU/CPU共通化)
        self.W = xp.random.normal(0, 1.78, (n_v, n_h))
        self.b = xp.ones(n_v) * 0.001   #xp.zeros(n_v)
        self.c = xp.ones(n_h) * 0.001   #xp.zeros(n_h)
        self.gamma = xp.ones(n_v) * xp.log(xp.exp(1.0) - 1.0)

    def sample_h_given_v(self, v):
        #条件付き確率のsigmoid関数への入力を計算
        pre_activation = xp.dot(v, self.W) + self.c
        h_prob = self.unit.activation(pre_activation)
        h_sample = self.unit.sample(h_prob)
        return h_prob, h_sample

    #gammaをsoftplus関数に入力して分散を得る
    def get_var(self):
        return xp.log(1.0 + xp.exp(xp.clip(self.gamma, -50, 50)))

    def sample_v_given_h(self, h):
        sig2 = self.get_var()
        v_mean = sig2 * (self.b + xp.dot(h, self.W.T))
        v_sample = xp.random.normal(v_mean, xp.sqrt(sig2))
        return v_mean, v_sample

    def update(self, v_batch, lr):
        # --- Positive Phase ---
        h0_prob, _ = self.sample_h_given_v(v_batch)

        # --- Negative Phase (サンプリング手法の切り替え) ---
        vk = self.sampler.run(self, v_batch)
        hk_prob, _ = self.sample_h_given_v(vk)
        
        # --- Gradient Calculation (解析的な式をここに実装) ---
        # 自分で求めた W, b, c, var の更新式を記述
        # 例: dW = (v_batch/var).T @ h0_prob - (vk/var).T @ hk_prob
        
        # 更新処理 ...
        batch_size = v_batch.shape[0]

        db = xp.mean(v_batch, axis=0) - xp.mean(vk, axis=0)
        self.b += lr * db

        dc = xp.mean(h0_prob, axis=0) - xp.mean(hk_prob, axis=0)
        self.c += lr * dc

        pos_grad_W = xp.dot(v_batch.T, h0_prob) / batch_size
        neg_grad_W = xp.dot(vk.T, hk_prob) / batch_size
        self.W += lr * (pos_grad_W - neg_grad_W)

        v_sq_mean = xp.mean(v_batch**2, axis=0)
        vk_sq_mean = xp.mean(vk**2, axis=0)
        sig2 = self.get_var()
        sigmoid_gamma = 1.0 / (1.0 + xp.exp(-xp.clip(self.gamma, -50, 50)))
        self.gamma += lr * (sigmoid_gamma * (v_sq_mean - vk_sq_mean)) / (2 * sig2**2)