import numpy as np
import matplotlib.pyplot as plt
import itertools 

# ==========================================
# 1. GPU用の計算パッケージのimport
# ==========================================
# 使用しない

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
    def activation(x): #隠れ変数が1になる確率を返す だから解析計算と異なるように見える。
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    
    @staticmethod
    def sample(prob): #計算した確率に応じて 0/1をサンプリングする
        return (np.random.uniform(size=prob.shape) < prob).astype(np.float32)

class IsingUnit(UnitType):
    """ {-1, 1} """
    @staticmethod 
    def activation(x): #隠れ変数の期待値を返す。
        return np.tanh(x)
    
    @staticmethod
    def sample(expected): #期待値
        prob_one = (expected + 1.0) / 2.0
        return 2.0 * (np.random.uniform(size=prob_one.shape) < prob_one) - 1.0

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
        pass

# ==========================================
# 4. GBRBM Core: 行列計算本体
# ==========================================
class GBRBM:
    # 変更点：weight_std (β) を引数で受け取る
    def __init__(self, n_v, n_h, unit_type: UnitType, sampler: Sampler, weight_std: float = 1.0):
        self.unit = unit_type
        self.sampler = sampler
        
        # パラメータ: 指定された標準偏差(weight_std)で初期化
        self.W = np.random.normal(0, weight_std/(np.sqrt(n_v + n_h)), (n_v, n_h))
        #print("weight_std:",weight_std, "n_v:",n_v, "n_h:",n_h, "sigma:",weight_std/(np.sqrt(n_v + n_h)))
        
        # 可視層のバイアスパラメータ
        self.b = np.ones(n_v) * 0.001
        self.gamma = np.ones(n_v) * np.log(np.exp(1.0) - 1.0)

        # 隠れ層のバイアスパラメータ
        self.c = np.ones(n_h) * 0.001

        # itertoolsではなく、ビットシフトを用いて_H_allを作成する
        num_states = 2 ** n_h
        n = np.arange(num_states, dtype=np.uint32)[:, None]
        # ビットシフト用の配列
        shifts = np.arange(n_h - 1, -1, -1).astype(np.uint32)
        # ビット演算で一気に 0/1 の行列を作成
        self._H_all = ((n >> shifts) & 1).astype(np.float32)
        
        # 渡された UnitType が IsingUnit の場合のみ {-1, 1} に変換
        if isinstance(self.unit, IsingUnit):
            self._H_all = self._H_all * 2.0 - 1.0

    def sample_h_given_v(self, v):
        pre_activation = np.dot(v, self.W) + self.c
        h_prob = self.unit.activation(pre_activation)
        h_sample = self.unit.sample(h_prob)
        return h_prob, h_sample

    def get_var(self): #softplus関数を使って、分散の計算を行う
        return np.log(1.0 + np.exp(np.clip(self.gamma, -50, 50)))

    def sample_v_given_h(self, h): #隠れ変数がbinaryでもIsingでも変わらない
        sig2 = self.get_var()
        v_mean = sig2 * (self.b + np.dot(h, self.W.T))
        v_sample = np.random.normal(v_mean, np.sqrt(sig2)) #平均v_mean、分散sig2の正規分布からサンプリング
        return v_mean, v_sample

    def update(self, v_batch, lr):
        # --- Positive Phase ---
        h0_prob, _ = self.sample_h_given_v(v_batch) #tanh関数を使って、サンプリング

        # --- Negative Phase ---
        vk = self.sampler.run(self, v_batch)
        hk_prob, _ = self.sample_h_given_v(vk)
        
        # --- Gradient Calculation ---
        batch_size = v_batch.shape[0]

        db = np.mean(v_batch, axis=0) - np.mean(vk, axis=0)
        self.b += lr * db

        dc = np.mean(h0_prob, axis=0) - np.mean(hk_prob, axis=0)
        self.c += lr * dc

        pos_grad_W = np.dot(v_batch.T, h0_prob) / batch_size
        neg_grad_W = np.dot(vk.T, hk_prob) / batch_size
        self.W += lr * (pos_grad_W - neg_grad_W)

        v_sq_mean = np.mean(v_batch**2, axis=0)
        vk_sq_mean = np.mean(vk**2, axis=0)
        sig2 = self.get_var()
        sigmoid_gamma = 1.0 / (1.0 + np.exp(-np.clip(self.gamma, -50, 50)))
        self.gamma += lr * (sigmoid_gamma * (v_sq_mean - vk_sq_mean)) / (2 * sig2**2)

    # ==========================================
    # 対数尤度計算メソッド
    # ==========================================
    def compute_log_likelihood(self, v_train):
        n_h = self.W.shape[1]
        
        #隠れ変数がbinaryの場合の対数尤度関数のpositive側
        #pos = -(v_train**2 @ (0.5/self.get_var())) + v_train @ self.b + np.log(1 + np.exp(self.c.T + v_train @ self.W)).sum(axis=1)
        pos = -(v_train**2 @ (0.5/self.get_var())) + v_train @ self.b + np.log(2 * np.cosh(self.c.T + v_train @ self.W)).sum(axis=1)
        LL_pos = pos.mean(axis=0)

        neg1 = 0.5*np.log(2*np.pi*self.get_var()).sum(axis=0) + (0.5*self.b**2) @ self.get_var()
        
        # 保存されている H_all を使う（転送ゼロ！）
        H_all = self._H_all
        
        neg2_1 = self.W.T @ (self.get_var()*self.b) + self.c
        neg2_1 = (H_all @ neg2_1) 
        
        neg2_2 = ((self.W @ H_all.T)**2 * 0.5*self.get_var()[:, None]).sum(axis=0) 
        
        exponents = neg2_1 + neg2_2
        max_val = np.max(exponents)
        neg2 = max_val + np.log(np.exp(exponents - max_val).sum())
        
        LL_neg = neg1 + neg2
        LL = LL_pos - LL_neg
        return LL

    # ==========================================
    # データの再構成（生成）メソッド
    # ==========================================
    def reconstruct(self, v, k=1):
        """
        元のデータ v を初期値として、k回のギブスサンプリングを行い
        生成（再構成）されたデータを返す。
        """
        vk = v
        for _ in range(k):
            _, hk = self.sample_h_given_v(vk)
            _, vk = self.sample_v_given_h(hk)
        return vk