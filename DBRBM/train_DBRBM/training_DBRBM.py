#最終的に、横軸にepoch、縦軸に対数尤度関数をプロットしたグラフを出力したい
#1枚の画像に対して、グラフは3つのモデルの結果を重ねて表示したい
#各グラフは、それぞれ重みの初期値分布（ガウス分布の標準偏差β）を変えている

#パッケージ、ライブラリ、クラスのインポート
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools #対数尤度関数（分配関数）計算時、隠れ変数の和をとる場合に使用

#実験設定の記述
CONFIG = {
    "num_visible": 8, 
    "alpha_list": [0.5],#, 1.0, 2.0, 2.5],
    "lerning_rate": 0.01,
    "num_epochs": 1000,
    "cd_k": 1,
    "batch_size": 1,
    "num_samples": 50
}

#可視変数はalphaの値によって変わるので、alphaの関数として記述している？
#main文で、num_hidden = CONFIG["num_hidden"](alpha) のように使用する。
CONFIG["num_hidden"] = lambda alpha: int(CONFIG["num_visible"] * alpha)  # 隠れ変数の数は可視変数の数＊alpha

CONFIG["beta_max"] = [
    1.66967 if alpha == 0.5 else
    1.59095 if alpha == 1.0 else
    1.60357 if alpha == 2.0 else
    1.62606
    for alpha in CONFIG["alpha_list"]
]

class DBRBM:
    def __init__(self, num_visible, num_hidden, learning_rate, beta_max):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.beta_max = beta_max
        
        # 重みの初期化（ガウス分布に従う乱数で初期化）
        self.weights = np.random.normal(0, beta_max / np.sqrt(num_visible + num_hidden), (num_visible, num_hidden)) # std = beta_max / sqrt(num_visible + num_hidden)
        self.visible_bias = np.random.rand(num_visible)
        self.hidden_bias = np.random.rand(num_hidden)

        # 隠れ変数の総和計算用行列の生成
        num_states = 2 ** self.num_hidden
        n = np.arange(num_states, dtype=np.int32)[:, None]
        # ビットシフト用の配列
        shifts = np.arange(self.num_hidden - 1, -1, -1, dtype=np.int32)
        # ビット演算で一気に 0/1 の行列を作成
        self._H_all = (2 * ((n >> shifts) & 1) - 1).astype(np.float32)

    #教師データでデータ生成

    #活性化関数
    """
    def sample_h(self, v):
        
        activation = np.dot(v, self.weights) + self.hidden_bias #expへの入力
        logit_1 = activation
        logit_m1 = -activation
        max_logit = np.maximum(logit_1, logit_m1)

        exp_1 = np.exp(logit_1 - max_logit)
        exp_m1 = np.exp(logit_m1 - max_logit)

        prob_1 = exp_1 / (exp_1 + exp_m1) #隠れ変数が1をとる確率

        #確率に応じて隠れ変数をサンプリング（CD法では確率ではなくサンプリングした値を用いる）
        random_vals = np.random.rand(*prob_1.shape)
        h_sample = np.where(random_vals < prob_1, 1, -1)

        #期待値計算
        h_expected = np.tanh(activation)
        
        return h_expected, h_sample
    """

    def sample_h(self, v):
        activation = np.dot(v, self.weights) + self.hidden_bias

        #期待値計算 numpyのtanhはoverflow起こしにくい
        h_expected = np.tanh(activation)

        #隠れ変数が1をとる確率を, 期待値から逆算
        prob_1 = (h_expected + 1.0) / 2.0
        random_vals = np.random.rand(*prob_1.shape)
        h_sample = np.where(random_vals < prob_1, 1, -1)

        return h_expected, h_sample

    def sample_v(self, h):
        activation = np.dot(h, self.weights.T) + self.visible_bias
        #各実現値{-1, 0, 1}に対応した, 活性化関数への入力
        logit_1 = activation
        logit_0 = np.zeros_like(activation)
        logit_m1 = -activation
        
        #overflow対策
        ##logit_1,0,m1で最も大きいものを見つける
        max_logit = np.maximum.reduce([logit_1, logit_0, logit_m1])
        ##処理前=処理後で結果は変わらない（分子分母をexp(max_logit)で割っているだけなので）
        exp_1 = np.exp(logit_1 - max_logit)
        exp_0 = np.exp(logit_0 - max_logit) 
        exp_m1 = np.exp(logit_m1 - max_logit)
        ##分母の計算
        sum_exp = exp_1 + exp_0 + exp_m1

        #各実現値{-1, 0, 1}に対応した条件付き確率P(v={-1,0,1}|h,θ)を計算
        prob_1 = exp_1 / sum_exp
        prob_0 = exp_0 / sum_exp
        prob_m1 = exp_m1 / sum_exp

        #期待値（勾配計算→期待値必要→サンプリング必要）
        v_expected = prob_1 - prob_m1

        #サンプリングブロック
        random_vals = np.random.rand(*activation.shape)
        v_sample = np.where( # np.where(condition, x, y) : conditionが真ならx, 偽ならyを出力
            random_vals < prob_1,
            1,
            np.where(random_vals < prob_1 + prob_0, 0, -1)
        )

        return v_expected, v_sample
    
    def cd(self, batch_data, k=1):
        v0 = batch_data
        h0_expected, _ = self.sample_h(v0)

        v = v0
        for _ in range(k):
            _, h = self.sample_h(v) #cd法では、必ずサンプリング結果（実現値）をやり取りする
            vk_expected, v = self.sample_v(h) #仮に期待値を渡してみたら学習結果はどうなるのだろうか？
        
        vk = v
        hk_expected, _ = self.sample_h(vk)
        
        return v0, h0_expected, vk_expected, hk_expected

    #勾配更新（勾配上昇法 or adamを選択可能）
    def update(self, batch_data, cd_k=1):
        batch_size = batch_data.shape[0] #与えられたデータからbatch_sizeを取得するか、CONFIGから取得するか。
        # print("batch_size", batch_size)
        v0, h0_expected, vk_expected, hk_expected = self.cd(batch_data, k=cd_k)
        # print(
        #     "v0.shape",v0.shape ,
        #     "\nh0_expected.shape", h0_expected.shape,
        #     "\nvk_expected.shape", vk_expected.shape,
        #     "\nhk_expected.shape", hk_expected.shape
        #     )
        
        #勾配計算
        db = np.mean(v0, axis=0) - np.mean(vk_expected, axis=0)
        # print("db.shape", db.shape)
        dc = np.mean(h0_expected, axis=0) - np.mean(hk_expected, axis=0)
        # print("dc.shape", dc.shape)
        dw = np.dot(v0.T, h0_expected)/batch_size - np.dot(vk_expected.T, hk_expected)/batch_size
        # print("dw.shape", dw.shape)
        
        #パラメータ更新
        self.visible_bias += self.learning_rate * db
        self.hidden_bias += self.learning_rate * dc
        self.weights += self.learning_rate * dw

    #評価指標：対数尤度関数（厳密計算）
    def log_likelihood(self, train_data):
        n_samples = train_data.shape[0]
        # print("n_samples", n_samples)
        # print("train_data.shape", train_data.shape)
        # print("visible_bias.shape", self.visible_bias.shape)
        # print("hidden_bias.shape", self.hidden_bias.shape)
        # print("weights.shape", self.weights.shape)
        # a = train_data @ self.visible_bias
        # b = train_data @ self.weights + self.hidden_bias
        # c = np.log(2 * np.cosh(b))
        # d = np.sum(c, axis=1)
        # e = np.mean(a + d, axis=0)
        # print("a\n", a, "\nb\n", b, "\nc\n", c, "\nd\n", d, "\ne\n", e)

        p_term = np.mean(train_data @ self.visible_bias + np.sum(np.log(2 * np.cosh(train_data @ self.weights + self.hidden_bias)), axis=1), axis=0)
        #print("p_term", p_term)

        n_term = self.ln_Z()

        return p_term - n_term
    
    def ln_Z(self):
        _H_all = self._H_all
        w = self.weights
        b = self.visible_bias
        c = self.hidden_bias
        
        phi = _H_all @ w.T + b
        abs_phi = np.abs(phi) #iについての総積（=exp上での総和）はとられているためサイズが(8, "6")
        #print("phi.shape", phi.shape, "\nphi\n", phi)
        term_A = abs_phi + np.log(1 + np.exp(-abs_phi) + np.exp(-2 * abs_phi))
        term_A = np.sum(term_A, axis=1)
        #print("term_A.shape", term_A.shape, "\nterm_A\n", term_A)

        term_B = _H_all @ c.T
        #print("term_B.shape", term_B.shape, "\nterm_B\n", term_B)

        psi = term_B + term_A
        #print("psi.shape", psi.shape,"\npsi\n", psi)
        psi_max = np.maximum.reduce(psi)
        #print("psi_max", psi_max)

        term_C = np.exp(psi - psi_max)
        ln_Z = psi_max + np.log(np.sum(term_C))
        #print("ln_Z", ln_Z)
        return ln_Z

#main文
for alpha, beta in zip(CONFIG["alpha_list"], CONFIG["beta_max"]):
    print(f"========== alpha: {alpha}, beta_max: {beta} ==========")
    
    n_v = CONFIG["num_visible"]
    n_h = CONFIG["num_hidden"](alpha)
    n_samples = CONFIG["num_samples"]

    #データセットの生成
    # dummy_data = np.random.choice([-1, 0, 1], size=(n_samples, n_v))
    # train_data = dummy_data
    # 規則性のあるデータセットの生成 (例: 前半半分が1、後半半分が-1になるパターンなど)
    train_data = np.zeros((n_samples, n_v))
    for i in range(n_samples):
        if np.random.rand() > 0.5:
            # パターンA: [1, 1, 1, 1, -1, -1, -1, -1] のような傾向
            train_data[i, :n_v//2] = np.random.choice([0, 1], size=n_v//2)
            train_data[i, n_v//2:] = np.random.choice([-1, 0], size=n_v - n_v//2)
        else:
            # パターンB: [-1, -1, -1, -1, 1, 1, 1, 1] のような傾向
            train_data[i, :n_v//2] = np.random.choice([-1, 0], size=n_v//2)
            train_data[i, n_v//2:] = np.random.choice([0, 1], size=n_v - n_v//2)
    print("train_data.shape\n", train_data.shape)

    model = DBRBM(
        num_visible=CONFIG["num_visible"],
        num_hidden=CONFIG["num_hidden"](alpha),
        learning_rate=CONFIG["lerning_rate"],
        beta_max=beta
    )

    epochs = CONFIG["num_epochs"]
    batch_size = CONFIG["batch_size"] #: データセットのバッチサイズ
    #n_samples : データセットのサンプル数

    #print("train_data\n", train_data)
    #print("log_likelihood", model.log_likelihood(train_data))
    for epoch in range(epochs):
        #print(f"===== epoch : {epoch} =====")
        
        #各epochでデータをシャッフル
        indices = np.random.permutation(n_samples)
        # print("indices", indices)
        shuffled_data = train_data[indices]
        # print("train_data\n", shuffled_data)

        for i in range(0, n_samples, batch_size):
            batch_data = shuffled_data[i : i + batch_size]
            model.update(batch_data, cd_k=1)
        
        if(epoch % 100 == 0):
            print(f"epoch {epoch} : ", model.log_likelihood(train_data))
    

    #結果出力
        #epochごとの対数尤度をcsvファイルに出力