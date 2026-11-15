#最終的に、横軸にepoch、縦軸に対数尤度関数をプロットしたグラフを出力したい
#1枚の画像に対して、グラフは3つのモデルの結果を重ねて表示したい
#各グラフは、それぞれ重みの初期値分布（ガウス分布の標準偏差β）を変えている

#パッケージ、ライブラリ、クラスのインポート
import numpy as np
import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import itertools #対数尤度関数（分配関数）計算時、隠れ変数の和をとる場合に使用
import csv
# from sklearn.datasets import load_diabetes
from sklearn.datasets import load_wine

# 実験設定の記述（人工データ）
CONFIG = {
    "num_visible": 10, 
    "alpha_list": [0.5],#, 1.0, 2.0
    "lerning_rate": 0.01,   # ← 少し大きめにして学習速度の差を開かせる
    "num_epochs": 10,      # ← 収束までの過程が見える程度の長さに
    "cd_k": 1,
    "batch_size": 32,       # ← 標準的なバッチサイズで勾配を安定させる
    "TRIALS": 1,           # ← 乱数によるブレを消すため、最低10回は平均をとる
    # 教師モデルの設定
    "num_samples": 100,    # ← データ数を増やして真の分布を安定させる
    "burn_in_steps": 1000,
    "teacher_beta": 3.0,
    "teacher_bias_scale": 0.5,
    "output_path": "results.csv", 
    "update_method": "sgd" #"exact" or "adam" or "sgd"
}

if CONFIG["update_method"] == "exact":
    CONFIG["batch_size"] = CONFIG["num_samples"]  # exact法では、全データを使うためバッチサイズはデータ数と同じにする


# train_data = load_diabetes()  # データセットをロード
# train_data = load_wine()
# X = train_data.data
# Y = train_data.target
# print("train_data.data.shape", X.shape)
# print("train_data.target.shape", Y.shape)

# X = pd.DataFrame(X)  # データをDataFrameに変換
# Y = pd.DataFrame(Y)  # ターゲットをDataFrameに変換
# df = pd.concat([X,Y], axis=1)
# print("train_data.shape", df.shape)
# print(df.head()) #先頭から5行を表示

# print("columns", len(df.columns))
# print("rows", len(df))
# # 実験設定（実データ）
# CONFIG = {
#     "num_visible": len(df.columns), 
#     "alpha_list": [0.5, 1.0, 2.0],
#     "lerning_rate": 0.01,   # ← 少し大きめにして学習速度の差を開かせる
#     "num_epochs":150,      # ← 収束までの過程が見える程度の長さに
#     "cd_k": 1,
#     "batch_size": 16,       # ← 標準的なバッチサイズで勾配を安定させる
#     "TRIALS": 50,           # ← 乱数によるブレを消すため、最低10回は平均をとる
#     # 教師モデルの設定
#     "num_samples": len(df),    # ← データ数を増やして真の分布を安定させる
#     "burn_in_steps": 1000,
#     # "teacher_beta": 3.0,
#     # "teacher_bias_scale": 0.5,
#     "output_path": "results_toy_2.csv",
#     "update_method": "adam"  # 勾配更新方法を指定（"sgd" または "adam"）
# }

#可視変数はalphaの値によって変わるので、alphaの関数として記述している？
#main文で、num_hidden = CONFIG["num_hidden"](alpha) のように使用する。
CONFIG["num_hidden"] = lambda alpha: int(CONFIG["num_visible"] * alpha)  # 隠れ変数の数は可視変数の数＊alpha

CONFIG["beta_max"] = [
    1.66967 if alpha == 0.5 else
    # 1.68779 if alpha == 0.5 else #diabetesデータセット使用時は可視変数が奇数になるので、転移点が少しずれる
    1.59095 if alpha == 1.0 else
    1.60357 if alpha == 2.0 else
    None
    for alpha in CONFIG["alpha_list"]
]

#教師データでデータ生成
def generate_train_data():
    n_v = CONFIG["num_visible"]
    n_h = CONFIG["num_hidden"](alpha)
    n_h = int(n_h*1.5) #教師モデルの表現力を高め、beta_maxを用いたモデルの学習に差が出るようにする
    n_samples = CONFIG["num_samples"]
    teacher_beta = CONFIG["teacher_beta"]
    teacher_bias_scale = CONFIG["teacher_bias_scale"]
    burn_in_steps = CONFIG["burn_in_steps"]

    teacher = DBRBM(
        num_visible = n_v,
        num_hidden=n_h,
        learning_rate=0.0, # 学習しないため0
        beta_max=teacher_beta,
        teacher=True
    )
    # print("teacher_model_size: n_v", n_v," n_h", n_h)
    teacher.visible_bias = np.random.normal(0, teacher_bias_scale, n_v)
    # print("teacher.visible_bias", teacher.visible_bias)
    teacher.hidden_bias = np.random.normal(0, teacher_bias_scale, n_h)
    # print("teacher.hidden_bias", teacher.hidden_bias)
    # print("teacher.weights", teacher.weights)

    v = np.random.choice([-1, 0, 1], size=(n_samples, n_v))

    for i in range(burn_in_steps):
        _, h = teacher.sample_h(v)
        _, v = teacher.sample_v(h)
    print("==== data generated\n")

    return v

def discretize_to_ternary(series):
    unique_vals = series.unique()
    
    # パターン1: 値が2種類しかない場合（例：性別）
    # → 最小値を -1、最大値を 1 に割り当てる（0は使わない）
    if len(unique_vals) <= 2:
        return np.where(series == series.min(), -1, 1)
    
    # パターン2: それ以外の連続値
    # → 下から33.3%のラインと、66.7%のラインを計算して3等分する
    p33 = np.percentile(series, 33.33)
    p67 = np.percentile(series, 66.67)
    
    conditions = [
        series < p33,                         # 下位33%
        (series >= p33) & (series <= p67),    # 中位33%
        series > p67                          # 上位33%
    ]
    choices = [-1, 0, 1]
    
    # 条件に当てはめて変換
    return np.select(conditions, choices, default=0)

class DBRBM:
    def __init__(self, num_visible, num_hidden, learning_rate, beta_max, teacher=False):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.beta_max = beta_max
        
        # 重みの初期化（ガウス分布に従う乱数で初期化）
        self.weights = np.random.normal(0, beta_max / np.sqrt(num_visible + num_hidden), (num_visible, num_hidden)) # std = beta_max / sqrt(num_visible + num_hidden)
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

        # ====================================================
        # 【追加】Adamオプティマイザ用のパラメータ初期化
        # ====================================================
        self.beta1 = 0.9      # 1次モーメント（平均）の減衰率
        self.beta2 = 0.999    # 2次モーメント（分散）の減衰率
        self.epsilon = 1e-8   # ゼロ割り算防止
        self.t = 0            # 更新ステップのカウンター
        
        # 重みとバイアスの勾配履歴（m:1次, v:2次）をゼロ配列で準備
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_vb = np.zeros_like(self.visible_bias)
        self.v_vb = np.zeros_like(self.visible_bias)
        self.m_hb = np.zeros_like(self.hidden_bias)
        self.v_hb = np.zeros_like(self.hidden_bias)
        # ====================================================

        if teacher == False:
            if (num_hidden / num_visible) <= (np.log(3)/np.log(2)):
                # αが1.58より小さい→可視層の計算量が大きい→可視層を周辺化したln_Zの計算
                # →隠れ層の総組み合わせ行列を使う
                # 隠れ変数の全組み合わせ行列の生成
                num_states = 2 ** self.num_hidden
                n = np.arange(num_states, dtype=np.int32)[:, None]
                # ビットシフト用の配列
                shifts = np.arange(self.num_hidden - 1, -1, -1, dtype=np.int32)
                # ビット演算で一気に 0/1 の行列を作成
                self._all = (2 * ((n >> shifts) & 1) - 1).astype(np.float32)

            else:
                # αが1.58より大きい→隠れ層の計算量が大きい→隠れ層を周辺化したln_Zの計算
                # →可視層の総組み合わせ行列を使う
                #可視変数の全組み合わせ行列の生成
                num_states = 3 ** self.num_visible
                # 1. 0 から 3^V - 1 までの整数 (サイズ: [3^V, 1])
                n = np.arange(num_states, dtype=np.int32)[:, None]
                # 2. 3を底とする「桁の重み」（3^0, 3^1, ...）の指数を作成
                powers = np.arange(self.num_visible - 1, -1, -1, dtype=np.int32)
                # 3. 3進数の各桁を抽出する: (n // 3^powers) % 3
                # これにより、要素が {0, 1, 2} の行列が一気に生成されます
                ternary_matrix = (n // (3 ** powers)) % 3
                # 4. {0, 1, 2} を {-1, 0, 1} に変換する (1を引くだけ！)
                self._all = (ternary_matrix - 1).astype(np.float32)
            # print("==== student model initialized :", "beta", beta)

    #活性化関数
    def sample_h(self, v):
        activation = np.dot(v, self.weights) + self.hidden_bias

        #期待値計算 numpyのtanhはoverflow起こしにくい
        h_expected = np.tanh(activation)

        #隠れ変数が1をとる確率を, 期待値から逆算
        prob_1 = (h_expected + 1.0) / 2.0
        random_vals = np.random.rand(*prob_1.shape)
        h_sample = np.where(random_vals < prob_1, 1, -1)

        return h_expected, h_sample #隠れ変数の期待値, 隠れ変数のサンプリング値, 

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

        return v_expected, v_sample #可視変数の期待値, 可視変数のサンプリング値
    
    def cd(self, batch_data, k=1):
        v0 = batch_data
        h0_expected, _ = self.sample_h(v0)

        v = v0
        for _ in range(k):
            _, h = self.sample_h(v) #cd法では、必ずサンプリング結果（実現値）をやり取りする
            vk_expected, v = self.sample_v(h) #仮に期待値を渡してみたら学習結果はどうなるのだろうか？
        
        vk = v
        hk_expected, hk = self.sample_h(vk)
        
        # 対数尤度関数のpostive_termは”データ平均”と”①Hの期待値”で計算、negative_termは”Vのサンプル値”と”Hのサンプル値”で計算
        # ①を用いるのは、サンプリングによるノイズを避けるため
        return v0, h0_expected, vk, hk

    # def update_exact(self, batch_data):
    #     batch_size = batch_data.shape[0] #与えられたデータからbatch_sizeを取得するか、CONFIGから取得するか。
    #     # print("batch_size", batch_size)
    #     v0, h0_expected, _, _ = self.cd(batch_data, k=cd_k)
    #     # print(
    #     #     "v0.shape",v0.shape ,
    #     #     "\nh0_expected.shape", h0_expected.shape,
    #     #     )
        
    #     #勾配計算
    #     db = np.mean(v0, axis=0) - #可視変数の期待値
    #     # print("db.shape", db.shape)
    #     dc = np.mean(h0_expected, axis=0) - #隠れ変数の期待値
    #     # print("dc.shape", dc.shape)
    #     dw = np.dot(v0.T, h0_expected)/batch_size - #可視変数と隠れ変数の期待値
    #     # print("dw.shape", dw.shape)
        
    #     #パラメータ更新
    #     self.visible_bias += self.learning_rate * db
    #     self.hidden_bias += self.learning_rate * dc
    #     self.weights += self.learning_rate * dw

    #勾配更新（勾配上昇法 or adamを選択可能）
    def update_sgd(self, batch_data, cd_k=1):
        batch_size = batch_data.shape[0] #与えられたデータからbatch_sizeを取得するか、CONFIGから取得するか。
        # print("batch_size", batch_size)
        v0, h0_expected, vk_sample, hk_sample = self.cd(batch_data, k=cd_k)
        # print(
        #     "v0.shape",v0.shape ,
        #     "\nh0_expected.shape", h0_expected.shape,
        #     "\nvk_expected.shape", vk_expected.shape,
        #     "\nhk_expected.shape", hk_expected.shape
        #     )
        
        #勾配計算
        db = np.mean(v0, axis=0) - np.mean(vk_sample, axis=0)
        # print("db.shape", db.shape)
        dc = np.mean(h0_expected, axis=0) - np.mean(hk_sample, axis=0)
        # print("dc.shape", dc.shape)
        dw = np.dot(v0.T, h0_expected)/batch_size - np.dot(vk_sample.T, hk_sample)/batch_size
        # print("dw.shape", dw.shape)
        
        #パラメータ更新
        self.visible_bias += self.learning_rate * db
        self.hidden_bias += self.learning_rate * dc
        self.weights += self.learning_rate * dw

    # ====================================================
    # 【追加】Adamによる勾配更新メソッド
    # ====================================================
    def update_adam(self, batch_data, cd_k=1):
        batch_size = batch_data.shape[0]
        
        # CD法によるサンプリング（通常のupdateと同じ）
        v0, h0_expected, vk_sample, hk_sample = self.cd(batch_data, k=cd_k)
        
        # 勾配の計算（通常のupdateと同じ）
        db = np.mean(v0, axis=0) - np.mean(vk_sample, axis=0)
        dc = np.mean(h0_expected, axis=0) - np.mean(hk_sample, axis=0)
        dw = np.dot(v0.T, h0_expected) / batch_size - np.dot(vk_sample.T, hk_sample) / batch_size
        
        # ステップ数のカウントアップ
        self.t += 1
        
        # --- 1. 重み (Weights) の更新 ---
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)  # バイアス補正
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)  # バイアス補正
        # 勾配上昇法なので、プラス（+）で更新します
        self.weights += self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        # --- 2. 可視バイアス (Visible Bias) の更新 ---
        self.m_vb = self.beta1 * self.m_vb + (1 - self.beta1) * db
        self.v_vb = self.beta2 * self.v_vb + (1 - self.beta2) * (db ** 2)
        m_vb_hat = self.m_vb / (1 - self.beta1 ** self.t)
        v_vb_hat = self.v_vb / (1 - self.beta2 ** self.t)
        self.visible_bias += self.learning_rate * m_vb_hat / (np.sqrt(v_vb_hat) + self.epsilon)

        # --- 3. 隠れバイアス (Hidden Bias) の更新 ---
        self.m_hb = self.beta1 * self.m_hb + (1 - self.beta1) * dc
        self.v_hb = self.beta2 * self.v_hb + (1 - self.beta2) * (dc ** 2)
        m_hb_hat = self.m_hb / (1 - self.beta1 ** self.t)
        v_hb_hat = self.v_hb / (1 - self.beta2 ** self.t)
        self.hidden_bias += self.learning_rate * m_hb_hat / (np.sqrt(v_hb_hat) + self.epsilon)

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
        
        w = self.weights
        b = self.visible_bias
        c = self.hidden_bias
        
        if alpha <= np.log(3)/np.log(2):
            # αが1.58より小さい→可視変数の計算量が大きい→可視層を周辺化したln_Zを計算
            # 隠れ変数の総組み合わせ行列を使用

            # if epoch == 0:
                # print("==== ln_Z_marge_v")
            _H_all = self._all
            # print("_H_all.shape", _H_all.shape)
            phi = _H_all @ w.T + b
            abs_phi = np.abs(phi) #iについての総積（=exp上での総和）はとられているためサイズが(2*num_hidden, "num_visible")
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


        else:
            # αが1.58より大きい→隠れ変数の計算量が大きい→隠れ層を周辺化したln_Zを計算
            # 可視変数の総組み合わせ行列を使用

            # if epoch == 0:
                # print("==== use ln_Z_marge_h")
            _V_all = self._all
            # print("_V_all.shape", _V_all.shape)
            phi = _V_all @ w + c
            # print("phi.shape", phi.shape)
            abs_phi = np.abs(phi)
            term_A = abs_phi + np.log(1+np.exp(-2*abs_phi))
            # print("term_A.shape", term_A.shape)
            term_A = np.sum(term_A, axis=1)
            # print("term_A.shape", term_A.shape)

            term_B = _V_all @ b.T
            # print("term_B.shape", term_B.shape)

            psi = term_B + term_A
            psi_max = np.maximum.reduce(psi)
            # print("psi_max", psi_max)

            term_C = np.exp(psi - psi_max)
            # print("term_C", term_C)
            # print("term_C.shape", term_C.shape)
            
            ln_Z = psi_max + np.log(np.sum(term_C))
        return ln_Z
        
    # def exact_expectation(self):


#main文
path = CONFIG["output_path"]
trials = CONFIG["TRIALS"]
epochs = CONFIG["num_epochs"]

results_dict = {"epoch": np.arange(epochs)}

# # データを3値化（-1, 0, 1）する
# df = df.apply(discretize_to_ternary)
# df = df.values  # DataFrameをNumPy配列に変換
# print(df[:, 1])
# print(df)


for alpha, beta in zip(CONFIG["alpha_list"], CONFIG["beta_max"]):
    print(f"\n========== alpha: {alpha}, beta_max: {beta} ==========")
    
    n_v = CONFIG["num_visible"]
    n_h = CONFIG["num_hidden"](alpha)
    n_samples = CONFIG["num_samples"]
    # print(f"{n_h}/{n_v}={alpha}")

    # α毎にキャンバスを更新
    # fig, ax = plt.subplots()

    # beta_conditions = [
    #     (f"{alpha}_/4", beta/4),
    #     (f"{alpha}_1", beta),
    #     (f"{alpha}_4", 4*beta)
    # ]  # βの条件をリストで定義

    beta_conditions = [
        (f"{alpha}_/4", beta/4)
    ]  # βの条件をリストで定義

    # 各モデルに対する3通りの初期化
    for label, beta in beta_conditions:
        print("===== beta condition:", label, beta, "=====")

        ll_array = np.zeros((trials, epochs))
        # print("ll_array.shape", ll_array.shape)

        for trial in range(CONFIG["TRIALS"]):
            train_data = generate_train_data()
            # train_data = df
            model = DBRBM(
                num_visible=CONFIG["num_visible"],
                num_hidden=CONFIG["num_hidden"](alpha),
                learning_rate=CONFIG["lerning_rate"],
                beta_max=beta
                )
            
            prog = epochs/10

            if CONFIG["update_method"] == "exact":
                print("update_method: exact")
                batch_size = CONFIG["batch_size"] #: データセットのバッチサイズ

                for epoch in range(epochs):
                    # print(f"===== epoch : {epoch} =====")
                    batch_data = train_data

                    # 分配関数の計算
                    # 厳密計算で分配関数の対数の計算を1回で抑えられるように、対数尤度関数計算→分配関数計算の順にしようとしているが、cd法を用いた勾配更新でも同じ順番でも変わらないのでは？
                    ln_Z = model.ln_Z()

                    if(epoch % 1 == 0):
                        ll = model.log_likelihood(train_data)
                        ll_array[trial, epoch] = ll

                    model.update_exact(batch_data)

                    if (epoch+1) % prog == 0:
                        print("■", end="", flush=True)




            elif CONFIG["update_method"] == "sgd":
                print("update_method: sgd")
                batch_size = CONFIG["batch_size"] #: データセットのバッチサイズ
                #n_samples : データセットのサンプル数

                for epoch in range(epochs):
                    #print(f"===== epoch : {epoch} =====")
                    
                    #各epochでデータをシャッフル
                    indices = np.random.permutation(n_samples)
                    # print("indices", indices)
                    shuffled_data = train_data[indices]
                    # print("train_data\n", shuffled_data)

                    for i in range(0, n_samples, batch_size):
                        batch_data = shuffled_data[i : i + batch_size]
                        model.update_sgd (batch_data, cd_k=1)
                    
                    
                    if(epoch % 1 == 0):
                        ll = model.log_likelihood(train_data)
                        ll_array[trial, epoch] = ll
                    
                    if (epoch+1) % prog == 0:
                        print("■", end="", flush=True)
                # print()
        ll_mean = np.mean(ll_array, axis=0)
        print()
        # データを用意
        x = np.arange(epochs)
        y = ll_mean
        results_dict[label] = ll_mean
# 単純にcsvファイルに出力
df = pd.DataFrame(results_dict)
df.to_csv(path, index=False, encoding='utf-8')

# 実験の設定をコメントとして書き込む
comment = f"# CONFIG: {CONFIG}"
print(comment)
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

with open(path, 'w', newline='', encoding='utf-8') as f:
    f.write(comment + "\n" + content)





#         # 描画オブジェクト（fig）とサブプロット（ax）を生成する
#         ax.plot(x,y, label=f"beta ={beta:.5f}")

#         # 折れ線グラフを表示
#         ax.plot(x, y)
#         # グラフにタイトルをつける
#         ax.set_title(f'alpha = {alpha}', fontname="Meiryo")
#         # x軸のラベルを設定
#         plt.xlabel("x軸ラベル")
#         # y軸のラベルを設定
#         plt.ylabel("y軸ラベル")
#         # 凡例を表示
#         ax.legend()
#         # グラフをカレントディレクトリに保存
#         # fig.savefig(f"result{alpha}.svg")
#         # グラフを表示
#         plt.show(block=False)

#         #結果出力
#         #epochごとの対数尤度をcsvファイルに出力
# plt.show()
