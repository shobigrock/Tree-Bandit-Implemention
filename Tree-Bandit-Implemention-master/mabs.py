import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random
from math import exp, sqrt, log
from scipy import optimize  # 最適パラメータ計算用(ニュートン法)

"""
各種バンディットアルゴリズムのクラスを置くファイル

init引数(n_actions, hyper_parameter, n_dims)で統一できる？
"""
def break_tie(_range):
    indices = np.argwhere(_range == np.max(_range))
    index = np.random.randint(0,len(indices))

    return indices[index][0]

class TreeBootstrap:
    # initialise values and raise input errors
    def __init__(self, n_actions, n_dims, tree=DecisionTreeClassifier(max_depth=2)):
        # 決定木はsklearnのDecisionTreeClassifier使用(CART分類木)
        # てっきり回帰木かと思ったが，これの.predict_proba()を利用して推定値を出力しているみたい

        if not type(n_dims) == int:
            raise TypeError("`n_dims` must be integer type")
        self.n_actions = n_actions
        self.n_dims = n_dims
        self.tree = tree
        self.D = [[[] for i in range(n_actions) ] for j in range(1)]  # 過去データ(文脈)．D[0][arm]で行動arm(0インデックス)の過去データを取得
        self.r = [[0 for i in range(n_actions) ] for j in range(1)]  # 最終的には[[0. ] [1. ] [1. ] [1. ] [0. ] [... [1. ]]のような形になってそう
        self.prob = np.zeros(self.n_actions)  # create zero array to save predicted probability from treeclassifier
        self.stopper = 0

        self.features = []
        self.thresholds = []
        self.values = []
        self.preds = []
        # feature = tree.feature
        # threshold = tree.threshold
        # value = tree.value  # 各クラスのデータ数

    def reset(self, n_actions, n_dims, tree=DecisionTreeClassifier(max_depth=2)):
        # 決定木はsklearnのDecisionTreeClassifier使用(CART分類木)
        # てっきり回帰木かと思ったが，これの.predict_proba()を利用して推定値を出力しているみたい

        if not type(n_dims) == int:
            raise TypeError("`n_dims` must be integer type")
        self.n_actions = n_actions
        self.n_dims = n_dims
        self.tree = tree
        self.D = [[[] for i in range(n_actions) ] for j in range(1)]  # 過去データ(文脈)．D[0][arm]で行動arm(0インデックス)の過去データを取得
        self.r = [[0 for i in range(n_actions) ] for j in range(1)]  # 最終的には[[0. ] [1. ] [1. ] [1. ] [0. ] [... [1. ]]のような形になってそう
        self.prob = np.zeros(self.n_actions)  # create zero array to save predicted probability from treeclassifier
        self.stopper = 0

        self.features = []
        self.thresholds = []
        self.values = []
        self.preds = []

    # return the best arm
    # context: 時刻tの文脈情報
    def play(self, context):
        shaped_context = context.values.reshape(1, -1)

        def vstack_for_bootstrap(older, newer):
            if len(older) == 0:
                return newer  # 定義
            else:
                return np.vstack((older, newer))  # 垂直方向に配列を結合

        # 全ての行動について以下forループでブートストラップサンプルを作りたい
        for kaisuu, arm in enumerate(range(self.n_actions)):
            # とりあえず各行動1回は実行してデータ(文脈, 報酬)を回収する
            if len(self.D[0][arm]) == 0:
                # set decision tree to predict 1 regardless of the input
                self.prob[arm] = 1.0  # predict 1
            else:
                # インデックス0を取ってくるだけで許されるのか？
                # 普通に許されなさそう．
                sample_context = self.D[0][arm]
                sample_reward = self.r[0][arm]

                # Bootstrapping
                # 最初の2回は先頭から2つを取る．残りは全体から選ぶ．
                b_context = np.vstack((sample_context[0], sample_context[1]))
                b_reward = np.vstack((sample_reward[0], sample_reward[1]))

                for i in range(len(sample_context)):
                    # i=2以降，ランダムに選んだインデックスsampling_numberをもってサンプリング
                    if i >= 2:
                        sampling_number = random.randint(0, len(sample_context)-1)
                        b_context = vstack_for_bootstrap(b_context, sample_context[sampling_number])
                        b_reward = vstack_for_bootstrap(b_reward, sample_reward[sampling_number])
                # Bootstrapping終了

                # tree = self.tree.fit(sample_context, sample_reward)
                tree = self.tree.fit(b_context, b_reward)          # train the tree classifier -> sample_からb_に変更
                temp_p = tree.predict_proba(shaped_context)      # predict the probability of the current context

                self.prob[arm] = temp_p[0][1]

        arm = break_tie(self.prob)  # [0.1, 0.01, 0.8, ...]行動ごと推定報酬値のargmax

        return arm

    # update
    def update(self, context, action, reward):
        shaped_context = context.values.reshape(1, -1)             # reshape the form

        if len(self.D[0][action]) == 0:
            self.D[0][action] = np.vstack((shaped_context, shaped_context))
            self.r[0][action] = np.vstack((np.array([0.]),np.array([1.])))

        self.D[0][action] = np.vstack((self.D[0][action], shaped_context))
        self.r[0][action] = np.vstack((self.r[0][action], reward))

class TreeUCB:
    # initialise values and raise input errors
    def __init__(self, n_actions, n_dims, tree=DecisionTreeClassifier(max_depth=2)):
        # 決定木はsklearnのDecisionTreeClassifier使用(CART分類木)
        # てっきり回帰木かと思ったが，これの.predict_proba()を利用して推定値を出力しているみたい

        if not type(n_dims) == int:
            raise TypeError("`n_dims` must be integer type")
        self.n_actions = n_actions
        self.n_dims = n_dims
        self.tree = tree
        self.D = [[[] for i in range(n_actions) ] for j in range(1)]  # 過去データ(文脈)．D[0][arm]で行動arm(0インデックス)の過去データを取得
        self.r = [[0 for i in range(n_actions) ] for j in range(1)]  # 最終的には[[0. ] [1. ] [1. ] [1. ] [0. ] [... [1. ]]のような形になってそう
        self.prob = np.zeros(self.n_actions)  # create zero array to save predicted probability from treeclassifier
        self.stopper = 0

        self.features = []
        self.thresholds = []
        self.values = []
        self.preds = []
        # feature = tree.feature
        # threshold = tree.threshold
        # value = tree.value  # 各クラスのデータ数

    def reset(self, n_actions, n_dims, tree=DecisionTreeClassifier(max_depth=2)):
        # 決定木はsklearnのDecisionTreeClassifier使用(CART分類木)
        # てっきり回帰木かと思ったが，これの.predict_proba()を利用して推定値を出力しているみたい

        if not type(n_dims) == int:
            raise TypeError("`n_dims` must be integer type")
        self.n_actions = n_actions
        self.n_dims = n_dims
        self.tree = tree
        self.D = [[[] for i in range(n_actions) ] for j in range(1)]  # 過去データ(文脈)．D[0][arm]で行動arm(0インデックス)の過去データを取得
        self.r = [[0 for i in range(n_actions) ] for j in range(1)]  # 最終的には[[0. ] [1. ] [1. ] [1. ] [0. ] [... [1. ]]のような形になってそう
        self.prob = np.zeros(self.n_actions)  # create zero array to save predicted probability from treeclassifier
        self.stopper = 0

        self.features = []
        self.thresholds = []
        self.values = []
        self.preds = []

    # return the best arm
    # context: 時刻tの文脈情報
    def play(self, context):
        shaped_context = context.values.reshape(1, -1)

        # 全ての行動について以下forループでブートストラップサンプルを作りたい
        for kaisuu, arm in enumerate(range(self.n_actions)):
            # とりあえず各行動1回は実行してデータ(文脈, 報酬)を回収する
            if len(self.D[0][arm]) == 0:
                # set decision tree to predict 1 regardless of the input
                self.prob[arm] = 1.0  # predict 1
            else:
                # インデックス0を取ってくるだけで許されるのか？
                # 普通に許されなさそう．
                sample_context = self.D[0][arm]
                sample_reward = self.r[0][arm]

                # tree = self.tree.fit(sample_context, sample_reward)
                tree = self.tree.fit(sample_context, sample_reward)          # train the tree classifier -> sample_からb_に変更
                temp_p = tree.predict_proba(shaped_context)      # predict the probability of the current context

                # UCB処理部分
                expected_reward = temp_p[0][1]  # p(θ_a, x_t)
                data_num_list = tree.tree_.value  # idxを指定することで各ノードに落ちているデータの数を探すことが出来るリスト．
                n_i = sum(data_num_list[tree.apply(shaped_context)[0]][0])  # 文脈を入力とした際に葉ノードに落ちるデータの数． sum([その葉ノードに落ちる報酬0のデータ数, その葉ノードに落ちる報酬1のデータ数])

                # upper_confidence_bound = delta * (np.log(parent_data_num) / n_i)**0.5  # 信頼区間上限の計算

                z_a = 1.96  # 2.5%信頼上界：1.96，5%信頼上界：1.645?
                upper_confidence_bound_true = z_a*((expected_reward*(1-expected_reward)/n_i)**0.5)
                self.prob[arm] = temp_p[0][1] + upper_confidence_bound_true  # 推定確率 + 信頼区間上限で推定スコア


        arm = break_tie(self.prob)  # [0.1, 0.01, 0.8, ...]行動ごと推定報酬値のargmax

        return arm

    # update
    def update(self, context, action, reward):
        shaped_context = context.values.reshape(1, -1)             # reshape the form

        if len(self.D[0][action]) == 0:
            self.D[0][action] = np.vstack((shaped_context, shaped_context))
            self.r[0][action] = np.vstack((np.array([0.]),np.array([1.])))

        self.D[0][action] = np.vstack((self.D[0][action], shaped_context))
        self.r[0][action] = np.vstack((self.r[0][action], reward))

class EpsGreedy:

    """Epsilon-Greedy multi-armed bandit

    Parameters
    ----------
    n_arms : int
        Number of arms

    epsilon : float
        Explore probability. Must be in the interval [0, 1].

    Q0 : float, default=np.inf
        Initial value for the arms.
    """
    # initialise values and raise input errors
    def __init__(self, n_arms, epsilon, Q0=np.inf):
        if not (epsilon >= 0 and epsilon <= 1):
            raise ValueError("`epsilon` must be a number in [0,1]")
        if not type(epsilon) == float:
            raise TypeError("`epsilon` must be float")
        if not type(Q0) == float:
            raise TypeError("`Q0` must be a float number or default value 'np.inf'")

        self.epsilon = epsilon
        self.q = np.full(n_arms, Q0)      # initialise q values
        self.rewards = np.zeros(n_arms)     # keep the total rewards per arm
        self.clicks = np.zeros(n_arms)      # count the pulled rounds per arm
        self.n_arms = n_arms

    def reset(self, n_arms, epsilon, Q0=np.inf):
        if not (epsilon >= 0 and epsilon <= 1):
            raise ValueError("`epsilon` must be a number in [0,1]")
        if not type(epsilon) == float:
            raise TypeError("`epsilon` must be float")
        if not type(Q0) == float:
            raise TypeError("`Q0` must be a float number or default value 'np.inf'")

        self.epsilon = epsilon
        self.q = np.full(n_arms, Q0)      # initialise q values
        self.rewards = np.zeros(n_arms)     # keep the total rewards per arm
        self.clicks = np.zeros(n_arms)      # count the pulled rounds per arm
        self.n_arms = n_arms

    # select a random arm to explore or a arm with best rewards to exploit, then return the arm
    def play(self, context=None):
        if np.random.random_sample() <= self.epsilon:           #explore
            arm = np.random.randint(0,self.n_arms)
        else:
            arm = break_tie(self.q)
        return arm

    # update values
    def update(self, context, action, reward):
        self.clicks[action] += 1
        self.rewards[action] += reward
        self.q[action] = self.rewards[action] / self.clicks[action]

class LogisticUCB:
    def __init__(self, n_actions, n_dims, delta, T):

        if not type(n_dims) == int:
            raise TypeError("`n_dims` must be integer type")
        self.n_actions = n_actions
        self.n_dims = n_dims
        self.D = [np.ones(self.n_dims) for _ in range(n_actions) ] # D[action_idx]: 各行動を実行したときの文脈情報
        self.r = [[] for _ in range(n_actions)]  # 行動ごとに得られた報酬
        self.X = [np.eye(self.n_dims) for _ in range(n_actions) ]  # 行動ごとに過去の文脈の和を取る．正則化のため単位行列が初期値
        self.prob = np.zeros(self.n_actions)  # 行動選択確率
        # -----------------{以下、LogisticUCB特有項}---------------------------------------------
        self.T = T  # 何回シミュレーション試行するのか
        self.delta = delta  # 信頼係数
        self.t = [0]*(n_actions)  # 各行動の試行回数
        # ロジスティック回帰パラメータ，行動ごとに文脈と同次元のパラメータ
        self.theta = [np.ones(self.n_dims) for _ in range(n_actions)] # * initial_value  # 回帰パラメータ，初期値わからんので長さd，すべて大きさ1のベクトルに

        # 以下，計算用関数
        self.L = lambda x: exp(x) / (1 + exp(x)) if x < 10 else 1  # x>=10はL(x)=1とみなす（オーバーフローしてしまうので，，）
        self.rho = lambda x: sqrt(self.n_dims*log(x)*log(x*self.T / self.delta))  


    # return the best arm
    # context: 時刻tの文脈情報 -> (DやXには含まれない) -> 擬似アルゴリズムと合わせてx_tと表記する
    def play(self, context):
        x_t = context.values.reshape(1, -1)  # like [1, 2, ...]: 1行にまとめる、-1は列数が要素数に対応することを意味

        for kaisuu, arm in enumerate(range(self.n_actions)):
            # とりあえず各行動1回は実行してデータ(文脈, 報酬)を回収する（他手法と同様）
            if (self.D[arm] == np.ones(self.n_dims)).all():
                self.prob[arm] = 1.0
            else:
                # アルゴリズム通り行動選択確率を計算
                # print(x_t)
                # print(self.theta[arm])
                # print(x_t @ self.theta[arm])
                UCB_score = self.L(x_t @ self.theta[arm]) + self.rho(max(len(self.D[arm])-2, 1)) * sqrt(x_t @ np.linalg.inv(self.X[arm]) @ x_t.T)
                self.prob[arm] = UCB_score
                # print(UCB_score)

        arm = break_tie(self.prob)  # [0.1, 0.01, 0.8, ...]行動ごと推定報酬値のargmax
        return arm

    # 選択行動の文脈・報酬情報更新＋ニュートン法によるパラメータ最適化
    def update(self, context, action, reward):
        shaped_context = context.values.reshape(1, -1)


        if (self.D[action] == np.ones(self.n_dims)).all():
            self.D[action] = np.vstack((shaped_context, shaped_context, shaped_context))
            self.r[action] = np.vstack((np.array([0.]),np.array([1.]), reward))  # 今回の結果も反映するが，最初は0と1を一つずつ入れる
        else:
            self.D[action] = np.vstack((self.D[action], shaped_context))
            self.r[action] = np.vstack((self.r[action], reward))

        # 行動actionの文脈に関する計画行列(n_dims × n_dims)
        self.X[action] = self.X[action] + shaped_context.T @ shaped_context
        # 行動actionの試行回数++;
        self.t[action] += 1

        # 【訂正】内積計算をnp.dot()で書く
        def func(theta_):
            # print(shaped_context)
            # print(self.theta)
            # print(np.dot(shaped_context, self.theta))
            tmp = exp(max(50, np.dot(shaped_context, self.theta[action])))  # オーバーフロー回避
            # print(self.D)
            # print(self.r)
            # print(self.t)
            return [  # 文脈の次元の数だけ方程式が立つ
                np.array([sum([self.D[action][i][j]*(self.r[action][i][0]-tmp) for i in range(self.t[action])]) for j in range(self.n_dims)
            ]).reshape(-1)]  # len(self.D)-2の-2は，D = D_1, D_1, D_1, D_2, D_3, ...とD_1が3個入るようになっているため
        
        # パラメータの初期値: 全ての要素=1
        result = optimize.root(func, np.array([1.0]*self.n_dims), method="broyden1")
        # パラメータ更新
        self.theta[action] = result.x