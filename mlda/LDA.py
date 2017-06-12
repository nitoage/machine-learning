#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy as np

class LDA:
    def __init__(self, K, alpha, beta, docs, V, smartinit=True, seed=0):
        np.random.seed(seed)
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs
        self.V = V

        self.z_m_n = [] # topics of words of documents
        self.n_m_z = np.zeros((len(self.docs), K)) + alpha     # word count of each document and topic
        self.n_z_t = np.zeros((K, V)) + beta # word count of each topic and vocabulary
        self.n_z = np.zeros(K) + V * beta    # word count of each topic

        self.N = 0
        for m, doc in enumerate(docs):
            self.N += len(doc)
            z_n = []
            for t in doc:
                if smartinit:
                    p_z = self.n_z_t[:, t] * self.n_m_z[m] / self.n_z
                    z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                else:
                    z = np.random.randint(0, K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_t[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(np.array(z_n))

    def inference(self):
        """learning once iteration"""
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                z = z_n[n]
                n_m_z[z] -= 1
                self.n_z_t[z, t] -= 1
                self.n_z[z] -= 1

                # sampling topic new_z for t
                p_z = self.n_z_t[:, t] * n_m_z / self.n_z
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                z_n[n] = new_z
                n_m_z[new_z] += 1
                self.n_z_t[new_z, t] += 1
                self.n_z[new_z] += 1

    def worddist(self):
        """get topic-word distribution"""
        return self.n_z_t / self.n_z[:, np.newaxis]

    def topicdist(self):
        """get topic-document distribution"""
        n_m = np.apply_along_axis(sum, 1, self.n_m_z)
        return self.n_m_z / n_m[:, np.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= np.log(np.inner(phi[:,w], theta))
            N += len(doc)
        return np.exp(log_per / N)

    def fit(self, thresold=1.0e-1, max_itr=1000):
        perp = [1e10]
        itr = 0
        while True:
            itr += 1
            for i in range(10):
                self.inference()
            perp.append(self.perplexity())
            if (perp[-2] - perp[-1] < threshold) and (perp[-2] - perp[-1] > 0):
                break
            if itr > max_itr:
                berak
        del perp[0]
        return perp



class MLDA:
    """ マルチモーダルＬＤＡ
        N種類の事象(ドキュメント)があり、各事象は
        M個のモーダルそれぞれについての観測系列を生成する
        各観測系列は、モーダル毎に異なる語彙数の単語系列とみなすことができ
        その語彙数を V1, V2, ..Vm..VMとする
        これらの観測系列は、潜在変数のK個のトピックに基づき生成されたとする
        事象ごとのトピック分布は、モーダルによらず同一とする
        これを gibbs samplingにより各事後確率を求める
    """
    def __init__(self, K, alpha, beta, docs, V, smartinit=True, seed=0):
        """コンストラクタ
           引数
             K: トピック数
             alpha: トピック生成に関するハイパーパラメータ(スカラ)
             beta: 各モダールの単語生成に関するハイパーパラメータ(M次元ベクトル)
             docs: モーダル×文章×単語の3次元リスト
             V: 各モーダルの語彙数 (M次元ベクトル)

        """
        np.random.seed(seed)
        self.K = K
        self.alpha = alpha # parameter of topics prior
        self.beta = beta   # parameter of words prior
        self.docs = docs
        self.V = V
        # モーダル数が同じかどうかのチェック
        if len(beta) != len(V) or len(beta) != len(docs):
            raise ValueError('# of modal mismatch')
        for i in range(1, len(beta)):
            if len(docs[i-1]) != len(docs[i]):
                print len(docs[i-1])
                print len(docs[i])
                raise ValueError('# of modal mismatch')
        self.M = len(V)

        # 各モーダル、文章ごとのトピック系列
        self.z_m_d_n = []
        # 事象ごとのトピックの出現数
        self.n_d_z = np.zeros((len(self.docs[0]), K)) + alpha
        self.n_z_t = []
        self.n_z = []
        for m in range(self.M):
            # モーダルmの語彙ごとのトピック出現数
            self.n_z_t.append( np.zeros((K, V[m])) + beta[m] )
            # モーダルmのトピック出現数
            self.n_z.append( np.zeros(K) + V[m] * beta[m] )

        self.N = [0] * self.M
        for m in range(self.M):
            z_d_n = []
            for d, doc in enumerate(docs[m]):
                self.N[m] += len(doc)
                z_n = []
                for t in doc:
                    if smartinit:
                        p_z = self.n_d_z[d] * self.n_z_t[m][:, t] / self.n_z[m]
                        z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                    else:
                        z = np.random.randint(0, K)
                    z_n.append(z)
                    self.n_d_z[d, z] += 1
                    self.n_z_t[m][z, t] += 1
                    self.n_z[m][z] += 1
                z_d_n.append(np.array(z_n))
            self.z_m_d_n.append(z_d_n)

    def inference(self):
        """learning once iteration"""
        for m in range(self.M):
            z_d_n = self.z_m_d_n[m]
            for d, doc in enumerate(self.docs[m]):
                z_n = z_d_n[d]
                n_d_z = self.n_d_z[d]
                for n, t in enumerate(doc):
                    # discount for n-th word t with topic z
                    z = z_n[n]
                    n_d_z[z] -= 1
                    self.n_z_t[m][z, t] -= 1
                    self.n_z[m][z] -= 1

                    # sampling topic new_z for t
                    p_z = n_d_z * self.n_z_t[m][:, t] / self.n_z[m]
                    new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                    # set z the new topic and increment counters
                    z_n[n] = new_z
                    n_d_z[new_z] += 1
                    self.n_z_t[m][new_z, t] += 1
                    self.n_z[m][new_z] += 1

    def worddist(self):
        """get topic-word distribution"""
        result = []
        for m in range(self.M):
            result.append( self.n_z_t[m] / self.n_z[m][:, np.newaxis] )
        return result

    def topicdist(self):
        """get topic-document distribution"""
        #Kalpha = self.K * self.alpha
        #n_m = Kalpha + self.M \
        #      * np.array([len(self.docs[i]) for i in range(len(self.docs))])
	n_m = np.apply_along_axis(sum, 1, self.n_d_z)
        return self.n_d_z / n_m[:, np.newaxis]

    def perplexity(self, docs=None):
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        theta = self.topicdist()
        for m in range(self.M):
            for d, doc in enumerate(docs[m]):
                for w in doc:
                    log_per -= np.log(np.inner(phi[m][:,w], theta[d]))
                N += len(doc)
        return np.exp(log_per / N)

    def fit(self, threshold=1.0e-1, max_itr=1000):
        perp = [1e10]
        itr = 0
        while True:
            itr += 1
            for i in range(10):
                self.inference()
            perp.append(self.perplexity())
            if (perp[-2] - perp[-1] < threshold) and (perp[-2] - perp[-1] > 0):
                break
            if itr > max_itr:
                break
        del perp[0]
        return perp
