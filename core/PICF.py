import numpy as np
from .PMF import PMF
from collections import defaultdict

class Qfunc():
    def fit(self, triples):
        self.triples = defaultdict(lambda: 0.)
        for tri in triples:
            Uid = int(tri[0])
            Vid = int(tri[1])
            rating = tri[2]
            self.triples[(Uid, Vid)] = rating

    def reward(self, Uid, Vid):
        return self.triples[(Uid, Vid)]

class ICF(PMF):
    def pretrain(self, tr_data, method="PMF"):
        # init
        self.rec_U = defaultdict(list)
        self.rec_V = defaultdict(list)
        for tri in tr_data:
            self.rec_U[int(tri[0])].append((int(tri[1]), tri[2]))
            self.rec_V[int(tri[1])].append((int(tri[0]), tri[2]))

        if method == "PMF":
            self.fit(tr_data)
            self.theta = dict()
            for v in self.rec_V.keys():
                Uid = []
                ratings = []
                for user, rating in self.rec_V[v]:
                    Uid.append(user)
                    ratings.append(rating)
                B = self.w_U[np.array(Uid)]
                r = np.array(ratings)[:,np.newaxis]
                G = np.dot(B.T, B) + self._lambda * np.eye(B.shape[1])
                mu = np.linalg.inv(G).dot(B.T).dot(r).ravel()
                cov = np.linalg.inv(G) * self.var
                self.theta[v] = (mu, cov)

    def interact(self, Uid, qfunc, iters=120):
        # init
        assert len(self.w_V) > 0, "No item vectors."
        Vid = []
        ratings = []
        log = []
        for item, rating in self.rec_U[Uid]:
            Vid.append(item)
            ratings.append(rating)
        D = self.w_V[np.array(Vid)]
        r = np.array(ratings)[:, np.newaxis]
        G = np.dot(D.T, D) + self._lambda * np.eye(D.shape[1])
        mu = np.linalg.inv(G).dot(D.T).dot(r).ravel()
        cov = np.linalg.inv(G) # set sigma as 1
        p = np.random.multivariate_normal(mu, cov)
        for i in range(iters):
            maxval = -1e4
            bestv = None
            for v, hyper in self.theta.items():
                if v in log:
                    continue
                q = np.random.multivariate_normal(hyper[0], hyper[1])
                val = np.dot(p.T, q)
                if val > maxval:
                    maxval = val
                    bestv = v
            rstar = qfunc(Uid, bestv)
            params = (str(i+1), str(Uid), str(bestv), rstar)
            print("Iter:%s User:%s Item:%s Reward:%.2f"%params)
            self.rec_U[Uid].append((bestv, rstar))
            log.append(bestv)




