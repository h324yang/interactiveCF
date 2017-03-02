import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle

class Base():
    def load_triples(self, path, sep="::"):
        triples = []
        for line in open(path):
            spl = line.strip().split(sep)
            triples.append(spl[:3])

        return np.array(triples).astype(np.float32)

    def save(self, path):
        with open(path, "wb") as out:
            pickle.dump(self, out)

class ICF(Base):
    def __init__(self, size=10, var=0.1, var_p=0.01, var_q=0.01):
        self.size = size
        self.var = var
        self.var_p = var_p
        self.var_q = var_q
        self.lambda_p = self.var / self.var_p
        self.lambda_q = self.var / self.var_q
        self.reward = defaultdict(self.get_zero)

    def get_zero(self):
        return 0.0

    def pretrain(self, tr_data, sample_epoch=5):
        # initializing.
        self.num_u = int(np.amax(tr_data[:, 0])) + 1 # +1 for index 0.
        self.num_i = int(np.amax(tr_data[:, 1])) + 1
        ini_mean = np.zeros(self.size)
        ini_cov_p = np.eye(self.size) * self.var_p
        ini_cov_q = np.eye(self.size) * self.var_q
        self.w_P = np.random.multivariate_normal(ini_mean, ini_cov_p, self.num_u)
        self.w_Q = np.random.multivariate_normal(ini_mean, ini_cov_q, self.num_i)
        self.u_rec = defaultdict(list)
        self.i_rec = defaultdict(list)
        self.mean_r = np.mean(tr_data[:, 2])

        # recording pairs of u and i.
        for tri in tr_data:
            u_id = int(tri[0])
            i_id = int(tri[1])
            r = tri[2]
            self.reward[(u_id, i_id)] = r
            self.u_rec[u_id].append(i_id)
            self.i_rec[i_id].append(u_id)

        # training via gibbs sampling
        for epoch in range(sample_epoch):
            print("[epoch %s]"% str(epoch+1))
            for index in tqdm(range(len(tr_data))):
                tri = tr_data[index]
                u_id = int(tri[0])
                i_id = int(tri[1])

                # sampling and updating user features
                p = self.sample_p(u_id)
                self.w_P[u_id] = p

                # sampling and updating item features
                q = self.sample_q(i_id)
                self.w_Q[i_id] = q

            # computing training error
            preds = self.predict(tr_data)
            RMSE = np.linalg.norm(preds - tr_data[:, 2])/np.sqrt(float(len(tr_data)))
            print("RMSE:%.3f"%RMSE)

    def sample_p(self, u_id):
        items = self.u_rec[u_id]
        D = self.w_Q[items]
        r = np.array([self.reward[u_id, i_id] for i_id in items]) - self.mean_r
        G = np.linalg.inv(np.dot(D.T, D) + np.eye(self.size) * self.lambda_p)
        mean = G.dot(D.T).dot(r).ravel()
        cov = G * self.var

        return np.random.multivariate_normal(mean, cov)

    def sample_q(self, i_id):
        users = self.i_rec[i_id]
        B = self.w_P[users]
        r = np.array([self.reward[u_id, i_id] for u_id in users]) - self.mean_r
        H = np.linalg.inv(np.dot(B.T, B) + np.eye(self.size) * self.lambda_q)
        mean = H.dot(B.T).dot(r).ravel()
        cov = H * self.var

        return np.random.multivariate_normal(mean, cov)

    def predict(self, data):
        u_id = data[:, 0].astype(np.int32)
        i_id = data[:, 1].astype(np.int32)
        preds = np.sum(self.w_P[u_id] * self.w_Q[i_id], 1) + self.mean_r

        return preds

class Interactor():
    def __init__(self, ICF_model):
        assert len(ICF_model.w_Q) > 0, "No item features."
        self.model = ICF_model

    def interact(self, u_id, iters=100):
        p = self.model.sample_p(u_id)
        log = list()
        ALL = len(self.model.u_rec[u_id])
        hit = 0
        for t in range(iters):
            maxval = -1e4
            best_i = None
            for i_id in self.model.i_rec.keys():
                if i_id in log:
                    continue
                q = self.model.sample_q(i_id)
                val = np.dot(p.T, q)
                if val > maxval:
                    maxval = val
                    best_i = i_id
            r_star = self.model.reward[(u_id, best_i)]
            if r_star >= 4:
                hit += 1
            params = (str(t+1), str(u_id), str(best_i), r_star, str(hit), str(ALL))
            print("Iter:%s User:%s Item:%s Reward:%.2f Recall:%s/%s"%params)
            self.model.u_rec[u_id].append(best_i)
            log.append(best_i)




