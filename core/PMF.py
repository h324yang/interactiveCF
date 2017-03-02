import numpy as np
import pickle

class Base():
    def load_triples(self, path, sep="::"):
        triples = []
        for line in open(path):
            spl = line.strip().split(sep)
            triples.append(spl[:3])

        return np.array(triples).astype(np.float32)

    def pop_batch_ind(self, data, batch_size):
        N = len(data)
        assert N > 0, "No data feeded."
        shuffled = np.arange(N)
        np.random.shuffle(shuffled)
        left = 0
        while left < N:
            right = left + batch_size
            if right < N:
                yield shuffled[left:right]
            else:
                yield shuffled[left:]
            left = right

    def save(self, path):
        with open(path, "wb") as out:
            pickle.dump(self, out)

class PMF(Base):
    def __init__(self, size=10, eta=0.1, var=0.1, var_u=0.01, var_v=0.01, maxepoch=20, batch_size=1000):
        self.size = size
        self._lambda = var / var_u
        self.var = var
        self.var_u = var_u
        self.var_v = var_v
        self.maxepoch = maxepoch
        self.batch_size = batch_size
        self.eta = eta
        self.iter = 0

    def fit(self, tr_data):
        # init
        epoch = 0
        self._mean = np.mean(tr_data[:,2])
        self.num_U = int(np.amax(tr_data[:,0])) + 1 # +1 for index 0.
        self.num_V = int(np.amax(tr_data[:,1])) + 1 # +1 for index 0.
        self.w_U = np.random.randn(self.num_U, self.size)
        self.w_V = np.random.randn(self.num_V, self.size)

        while epoch < self.maxepoch:
            print("Epoch:%s"%int(epoch+1))
            self.batch_learn(tr_data)
            epoch += 1

    def batch_learn(self, tr_data):
        for b_ind in self.pop_batch_ind(tr_data, self.batch_size):
            self.iter += 1
            # batch init
            batch = tr_data[b_ind]
            b_Uid = batch[:,0].astype(np.int32)
            b_Vid = batch[:,1].astype(np.int32)
            b_U = self.w_U[b_Uid]
            b_V = self.w_V[b_Vid]
            grad_U = np.zeros((self.num_U, self.size))
            grad_V = np.zeros((self.num_V, self.size))

            # compute error
            b_preds = np.sum(b_U * b_V, 1) + self._mean # default is mean.
            b_err = b_preds - batch[:,2]

            # compute gradients
            b_grad_U = b_err[:,np.newaxis] * b_V + self._lambda * b_U
            b_grad_V = b_err[:,np.newaxis] * b_U + self._lambda * b_V
            for i in range(len(batch)):
                grad_U[b_Uid[i]] += b_grad_U[i]
                grad_V[b_Vid[i]] += b_grad_V[i]

            self.w_U -= self.eta * grad_U
            self.w_V -= self.eta * grad_V

            # compute training err
            if self.iter % 50 == 0:
                tr_preds = self.predict(tr_data)
                RMSE = np.linalg.norm(tr_preds - tr_data[:,2])/np.sqrt(float(len(tr_data)))
                print("iter:%s RMSE:%.3f"%(self.iter, RMSE))

    def predict(self, data):
        Uid = data[:,0].astype(np.int32)
        Vid = data[:,1].astype(np.int32)
        preds = np.sum(self.w_U[Uid] * self.w_V[Vid], 1) + self._mean

        return preds
