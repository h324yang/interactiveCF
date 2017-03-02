from core.PICF import Qfunc, ICF
import pickle

D = "ml-1m/ratings.dat"
OUT = "model/s10r10.pimf"
model = ICF(size=20, eta=0.05, var=0.001, var_u=10, maxepoch=100)
tr = model.load_triples(D)
q = Qfunc()
q.fit(tr)

model.pretrain(tr)
model.save(OUT)

# with open(OUT, "rb") as f:
    # model = pickle.load(f)

model.interact(2, q.reward)

