from core.ICF import ICF, Interactor
import pickle

DATA = "ml-1m/ratings.dat"
OUT = ""

# # pretraining model
# model = ICF(var=0.2, var_p=0.01, var_q=0.01)
# tr_data = model.load_triples(DATA)
# model.pretrain(tr_data)
# model.save(OUT)

# # if model is pretrained already, then...
with open(OUT, "rb") as f:
    model = pickle.load(f)

interactor = Interactor(model)
interactor.interact(2)

