import pickle

with open("../gym-pusht/demonstrations.pkl", "rb") as f:
    expert = pickle.load(f)


print(expert)