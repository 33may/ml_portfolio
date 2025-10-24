import pickle

with open("demonstrations.pkl", "rb") as f:
    expert = pickle.load(f)


print(expert)