import pickle

model = pickle.load(open("final_leakage_free_model.pkl", "rb"))
print(model.feature_names_in_)