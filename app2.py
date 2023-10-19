import pickle

model = pickle.load(open("model.pkl", "rb"))
cols_when_model_builds = model.get_booster().feature_names
print(cols_when_model_builds)