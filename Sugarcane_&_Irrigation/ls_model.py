import joblib


data = joblib.load(r"D:\study\ai\Projects\4-ML Project\2-Predicting Irrigation Need\lr_model_st.pkl")

model = data["model"]
features = data["features"]

print(model)