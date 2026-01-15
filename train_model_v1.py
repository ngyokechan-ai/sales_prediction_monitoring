import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("Details.csv")

X = df[["Amount", "Quantity"]]
y = df["Profit"]

model_v1 = LinearRegression()
model_v1.fit(X, y)

joblib.dump(model_v1, "model_v1.pkl")
print("Baseline model saved.")
