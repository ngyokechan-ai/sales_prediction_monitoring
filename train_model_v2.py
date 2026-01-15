# train_model_v2.py
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("Details.csv")  # <-- updated filename

# Drop unused columns
df = df.drop(columns=["Order ID"])

# Features and target
X = df.drop(columns=["Profit"])
y = df["Profit"]

# Identify categorical columns
cat_cols = ["Category", "Sub-Category", "PaymentMode"]

# Preprocessing: one-hot encode categorical columns
preprocess = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

# Build pipeline: preprocessing + Gradient Boosting Regressor
model_v2 = Pipeline([
    ("preprocess", preprocess),
    ("regressor", GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

# Train the model
model_v2.fit(X, y)

# Save the trained model
joblib.dump(model_v2, "model_v2.pkl")
print("Improved Gradient Boosting Regressor model saved as model_v2.pkl")
