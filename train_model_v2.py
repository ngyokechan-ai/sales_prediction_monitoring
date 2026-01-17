# train_model_v2.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import logging
import numpy as np

# -----------------------------
# Setup logging
# -----------------------------
log_file = 'model_training.log'
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Starting training for Model V2")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("Details.csv")

# Drop unused columns
drop_cols = ["Order ID"]
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Features and target
features = ["Amount", "Quantity", "Category"]

X = df[features].copy()
y = df["Profit"]

# Identify categorical columns
cat_cols = [col for col in X.columns if X[col].dtype == "object"]
X[cat_cols] = X[cat_cols].astype(str)

# -----------------------------
# Preprocessing: one-hot encode categorical columns
# -----------------------------
preprocess = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

# -----------------------------
# Split dataset into train and test sets (80% train, 20% test)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# -----------------------------
# Build pipeline: preprocessing + Random Forest Regressor
# -----------------------------
model_v2 = Pipeline([
    ("preprocess", preprocess),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,          # limit tree depth
        min_samples_leaf=5,    # minimum samples per leaf
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# Train the model
# -----------------------------
model_v2.fit(X_train, y_train)
logging.info("Model V2 trained successfully.")

# -----------------------------
# Evaluate R²
# -----------------------------
train_r2 = model_v2.score(X_train, y_train)
test_r2 = model_v2.score(X_test, y_test)
cv_r2 = np.mean(cross_val_score(model_v2, X, y, cv=5, scoring="r2"))

logging.info(f"V2 Training R2: {train_r2:.3f}")
logging.info(f"V2 Testing R2: {test_r2:.3f}")
logging.info(f"V2 CV R2: {cv_r2:.3f}")

print(f"Training R²: {train_r2:.3f}, Testing R²: {test_r2:.3f}, CV R²: {cv_r2:.3f}")

# -----------------------------
# Save the trained model
# -----------------------------
joblib.dump(model_v2, "model_v2.pkl")
logging.info("Model V2 saved as model_v2.pkl")
print("Improved Random Forest Regressor model saved as model_v2.pkl")
