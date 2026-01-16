# train_model_v1.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import logging
import os
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
logging.info("Starting training for Model V1")

# Load dataset
df = pd.read_csv("Details.csv")
logging.info(f"Dataset loaded with shape {df.shape}")

# Features and target
X = df[["Amount", "Quantity"]]  # v1 uses only these
y = df["Profit"]
logging.info(f"Features used: {X.columns.tolist()}")

# Split dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Initialize and train model
model_v1 = LinearRegression()
model_v1.fit(X_train, y_train)
logging.info("Model V1 trained successfully.")

# -----------------------------
# Evaluate R²
# -----------------------------
train_r2 = model_v1.score(X_train, y_train)
test_r2 = model_v1.score(X_test, y_test)
cv_r2 = np.mean(cross_val_score(model_v1, X, y, cv=5, scoring="r2"))

logging.info(f"V1 Training R2: {train_r2:.3f}")
logging.info(f"V1 Testing R2: {test_r2:.3f}")
logging.info(f"V1 CV R2: {cv_r2:.3f}")

print(f"Training R²: {train_r2:.3f}, Testing R²: {test_r2:.3f}, CV R²: {cv_r2:.3f}")


# Save the trained model
joblib.dump(model_v1, "model_v1.pkl")
logging.info("Model V1 saved as model_v1.pkl")
print("Baseline Linear Regression model saved.")