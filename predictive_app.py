# predictive_app.py
import time
import pandas as pd
import joblib
import streamlit as st
import json
from log_utils import log_prediction

st.set_page_config(page_title="Profit Prediction App", layout="centered")
st.title("Profit Prediction App with Monitoring")

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    # V1: baseline – Amount + Quantity
    model_v1 = joblib.load("model_v1.pkl")
    # V2: improved – uses Amount, Quantity + Category, Sub-Category, PaymentMode
    model_v2 = joblib.load("model_v2.pkl")
    return model_v1, model_v2

model_v1, model_v2 = load_models()

# ---------- Load R2 from training log ----------
# ---------- Load R2 from training log ----------
def extract_r2_from_log(model_version):
    try:
        with open("model_training.log") as f:
            lines = f.readlines()
        train_r2 = test_r2 = cv_r2 = None
        for line in reversed(lines):
            if model_version == "v1" and "V1 Training R2" in line:
                train_r2 = float(line.split(":")[-1])
            elif model_version == "v1" and "V1 Testing R2" in line:
                test_r2 = float(line.split(":")[-1])
            elif model_version == "v1" and "V1 CV R2" in line:
                cv_r2 = float(line.split(":")[-1])
            elif model_version == "v2" and "V2 Training R2" in line:
                train_r2 = float(line.split(":")[-1])
            elif model_version == "v2" and "V2 Testing R2" in line:
                test_r2 = float(line.split(":")[-1])
            elif model_version == "v2" and "V2 CV R2" in line:
                cv_r2 = float(line.split(":")[-1])
        return train_r2, test_r2, cv_r2
    except FileNotFoundError:
        return None, None, None


train_r2_v1, test_r2_v1, cv_r2_v1 = extract_r2_from_log("v1")
train_r2_v2, test_r2_v2, cv_r2_v2 = extract_r2_from_log("v2")

# ---------- Initialize session state ----------
if "pred_ready" not in st.session_state:
    st.session_state["pred_ready"] = False
if "old_pred" not in st.session_state:
    st.session_state["old_pred"] = None
if "new_pred" not in st.session_state:
    st.session_state["new_pred"] = None
if "latency_ms" not in st.session_state:
    st.session_state["latency_ms"] = None
if "input_summary" not in st.session_state:
    st.session_state["input_summary"] = ""

# ---------- Sidebar: Inputs ----------
st.sidebar.header("Input Parameters")

amount = st.sidebar.number_input("Amount", min_value=1, value=100)
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)

# Load unique values from your dataset for dropdowns
df = pd.read_csv("Details.csv")
category = st.sidebar.selectbox("Category", df["Category"].unique())

# Input dataframe
input_df = pd.DataFrame({
    "Amount": [amount],
    "Quantity": [quantity],
    "Category": [category],
})

st.subheader("Input Summary")
st.write(input_df)

# ---------- Run Prediction ----------
if st.button("Run Prediction"):
    start_time = time.time()

    # V1: baseline
    input_v1 = input_df[["Amount", "Quantity"]]
    old_pred = model_v1.predict(input_v1)[0]

    # V2: improved
    input_v2 = input_df[["Amount", "Quantity", "Category"]]
    new_pred = model_v2.predict(input_v2)[0]

    latency_ms = (time.time() - start_time) * 1000.0

    # Store predictions in session
    st.session_state["old_pred"] = float(old_pred)
    st.session_state["new_pred"] = float(new_pred)
    st.session_state["latency_ms"] = float(latency_ms)
    st.session_state["input_summary"] = (
        f"Amount={amount}, Quantity={quantity}, "
        f"Category={category}"
    )
    st.session_state["pred_ready"] = True

# ---------- Show Predictions ----------
if st.session_state.get("pred_ready", False):
    st.subheader("Predictions")
    st.write(f"V1 Model: ${st.session_state['old_pred']:,.2f}")
    st.write(f"Training R²: {train_r2_v1}, Testing R²: {test_r2_v1}, CV R²: {cv_r2_v1}")

    st.write(f"V2 Model: ${st.session_state['new_pred']:,.2f}")
    st.write(f"Training R²: {train_r2_v2}, Testing R²: {test_r2_v2}, CV R²: {cv_r2_v2}")

    st.write(f"Latency: {st.session_state['latency_ms']:.1f} ms")
else:
    st.info("Click **Run Prediction** to see model outputs.")

# ---------- Feedback ----------
st.subheader("Your Feedback")

feedback_score = st.slider(
    "How useful were these predictions? (1 = Poor, 5 = Excellent)",
    min_value=1, max_value=5, value=4
)
feedback_text = st.text_area("Comments (optional)")

if st.button("Submit Feedback"):
    if not st.session_state["pred_ready"]:
        st.warning("Please run the prediction first!")
    else:
        # Log V1 prediction with three R² values
        log_prediction(
            model_version="v1",
            model_type="baseline",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["old_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
            train_r2=train_r2_v1,
            test_r2=test_r2_v1,
            cv_r2=cv_r2_v1
        )

        # Log V2 prediction with three R² values
        log_prediction(
            model_version="v2",
            model_type="improved",  # <-- this was missing
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["new_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
            train_r2=train_r2_v2,
            test_r2=test_r2_v2,
            cv_r2=cv_r2_v2
        )

        st.success("Feedback saved! You can now check the monitoring logs.")