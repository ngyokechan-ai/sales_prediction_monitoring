# predictive_app.py
import time
import pandas as pd
import joblib
import streamlit as st
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
sub_category = st.sidebar.selectbox("Sub-Category", df["Sub-Category"].unique())
payment = st.sidebar.selectbox("Payment Mode", df["PaymentMode"].unique())

# Input dataframe
input_df = pd.DataFrame({
    "Amount": [amount],
    "Quantity": [quantity],
    "Category": [category],
    "Sub-Category": [sub_category],
    "PaymentMode": [payment]
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
    input_v2 = input_df[["Amount", "Quantity", "Category", "Sub-Category", "PaymentMode"]]
    new_pred = model_v2.predict(input_v2)[0]

    latency_ms = (time.time() - start_time) * 1000.0

    # Store predictions in session
    st.session_state["old_pred"] = float(old_pred)
    st.session_state["new_pred"] = float(new_pred)
    st.session_state["latency_ms"] = float(latency_ms)
    st.session_state["input_summary"] = (
        f"Amount={amount}, Quantity={quantity}, "
        f"Category={category}, Sub-Category={sub_category}, Payment={payment}"
    )
    st.session_state["pred_ready"] = True

# ---------- Show Predictions ----------
if st.session_state["pred_ready"]:
    st.subheader("Predictions")
    st.write(f"V1 Model (Amount + Quantity): **${st.session_state['old_pred']:,.2f}**")
    st.write(f"V2 Model (All features): **${st.session_state['new_pred']:,.2f}**")
    st.write(f"Latency: {st.session_state['latency_ms']:.1f} ms")
else:
    st.info("Click **Run Prediction** to see model outputs before giving feedback.")

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
        log_prediction(
            model_version="v1",
            model_type="baseline",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["old_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )
        log_prediction(
            model_version="v2",
            model_type="improved",
            input_summary=st.session_state["input_summary"],
            prediction=st.session_state["new_pred"],
            latency_ms=st.session_state["latency_ms"],
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )
        st.success("Feedback saved! You can now check the monitoring logs.")
