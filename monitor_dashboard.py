# monitor_dashboard.py
import os
import pandas as pd
import streamlit as st
from log_utils import LOG_PATH

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="Model Monitoring & Feedback", layout="wide")
st.title("Model Monitoring & Feedback Dashboard")


# -------------------- Load Logs --------------------
@st.cache_data
def load_logs():
    # Expected columns
    cols = [
        "timestamp", "model_version", "model_type", "input_summary",
        "prediction", "latency_ms", "feedback_score", "feedback_text",
        "train_r2", "test_r2", "cv_r2"
    ]
    
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(LOG_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)
    
    # Ensure all expected columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    
    # Convert timestamp column to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Convert numeric columns
    numeric_cols = ["prediction","latency_ms","feedback_score","train_r2","test_r2","cv_r2"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Compute average R²
    df["r2_value"] = df[["train_r2","test_r2","cv_r2"]].mean(axis=1, skipna=True)

    return df.sort_values("timestamp")

logs = load_logs()

# -------------------- Handle "no logs yet" --------------------
if logs.empty or logs["timestamp"].isna().all():
    st.warning(
        "No monitoring logs found yet. "
        "Please run the prediction app, submit feedback at least once, and then refresh this page."
    )
    st.stop()

# -------------------- Sidebar Filters --------------------
st.sidebar.header("Filters")
models = ["All"] + sorted(logs["model_version"].dropna().unique())
selected_model = st.sidebar.selectbox("Model version", models)

if selected_model == "All":
    filtered = logs.copy()
else:
    filtered = logs[logs["model_version"] == selected_model]

# -------------------- Key Metrics --------------------
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Predictions", len(filtered))

avg_fb = filtered["feedback_score"].mean() if filtered["feedback_score"].notna().any() else None
col2.metric("Avg Feedback Score", f"{avg_fb:.2f}" if avg_fb is not None else "N/A")

avg_latency = filtered["latency_ms"].mean() if filtered["latency_ms"].notna().any() else None
col3.metric("Avg Latency (ms)", f"{avg_latency:.1f}" if avg_latency is not None else "N/A")

avg_r2 = filtered["r2_value"].mean() if filtered["r2_value"].notna().any() else None
col4.metric("Avg R²", f"{avg_r2:.3f}" if avg_r2 is not None else "N/A")

st.markdown("---")

# -------------------- Tabs --------------------
tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "💬 Feedback Analysis", "📄 Raw Logs"])

# ----------- Tab 1: Model Comparison -----------
with tab1:
    st.subheader("Model Version Comparison (Aggregated)")
    summary = filtered.groupby("model_version").agg({
        "feedback_score": "mean",
        "latency_ms": "mean",
        "train_r2": "mean",
        "test_r2": "mean",
        "cv_r2": "mean",
        "r2_value": "mean"
    }).rename(columns={
        "feedback_score": "avg_feedback_score",
        "latency_ms": "avg_latency_ms",
        "train_r2": "avg_train_r2",
        "test_r2": "avg_test_r2",
        "cv_r2": "avg_cv_r2",
        "r2_value": "avg_r2"
    })

    st.dataframe(summary.style.format({
        "avg_feedback_score": "{:.2f}",
        "avg_latency_ms": "{:.1f}",
        "avg_train_r2": "{:.3f}",
        "avg_test_r2": "{:.3f}",
        "avg_cv_r2": "{:.3f}",
        "avg_r2": "{:.3f}"
    }))

# ----------- Tab 2: Feedback Analysis -----------
with tab2:
    st.subheader("Average Feedback Score by Model Version")
    fb = filtered.groupby("model_version")["feedback_score"].mean().reset_index()
    if not fb.empty:
        st.bar_chart(fb.set_index("model_version"))
    else:
        st.info("No feedback scores available yet.")

    st.subheader("Recent Comments")
    comments = filtered[filtered["feedback_text"].astype(str).str.strip() != ""]
    comments = comments.sort_values("timestamp", ascending=False).head(10)
    if comments.empty:
        st.info("No qualitative comments yet.")
    else:
        for _, row in comments.iterrows():
            st.write(f"**[{row['timestamp']}] {row['model_version']} – Score: {row['feedback_score']}**")
            st.write(row["feedback_text"])
            st.markdown("---")

# ----------- Tab 3: Raw Logs -----------
with tab3:
    st.subheader("Raw Monitoring Logs")
    st.dataframe(filtered)
