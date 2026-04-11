import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Load model
model = joblib.load("final_model.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Page config
st.set_page_config(page_title="Churn Prediction", layout="centered")

# Styling
st.markdown("""
    <style>
    h1 {
        text-align: center;
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>Customer Churn Prediction System</h1>", unsafe_allow_html=True)

st.write("### 📊 Enter Customer Details")

# Inputs
frequency = st.slider("📦 Frequency (Number of Purchases)", 0, 100, 5)
monetary = st.slider("💰 Monetary Value (Total Spend)", 0, 10000, 500)
cluster = st.selectbox("👥 Customer Segment (Cluster)", [0, 1, 2, 3])

st.markdown("---")

# Prediction
if st.button("🔍 Predict Churn"):
    data = np.array([[frequency, monetary, cluster]])

    prob = model.predict_proba(data)[0][1]

    st.markdown("## 🔎 Prediction Result")

    if prob > 0.5:
        st.error(f"⚠️ High Churn Risk: {prob:.2f}")
        st.warning("💡 Suggestion: Offer discounts or re-engagement campaigns.")
    else:
        st.success(f"✅ Low Churn Risk: {prob:.2f}")
        st.info("💡 Suggestion: Customer is loyal. Try upselling strategies.")

    # 🔥 SHAP Explanation
    st.markdown("### 🧠 Why this prediction?")

    shap_values = explainer.shap_values(data)

    # Handle different SHAP formats safely
    if isinstance(shap_values, list):
        values = shap_values[1]  # class 1
    else:
        values = shap_values

    # Convert to numpy and flatten
    values = np.array(values).reshape(-1)

    # Ensure correct length (match features)
    feature_names = ['Frequency', 'Monetary', 'Cluster']

    # 🔥 FIX: Trim or match length
    values = values[:len(feature_names)]

    # Create DataFrame
    shap_df = pd.DataFrame({
    'Feature': feature_names,
    'Impact': values
    }).sort_values(by='Impact', key=abs, ascending=False)

    # Plot
    fig, ax = plt.subplots()
    ax.barh(shap_df['Feature'], shap_df['Impact'],color=['green' if v < 0 else 'red' for v in shap_df['Impact']])
    ax.set_title("Feature Impact on Prediction")
    st.pyplot(fig)