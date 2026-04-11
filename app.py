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

# Cluster labels
cluster_names = {
    0: "High Value",
    1: "Low Value",
    2: "At Risk",
    3: "New Customers"
}

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

# Title + Description
st.markdown("<h1>Customer Churn Prediction System</h1>", unsafe_allow_html=True)
st.write("This app predicts customer churn based on behavioral patterns and provides actionable business insights.")

st.write("### 📊 Enter Customer Details")

# Inputs
frequency = st.slider("📦 Frequency (Number of Purchases)", 0, 100, 5)

monetary = st.slider("💰 Total Spend ($)", 0, 10000, 500)
st.write(f"💵 Selected Spending: ${monetary:,.2f}")

cluster_label = st.selectbox(
    "👥 Customer Segment",
    list(cluster_names.values())
)

# Convert label → numeric cluster
cluster = [k for k, v in cluster_names.items() if v == cluster_label][0]

st.info(f"Selected Segment: {cluster_label}")

st.markdown("---")

# Prediction
if st.button("🔍 Predict Churn"):
    
    data = np.array([[frequency, monetary, cluster]])

    prob = model.predict_proba(data)[0][1]

    st.markdown("## 🔎 Prediction Result")

    # Show inputs nicely
    st.write(f"📦 Frequency: {frequency}")
    st.write(f"💰 Spending: ${monetary:,.2f}")
    st.write(f"👥 Segment: {cluster_label}")

    # Risk levels
    if prob > 0.7:
        st.error(f"🔥 High Risk Customer ({prob:.2f})")
        st.warning("💡 Suggestion: Immediate retention campaign required.")
    elif prob > 0.4:
        st.warning(f"⚠️ Medium Risk Customer ({prob:.2f})")
        st.info("💡 Suggestion: Engage with offers and recommendations.")
    else:
        st.success(f"💎 Loyal Customer ({prob:.2f})")
        st.info("💡 Suggestion: Upsell and reward loyalty.")

    # 🔥 SHAP Explanation with side panel
    st.markdown("### 🧠 Why this prediction?")

    shap_values = explainer.shap_values(data)

    if isinstance(shap_values, list):
        values = shap_values[1]
    else:
        values = shap_values

    values = np.array(values).reshape(-1)
    values = values[:3]

    feature_names = ['Frequency', 'Monetary ($)', 'Cluster']

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact': values
    }).sort_values(by='Impact', key=abs, ascending=False)

    # Create two columns
    col1, col2 = st.columns([2, 1])

    # LEFT: SHAP chart
    with col1:
        fig, ax = plt.subplots()
        colors = ['green' if v < 0 else 'red' for v in shap_df['Impact']]
        ax.barh(shap_df['Feature'], shap_df['Impact'], color=colors)
        ax.set_title("Feature Impact on Prediction")
        st.pyplot(fig)

    # RIGHT: Explanation
    with col2:
        st.write("### 📖 Explanation")

        for _, row in shap_df.iterrows():
            feature = row['Feature']
            impact = row['Impact']

            if impact > 0:
                st.write(f"🔴 **{feature}** increases churn risk")
            else:
                st.write(f"🟢 **{feature}** reduces churn risk")

# Feature Importance (Global)
st.markdown("---")
st.write("### 📊 Model Feature Importance")

importance = pd.DataFrame({
    'Feature': ['Cluster', 'Frequency', 'Monetary'],
    'Importance': [0.73, 0.16, 0.07]
})

fig, ax = plt.subplots()
ax.bar(importance['Feature'], importance['Importance'])
ax.set_ylabel("Importance")
st.pyplot(fig)

# Footer
st.markdown("---")
st.write("Built by Vignesh M Naik | Data Science Project 🚀")
