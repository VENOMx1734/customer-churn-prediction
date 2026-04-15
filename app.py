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
    0: "High Value (Frequent + High Spend)",
    1: "Low Value (Infrequent + Low Spend)",
    2: "At Risk (Previously active, now inactive)",
    3: "New Customers (Recently acquired)"
}

# Page config
st.set_page_config(page_title="Churn Prediction", layout="centered")

# Title
st.title("Customer Churn Prediction System")
st.write("Predict churn, understand why, and take action.")

# ---------------- SINGLE CUSTOMER ---------------- #
st.markdown("## 🔹 Single Customer Prediction")

frequency = st.slider("📦 Monthly Orders", 0, 50, 5)

# ✅ UPDATED HERE
monetary = st.slider("💰 Avg Spend per Customer ($)", 0, 1000, 100)
st.write(f"💵 Selected Spending: ${monetary:,.2f}")

cluster_label = st.selectbox("👥 Customer Segment", list(cluster_names.values()))
cluster = [k for k, v in cluster_names.items() if v == cluster_label][0]

st.info(f"Selected Segment: {cluster_label}")

if st.button("🔍 Predict Churn"):
    
    data = np.array([[frequency, monetary, cluster]])
    prob = model.predict_proba(data)[0][1]

    st.markdown("## 🔎 Prediction Result")

    if prob > 0.7:
        st.error(f"🔥 High Risk ({prob:.2f})")
        st.warning("💡 Immediate retention campaign required.")
    elif prob > 0.4:
        st.warning(f"⚠️ Medium Risk ({prob:.2f})")
        st.info("💡 Engage with offers.")
    else:
        st.success(f"💎 Low Risk ({prob:.2f})")
        st.info("💡 Upsell opportunity.")

    # SHAP
    st.markdown("### 🧠 Why this prediction?")

    shap_values = explainer.shap_values(data)
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    values = np.array(values).reshape(-1)[:3]

    feature_names = ['Frequency', 'Monetary ($)', 'Cluster']

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact': values
    }).sort_values(by='Impact', key=abs, ascending=False)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots()
        colors = ['green' if v < 0 else 'red' for v in shap_df['Impact']]
        ax.barh(shap_df['Feature'], shap_df['Impact'], color=colors)
        ax.set_title("Feature Impact")
        st.pyplot(fig)

    with col2:
        st.write("### 📖 Explanation")
        for _, row in shap_df.iterrows():
            if row['Impact'] > 0:
                st.write(f"🔴 {row['Feature']} increases churn")
            else:
                st.write(f"🟢 {row['Feature']} reduces churn")

# ---------------- BULK PREDICTION ---------------- #
st.markdown("---")
st.markdown("## 📊 Bulk Prediction Dashboard")

uploaded_file = st.file_uploader("Upload CSV (Frequency, Monetary, Cluster)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Uploaded Data")
    st.dataframe(df.head())

    try:
        X = df[['Frequency', 'Monetary', 'Cluster']]

        probs = model.predict_proba(X)[:, 1]

        df['Churn Probability'] = probs
        df['Risk Level'] = pd.cut(
            probs,
            bins=[0, 0.4, 0.7, 1],
            labels=["Low", "Medium", "High"]
        )

        # Dashboard
        st.markdown("## 📊 Summary Dashboard")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Customers", len(df))
        col2.metric("High Risk", (df['Risk Level'] == "High").sum())
        col3.metric("Avg Churn", f"{df['Churn Probability'].mean():.2f}")

        # Chart
        st.markdown("### 📈 Risk Distribution")

        fig, ax = plt.subplots()
        df['Risk Level'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

        # Insights
        st.markdown("## 🧠 Insights")

        high_pct = (df['Risk Level'] == "High").mean() * 100

        if high_pct > 30:
            st.error("⚠️ High churn risk across customers!")
        elif high_pct > 15:
            st.warning("⚠️ Moderate churn risk detected.")
        else:
            st.success("✅ Customer base stable.")

        # Top customers
        st.markdown("## 🔥 Top At-Risk Customers")

        top_risk = df[df['Risk Level'] == "High"].sort_values(
            by='Churn Probability', ascending=False
        )

        st.dataframe(top_risk.head(10))

        # Download
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Download Results",
            csv,
            "churn_predictions.csv",
            "text/csv"
        )

    except:
        st.error("⚠️ CSV must contain: Frequency, Monetary, Cluster")

# Footer
st.markdown("---")
st.write("Built by Vignesh M Naik | Data Science Project 🚀")
