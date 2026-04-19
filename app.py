import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("final_model.pkl")

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ---------------- HERO ---------------- #
st.title("🚀 Find Customers You’re About to Lose")

st.write("""
Identify high-risk customers in seconds and take action before they churn.

👉 Upload your data or try a sample dataset  
👉 Get insights + recommended actions instantly  
""")

# ---------------- SAMPLE DATA ---------------- #
if st.button("⚡ Try with Sample Data"):
    st.session_state['data'] = pd.DataFrame({
        'Frequency': [2, 10, 5, 1, 8, 3, 12],
        'Monetary': [50, 300, 120, 20, 250, 80, 400],
        'Cluster': [3, 0, 2, 1, 0, 2, 0]
    })

# ---------------- FILE UPLOAD ---------------- #
st.markdown("## 📄 Upload Your Customer Data")

st.markdown("""
Required columns:
- Frequency → number of purchases  
- Monetary → average spend ($)  
- Cluster → optional (default will be used if missing)
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# ---------------- DATA SOURCE ---------------- #
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)

elif 'data' in st.session_state:
    df = st.session_state['data']

# ---------------- MAIN DASHBOARD ---------------- #
if df is not None:

    # Handle missing cluster
    if 'Cluster' not in df.columns:
        df['Cluster'] = 2  # default (At Risk assumption)

    try:
        X = df[['Frequency', 'Monetary', 'Cluster']]
        probs = model.predict_proba(X)[:, 1]

        df['Churn Probability'] = probs
        df['Risk Level'] = pd.cut(
            probs,
            bins=[0, 0.4, 0.7, 1],
            labels=["Low", "Medium", "High"]
        )

        # ---------------- METRICS ---------------- #
        st.markdown("## 📊 Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Customers", len(df))
        col2.metric("High Risk", (df['Risk Level'] == "High").sum())
        col3.metric("Avg Churn", f"{df['Churn Probability'].mean():.0%}")
        col4.metric("Max Risk", f"{df['Churn Probability'].max():.0%}")

        # ---------------- CHART ---------------- #
        st.markdown("## 📈 Risk Distribution")

        fig, ax = plt.subplots()
        df['Risk Level'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

        # ---------------- INSIGHTS ---------------- #
        st.markdown("## 🧠 Key Insights")

        high_pct = (df['Risk Level'] == "High").mean() * 100

        if high_pct > 30:
            st.error("⚠️ High churn risk — immediate action needed")
        elif high_pct > 15:
            st.warning("⚠️ Moderate churn risk — monitor closely")
        else:
            st.success("✅ Customer base is stable")

        # ---------------- ACTIONS ---------------- #
        st.markdown("## 🎯 What Should You Do?")

        high_count = (df['Risk Level'] == "High").sum()

        if high_count > 0:
            st.write(f"👉 Focus on {high_count} high-risk customers")
            st.write("👉 Send targeted discounts or offers")
            st.write("👉 Re-engage inactive users via email")
            st.write("👉 Prioritize high-value customers")
        else:
            st.write("✅ No urgent churn risk — focus on growth")

        # ---------------- TOP CUSTOMERS ---------------- #
        st.markdown("## 🔥 Top At-Risk Customers")

        top_risk = df[df['Risk Level'] == "High"].sort_values(
            by='Churn Probability', ascending=False
        )

        st.dataframe(top_risk.head(10))

        # ---------------- OPTIONAL CLUSTER VIEW ---------------- #
        st.markdown("## ⚙️ Advanced (Optional)")

        if st.checkbox("Show Customer Segments (Clusters)"):
            st.write("Cluster 0 = High Value")
            st.write("Cluster 1 = Low Value")
            st.write("Cluster 2 = At Risk")
            st.write("Cluster 3 = New Customers")

        # ---------------- DOWNLOAD ---------------- #
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "📥 Download Results",
            csv,
            "churn_results.csv",
            "text/csv"
        )

    except:
        st.error("⚠️ Ensure CSV has Frequency, Monetary, Cluster")

# ---------------- HOW IT WORKS ---------------- #
st.markdown("---")
st.markdown("## ⚙️ How It Works")

st.write("""
We analyze customer behavior (orders + spending),
predict churn risk using machine learning,
and highlight who you should act on.
""")

# Footer
st.markdown("---")
st.caption("Built using customer behavior data (RFM + ML)")
st.write("Built by Vignesh M Naik | Data Science Project 🚀")
