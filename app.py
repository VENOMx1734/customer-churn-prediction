import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("final_model.pkl")

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ---------------- HERO ---------------- #
st.title("🚀 Stop Losing Customers Before It Happens")

st.write("""
Understand which customers are likely to churn and what actions to take — instantly.

👉 Simulate your business  
👉 See churn risk  
👉 Take action  
""")

# ---------------- SIMULATION ---------------- #
st.markdown("## ⚡ Simulate Your Business")

col1, col2, col3 = st.columns(3)

num_customers = col1.slider("Customers", 50, 1000, 200)
avg_orders = col2.slider("Monthly Orders", 1, 20, 5)
avg_spend = col3.slider("Avg Spend ($)", 10, 500, 100)

if st.button("Generate Insights"):

    np.random.seed(42)

    df = pd.DataFrame({
        'Frequency': np.random.poisson(avg_orders, num_customers),
        'Monetary': np.random.normal(avg_spend, 30, num_customers).clip(10),
        'Cluster': np.random.choice([0,1,2,3], num_customers)
    })

    st.session_state['data'] = df

# ---------------- OPTIONAL CSV ---------------- #
st.markdown("### 📄 Or Upload Your Data (Optional)")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state['data'] = df

# ---------------- DASHBOARD ---------------- #
if 'data' in st.session_state:

    df = st.session_state['data']

    if 'Cluster' not in df.columns:
        df['Cluster'] = 2

    X = df[['Frequency', 'Monetary', 'Cluster']]
    probs = model.predict_proba(X)[:, 1]

    df['Churn Probability'] = probs
    df['Risk Level'] = pd.cut(
        probs,
        bins=[0, 0.4, 0.7, 1],
        labels=["Low", "Medium", "High"]
    )

    # ---------------- METRICS (CARDS STYLE) ---------------- #
    st.markdown("## 📊 Overview")

    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    high = (df['Risk Level'] == "High").sum()
    avg = df['Churn Probability'].mean()
    max_risk = df['Churn Probability'].max()

    col1.metric("Customers", total)
    col2.metric("High Risk", high)
    col3.metric("Avg Churn", f"{avg:.0%}")
    col4.metric("Max Risk", f"{max_risk:.0%}")

    # ---------------- SMALL CHART ---------------- #
    st.markdown("## 📈 Risk Breakdown")

    col1, col2 = st.columns([1,2])

    with col1:
        fig, ax = plt.subplots(figsize=(3,3))
        df['Risk Level'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.markdown("### 🧠 What This Means")

        high_pct = (df['Risk Level'] == "High").mean() * 100

        if high_pct > 30:
            st.error("⚠️ High churn risk — you're losing customers fast")
        elif high_pct > 15:
            st.warning("⚠️ Moderate churn — needs attention")
        else:
            st.success("✅ Customer base looks stable")

    # ---------------- ACTIONS ---------------- #
    st.markdown("## 🎯 What Should You Do?")

    if high > 0:
        st.write(f"👉 {high} customers are at risk — prioritize them")
        st.write("👉 Send targeted discounts or offers")
        st.write("👉 Re-engage inactive users")
        st.write("👉 Focus on high-value segments first")
    else:
        st.write("✅ No urgent churn risk — focus on growth")

    # ---------------- TOP USERS ---------------- #
    st.markdown("## 🔥 Customers to Act On")

    top_risk = df[df['Risk Level'] == "High"].sort_values(
        by='Churn Probability', ascending=False
    )

    st.dataframe(top_risk.head(10))

    # ---------------- DOWNLOAD ---------------- #
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "📥 Download Results",
        csv,
        "churn_results.csv",
        "text/csv"
    )

# ---------------- HOW IT WORKS ---------------- #
st.markdown("---")
st.markdown("## ⚙️ How It Works")

st.write("""
We analyze customer behavior (orders + spending),
predict churn risk using machine learning,
and show you exactly who to focus on.
""")

st.caption("Built using customer behavior modeling (RFM + ML)")
st.write("Built by Vignesh M Naik | Data Science Project 🚀")
