import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("final_model.pkl")

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ---------------- STYLE ---------------- #
st.markdown("""
<style>
.metric-card {
    padding: 15px;
    border-radius: 12px;
    background-color: #111827;
    border: 1px solid #1f2937;
    text-align: center;
}
.metric-title {
    font-size: 14px;
    color: #9ca3af;
}
.metric-value {
    font-size: 28px;
    font-weight: bold;
}
.card {
    padding: 15px;
    border-radius: 12px;
    background-color: #111827;
    border: 1px solid #1f2937;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

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
avg_spend = col3.slider("Avg Spend ($)", 10, 1000, 100)

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

    # ---------------- METRICS ---------------- #
    st.markdown("## 📊 Overview")

    col1, col2, col3, col4 = st.columns(4)

    total = len(df)
    high = (df['Risk Level'] == "High").sum()
    avg = df['Churn Probability'].mean()
    revenue = df['Monetary'].sum()

    col1.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">Customers</div>
    <div class="metric-value">{total}</div>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">High Risk</div>
    <div class="metric-value">{high}</div>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">Avg Churn</div>
    <div class="metric-value">{avg:.0%}</div>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">Revenue</div>
    <div class="metric-value">${revenue:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- CHART ---------------- #
    st.markdown("## 📉 Revenue at Risk")

    risk_summary = df.groupby('Risk Level')['Monetary'].sum()

    col1, col2 = st.columns([1,2])

    with col1:
        fig, ax = plt.subplots(figsize=(3,3))
        risk_summary.plot(kind='bar', ax=ax)
        ax.set_title("")
        ax.set_ylabel("$")
        ax.set_xlabel("")
        st.pyplot(fig)

    with col2:
        revenue_at_risk = risk_summary.get("High", 0)

        st.markdown("### 💡 Key Insight")

        st.error(f"⚠️ ${revenue_at_risk:,.0f} revenue is at risk")

        high_pct = (df['Risk Level'] == "High").mean() * 100

        if high_pct > 30:
            st.write("You are losing a large portion of customers.")
        elif high_pct > 15:
            st.write("Churn is moderate and needs attention.")
        else:
            st.write("Customer base looks stable.")

    # ---------------- ACTIONS ---------------- #
    st.markdown("## 🎯 What Should You Do?")

    if high > 0:
        st.write(f"👉 Focus on {high} high-risk customers")
        st.write("👉 Offer discounts or incentives")
        st.write("👉 Re-engage inactive users")
        st.write("👉 Prioritize high-value users")
    else:
        st.write("✅ No urgent churn risk — focus on growth")

    # ---------------- ACTION CARDS ---------------- #
    st.markdown("## 🔥 Customers You Should Act On")

    top_risk = df[df['Risk Level'] == "High"].sort_values(
        by='Churn Probability', ascending=False
    ).head(5)

    for i, row in top_risk.iterrows():
        st.markdown(f"""
        <div class="card">
            <b>Customer #{i}</b><br><br>
            🔴 Risk: <b>{row['Churn Probability']:.0%}</b><br>
            📦 Orders: {int(row['Frequency'])}<br>
            💰 Spend: ${row['Monetary']:.0f}<br><br>
            👉 <b>Action:</b> Send targeted offer / re-engagement email
        </div>
        """, unsafe_allow_html=True)

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
