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
    padding: 18px;
    border-radius: 14px;
    background: linear-gradient(145deg, #0f172a, #020617);
    border: 1px solid #1e293b;
    text-align: center;
}
.metric-title {
    font-size: 13px;
    color: #94a3af;
}
.metric-value {
    font-size: 30px;
    font-weight: bold;
}
.card {
    padding:15px;
    border-radius:12px;
    background:#0f172a;
    border:1px solid #1e293b;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ---------------- #
st.title("🚀 Stop Losing Customers Before It Happens")
st.write("Predict churn, understand risk, and take action instantly.")

# ---------------- EXACT GUIDE ---------------- #
with st.expander("📘 How to Use This Tool (Detailed Guide)"):

    st.markdown("""
## 🎯 What This Tool Actually Does

This tool simulates your customer base and predicts **which customers are likely to stop buying (churn)**.

It helps you:
- Identify risky customers  
- Understand how serious the problem is  
- Take action before losing revenue  

---

## 🧩 Understanding the Inputs

### 👥 Total Customers
- Total number of customers in your business  
- Used to estimate overall churn impact  
- More customers = more total risk (but not always worse %)

---

### 📦 Avg Orders per Customer (Monthly)
- How often customers buy from you  
- Higher = more engaged customers  
- Lower = more likely to churn  

👉 Example:
- 2 orders/month → weak engagement  
- 10 orders/month → strong retention  

---

### 💰 Avg Spend per Customer ($)
- How much revenue each customer brings  
- This determines **how valuable each customer is**

👉 Important:
Losing 1 high-spend customer can hurt more than losing 10 low-spend customers  

---

### 🎚 Customer Variation
- Controls how different your customers are  

👉 Low variation:
- Everyone behaves similarly  
- Stable business  

👉 High variation:
- Mix of:
  - loyal customers  
  - inactive users  
  - risky users  

👉 This is **very realistic** for most startups  

---

## 📊 What Matters Most (Focus Here)

✔ High churn percentage  
✔ High-value customers churning  
✔ Low-frequency customers  

---

## ⚠️ What Doesn’t Matter Much

❌ Exact customer count  
❌ Small slider changes  
❌ Perfect accuracy  

---

## 🎯 How to Use This Tool (Step-by-Step)

1. Adjust sliders to match your business  
2. Click **Generate Insights**  
3. Focus on:
   - High-risk customers  
   - Revenue at risk  
   - Recommended actions  

---

## 💡 Key Insight

Not all customers are equal.

Losing a few **high-value customers** is often worse than losing many low-value ones.
""")

# ---------------- CSV GUIDE ---------------- #
with st.expander("📄 How to Upload Your CSV File"):

    st.markdown("""
### 📁 Required Columns

Your CSV must include:

- Frequency → Number of purchases  
- Monetary → Amount spent  
- Cluster → Optional (if missing, system assumes default)

---

### 📊 Example Format

Frequency,Monetary,Cluster  
5,200,0  
2,50,3  
10,500,1  

---

### ⚠️ Common Mistakes

❌ Wrong column names  
❌ Missing columns  
❌ Extra spaces  

---

### 💡 Where to Get This Data

- Shopify  
- Stripe  
- CRM systems  
- Excel exports  
""")

# ---------------- SIMULATION ---------------- #
st.markdown("## ⚡ Simulate Your Business")

col1, col2, col3, col4 = st.columns(4)

num_customers = col1.slider("Total Customers", 50, 1000, 200)
avg_orders = col2.slider("Avg Orders per Customer (Monthly)", 1, 20, 5)
avg_spend = col3.slider("Avg Spend per Customer ($)", 10, 1000, 100)
variation = col4.slider("Customer Variation", 0.1, 1.0, 0.5)

# ---------------- CSV UPLOAD ---------------- #
st.markdown("## 📄 Or Upload Your Data")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if st.button("Generate Insights"):

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        np.random.seed(42)

        freq_std = avg_orders * variation
        spend_std = avg_spend * variation

        df = pd.DataFrame({
            'Frequency': np.random.normal(avg_orders, freq_std, num_customers).clip(1),
            'Monetary': np.random.normal(avg_spend, spend_std, num_customers).clip(10),
            'Cluster': np.random.choice([0,1,2,3], num_customers)
        })

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

    total = len(df)
    high = (df['Risk Level'] == "High").sum()
    avg = df['Churn Probability'].mean()
    revenue = df['Monetary'].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"<div class='metric-card'><div class='metric-title'>Customers</div><div class='metric-value'>{total}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='metric-title'>High Risk</div><div class='metric-value'>{high}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='metric-title'>Avg Churn</div><div class='metric-value'>{avg:.0%}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='metric-title'>Revenue</div><div class='metric-value'>${revenue:,.0f}</div></div>", unsafe_allow_html=True)

    # ---------------- CHART ---------------- #
    st.markdown("## 📊 Churn Risk Distribution")

    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(df['Churn Probability'], bins=20)
    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Customers")

    st.pyplot(fig)

    # ---------------- EXACT CHART GUIDE ---------------- #
    st.markdown("### 🧠 How to Read This Chart")

    st.markdown("""
This chart shows how your customers are distributed based on their churn risk.

---

### 📈 What You're Seeing

Each bar represents how many customers fall into a certain churn probability range.

- Left side → Low risk customers  
- Middle → Medium risk  
- Right side → High risk customers  

---

### 🎯 How to Interpret It

#### 🟢 If most customers are on the LEFT:
- Your business is stable  
- Customers are engaged  
- Low immediate risk  

---

#### 🟡 If many are in the MIDDLE:
- Warning zone  
- Customers may churn soon  
- Needs attention  

---

#### 🔴 If many are on the RIGHT:
- Serious churn problem  
- Immediate action needed  
- Revenue at risk  

---

### 💡 What Good Looks Like

- Majority of customers below 0.4  
- Very few above 0.7  

---

### 🚨 What Bad Looks Like

- Large cluster above 0.7  
- Spread evenly across all ranges  

---

### ⚡ Key Insight

This chart helps you understand:

👉 “Is churn concentrated or widespread?”

- Concentrated → easier to fix  
- Widespread → deeper problem in product/business  
""")

    st.info("💡 Tip: Focus on customers above 0.7 churn probability — they are most likely to leave soon.")

    # ---------------- REVENUE ---------------- #
    st.markdown("## 📉 Revenue at Risk")

    risk_summary = df.groupby('Risk Level')['Monetary'].sum()
    revenue_at_risk = risk_summary.get("High", 0)

    if revenue_at_risk > 0:
        st.error(f"⚠️ ${revenue_at_risk:,.0f} revenue is at risk")
    else:
        st.success("✅ No revenue is at risk")

    # ---------------- ACTIONS ---------------- #
    st.markdown("## 🎯 What Should You Do?")

    if high > 0:
        st.write("👉 Target high-risk customers immediately")
        st.write("👉 Offer discounts / incentives")
        st.write("👉 Improve engagement")
    else:
        st.write("✅ Focus on growth")

    # ---------------- CARDS ---------------- #
    st.markdown("## 🔥 Customers You Should Act On")

    top_risk = df[df['Risk Level'] == "High"].sort_values(
        by='Churn Probability', ascending=False
    ).head(6)

    cols = st.columns(3)

    for i, (_, row) in enumerate(top_risk.iterrows()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card">
            🔴 Risk: <b>{row['Churn Probability']:.0%}</b><br>
            📦 Orders: {int(row['Frequency'])}<br>
            💰 Spend: ${row['Monetary']:.0f}<br><br>
            👉 Action: Retention campaign
            </div>
            """, unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Built for founders to understand churn instantly 🚀")
st.write("Built by Vignesh M Naik | Data Science Project 🚀")
