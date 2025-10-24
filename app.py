import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# ---------- Branding & Intro Section ----------

st.set_page_config(page_title="Wind Farm Predictive Maintenance", layout="wide")
st.title("🌬️ Wind Turbine Predictive Maintenance Dashboard")
st.markdown("""
#### By Team TECH_TONIC
*Using AI & Data Analytics to reduce downtime and optimize wind energy systems*
""")
with st.expander("ℹ️ About this Project"):
    st.write("""
    This tool predicts wind turbine maintenance needs using SCADA sensor data and AI. 
    Early fault detection saves resources by scheduling proactive repairs, minimizing downtime, and boosting renewable generation.

    **How it works:**  
    - The model predicts failure risk from real-time turbine features (wind speed, power curve, direction).
    - Maintenance recommendations are generated instantly for operators.
    """)

st.divider()

# ---------- Sidebar: User Inputs ----------

st.sidebar.header("Enter Turbine Sensor Data")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 25.0, 10.0)
theoretical_power = st.sidebar.slider("Theoretical Power Curve (KWh)", 0.0, 1000.0, 400.0)
wind_direction = st.sidebar.slider("Wind Direction (°)", 0.0, 360.0, 180.0)

# ---------- AI/ML Prediction ----------

model = pickle.load(open('turbine_model.pkl', 'rb'))
input_instance = [[wind_speed, theoretical_power, wind_direction]]
pred = model.predict(input_instance)[0]
risk = model.predict_proba(input_instance)[0][1]

# ---------- Main Prediction Panel ----------

st.markdown("### 🛠️ Turbine Health Status:")
if risk > 0.7:
    st.error("⚠️ Maintenance Required Soon", icon="⚠️")
    st.metric("Failure Risk (%)", f"{risk*100:.1f}", delta_color="inverse")
elif risk > 0.4:
    st.warning("🔶 Moderate Risk Detected", icon="🔶")
    st.metric("Failure Risk (%)", f"{risk*100:.1f}", delta_color="off")
else:
    st.success("✅ Turbine Healthy", icon="✅")
    st.metric("Failure Risk (%)", f"{risk*100:.1f}", delta_color="off")

cols = st.columns(3)
cols[0].metric("Wind Speed (m/s)", f"{wind_speed:.2f}")
cols[1].metric("Theoretical Power (KWh)", f"{theoretical_power:.2f}")
cols[2].metric("Wind Direction (°)", f"{wind_direction:.2f}")

with st.expander("🧮🤔 What does this mean?"):
    st.info("""
    - **Risk > 70%**: Take action urgently—component failure likely soon!
    - **Risk 40–70%**: Consider inspection/maintenance cycle.
    - **Risk < 40%**: Turbine in healthy range.
    """)

st.divider()

# ---------- Fleet Analytics Demo ----------

st.subheader("Fleet-Wide Risk Overview (Demo)")
# Simulate a small fleet with current/related risk
turbine_ids = [f"Turbine-{i+1}" for i in range(5)]
fleet_risks = [risk*100, np.clip(risk*110,0,100), np.clip(risk*60,0,100),
               np.clip(risk*45,0,100), np.clip(risk*80,0,100)]
fleet_df = pd.DataFrame({
    "Turbine": turbine_ids,
    "Failure Risk (%)": fleet_risks
})
bar_color = fleet_df['Failure Risk (%)'].apply(lambda r: "red" if r>70 else "orange" if r>40 else "green")
fig = px.bar(fleet_df, x="Turbine", y="Failure Risk (%)", color=bar_color,
             color_discrete_map={"green":"#2ecc40","orange":"#ffa500","red":"#ff4136"},
             title="Fleet Failure Risk Snapshot")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------- Historical Data Upload (Optional) ----------

st.subheader("📈 Analyze Real Data & Trends")
uploaded_file = st.file_uploader("Upload Turbine History CSV (See example format in README)", type="csv")
if uploaded_file:
    hist_df = pd.read_csv(uploaded_file)
    if all(col in hist_df.columns for col in ["Wind Speed (m/s)", "Theoretical_Power_Curve (KWh)", "Wind Direction (°)"]):
        hist_X = hist_df[["Wind Speed (m/s)", "Theoretical_Power_Curve (KWh)", "Wind Direction (°)"]]
        hist_df["Predicted Failure Risk"] = model.predict_proba(hist_X)[:,1]
        st.line_chart(hist_df["Predicted Failure Risk"], use_container_width=True)
        st.write(hist_df.head())
    else:
        st.warning("CSV must contain columns: Wind Speed (m/s), Theoretical_Power_Curve (KWh), and Wind Direction (°)")

st.divider()

# ---------- Business Impact Card ----------

with st.container():
    st.markdown("### 💸 Why Does This Matter?")
    st.info("""
    - **Early detection reduces annual downtime losses by an estimated $500,000 per turbine.**
    - Scheduled maintenance = 60% less production interruption vs. unplanned failure.
    - Data-driven diagnostics help achieve national renewable energy targets.
    """)
    st.markdown("_Built for Cypher Hackathon 2025 — Modernizing wind energy with AI!_")

# End of app.py
