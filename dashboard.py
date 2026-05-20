import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, zipfile, requests
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="GridSense", page_icon="🔌", layout="wide")

# Title
st.title("🔌 GridSense — Electricity Anomaly Detection")
st.markdown("Detects **theft**, **faults**, and **surges** from smart meter data using Machine Learning.")
st.markdown("---")

# Load and train
@st.cache_data
def load_and_train():
    # Download from UCI directly — works on Streamlit Cloud
    url = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
    r   = requests.get(url, timeout=120)
    z   = zipfile.ZipFile(io.BytesIO(r.content))

    with z.open('household_power_consumption.txt') as f:
        df = pd.read_csv(f,
                          sep=';',
                          parse_dates={'DateTime': ['Date', 'Time']},
                          dayfirst=True,
                          low_memory=False,
                          na_values=['?'])

    # Clean
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Feature engineering
    df['Hour']    = df['DateTime'].dt.hour
    df['Month']   = df['DateTime'].dt.month
    df['Weekday'] = df['DateTime'].dt.weekday
    df['Date']    = df['DateTime'].dt.date

    # MLR — Expected power
    features_mlr = ['Voltage', 'Global_intensity',
                     'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    mlr = LinearRegression()
    mlr.fit(df[features_mlr], df['Global_active_power'])
    df['Expected_Power'] = mlr.predict(df[features_mlr])
    df['Deviation']      = df['Global_active_power'] - df['Expected_Power']

    # Labels
    def label_anomaly(row):
        if row['Deviation'] < -0.3 and row['Global_reactive_power'] < 0.1:
            return 'Theft'
        elif row['Voltage'] < 220 or row['Voltage'] > 245:
            return 'Fault'
        elif row['Deviation'] > 0.1:
            return 'Surge'
        else:
            return 'Normal'

    df['Anomaly_Label'] = df.apply(label_anomaly, axis=1)

    # Train Random Forest
    features_rf = ['Global_active_power', 'Global_reactive_power',
                    'Voltage', 'Global_intensity', 'Deviation',
                    'Hour', 'Month', 'Weekday']
    rf = RandomForestClassifier(n_estimators=50,
                                 max_depth=10,
                                 random_state=42,
                                 n_jobs=-1)
    rf.fit(df[features_rf], df['Anomaly_Label'])

    return df, rf, features_rf

# Show loading spinner
with st.spinner("Loading dataset and training models... ⏳ (first load takes ~2 mins)"):
    df, rf, features_rf = load_and_train()

st.success(f"✅ Dataset loaded — {len(df):,} rows | All models trained!")

# ── SECTION 1 — Overview Metrics ──
st.markdown("---")
st.subheader("📊 Overview")

counts = df['Anomaly_Label'].value_counts()
col1, col2, col3, col4 = st.columns(4)
col1.metric("✅ Normal", f"{counts.get('Normal', 0):,}")
col2.metric("🚨 Theft",  f"{counts.get('Theft',  0):,}")
col3.metric("⚡ Fault",  f"{counts.get('Fault',  0):,}")
col4.metric("📈 Surge",  f"{counts.get('Surge',  0):,}")

# ── SECTION 2 — Consumption Graph ──
st.markdown("---")
st.subheader("📈 Daily Power Consumption (2006–2010)")

daily_avg = df.groupby('Date')['Global_active_power'].mean()
fig1, ax1  = plt.subplots(figsize=(14, 4))
ax1.plot(daily_avg.index, daily_avg.values,
         color='steelblue', linewidth=0.8)
ax1.set_xlabel('Date')
ax1.set_ylabel('Power (kW)')
ax1.set_title('Daily Average Power Consumption')
plt.tight_layout()
st.pyplot(fig1)

# ── SECTION 3 — Patterns ──
st.markdown("---")
st.subheader("🕐 Consumption Patterns")

col_a, col_b = st.columns(2)

with col_a:
    hourly_avg = df.groupby('Hour')['Global_active_power'].mean()
    fig2, ax2  = plt.subplots(figsize=(7, 4))
    ax2.bar(hourly_avg.index, hourly_avg.values, color='coral')
    ax2.set_title('Average by Hour of Day')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Power (kW)')
    plt.tight_layout()
    st.pyplot(fig2)

with col_b:
    monthly_avg = df.groupby('Month')['Global_active_power'].mean()
    fig3, ax3   = plt.subplots(figsize=(7, 4))
    ax3.bar(monthly_avg.index, monthly_avg.values, color='mediumseagreen')
    ax3.set_title('Average by Month')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Power (kW)')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec'])
    plt.tight_layout()
    st.pyplot(fig3)

# ── SECTION 4 — Anomaly Table ──
st.markdown("---")
st.subheader("🚨 Detected Anomalies")

anomaly_type = st.selectbox("Filter by type:",
                             ['All', 'Theft', 'Fault', 'Surge'])

if anomaly_type == 'All':
    anomalies = df[df['Anomaly_Label'] != 'Normal']
else:
    anomalies = df[df['Anomaly_Label'] == anomaly_type]

st.write(f"Showing {len(anomalies):,} anomalies")
st.dataframe(
    anomalies[['DateTime', 'Global_active_power', 'Voltage',
               'Global_reactive_power', 'Deviation', 'Anomaly_Label']]
    .head(50)
    .reset_index(drop=True)
)

# ── SECTION 5 — Live Predictor ──
st.markdown("---")
st.subheader("🔍 Live Anomaly Predictor")
st.markdown("Enter meter readings and get instant prediction.")

c1, c2, c3, c4 = st.columns(4)
inp_power     = c1.number_input("Active Power (kW)",    0.0, 10.0,  1.5)
inp_reactive  = c2.number_input("Reactive Power (kW)",  0.0,  2.0,  0.2)
inp_voltage   = c3.number_input("Voltage (V)",        200.0,260.0,235.0)
inp_intensity = c4.number_input("Intensity (A)",        0.0, 50.0,  7.0)

c5, c6, c7 = st.columns(3)
inp_sub1 = c5.number_input("Sub Meter 1 (Wh)", 0.0, 100.0,  0.0)
inp_sub2 = c6.number_input("Sub Meter 2 (Wh)", 0.0, 100.0,  0.0)
inp_sub3 = c7.number_input("Sub Meter 3 (Wh)", 0.0, 100.0, 17.0)

if st.button("🔍 Predict", use_container_width=True):
    features_mlr2 = ['Voltage', 'Global_intensity',
                      'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    mlr2     = LinearRegression()
    mlr2.fit(df[features_mlr2], df['Global_active_power'])
    expected  = mlr2.predict([[inp_voltage, inp_intensity,
                                inp_sub1, inp_sub2, inp_sub3]])[0]
    deviation = inp_power - expected

    now        = pd.Timestamp.now()
    input_data = pd.DataFrame([{
        'Global_active_power'  : inp_power,
        'Global_reactive_power': inp_reactive,
        'Voltage'              : inp_voltage,
        'Global_intensity'     : inp_intensity,
        'Deviation'            : deviation,
        'Hour'                 : now.hour,
        'Month'                : now.month,
        'Weekday'              : now.weekday()
    }])

    prediction = rf.predict(input_data[features_rf])[0]
    confidence = max(rf.predict_proba(input_data[features_rf])[0]) * 100

    if prediction == 'Normal':
        st.success(f"✅ Normal Reading — {confidence:.1f}% confident")
    elif prediction == 'Theft':
        st.error(f"🚨 Theft Detected! — {confidence:.1f}% confident")
    elif prediction == 'Fault':
        st.warning(f"⚡ Equipment Fault — {confidence:.1f}% confident")
    else:
        st.warning(f"📈 Surge Detected — {confidence:.1f}% confident")

    st.info(f"Expected Power: {expected:.3f} kW | Deviation: {deviation:.3f} kW")

# ── SECTION 6 — Bill Prediction ──
st.markdown("---")
st.subheader("💸 Next Month Bill Prediction")

last_3_avg = df.groupby('Month')['Global_active_power'].mean().tail(3).mean()
units      = last_3_avg * 24 * 30
bill       = units * 6.5

col_b1, col_b2, col_b3 = st.columns(3)
col_b1.metric("Avg Power (last 3 months)", f"{last_3_avg:.3f} kW")
col_b2.metric("Estimated Units",            f"{units:.0f} kWh")
col_b3.metric("Predicted Bill",             f"₹{bill:,.0f}")
st.caption("Based on TANGEDCO rate ₹6.5/unit")

# Footer
st.markdown("---")
st.caption("GridSense — Built with Python, scikit-learn & Streamlit | ECE Project")