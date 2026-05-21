from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="GridSense", page_icon="🔌", layout="wide")

st.markdown(
    "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;} .css-18e3th9 {padding: 0rem 0rem 0rem 0rem;}</style>",
    unsafe_allow_html=True,
)

page_html = Path(__file__).with_name("gridsense_redesigned.html").read_text(encoding="utf-8")
components.html(page_html, height=5200, scrolling=True)

with st.spinner("Loading optimized dataset and training models... ⏳ (first load takes ~2 mins)"):
    try:
        df, mlr, rf, features_mlr, features_rf = load_and_train(MAX_ROWS)
        st.success(
            f"✅ App running in low-memory mode — {len(df):,} rows loaded | Models trained successfully!"
        )
        st.caption(
            "Note: This Streamlit Cloud version uses a sample of the dataset to avoid memory-limit errors."
        )
    except Exception as e:
        st.error("❌ App failed while loading data or training the model.")
        st.exception(e)
        st.stop()

# SECTION 1 - OVERVIEW
st.markdown("---")
st.subheader("📊 Overview")

counts = df["Anomaly_Label"].value_counts()

col1, col2, col3, col4 = st.columns(4)
col1.metric("✅ Normal", f"{counts.get('Normal', 0):,}")
col2.metric("🚨 Theft", f"{counts.get('Theft', 0):,}")
col3.metric("⚡ Fault", f"{counts.get('Fault', 0):,}")
col4.metric("📈 Surge", f"{counts.get('Surge', 0):,}")


# ============================================================
# SECTION 2 - DAILY POWER CONSUMPTION
# ============================================================
st.markdown("---")
st.subheader("📈 Daily Power Consumption")

daily_avg = df.groupby("Date_Only", observed=True)["Global_active_power"].mean()

fig1, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(daily_avg.index, daily_avg.values, linewidth=0.9)
ax1.set_xlabel("Date")
ax1.set_ylabel("Power (kW)")
ax1.set_title("Daily Average Power Consumption")
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)


# ============================================================
# SECTION 3 - CONSUMPTION PATTERNS
# ============================================================
st.markdown("---")
st.subheader("🕐 Consumption Patterns")

col_a, col_b = st.columns(2)

with col_a:
    hourly_avg = df.groupby("Hour", observed=True)["Global_active_power"].mean()

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.bar(hourly_avg.index, hourly_avg.values)
    ax2.set_title("Average by Hour of Day")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Power (kW)")
    ax2.set_xticks(range(0, 24, 2))
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

with col_b:
    monthly_avg = df.groupby("Month", observed=True)["Global_active_power"].mean()

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.bar(monthly_avg.index, monthly_avg.values)
    ax3.set_title("Average by Month")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Power (kW)")
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


# ============================================================
# SECTION 4 - ANOMALY TABLE
# ============================================================
st.markdown("---")
st.subheader("🚨 Detected Anomalies")

anomaly_type = st.selectbox("Filter by type:", ["All", "Theft", "Fault", "Surge"])

if anomaly_type == "All":
    anomalies = df.loc[df["Anomaly_Label"] != "Normal"]
else:
    anomalies = df.loc[df["Anomaly_Label"] == anomaly_type]

st.write(f"Found {len(anomalies):,} anomalies. Displaying first 50 rows only.")

display_cols = [
    "DateTime",
    "Global_active_power",
    "Voltage",
    "Global_reactive_power",
    "Deviation",
    "Anomaly_Label",
]

st.dataframe(
    anomalies[display_cols].head(50).reset_index(drop=True),
    use_container_width=True,
)


# ============================================================
# SECTION 5 - LIVE PREDICTOR
# ============================================================
st.markdown("---")
st.subheader("🔍 Live Anomaly Predictor")
st.markdown("Enter meter readings and get instant prediction.")

c1, c2, c3, c4 = st.columns(4)

inp_power = c1.number_input(
    "Active Power (kW)",
    min_value=0.0,
    max_value=10.0,
    value=1.5,
    step=0.1,
)

inp_reactive = c2.number_input(
    "Reactive Power (kW)",
    min_value=0.0,
    max_value=2.0,
    value=0.2,
    step=0.01,
)

inp_voltage = c3.number_input(
    "Voltage (V)",
    min_value=200.0,
    max_value=260.0,
    value=235.0,
    step=0.5,
)

inp_intensity = c4.number_input(
    "Intensity (A)",
    min_value=0.0,
    max_value=50.0,
    value=7.0,
    step=0.1,
)

c5, c6, c7 = st.columns(3)

inp_sub1 = c5.number_input(
    "Sub Meter 1 (Wh)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=1.0,
)

inp_sub2 = c6.number_input(
    "Sub Meter 2 (Wh)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=1.0,
)

inp_sub3 = c7.number_input(
    "Sub Meter 3 (Wh)",
    min_value=0.0,
    max_value=100.0,
    value=17.0,
    step=1.0,
)

if st.button("🔍 Predict", use_container_width=True):
    mlr_input = pd.DataFrame(
        [
            {
                "Voltage": inp_voltage,
                "Global_intensity": inp_intensity,
                "Sub_metering_1": inp_sub1,
                "Sub_metering_2": inp_sub2,
                "Sub_metering_3": inp_sub3,
            }
        ]
    )

    expected = mlr.predict(mlr_input[features_mlr])[0]
    deviation = inp_power - expected

    now = pd.Timestamp.now()

    input_data = pd.DataFrame(
        [
            {
                "Global_active_power": inp_power,
                "Global_reactive_power": inp_reactive,
                "Voltage": inp_voltage,
                "Global_intensity": inp_intensity,
                "Deviation": deviation,
                "Hour": now.hour,
                "Month": now.month,
                "Weekday": now.weekday(),
            }
        ]
    )

    prediction = rf.predict(input_data[features_rf])[0]
    confidence = np.max(rf.predict_proba(input_data[features_rf])[0]) * 100

    if prediction == "Normal":
        st.success(f"✅ Normal Reading — {confidence:.1f}% confident")
    elif prediction == "Theft":
        st.error(f"🚨 Theft Detected! — {confidence:.1f}% confident")
    elif prediction == "Fault":
        st.warning(f"⚡ Equipment Fault — {confidence:.1f}% confident")
    else:
        st.warning(f"📈 Surge Detected — {confidence:.1f}% confident")

    st.info(f"Expected Power: {expected:.3f} kW | Deviation: {deviation:.3f} kW")


# ============================================================
# SECTION 6 - BILL PREDICTION
# ============================================================
st.markdown("---")
st.subheader("💸 Next Month Bill Prediction")

# With sampled rows, estimate using the most recent available rows.
recent_df = df.sort_values("DateTime").tail(min(len(df), 20_000))

last_avg_power = recent_df["Global_active_power"].mean()
estimated_units = last_avg_power * 24 * 30
estimated_bill = estimated_units * 6.5

col_b1, col_b2, col_b3 = st.columns(3)

col_b1.metric("Avg Power", f"{last_avg_power:.3f} kW")
col_b2.metric("Estimated Units", f"{estimated_units:.0f} kWh")
col_b3.metric("Predicted Bill", f"₹{estimated_bill:,.0f}")

st.caption("Estimated using ₹6.5/unit")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("GridSense — Built with Python, scikit-learn & Streamlit | ECE Project")
