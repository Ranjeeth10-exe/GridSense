import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="GridSense", page_icon="🔌", layout="wide")

DATA_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
DATA_FILE = "household_power_consumption.txt"

# Streamlit Cloud has limited RAM. Do not load the full 2M+ rows.
MAX_ROWS = 80_000
TRAIN_PER_CLASS = 6_000


# ============================================================
# LOAD DATA + TRAIN MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_and_train(max_rows: int = MAX_ROWS):
    """
    Optimized for Streamlit Cloud.

    Fixes included:
    1. Does not use deprecated/unsupported parse_dates dictionary format.
    2. Loads only a limited number of rows to avoid Streamlit memory crash.
    3. Uses only required columns.
    4. Uses vectorized anomaly labeling instead of slow df.apply().
    5. Uses a smaller Random Forest with n_jobs=1 to reduce memory usage.
    6. Uses st.cache_resource instead of st.cache_data to avoid heavy serialization.
    """

    usecols = [
        "Date",
        "Time",
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    response = requests.get(DATA_URL, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open(DATA_FILE) as f:
            df = pd.read_csv(
                f,
                sep=";",
                usecols=usecols,
                nrows=max_rows,
                na_values=["?"],
                low_memory=False,
            )

    # Create DateTime manually. This avoids the Streamlit pandas parse_dates error.
    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["DateTime"]).copy()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Feature engineering
    df["Hour"] = df["DateTime"].dt.hour.astype("int8")
    df["Month"] = df["DateTime"].dt.month.astype("int8")
    df["Weekday"] = df["DateTime"].dt.weekday.astype("int8")
    df["Date_Only"] = df["DateTime"].dt.date

    # Linear Regression for expected power
    features_mlr = [
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    mlr = LinearRegression()
    mlr.fit(df[features_mlr], df["Global_active_power"])

    df["Expected_Power"] = mlr.predict(df[features_mlr])
    df["Deviation"] = df["Global_active_power"] - df["Expected_Power"]

    # Fast vectorized labeling instead of df.apply()
    conditions = [
        (df["Deviation"] < -0.3) & (df["Global_reactive_power"] < 0.1),
        (df["Voltage"] < 220) | (df["Voltage"] > 245),
        df["Deviation"] > 0.1,
    ]

    choices = ["Theft", "Fault", "Surge"]
    df["Anomaly_Label"] = np.select(conditions, choices, default="Normal")

    features_rf = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Deviation",
        "Hour",
        "Month",
        "Weekday",
    ]

    # Balanced lightweight training set
    train_parts = []
    for label in ["Normal", "Theft", "Fault", "Surge"]:
        part = df[df["Anomaly_Label"] == label]
        if not part.empty:
            train_parts.append(
                part.sample(n=min(TRAIN_PER_CLASS, len(part)), random_state=42)
            )

    if not train_parts:
        raise ValueError("No training data available after preprocessing.")

    train_df = pd.concat(train_parts, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    rf = RandomForestClassifier(
        n_estimators=25,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=1,
        class_weight="balanced",
    )
    rf.fit(train_df[features_rf], train_df["Anomaly_Label"])

    return df, mlr, rf, features_mlr, features_rf


with st.spinner("Loading optimized dataset and training models... ⏳"):
    try:
        df, mlr, rf, features_mlr, features_rf = load_and_train(MAX_ROWS)
    except Exception as e:
        st.error("❌ App failed while loading data or training the model.")
        st.exception(e)
        st.stop()

st.success(
    f"✅ App running in low-memory mode — {len(df):,} rows loaded | Models trained successfully!"
)
st.caption(
    "Note: This Streamlit Cloud version uses a sample of the dataset to avoid memory-limit errors."
)

page_html = Path(__file__).with_name("gridsense_redesigned.html").read_text(encoding="utf-8")
components.html(page_html, height=3400, scrolling=True)


# ============================================================
# SECTION 1 - OVERVIEW
# ============================================================
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
