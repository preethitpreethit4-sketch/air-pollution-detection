import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import folium

from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Air Pollution Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PROFESSIONAL DASHBOARD DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111c2e;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label, .stMarkdown {
    color: #cbd5e1;
}

[data-testid="stMetric"] {
    background-color: #17243a;
    border: 1px solid #263852;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #a5b4fc !important;
}

[data-testid="stMetricValue"] {
    color: #4ade80 !important;
}

div.stButton > button {
    background-color: #16a34a;
    color: white;
    border-radius: 8px;
    border: none;
    min-height: 45px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #15803d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================

st.title("🌍 Air Pollution Intelligence")
st.markdown(
    "### District-Wise Air Quality Monitoring & Analytics"
)

st.caption(
    "Karnataka, India | Interactive Mapping | Machine Learning"
)

st.divider()

# ==================================================
# KARNATAKA DISTRICTS
# Approximate district-center coordinates
# ==================================================

districts = {
    "Bagalkot": (16.18, 75.70),
    "Ballari": (15.14, 76.92),
    "Belagavi": (15.85, 74.50),
    "Bengaluru Rural": (13.29, 77.53),
    "Bengaluru Urban": (12.97, 77.59),
    "Bidar": (17.91, 77.52),
    "Chamarajanagar": (11.92, 76.94),
    "Chikkaballapur": (13.44, 78.06),
    "Chikkamagaluru": (13.32, 75.77),
    "Chitradurga": (14.23, 76.40),
    "Dakshina Kannada": (12.87, 74.88),
    "Davanagere": (14.47, 75.92),
    "Dharwad": (15.46, 75.01),
    "Gadag": (15.43, 75.63),
    "Hassan": (13.00, 76.10),
    "Haveri": (14.79, 75.40),
    "Kalaburagi": (17.33, 76.83),
    "Kodagu": (12.42, 75.74),
    "Kolar": (13.14, 78.13),
    "Koppal": (15.35, 76.15),
    "Mandya": (12.52, 76.90),
    "Mysuru": (12.30, 76.64),
    "Raichur": (16.21, 77.36),
    "Ramanagara": (12.72, 77.28),
    "Shivamogga": (13.93, 75.57),
    "Tumakuru": (13.34, 77.10),
    "Udupi": (13.34, 74.74),
    "Uttara Kannada": (14.80, 74.13),
    "Vijayapura": (16.83, 75.71),
    "Yadgir": (16.77, 77.14),
    "Vijayanagara": (15.27, 76.39)
}

# ==================================================
# MACHINE LEARNING TRAINING
# Synthetic data for educational demonstration
# ==================================================

@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 4000

    training = pd.DataFrame({
        "PM2.5": rng.uniform(0, 250, n),
        "PM10": rng.uniform(0, 350, n),
        "NO2": rng.uniform(0, 200, n)
    })

    # Simplified synthetic labels for demonstration
    score = (
        0.50 * training["PM2.5"] / 250
        + 0.30 * training["PM10"] / 350
        + 0.20 * training["NO2"] / 200
    )

    training["Category"] = np.select(
        [
            score < 0.20,
            score < 0.40,
            score < 0.60,
            score < 0.80
        ],
        [
            "Good",
            "Moderate",
            "Poor",
            "Very Poor"
        ],
        default="Severe"
    )

    X = training[["PM2.5", "PM10", "NO2"]]
    y = training["Category"]

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42
    )

    model.fit(X, y)

    return model


model = train_model()

# ==================================================
# AQI CATEGORY
# US AQI categories
# ==================================================

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#22c55e"
    elif aqi <= 100:
        return "Moderate", "#eab308"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#f97316"
    elif aqi <= 200:
        return "Unhealthy", "#ef4444"
    elif aqi <= 300:
        return "Very Unhealthy", "#a855f7"
    return "Hazardous", "#991b1b"


# ==================================================
# API DATA
# Open-Meteo Air Quality API
# ==================================================

@st.cache_data(ttl=1800)
def fetch_air_quality(lat, lon):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "us_aqi,pm2_5,pm10,nitrogen_dioxide",
        "hourly": "us_aqi,pm2_5,pm10",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(
        url,
        params=params,
        timeout=25
    )

    response.raise_for_status()
    result = response.json()

    current = result.get("current", {})

    if current.get("us_aqi") is None:
        raise ValueError("Current AQI is unavailable.")

    readings = {
        "US AQI": float(current["us_aqi"]),
        "PM2.5": float(current["pm2_5"])
        if current.get("pm2_5") is not None else np.nan,
        "PM10": float(current["pm10"])
        if current.get("pm10") is not None else np.nan,
        "NO2": float(current["nitrogen_dioxide"])
        if current.get("nitrogen_dioxide") is not None else np.nan,
        "Time": current.get("time", "Unavailable")
    }

    hourly = result.get("hourly", {})

    trend = pd.DataFrame({
        "Time": hourly.get("time", []),
        "US AQI": hourly.get("us_aqi", []),
        "PM2.5": hourly.get("pm2_5", []),
        "PM10": hourly.get("pm10", [])
    })

    if not trend.empty:
        trend["Time"] = pd.to_datetime(
            trend["Time"], errors="coerce"
        )

    return readings, trend


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("⚙️ Dashboard Controls")

district = st.sidebar.selectbox(
    "Select Karnataka District",
    sorted(districts.keys()),
    index=sorted(districts.keys()).index("Bengaluru Urban")
)

latitude, longitude = districts[district]

st.sidebar.markdown("---")
st.sidebar.write("**Selected District**")
st.sidebar.success(district)

st.sidebar.write("**Data Source**")
st.sidebar.caption("Open-Meteo modeled air-quality estimates")

st.sidebar.write("**Map Center**")
st.sidebar.caption(
    f"{latitude:.3f}, {longitude:.3f}"
)

refresh = st.sidebar.button(
    "🔄 Refresh Air Quality Data"
)

if refresh:
    fetch_air_quality.clear()

# ==================================================
# LOCATION GRID
# Approximate sample points, not district boundaries
# ==================================================

locations = [
    ("Center", latitude, longitude),
    ("North", latitude + 0.07, longitude),
    ("South", latitude - 0.07, longitude),
    ("East", latitude, longitude + 0.07),
    ("West", latitude, longitude - 0.07)
]

results = []
trend = pd.DataFrame()
errors = []

with st.spinner(f"Loading air-quality data for {district}..."):
    for name, lat, lon in locations:
        try:
            readings, location_trend = fetch_air_quality(lat, lon)

            results.append({
                "Location": name,
                "Latitude": lat,
                "Longitude": lon,
                **readings
            })

            if name == "Center":
                trend = location_trend

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError
        ) as error:
            errors.append(f"{name}: {error}")

# ==================================================
# MAIN DASHBOARD
# ==================================================

if not results:
    st.error(
        "Unable to retrieve air-quality data. "
        "Check your internet connection and try again."
    )

    st.info(
        "This dashboard needs internet access to retrieve "
        "the selected district's modeled air-quality data."
    )

    st.stop()

df = pd.DataFrame(results)

avg_aqi = df["US AQI"].mean()
category, category_color = get_aqi_category(avg_aqi)

# ==================================================
# TOP METRIC CARDS
# ==================================================

st.subheader(f"📍 Air Quality Overview — {district}")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Average US AQI",
    f"{avg_aqi:.0f}"
)

m2.metric(
    "Average PM2.5",
    f"{df['PM2.5'].mean():.1f} µg/m³"
    if df["PM2.5"].notna().any() else "N/A"
)

m3.metric(
    "Average PM10",
    f"{df['PM10'].mean():.1f} µg/m³"
    if df["PM10"].notna().any() else "N/A"
)

m4.metric(
    "Locations Available",
    f"{len(df)}/{len(locations)}"
)

st.markdown(
    f"""
    <div style="
        background:#17243a;
        border-left:6px solid {category_color};
        padding:18px;
        border-radius:10px;
        margin-top:15px;
        margin-bottom:20px;">
        <h3 style="margin:0;color:#f8fafc;">
            Air Quality: {category}
        </h3>
        <p style="margin:6px 0 0 0;">
            Average US AQI across the available approximate grid points.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================================
# INTERACTIVE MAP
# ==================================================

st.divider()
st.subheader("🗺️ Interactive District Pollution Map")

air_map = folium.Map(
    location=[latitude, longitude],
    zoom_start=10,
    tiles="CartoDB dark_matter"
)

for _, row in df.iterrows():
    _, color = get_aqi_category(row["US AQI"])

    popup = f"""
    <b>{district} — {row['Location']}</b><br>
    US AQI: {row['US AQI']:.0f}<br>
    PM2.5: {row['PM2.5']:.2f} µg/m³<br>
    PM10: {row['PM10']:.2f} µg/m³<br>
    NO2: {row['NO2']:.2f} µg/m³<br>
    Model time: {row['Time']}
    """

    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=11,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        tooltip=f"{row['Location']} | US AQI {row['US AQI']:.0f}",
        popup=folium.Popup(popup, max_width=300)
    ).add_to(air_map)

folium.Marker(
    [latitude, longitude],
    tooltip=f"{district} approximate center",
    icon=folium.Icon(color="blue", icon="info-sign")
).add_to(air_map)

st_folium(
    air_map,
    width=None,
    height=500,
    key=f"map_{district}"
)

st.caption(
    "Map markers represent approximate sample coordinates. "
    "They are not verified monitoring stations or district boundaries."
)

# ==================================================
# GRAPHS
# ==================================================

st.divider()
st.subheader("📊 Pollution Analytics")

left, right = st.columns(2)

with left:
    pollutants = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO2"],
        "Average Reading": [
            df["PM2.5"].mean(),
            df["PM10"].mean(),
            df["NO2"].mean()
        ]
    })

    pollutants = pollutants.dropna()

    fig_bar = px.bar(
        pollutants,
        x="Pollutant",
        y="Average Reading",
        color="Pollutant",
        title="Average Pollutant Concentrations",
        labels={
            "Average Reading": "Concentration (µg/m³)"
        },
        template="plotly_dark"
    )

    fig_bar.update_layout(
        showlegend=False,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111c2e"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

with right:
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_aqi,
            title={"text": "Average US AQI"},
            gauge={
                "axis": {"range": [0, 300]},
                "bar": {"color": category_color},
                "steps": [
                    {"range": [0, 50], "color": "#166534"},
                    {"range": [50, 100], "color": "#854d0e"},
                    {"range": [100, 150], "color": "#9a3412"},
                    {"range": [150, 200], "color": "#991b1b"},
                    {"range": [200, 300], "color": "#6b21a8"}
                ]
            }
        )
    )

    fig_gauge.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        height=350
    )

    st.plotly_chart(
        fig_gauge,
        use_container_width=True
    )

# ==================================================
# HISTORICAL / HOURLY TREND
# ==================================================

st.divider()
st.subheader("📈 Air Quality Trend")

if not trend.empty and "US AQI" in trend.columns:
    trend_plot = trend.dropna(subset=["US AQI"]).copy()

    if not trend_plot.empty:
        fig_line = px.line(
            trend_plot,
            x="Time",
            y="US AQI",
            markers=True,
            title="Modeled Hourly US AQI",
            template="plotly_dark"
        )

        fig_line.update_traces(
            line_width=3
        )

        fig_line.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#111c2e",
            xaxis_title="Time",
            yaxis_title="US AQI"
        )

        st.plotly_chart(
            fig_line,
            use_container_width=True
        )
    else:
        st.info("Hourly AQI data is not available for plotting.")
else:
    st.info("Historical or hourly trend data is unavailable.")

# ==================================================
# MACHINE LEARNING PREDICTION
# ==================================================

st.divider()
st.subheader("🤖 Machine Learning Prediction")

prediction_data = df[
    ["PM2.5", "PM10", "NO2"]
].mean().to_frame().T

if prediction_data.notna().all(axis=1).all():
    prediction = model.predict(prediction_data)[0]

    st.markdown(
        f"""
        <div style="
            background:#17243a;
            padding:20px;
            border-radius:12px;
            border:1px solid #263852;">
            <h3 style="color:#4ade80;">
                Predicted Category: {prediction}
            </h3>
            <p>
                Random Forest classification using PM2.5, PM10,
                and NO2 concentrations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "The classifier was trained on synthetic educational data "
        "and simplified labels. Its prediction is experimental, "
        "not an official AQI classification."
    )
else:
    st.warning(
        "Machine-learning prediction unavailable because "
        "one or more pollutant readings are missing."
    )

# ==================================================
# DATA TABLE
# ==================================================

st.divider()
st.subheader("📋 Detailed Location Readings")

st.dataframe(
    df[
        [
            "Location",
            "US AQI",
            "PM2.5",
            "PM10",
            "NO2",
            "Time"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

# ==================================================
# DATA SOURCE AND FOOTER
# ==================================================

if errors:
    with st.expander("Locations where data could not be retrieved"):
        for error in errors:
            st.write(error)

st.divider()

st.caption(
    "Air Pollution Intelligence | Educational CSE Project | "
    "Powered by Python, Streamlit, Plotly, Folium and Scikit-learn."
)

st.caption(
    "Data source: Open-Meteo Air Quality API. "
    "Data represents atmospheric model estimates, not necessarily "
    "ground-station measurements. US AQI is distinct from India's "
    "official CPCB National AQI. District-wide averages are "
    "approximate and are not official district measurements."
)