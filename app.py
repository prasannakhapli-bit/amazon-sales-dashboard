import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# ===============================
# ✅ PAGE CONFIG
# ===============================
st.set_page_config(page_title="Amazon Sales Dashboard", layout="wide")

st.title("📊 Amazon Sales Dashboard")

# ===============================
# ✅ CUSTOM CSS
# ===============================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
h1, h2, h3, h4 { color: white; }
div[data-testid="metric-container"] {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# ✅ LOAD DATA
# ===============================
df = pd.read_excel(r"Amazon Store Sales Data.xlsx")

# ===============================
# ✅ DATA PREP
# ===============================
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Month"] = df["Order Date"].dt.to_period("M").astype(str)

# ===============================
# ✅ FILTERS
# ===============================
st.sidebar.header("🔍 Filters")

selected_region = st.sidebar.selectbox("Select Region", df["Region"].unique())

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["Order Date"].min(), df["Order Date"].max()]
)

months_to_predict = st.sidebar.slider("📆 Months to Predict", 1, 12, 3)

# Apply filters
df_filtered = df[
    (df["Region"] == selected_region) &
    (df["Order Date"] >= pd.to_datetime(date_range[0])) &
    (df["Order Date"] <= pd.to_datetime(date_range[1]))
]

# ===============================
# ✅ KPI
# ===============================
st.markdown("## 📈 Key Metrics")

col1, col2 = st.columns(2)

col1.metric("💰 Total Sales", f"{df_filtered['Sales'].sum():,.0f}")
col2.metric("📊 Total Profit", f"{df_filtered['Profit'].sum():,.0f}")

# ===============================
# ✅ ✅ IMPORTANT FIX (YOU MISSED THIS)
# ===============================
monthly_sales = df_filtered.groupby("Month")["Sales"].sum().reset_index()
category_sales = df_filtered.groupby("Category")["Sales"].sum().reset_index()

# ===============================
# ✅ SIDE BY SIDE CHARTS
# ===============================
st.markdown("## 📊 Sales & Category Analysis")

colA, colB = st.columns(2)

with colA:
    fig1 = px.line(
        monthly_sales,
        x="Month",
        y="Sales",
        title="Monthly Sales Trend",
        markers=True,
        template="plotly_dark"
    )
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    fig2 = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        color="Category",
        title="Category Performance",
        template="plotly_dark"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ===============================
# ✅ PROFIT VS SALES
# ===============================
profit_vs_sales = df_filtered.groupby("Category")[["Sales", "Profit"]].sum().reset_index()

fig3 = px.bar(
    profit_vs_sales,
    x="Category",
    y=["Sales", "Profit"],
    barmode="group",
    template="plotly_dark"
)

st.plotly_chart(fig3, use_container_width=True)

# ===============================
# ✅ TOP PRODUCTS
# ===============================
top_products = (
    df_filtered.groupby("Product Name")["Sales"]
    .sum()
    .nlargest(10)
    .reset_index()
)

fig4 = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    color="Sales",
    template="plotly_dark"
)

st.plotly_chart(fig4, use_container_width=True)

# ===============================
# ✅ PROPHET MODEL
# ===============================
st.markdown("## 🔮 AI Sales Forecast")

ml_data = df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index()

ml_data.rename(columns={"Order Date": "ds", "Sales": "y"}, inplace=True)

model = Prophet(yearly_seasonality=True)
model.fit(ml_data)

future = model.make_future_dataframe(periods=months_to_predict, freq="ME")
forecast = model.predict(future)

# ===============================
# ✅ METRICS
# ===============================
pred_values = forecast.tail(months_to_predict)["yhat"].values

col3, col4 = st.columns(2)
col3.metric("Next Month", f"{int(pred_values[0]):,}")

if len(pred_values) > 1:
    col4.metric("Following Month", f"{int(pred_values[1]):,}")

# ===============================
# ✅ FORECAST VISUAL
# ===============================
forecast_df = forecast.copy()

forecast_df["Type"] = "Actual"
forecast_df.loc[forecast_df["ds"] > ml_data["ds"].max(), "Type"] = "Prediction"

forecast_df.rename(columns={"yhat": "Sales Forecast"}, inplace=True)

fig_prophet = px.line(
    forecast_df,
    x="ds",
    y="Sales Forecast",
    color="Type",
    template="plotly_dark",
    title="Sales Forecast (Prophet)"
)

st.plotly_chart(fig_prophet, use_container_width=True)

# ===============================
# ✅ COMPONENTS
# ===============================
st.markdown("## 📊 Model Components")

st.pyplot(model.plot_components(forecast))

# ===============================
# ✅ NOTE
# ===============================
st.caption("📌 Forecast uses Prophet (trend + seasonality)")
