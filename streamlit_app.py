
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("NDIS Cost Forecasting Dashboard")
st.markdown("This dashboard visualizes average monthly spend by support category and provides forecasts for Core support.")

# Load data
df = pd.read_csv("simulated_ndis_data.csv")
df['month'] = pd.to_datetime(df['month'])

# Average monthly spend by category
monthly_avg = df.groupby(['month', 'support_category'])['monthly_spend'].mean().reset_index()
monthly_pivot = monthly_avg.pivot(index='month', columns='support_category', values='monthly_spend').sort_index()

# User selection
category = st.selectbox("Select Support Category", monthly_pivot.columns.tolist())

# Plot historical trend
st.subheader(f"Historical Monthly Spend: {category}")
fig, ax = plt.subplots()
monthly_pivot[category].plot(ax=ax, marker='o', title=f"{category} Spend Over Time")
ax.set_ylabel("Spend ($AUD)")
st.pyplot(fig)

# Forecasting
st.subheader(f"6-Month Forecast for {category}")
series = monthly_pivot[category].dropna()
model = ARIMA(series, order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=6)
forecast.index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=6, freq='M')

# Plot forecast
fig2, ax2 = plt.subplots()
series.plot(ax=ax2, label="Historical")
forecast.plot(ax=ax2, label="Forecast", linestyle='--', marker='o')
ax2.set_title(f"{category} Spend Forecast")
ax2.set_ylabel("Spend ($AUD)")
ax2.legend()
st.pyplot(fig2)
