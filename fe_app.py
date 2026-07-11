import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns

# Page 2 — Forecast Explorer
#     • Dropdown to select: Category or Region
#     • Date range slider to select forecast horizon (1, 2, or 3 months ahead)
#     • Display the forecast output from your best model for the selected inputs
#     • Show MAE and RMSE of the model below the chart

st.set_page_config(page_title="Forecast Explorer", layout="wide")
st.markdown("# Forecast Explorer")
st.sidebar.header("Forecast Explorer")
forecast_type = st.selectbox("Select Forecast Type", ["Category", "Region"], key="forecast_type")
if forecast_type == "Category":
    available_options = ["Technology", "Furniture", "Office Supplies"]
else:
    available_options = ["North", "South", "East", "West"]
selected_segment = st.sidebar.selectbox(
    f"Select Specific {forecast_type}", 
    available_options
)
forecast_horizon = st.select_slider("Select Forecast Horizon (Months Ahead)", options=[1, 2, 3], format_func=lambda x: f"{x} Month{'s' if x > 1 else ''} Ahead",key="forecast_horizon")

@st.cache_resource
def load_forecasting_model():
    # Using st.cache_resource ensures the app doesn't re-read the disk on every click
    try:
        return joblib.load('sari_model.pkl')
    except FileNotFoundError:
        st.error("🚨 `sari_model.pkl` file not found! Please check your file path.")
        return None

model = load_forecasting_model()

if model is not None:
    try:
        forecast_output = model.forecast(steps=forecast_horizon)
        forecast_dates = pd.date_range(start=pd.Timestamp.now(), periods=forecast_horizon, freq='ME')
        forecast_df = pd.DataFrame({'Forecasted Sales': forecast_output},index=forecast_dates)
        col1, col2 = st.columns([3, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            sns.lineplot(data=forecast_df, x=forecast_df.index, y='Forecasted Sales', marker='o', color='#1E88E5', ax=ax, linewidth=2.5)
            
            ax.set_title(f"{forecast_horizon}-Month Demand Outlook", fontsize=12, fontweight='bold')
            ax.set_xlabel("Timeline")
            ax.set_ylabel("Sales ($)")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: pd.to_datetime(x).strftime('%b %d')))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
            plt.grid(True, linestyle=':', alpha=0.6)
            
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Execution Error during forecasting step: {str(e)}")
        st.info("Ensure your saved `sari_model.pkl` is an active Results object from statsmodels.")