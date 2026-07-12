import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

# Deployment: Interactive Dashboard using Streamlit
#  Build a working Streamlit web app with the following features:
# Page 1 — Sales Overview Dashboard
#     • Total sales by year (bar chart)
#     • Monthly sales trend line chart
#     • Sales by region and category (interactive filters)
# Page 2 — Forecast Explorer
#     • Dropdown to select: Category or Region
#     • Date range slider to select forecast horizon (1, 2, or 3 months ahead)
#     • Display the forecast output from your best model for the selected inputs
#     • Show MAE and RMSE of the model below the chart
# Page 3 — Anomaly Report
#     • Display the anomaly chart from Task 5
#     • List detected anomaly dates in a table with their sales values
# Page 4 — Product Demand Segments
#     • Display the cluster chart from Task 6
#     • Show which sub-categories belong to which demand cluster in a table

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.sidebar.title("Navigation")

st.markdown("# Sales Forecasting Dashboard")

st.sidebar.header("Sales Foreasting Dashboard")

df = pd.read_csv('train.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Order_Year'] = df['Order Date'].dt.year
chart_data = df.groupby('Order_Year')['Sales'].sum().reset_index()

chart = st.bar_chart(chart_data, x='Order_Year', y='Sales', x_label='Year', y_label='Total Sales', sort='Order_Year', height=700, width=500, use_container_width=True, color="#5c138d")