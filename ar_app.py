import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import joblib

# Page 3 — Anomaly Report
#     • Display the anomaly chart from Task 5
#     • List detected anomaly dates in a table with their sales values

st.set_page_config(page_title="Anomaly Report", layout="wide")
st.sidebar.title("Navigation")
st.markdown("# Anomaly Report")
st.sidebar.header("Anomaly Report")

# ---------------------------------------------------------------------------
# 1. Load & prepare weekly sales (train.csv)
# ---------------------------------------------------------------------------
df = pd.read_csv("train.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Order_Year'] = df['Order Date'].dt.year
df.set_index("Order Date", inplace=True)

# resample -> weekly sums, then pull the date back out as a real column
weekly_sales = df['Sales'].resample('W').sum().reset_index()
weekly_sales['Order_Year'] = weekly_sales['Order Date'].dt.year
weekly_sales = weekly_sales.rename(columns={'Order Date': 'Date'})
weekly_sales['Source'] = 'Weekly Sales'

# ---------------------------------------------------------------------------
# 2. Load & prepare video game sales (vgsales.csv)
# ---------------------------------------------------------------------------
vg_df = pd.read_csv("vgsales.csv")
vg_df_simple = vg_df.drop(columns=['Rank', 'Name', 'Platform', 'Genre', 'Publisher',
                                    'Other_Sales', 'EU_Sales', 'JP_Sales', 'Global_Sales'])
vg_df_simple.sort_values(by='Year', inplace=True)

selected_vg_df = vg_df_simple[(vg_df_simple['Year'] >= 2015) & (vg_df_simple['Year'] < 2020)].copy()
selected_vg_df = selected_vg_df.rename(columns={'NA_Sales': 'Sales', 'Year': 'Order_Year'})
selected_vg_df['Date'] = pd.NaT
selected_vg_df['Source'] = 'Video Game Sales'

# ---------------------------------------------------------------------------
# 3. Combine both as extra candidate points for the anomaly model
#    (concat, not merge — merging on exact Sales values made almost nothing
#    line up since the two datasets are on completely different scales)
# ---------------------------------------------------------------------------
combined_df = pd.concat(
    [weekly_sales[['Date', 'Order_Year', 'Sales', 'Source']],
     selected_vg_df[['Date', 'Order_Year', 'Sales', 'Source']]],
    ignore_index=True
)
combined_df.sort_values(by='Order_Year', inplace=True)
combined_df.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------------------------
# 4. Run the trained Isolation Forest model
# ---------------------------------------------------------------------------
model = joblib.load("isof_clf.pkl")
prediction = model.predict(combined_df['Sales'].values.reshape(-1, 1))
combined_df['Anomaly'] = prediction  # -1 = anomaly, 1 = normal
abnormal_indices = np.where(prediction < 0)[0]

normal_df = combined_df.drop(index=abnormal_indices)
anomaly_df = combined_df.iloc[abnormal_indices]

# ---------------------------------------------------------------------------
# 5. Chart — normal points in blue, anomalies highlighted in red
#    (using matplotlib since st.scatter_chart can't overlay two
#    differently-colored series on one axis)
# ---------------------------------------------------------------------------
st.subheader("Sales Over Time — Anomalies Highlighted")

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(normal_df['Order_Year'], normal_df['Sales'],
           color='steelblue', alpha=0.5, label='Normal')
ax.scatter(anomaly_df['Order_Year'], anomaly_df['Sales'],
           color='red', alpha=0.8, label='Anomaly')
ax.set_title('Yearly Sales Plot with Outliers Highlighted', fontsize=14, fontweight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Sales', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

st.pyplot(fig)

# ---------------------------------------------------------------------------
# 6. Table of detected anomalies
# ---------------------------------------------------------------------------
st.subheader("Detected Anomalies")

display_df = anomaly_df[['Date', 'Order_Year', 'Sales', 'Source']].copy()
display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')  # NaT -> None for vgsales rows
display_df = display_df.sort_values(by=['Order_Year']).reset_index(drop=True)

st.dataframe(display_df, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Total Points Evaluated", len(combined_df))
col2.metric("Anomalies Detected", len(anomaly_df))
col3.metric("Anomaly Rate", f"{len(anomaly_df) / len(combined_df):.1%}")