import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

# Page 4 — Product Demand Segments
# Display the cluster chart from Task 6
# Show which sub-categories belong to which demand cluster in a table

st.set_page_config(page_title="Product Demand Segments", layout="wide")
st.markdown("# Product Demand Segments")
st.sidebar.header("Product Demand Segments")

FEATURES_DISPLAY = ['Total_Sales_Volume', 'Average_Order_Value', 'Sales_Volatility', 'YoY_Sales_Growth_Rate_Pct']

# --- Load pre-computed cluster assignments (already scored in analysis.ipynb) ---
@st.cache_data
def load_data():
    return pd.read_csv('data/product_profile_with_clusters.csv')

# --- Load model artifacts (kept for scoring NEW products in-app, not for the plot below) ---
@st.cache_resource
def load_model_artifacts():
    scaler = joblib.load('models/scaler.pkl')
    kmeans_model = joblib.load('models/kmeans_model.pkl')
    cluster_label_mapping = joblib.load('models/cluster_label_mapping.pkl')
    features = joblib.load('models/cluster_features.pkl')
    return scaler, kmeans_model, cluster_label_mapping, features

df = load_data()
scaler, kmeans_model, cluster_label_mapping, features = load_model_artifacts()

# --- Chart: scatter of sub-categories, colored by Demand_Group ---
st.subheader("Demand Segmentation Matrix")

fig = px.scatter(
    df,
    x='Total_Sales_Volume',
    y='Sales_Volatility',
    color='Demand_Group',
    size='Average_Order_Value',
    hover_name='Sub_Category',
    hover_data=FEATURES_DISPLAY,
)
fig.update_traces(marker=dict(line=dict(width=1, color='white')))
fig.update_layout(
    legend_title_text='Demand Group',
    xaxis_title='Total Sales Volume ($)',
    yaxis_title='Sales Volatility (Std Dev)',
)
st.plotly_chart(fig, use_container_width=True)

# --- Table: which sub-categories fall into which cluster ---
st.subheader("Sub-Category → Demand Cluster Mapping")
st.dataframe(
    df[['Sub_Category', 'Demand_Group'] + FEATURES_DISPLAY].sort_values('Demand_Group'),
    use_container_width=True,
    hide_index=True,
)

