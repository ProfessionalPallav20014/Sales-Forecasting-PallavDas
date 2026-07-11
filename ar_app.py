import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

# Page 3 — Anomaly Report
#     • Display the anomaly chart from Task 5
#     • List detected anomaly dates in a table with their sales values

st.set_page_config(page_title="Anomaly Report", layout="wide")
st.sidebar.title("Navigation")
st.markdown("# Anomaly Report")
st.sidebar.header("Anomaly Report")

