import streamlit as st

# ---------------------------------------------------------------------------
# Single entry point for the multi-page Sales Forecasting Dashboard.
# Each business page lives in pages/ as its own script; st.navigation wires
# them together into one app with a shared sidebar.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

overview_page = st.Page(
    "pages/sfd_app.py",
    title="Sales Overview",
    icon="📊",
    default=True,
)
forecast_page = st.Page(
    "pages/fe_app.py",
    title="Forecast Explorer",
    icon="🔮",
)
anomaly_page = st.Page(
    "pages/ar_app.py",
    title="Anomaly Report",
    icon="🚨",
)
segments_page = st.Page(
    "pages/pds_app.py",
    title="Product Demand Segments",
    icon="🧩",
)

pg = st.navigation(
    {
        "Sales Forecasting Dashboard": [
            overview_page,
            forecast_page,
            anomaly_page,
            segments_page,
        ]
    }
)

pg.run()
