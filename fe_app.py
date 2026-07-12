import os
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
#
# IMPORTANT: SARIMAX was trained per-segment (per Category value and per
# Region value), not once on the whole dataset. So this app must load a
# *different* saved model depending on which segment the user picks, not
# a single sari_model.pkl.

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

forecast_horizon = st.select_slider(
    "Select Forecast Horizon (Months Ahead)",
    options=[1, 2, 3],
    format_func=lambda x: f"{x} Month{'s' if x > 1 else ''} Ahead",
    key="forecast_horizon"
)

MODEL_DIR = "models"  # folder holding all per-segment SARIMAX artifacts


def get_model_filename(f_type: str, segment: str) -> str:
    """
    Deterministic filename per (type, segment) combo, e.g.:
      models/sari_model_category_technology.pkl
      models/sari_model_category_office_supplies.pkl
      models/sari_model_region_north.pkl
    Each file must be created with:
      joblib.dump({'model': fitted_results, 'mae': mae, 'rmse': rmse}, path)
    in your training notebook.
    """
    safe_segment = segment.lower().replace(" ", "_")
    return os.path.join(MODEL_DIR, f"sari_model_{f_type.lower()}_{safe_segment}.pkl")


@st.cache_resource(show_spinner=False)
def load_forecasting_artifact(f_type: str, segment: str):
    """
    Loads the joblib artifact for exactly one segment's model.
    Cached per (f_type, segment) key, so Streamlit keeps up to 7 models
    warm in memory across reruns instead of re-reading disk every click,
    while still only loading the one that's actually needed.
    """
    path = get_model_filename(f_type, segment)
    try:
        artifact = joblib.load(path)
        return artifact, None
    except FileNotFoundError:
        return None, f"🚨 Model file not found: `{path}`. Train and save it first (see below)."
    except Exception as e:
        return None, f"🚨 Failed to load model: {str(e)}"


artifact, load_error = load_forecasting_artifact(forecast_type, selected_segment)

if load_error:
    st.error(load_error)
    st.info(
        "Expected file layout — one artifact per segment:\n\n"
        "```\n"
        "models/sari_model_category_technology.pkl\n"
        "models/sari_model_category_furniture.pkl\n"
        "models/sari_model_category_office_supplies.pkl\n"
        "models/sari_model_region_north.pkl\n"
        "models/sari_model_region_south.pkl\n"
        "models/sari_model_region_east.pkl\n"
        "models/sari_model_region_west.pkl\n"
        "```\n"
        "Each saved with `joblib.dump({'model': fitted_results, 'mae': mae, 'rmse': rmse}, path)`."
    )
else:
    model = artifact.get("model")
    mae = artifact.get("mae")
    rmse = artifact.get("rmse")

    if model is None:
        st.error("Loaded artifact has no 'model' key — check how it was saved in the notebook.")
    else:
        try:
            forecast_output = model.forecast(steps=forecast_horizon)
            forecast_dates = pd.date_range(
                start=pd.Timestamp.now(), periods=forecast_horizon, freq='ME'
            )
            forecast_df = pd.DataFrame(
                {'Forecasted Sales': forecast_output}, index=forecast_dates
            )

            col1, col2 = st.columns([3, 1])

            with col1:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                sns.lineplot(
                    data=forecast_df, x=forecast_df.index, y='Forecasted Sales',
                    marker='o', color='#1E88E5', ax=ax, linewidth=2.5
                )
                ax.set_title(
                    f"{selected_segment} — {forecast_horizon}-Month Demand Outlook",
                    fontsize=12, fontweight='bold'
                )
                ax.set_xlabel("Timeline")
                ax.set_ylabel("Sales ($)")
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: pd.to_datetime(x).strftime('%b %d'))
                )
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
                plt.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)

            with col2:
                st.markdown("**Model Accuracy**")
                metrics_df = pd.DataFrame(
                    {
                        "Metric": ["MAE", "RMSE"],
                        "Value": [
                            f"${mae:,.2f}" if mae is not None else "N/A",
                            f"${rmse:,.2f}" if rmse is not None else "N/A",
                        ],
                    }
                ).set_index("Metric")
                st.table(metrics_df)

                st.markdown("**Forecasted Values**")
                st.dataframe(
                    forecast_df.style.format({"Forecasted Sales": "${:,.0f}"}),
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Execution Error during forecasting step: {str(e)}")
            st.info("Ensure the saved model is a fitted SARIMAXResults object from statsmodels.")

st.markdown("---")
with st.expander("Compare accuracy across all segments"):
    rows = []
    for f_type, segments in [
        ("Category", ["Technology", "Furniture", "Office Supplies"]),
        ("Region", ["North", "South", "East", "West"]),
    ]:
        for seg in segments:
            path = get_model_filename(f_type, seg)
            try:
                a = joblib.load(path)
                rows.append({
                    "Type": f_type,
                    "Segment": seg,
                    "MAE": a.get("mae"),
                    "RMSE": a.get("rmse"),
                })
            except Exception:
                rows.append({"Type": f_type, "Segment": seg, "MAE": None, "RMSE": None})

    summary_df = pd.DataFrame(rows).set_index(["Type", "Segment"])
    st.dataframe(
        summary_df.style.format({"MAE": "${:,.2f}", "RMSE": "${:,.2f}"}, na_rep="N/A"),
        use_container_width=True,
    )