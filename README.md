# Sales Forecasting Dashboard

A single Streamlit multi-page app combining all four pages:

1. **Sales Overview** — total sales by year (bar chart)
2. **Forecast Explorer** — per-segment SARIMAX forecast, MAE/RMSE, accuracy comparison
3. **Anomaly Report** — Isolation Forest anomaly detection on sales
4. **Product Demand Segments** — KMeans cluster view of sub-category demand

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Live Demo

You can access the deployed application here:
👉 **[Live Application Demo](https://sales-forecasting-pallavdas-kdsdmpogvyd2wqkq92tcsk.streamlit.app/)**

`app.py` is the only entry point — don't run the files inside `pages/` directly.
Streamlit's built-in multi-page navigation (via `st.navigation`) reads the
number prefix on each filename to fix page order, and shows the `title=`
you see in `app.py` in the sidebar.

## Files each page expects to find (relative to the folder you launch from)

| Page | Needs |
|---|---|
| Sales Overview | `train.csv` |
| Forecast Explorer | `models/sari_model_category_technology.pkl`, `..._furniture.pkl`, `..._office_supplies.pkl`, `models/sari_model_region_north.pkl`, `..._south.pkl`, `..._east.pkl`, `..._west.pkl` — each a `joblib.dump({'model': fitted_results, 'mae': mae, 'rmse': rmse}, path)` |
| Anomaly Report | `train.csv`, `vgsales.csv`, `isof_clf.pkl` |
| Product Demand Segments | `data/product_profile_with_clusters.csv`, `models/scaler.pkl`, `models/kmeans_model.pkl`, `models/cluster_label_mapping.pkl`, `models/cluster_features.pkl` |

So the working directory when you run `streamlit run app.py` needs:

```
your_project/
├── app.py
├── pages/
│   ├── 1_Sales_Overview.py
│   ├── 2_Forecast_Explorer.py
│   ├── 3_Anomaly_Report.py
│   └── 4_Product_Demand_Segments.py
├── train.csv
├── vgsales.csv
├── isof_clf.pkl
├── models/
│   ├── sari_model_category_technology.pkl
│   ├── sari_model_category_furniture.pkl
│   ├── sari_model_category_office_supplies.pkl
│   ├── sari_model_region_north.pkl
│   ├── sari_model_region_south.pkl
│   ├── sari_model_region_east.pkl
│   ├── sari_model_region_west.pkl
│   ├── scaler.pkl
│   ├── kmeans_model.pkl
│   ├── cluster_label_mapping.pkl
│   └── cluster_features.pkl
└── data/
    └── product_profile_with_clusters.csv
```

## What changed from the original 4 standalone scripts

- Each page had its own `st.set_page_config(...)` and its own `st.sidebar.title("Navigation")` —
  these clash when multiple pages exist in one app, so `st.set_page_config` now
  lives once in `app.py`, and the duplicate sidebar nav titles were removed
  (Streamlit's own page navigation replaces them).
- Everything else — data loading, model loading, charts, tables — is untouched
  from your original `sfd_app.py`, `fe_app.py`, `ar_app.py`, `pds_app.py`.

## Note on Sales Overview

The spec for that page also calls for a monthly sales trend line chart and
an interactive region/category filter, but the original `sfd_app.py` only
implemented the yearly bar chart — that gap carried over as-is. Say the word
if you'd like those two pieces added.
