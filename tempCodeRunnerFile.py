anomalous_df = pd.merge(selected_weekly_sales, selected_vg_df, how='outer', on=['Order_Year', 'Sales'])
# anomalous_df.sort_values(by='Order_Year', inplace=True)
# anomalous_df.reset_index(inplace=True, drop=True)

# model=joblib.load("isof_clf.pkl")
# prediction = model.predict(anomalous_df['Sales'].values.reshape(-1, 1))
# abnormal_indices = np.where(prediction < 0)