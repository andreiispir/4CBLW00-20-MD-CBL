import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

# LOAD AND PREPARE DATA
csv_path = 'london_crime_with_wards.csv'
df = pd.read_csv(csv_path)
df['Month'] = pd.to_datetime(df['Month'])
df['Year'] = df['Month'].dt.year
df['MonthNum'] = df['Month'].dt.month

# Filter for Burglary crimes from 2020 onward
df_burglary = df[(df['Crime type'] == 'Burglary') & (df['Year'] >= 2020)].copy()

# Define base features and add lag feature later
base_features = ['Year', 'MonthNum', 'TimeIndex', 'Month_sin', 'Month_cos']
features_with_lag = base_features + ['Lag1']

# PER-WARD FORECASTS
wards = df_burglary['NAME'].dropna().unique()
ward_forecasts = []

# Collect metrics per ward here
metrics_list = []

for ward in wards:
    ward_data = df_burglary[df_burglary['NAME'] == ward]
    ward_monthly = (
        ward_data.groupby([ward_data['Month'].dt.to_period('M')])
        .size()
        .reset_index(name='Count')
    )
    ward_monthly['Date'] = ward_monthly['Month'].dt.to_timestamp()

    if len(ward_monthly) < 24:
        continue  # skip wards with too little data

    ward_monthly = ward_monthly.sort_values('Date').reset_index(drop=True)
    ward_monthly['Year'] = ward_monthly['Date'].dt.year
    ward_monthly['MonthNum'] = ward_monthly['Date'].dt.month
    ward_monthly['TimeIndex'] = np.arange(len(ward_monthly))
    ward_monthly['Month_sin'] = np.sin(2 * np.pi * ward_monthly['MonthNum'] / 12)
    ward_monthly['Month_cos'] = np.cos(2 * np.pi * ward_monthly['MonthNum'] / 12)
    ward_monthly['Lag1'] = ward_monthly['Count'].shift(1)

    ward_monthly = ward_monthly.dropna().reset_index(drop=True)

    X_ward = ward_monthly[features_with_lag]
    y_ward = ward_monthly['Count']

    model = XGBRegressor(objective='reg:squarederror', max_depth=3, learning_rate=0.1,
                         n_estimators=100, random_state=42)
    model.fit(X_ward, y_ward)

    # Predict past (train data)
    ward_monthly['Predicted'] = model.predict(X_ward)
    ward_monthly['Ward'] = ward

    # Calculate MAE and RMSE for training period
    mae = mean_absolute_error(ward_monthly['Count'], ward_monthly['Predicted'])
    mse = mean_squared_error(ward_monthly['Count'], ward_monthly['Predicted'])
    rmse = np.sqrt(mse)

    metrics_list.append({'Ward': ward, 'MAE': mae, 'RMSE': rmse})

    # Forecast future
    last_index = ward_monthly['TimeIndex'].max()
    last_known_lag = ward_monthly['Count'].iloc[-1]

    future_dates = pd.date_range(start=ward_monthly['Date'].max() + pd.offsets.MonthBegin(1),
                                 periods=24, freq='MS')
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Year'] = future_df['Date'].dt.year
    future_df['MonthNum'] = future_df['Date'].dt.month
    future_df['TimeIndex'] = np.arange(last_index + 1, last_index + 25)
    future_df['Month_sin'] = np.sin(2 * np.pi * future_df['MonthNum'] / 12)
    future_df['Month_cos'] = np.cos(2 * np.pi * future_df['MonthNum'] / 12)

    preds = []
    lag = last_known_lag

    for i in range(len(future_df)):
        row = future_df.iloc[i][base_features].tolist()
        input_row = pd.DataFrame([row + [lag]], columns=features_with_lag)
        pred = model.predict(input_row)[0]
        preds.append(pred)
        lag = pred  # Use predicted value as next lag

    future_df['Predicted'] = preds
    future_df['Ward'] = ward

    # Combine past + future predictions
    combined_df = pd.concat([
        ward_monthly[['Date', 'Ward', 'Predicted']],
        future_df[['Date', 'Ward', 'Predicted']]
    ])
    ward_forecasts.append(combined_df)

# Save forecasts to CSV
if ward_forecasts:
    df_all_wards = pd.concat(ward_forecasts)
    df_all_wards.to_csv('ward_level_xgb_forecasts.csv', index=False)
    print("\nSaved ward-level XGBoost forecasts to 'ward_level_xgb_forecasts.csv'")
else:
    print("\nNo ward forecasts generated (insufficient data)")

# Save metrics to DataFrame and print total MAE
metrics_df = pd.DataFrame(metrics_list)
print("\nMAE and RMSE per ward:")
print(metrics_df)

total_mae = metrics_df['MAE'].sum()
print(f"\nTotal (average) MAE across all wards: {total_mae:.4f}")

# INTERACTIVE DROPDOWN PLOTLY FOR WARDS
actuals_df = (
    df_burglary
    .groupby(['NAME', df_burglary['Month'].dt.to_period('M')])
    .size()
    .reset_index(name='Count')
)
actuals_df['Date'] = actuals_df['Month'].dt.to_timestamp()
wards = sorted(actuals_df['NAME'].unique())

forecast_df = df_all_wards.copy()
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

fig = go.Figure()
for i, ward in enumerate(wards):
    actual = actuals_df[actuals_df['NAME'] == ward]
    forecast = forecast_df[forecast_df['Ward'] == ward]

    fig.add_trace(go.Scatter(
        x=actual['Date'], y=actual['Count'],
        mode='lines+markers', name='Actual',
        visible=(i == 0), line=dict(color='blue'),
        legendgroup=ward, showlegend=(i == 0)
    ))

    fig.add_trace(go.Scatter(
        x=forecast['Date'], y=forecast['Predicted'],
        mode='lines+markers', name='Forecast',
        visible=(i == 0), line=dict(color='orange', dash='dash'),
        legendgroup=ward, showlegend=(i == 0)
    ))

dropdown_buttons = []
for i, ward in enumerate(wards):
    visibility = [False] * len(wards) * 2
    visibility[i * 2] = True
    visibility[i * 2 + 1] = True

    dropdown_buttons.append(dict(
        label=ward,
        method='update',
        args=[{'visible': visibility},
              {'title': f"Burglary Forecast for {ward}"}]
    ))

fig.update_layout(
    updatemenus=[{
        'active': 0,
        'buttons': dropdown_buttons,
        'x': 1.05,
        'xanchor': 'left',
        'y': 1,
        'yanchor': 'top'
    }],
    title=f"Burglary Forecast for {wards[0]}",
    xaxis_title='Date',
    yaxis_title='Burglary Count',
    width=1000,
    height=600
)

fig.show()
