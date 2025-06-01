import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# 
# LOAD AND PREPARE GLOBAL DATA
# 

csv_path = 'london_crime_with_wards.csv'
df = pd.read_csv(csv_path)
df['Month'] = pd.to_datetime(df['Month'])
df['Year'] = df['Month'].dt.year
df['MonthNum'] = df['Month'].dt.month

# Filter burglary data from 2020+
df_burglary = df[df['Crime type'] == 'Burglary'].copy()
df_burglary = df_burglary[df_burglary['Year'] >= 2020]

# Global monthly grouping
df_global = df_burglary.groupby(['Year', 'MonthNum']).size().reset_index(name='Count')
df_global['Date'] = pd.to_datetime(df_global['Year'].astype(str) + '-' + df_global['MonthNum'].astype(str)) + pd.offsets.MonthBegin(0)
df_global = df_global.sort_values('Date').reset_index(drop=True)

# Add TimeIndex and cyclical features
df_global['TimeIndex'] = np.arange(len(df_global))
df_global['Month_sin'] = np.sin(2 * np.pi * df_global['MonthNum'] / 12)
df_global['Month_cos'] = np.cos(2 * np.pi * df_global['MonthNum'] / 12)

# 
# GLOBAL FORECASTING (XGBoost)
# 

features = ['Year', 'MonthNum', 'TimeIndex', 'Month_sin', 'Month_cos']
X = df_global[features]
y = df_global['Count']
dates = df_global['Date']

tscv = TimeSeriesSplit(n_splits=5)
all_actuals, all_predictions, all_dates = [], [], []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    dates_test = dates.iloc[test_idx]

    model = XGBRegressor(objective='reg:squarederror', max_depth=3, learning_rate=0.1, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    all_actuals.extend(y_test)
    all_predictions.extend(y_pred)
    all_dates.extend(dates_test)

# Results
results_df = pd.DataFrame({'Date': all_dates, 'Actual': all_actuals, 'Predicted': all_predictions}).sort_values('Date')

# Evaluation
mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
print(f'MSE (Global): {mse:.2f}, MAE (Global): {mae:.2f}')

#
# Plots
#

# Plot 1: Scatter
fig = px.scatter(results_df, x='Actual', y='Predicted', custom_data=[results_df['Date'].dt.strftime('%Y-%m')],
                 title='Global: Actual vs Predicted Burglary Counts')
fig.update_traces(marker=dict(size=8, opacity=0.7), hovertemplate='<b>Actual</b>: %{x}<br><b>Predicted</b>: %{y}<br><b>Date</b>: %{customdata[0]}<extra></extra>')
fig.add_shape(type='line', line=dict(dash='dash', color='red'),
              x0=results_df['Actual'].min(), y0=results_df['Actual'].min(),
              x1=results_df['Actual'].max(), y1=results_df['Actual'].max())
fig.show()

# Plot 2: Time-based line plot
plt.figure(figsize=(14, 6))
plt.plot(results_df['Date'], results_df['Actual'], label='Actual', marker='o')
plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted', marker='x', linestyle='--')
plt.title('Global: Actual vs Predicted Burglary Counts Over Time')
plt.xlabel('Date'); plt.ylabel('Burglary Count')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# Plot 3: Forecast Next 24 Months
last_date = df_global['Date'].max()
last_time_index = df_global['TimeIndex'].max()
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=24, freq='MS')
future_df = pd.DataFrame({'Date': future_dates})
future_df['Year'] = future_df['Date'].dt.year
future_df['MonthNum'] = future_df['Date'].dt.month
future_df['TimeIndex'] = np.arange(last_time_index + 1, last_time_index + 25)
future_df['Month_sin'] = np.sin(2 * np.pi * future_df['MonthNum'] / 12)
future_df['Month_cos'] = np.cos(2 * np.pi * future_df['MonthNum'] / 12)
future_df['Predicted'] = model.predict(future_df[features])

forecast_df = pd.DataFrame({'Date': future_df['Date'], 'Actual': np.nan, 'Predicted': future_df['Predicted']})
full_df = pd.concat([results_df, forecast_df]).sort_values('Date').reset_index(drop=True)

plt.figure(figsize=(14, 6))
plt.plot(full_df['Date'], full_df['Predicted'], label='Predicted', linestyle='--', marker='x')
plt.plot(results_df['Date'], results_df['Actual'], label='Actual', linestyle='-', marker='o')
plt.axvline(x=last_date, color='gray', linestyle='dashed', label='Forecast Start')
plt.title('Global Forecast: Burglary Counts (Next 24 Months)')
plt.xlabel('Date'); plt.ylabel('Burglary Count'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# Plot 4: Residuals
results_df['Residual'] = results_df['Actual'] - results_df['Predicted']
plt.figure(figsize=(14, 5))
plt.plot(results_df['Date'], results_df['Residual'], marker='o')
plt.axhline(0, color='red', linestyle='--')
plt.title('Global: Residuals Over Time (Actual - Predicted)')
plt.xlabel('Date'); plt.ylabel('Residual'); plt.grid(True); plt.tight_layout(); plt.show()

# Plot 5: Feature Importance
importances = model.feature_importances_
plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.title('Global: Feature Importances (XGBoost)')
plt.xlabel('Importance'); plt.grid(True); plt.tight_layout(); plt.show()

# 
# PER-WARD FORECASTS
# 

wards = df_burglary['NAME'].dropna().unique()
ward_forecasts = []

for ward in wards:
    ward_data = df_burglary[df_burglary['NAME'] == ward]
    ward_monthly = (
        ward_data.groupby([ward_data['Month'].dt.to_period('M')])
        .size()
        .reset_index(name='Count')
    )
    ward_monthly['Date'] = ward_monthly['Month'].dt.to_timestamp()
    if len(ward_monthly) < 24:
        continue  # skip if not enough data

    ward_monthly = ward_monthly.sort_values('Date').reset_index(drop=True)
    ward_monthly['Year'] = ward_monthly['Date'].dt.year
    ward_monthly['MonthNum'] = ward_monthly['Date'].dt.month
    ward_monthly['TimeIndex'] = np.arange(len(ward_monthly))
    ward_monthly['Month_sin'] = np.sin(2 * np.pi * ward_monthly['MonthNum'] / 12)
    ward_monthly['Month_cos'] = np.cos(2 * np.pi * ward_monthly['MonthNum'] / 12)

    X_ward = ward_monthly[features]
    y_ward = ward_monthly['Count']

    model = XGBRegressor(objective='reg:squarederror', max_depth=3, learning_rate=0.1, n_estimators=100, random_state=42)
    model.fit(X_ward, y_ward)

    # Forecast future
    last_index = ward_monthly['TimeIndex'].max()
    future_dates = pd.date_range(start=ward_monthly['Date'].max() + pd.offsets.MonthBegin(1), periods=24, freq='MS')
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Year'] = future_df['Date'].dt.year
    future_df['MonthNum'] = future_df['Date'].dt.month
    future_df['TimeIndex'] = np.arange(last_index + 1, last_index + 25)
    future_df['Month_sin'] = np.sin(2 * np.pi * future_df['MonthNum'] / 12)
    future_df['Month_cos'] = np.cos(2 * np.pi * future_df['MonthNum'] / 12)

    future_df['Predicted'] = model.predict(future_df[features])
    future_df['Ward'] = ward
    ward_forecasts.append(future_df[['Date', 'Ward', 'Predicted']])

# Save to CSV
if ward_forecasts:
    df_all_wards = pd.concat(ward_forecasts)
    df_all_wards.to_csv('ward_level_xgb_forecasts.csv', index=False)
    print("\n Saved ward-level XGBoost forecasts to 'ward_level_xgb_forecasts.csv'")
else:
    print("\n No ward forecasts generated (insufficient data)")

#
# Interactive plot for ward predictions
#

# Read forecasts
forecast_df = pd.read_csv('ward_level_xgb_forecasts.csv')
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

# Prepare actuals from original data
actuals_df = (
    df_burglary
    .groupby(['NAME', df_burglary['Month'].dt.to_period('M')])
    .size()
    .reset_index(name='Count')
)
actuals_df['Date'] = actuals_df['Month'].dt.to_timestamp()
wards = sorted(actuals_df['NAME'].unique())

# Create figure
fig = go.Figure()

# Add traces for each ward (hidden by default)
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

# Create dropdown menu
dropdown_buttons = []
for i, ward in enumerate(wards):
    visibility = [False] * len(wards) * 2
    visibility[i * 2] = True     # Actual
    visibility[i * 2 + 1] = True # Forecast

    dropdown_buttons.append(dict(
        label=ward,
        method='update',
        args=[{'visible': visibility},
              {'title': f"Burglary Forecast for {ward}"}]
    ))

# Update layout with dropdown
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
