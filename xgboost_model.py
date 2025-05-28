import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import numpy as np

# Load the dataset
csv_path = 'london_data.csv'
df_london = pd.read_csv(csv_path)

# Convert 'Month' to datetime
df_london['Month'] = pd.to_datetime(df_london['Month'])
df_london['Year'] = df_london['Month'].dt.year
df_london['Month'] = df_london['Month'].dt.month

# Filter for burglary crimes
df_burglary = df_london[df_london['Crime type'] == 'Burglary']

# Filter data to only include entries from 2020 onward
df_burglary = df_burglary[df_burglary['Year'] >= 2020]

# Group by year and month
df_burglary_grouped = df_burglary.groupby(['Year', 'Month']).size().reset_index(name='Count')

# Combine Year and Month into datetime
df_burglary_grouped['Date'] = pd.to_datetime(
    df_burglary_grouped['Year'].astype(str) + '-' + df_burglary_grouped['Month'].astype(str).str.zfill(2)
    ) + pd.offsets.MonthBegin(0)

# Sort by Date
df_burglary_grouped = df_burglary_grouped.sort_values('Date').reset_index(drop=True)

# Filter to keep only data from 2020 onward
df_burglary_grouped = df_burglary_grouped[df_burglary_grouped['Date'] >= '2020-01-01'].reset_index(drop=True)

# Add TimeIndex
df_burglary_grouped['TimeIndex'] = np.arange(len(df_burglary_grouped))

# Add cyclical month features
df_burglary_grouped['Month_sin'] = np.sin(2 * np.pi * df_burglary_grouped['Month'] / 12)
df_burglary_grouped['Month_cos'] = np.cos(2 * np.pi * df_burglary_grouped['Month'] / 12)

# Define features and target
features = ['Year', 'Month', 'TimeIndex', 'Month_sin', 'Month_cos']
X = df_burglary_grouped[features]
y = df_burglary_grouped['Count']
dates = df_burglary_grouped['Date']

# TimeSeriesSplit cross-validation
tscv = TimeSeriesSplit(n_splits=5)

all_actuals = []
all_predictions = []
all_dates = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    dates_test = dates.iloc[test_index]

    model = XGBRegressor(
        objective='reg:squarederror',
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    all_actuals.extend(y_test)
    all_predictions.extend(y_pred)
    all_dates.extend(dates_test)

# Combine results
results_df = pd.DataFrame({
    'Date': all_dates,
    'Actual': all_actuals,
    'Predicted': all_predictions
}).sort_values('Date').reset_index(drop=True)

# Evaluation
mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Plot 1: Scatter plot
fig = px.scatter(
    results_df,
    x='Actual',
    y='Predicted',
    custom_data=[results_df['Date'].dt.strftime('%Y-%m')],
    title='Actual vs Predicted Burglary Counts',
    width=1000,
    height=600
)

fig.update_traces(
    marker=dict(size=8, opacity=0.7),
    hovertemplate=
        '<b>Actual</b>: %{x}<br>' +
        '<b>Predicted</b>: %{y}<br>' +
        '<b>Date</b>: %{customdata[0]}<br>' +
        '<extra></extra>'
)
fig.add_shape(
    type='line',
    line=dict(dash='dash', color='red'),
    x0=results_df['Actual'].min(), y0=results_df['Actual'].min(),
    x1=results_df['Actual'].max(), y1=results_df['Actual'].max()
)

fig.show()

# Plot 2: Time-based line plot
plt.figure(figsize=(14, 6))
plt.plot(results_df['Date'], results_df['Actual'], label='Actual', marker='o', linestyle='-')
plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted', marker='x', linestyle='--')
plt.title('Actual vs Predicted Burglary Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Burglary Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3 - Forecast

# Step 1: Get the last date and index
last_date = df_burglary_grouped['Date'].max()
last_time_index = df_burglary_grouped['TimeIndex'].max()

# Step 2: Create 24 months of future data
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=24, freq='MS')
future_df = pd.DataFrame({'Date': future_dates})

# Step 3: Extract features
future_df['Year'] = future_df['Date'].dt.year
future_df['Month'] = future_df['Date'].dt.month
future_df['TimeIndex'] = np.arange(last_time_index + 1, last_time_index + 25)
future_df['Month_sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
future_df['Month_cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)

# Step 4: Define feature columns and predict
future_X = future_df[['Year', 'Month', 'TimeIndex', 'Month_sin', 'Month_cos']]
future_df['Predicted'] = model.predict(future_X)

# Step 5: Combine with original results for plotting
forecast_df = pd.DataFrame({
    'Date': future_df['Date'],
    'Actual': np.nan,
    'Predicted': future_df['Predicted']
})

full_df = pd.concat([results_df, forecast_df]).sort_values('Date').reset_index(drop=True)

# Step 6: Plot full forecast
plt.figure(figsize=(14, 6))
plt.plot(full_df['Date'], full_df['Predicted'], label='Predicted', linestyle='--', marker='x')
plt.plot(results_df['Date'], results_df['Actual'], label='Actual', linestyle='-', marker='o')
plt.axvline(x=last_date, color='gray', linestyle='dashed', label='Forecast Start')
plt.title('Actual and Forecasted Burglary Counts (Next 24 Months)')
plt.xlabel('Date')
plt.ylabel('Burglary Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
