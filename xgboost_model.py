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

# Convert 'Month' to datetime and extract features
df_london['Month'] = pd.to_datetime(df_london['Month'])
df_london['Year'] = df_london['Month'].dt.year
df_london['Month'] = df_london['Month'].dt.month

# Filter for burglary crimes
df_burglary = df_london[df_london['Crime type'] == 'Burglary']

# Group by year and month, and count number of burglaries
df_burglary_grouped = df_burglary.groupby(['Year', 'Month']).size().reset_index(name='Count')

# Define features and target
X = df_burglary_grouped[['Year', 'Month']].copy()
y = df_burglary_grouped['Count'].copy()

# Combine Year and Month into datetime
X['Date'] = pd.to_datetime(X['Year'].astype(str) + '-' + X['Month'].astype(str).str.zfill(2))

# Sort by Date (important for TimeSeriesSplit)
X = X.sort_values('Date')
y = y.loc[X.index].reset_index(drop=True)
X = X.reset_index(drop=True)

# Drop 'Date' from features before training
X_features = X[['Year', 'Month']]

# Use TimeSeriesSplit for time-aware cross-validation
tscv = TimeSeriesSplit(n_splits=5)

all_actuals = []
all_predictions = []
all_dates = []

for train_index, test_index in tscv.split(X_features):
    X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    dates_test = X.iloc[test_index]['Date']

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

# Convert results to DataFrame
results_df = pd.DataFrame({
    'Date': all_dates,
    'Actual': all_actuals,
    'Predicted': all_predictions
}).sort_values('Date').reset_index(drop=True)

# Evaluate performance
mse = mean_squared_error(results_df['Actual'], results_df['Predicted'])
mae = mean_absolute_error(results_df['Actual'], results_df['Predicted'])

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Plot 1: Interactive actual vs predicted scatter

fig = px.scatter(
    results_df,
    x='Actual',
    y='Predicted',
    hover_data=['Date'],
    title='Cross-Validated: Actual vs Predicted Burglary Counts',
    width=1000,
    height=600
)

fig.add_shape(
    type='line',
    line=dict(dash='dash', color='red'),
    x0=results_df['Actual'].min(), y0=results_df['Actual'].min(),
    x1=results_df['Actual'].max(), y1=results_df['Actual'].max()
)

fig.update_traces(marker=dict(size=8, opacity=0.7))
fig.show()

# Plot 2: Time-based line plot

plt.figure(figsize=(14, 6))
plt.plot(results_df['Date'], results_df['Actual'], label='Actual', marker='o', linestyle='-')
plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted', marker='x', linestyle='--')
plt.title('Cross-Validated: Actual vs Predicted Burglary Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Burglary Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
