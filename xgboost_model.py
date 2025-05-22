import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px

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
X = df_burglary_grouped[['Year', 'Month']]
y = df_burglary_grouped['Count']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBRegressor
model = XGBRegressor(
    objective='reg:squarederror',
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Prepare DataFrame with results
results_df = X_test.copy()
results_df['Actual'] = y_test.values
results_df['Predicted'] = y_pred
results_df['Date'] = results_df['Year'].astype(str) + '-' + results_df['Month'].astype(str).str.zfill(2)

# Create interactive scatter plot
fig = px.scatter(
    results_df,
    x='Actual',
    y='Predicted',
    hover_data=['Date'],
    title='Actual vs Predicted Burglary Counts',
    width=1500,   # ⬅️ Increase figure width
    height=900,   # ⬅️ Increase figure height
)

# Add diagonal line of perfect prediction
fig.add_shape(
    type='line',
    line=dict(dash='dash', color='red'),
    x0=results_df['Actual'].min(), y0=results_df['Actual'].min(),
    x1=results_df['Actual'].max(), y1=results_df['Actual'].max()
)

# Optional: Tweak marker size and opacity
fig.update_traces(marker=dict(size=8, opacity=0.7))

fig.show()