import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
csv_path = 'london_data.csv'
df_london = pd.read_csv(csv_path)

# Convert 'Month' to datetime and extract features
df_london['Month'] = pd.to_datetime(df_london['Month'])
df_london['Year'] = df_london['Month'].dt.year
df_london['Month'] = df_london['Month'].dt.month

# Filter for burglary crimes
df_burglary = df_london[df_london['Crime type'] == 'Burglary']

# Define features and target variable
X = df_burglary[['Year', 'Month']]
y = df_burglary['Count']  # Still need to define count

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the training and testing sets into DMatrix objects
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100
}

# Train the XGBoost model
model = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions on the test set
y_pred = model.predict(dtest)

# Evaluate the model using Mean Squared Error (MSE) and Mean Absolute Error (MAE)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
