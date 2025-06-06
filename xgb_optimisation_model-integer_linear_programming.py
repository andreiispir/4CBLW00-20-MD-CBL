import pulp
import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

global_allocation = {}

def ilp_optimisation_model_xgboost(csv_path):
    """
    Optimisation model for allocating police officers to wards based on burglary forecasts.
    
    Parameters:
    csv_path (str): Path to the CSV file containing ward-level burglary forecasts.
    
    Returns:
    dict: A dictionary with ward names as keys and the number of officers allocated as values.
    """
    global global_allocation
    df = pd.read_csv(csv_path)

    # Calculate average predictions
    crime_counts = df.groupby("Ward")['Predicted'].mean().to_dict()
    wards = list(crime_counts.keys())

    # Calculate total crimes and crime proportions
    total_crimes = sum(crime_counts.values())
    crime_proportions = {w: count/total_crimes for w, count in crime_counts.items()}

    # Constraints
    max_officers_per_ward = 100
    min_officers_per_ward = 0
    burglary_hours_per_officer = 8  # 2h/day × 4 days
    total_officer_limit = 100 * len(wards) 

    # Calculate minimum total officers needed based on total crimes and available hours
    min_total_officers = int(total_crimes / burglary_hours_per_officer)  # Assuming that each crime needs at least 1 officer-hour

    # Decision variables
    officers = pulp.LpVariable.dicts("Officers", wards, lowBound=min_officers_per_ward, upBound=max_officers_per_ward, cat='Integer')

    # Create the model
    prob = pulp.LpProblem("Optimize_Officer_Allocation", pulp.LpMinimize)

    # Calculate target officers based on crime proportions
    total_available_officers = total_officer_limit * 0.8  # Use 80% of maximum capacity
    target_officers = {w: min(max_officers_per_ward, max(min_officers_per_ward, int(crime_proportions[w] * total_available_officers))) for w in wards}

    # Deviation variables
    under_staff = pulp.LpVariable.dicts("Under_Staff", wards, lowBound=0, upBound=100, cat='Integer')
    over_staff = pulp.LpVariable.dicts("Over_Staff", wards, lowBound=0, upBound=100, cat='Integer')

    # Adding constraints, so the model sticks to the target resources
    for w in wards:
        prob += officers[w] <= max_officers_per_ward, f"Max_Officers_{w}"
        prob += officers[w] >= min_officers_per_ward, f"Min_Officers_{w}"
        
        # Deviation constraints
        prob += officers[w] - target_officers[w] == over_staff[w] - under_staff[w], f"Deviation_{w}"
        
        # Minimum officers based on crime rate
        min_required = max(1, int(crime_proportions[w] * 50))  # At least 1 officer, scaled by crime rate
        prob += officers[w] >= min_required, f"Min_Required_{w}"

    # Modified objective function with balanced weights
    prob += (10.0 * pulp.lpSum(crime_proportions[w] * under_staff[w] for w in wards) + 2.0 * pulp.lpSum(over_staff[w] for w in wards) + 1.0 * pulp.lpSum(under_staff[w] + over_staff[w] for w in wards)), "Balanced_Objective"

    # Total officer limit constraint
    prob += pulp.lpSum(officers[w] for w in wards) <= total_officer_limit, "Total_Officers"

    # Solve the optimization problem
    prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=120, gapRel=0.02))

    # Check if optimization was successful
    if pulp.LpStatus[prob.status] != 'Optimal':
        print(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
        exit()

    # Results
    global_allocation = {w: int(officers[w].varValue) for w in wards}
    return global_allocation
    # print(allocation)


current_dir = os.getcwd()
csv_filename = 'ward_level_xgb_forecasts.csv'
csv_path = os.path.join(current_dir, csv_filename)

optimisation = ilp_optimisation_model_xgboost(csv_path)







# Create visualization
wards_gdf = gpd.read_file("London-wards-2018/London-wards-2018_ESRI/London_Ward.shp")

# Create allocation DataFrame
allocation_df = pd.DataFrame.from_dict(optimisation, orient='index', columns=['Officers'])
allocation_df.to_csv("officer_allocation_forecast.csv")
allocation_df.index.name = 'NAME'
allocation_df.reset_index(inplace=True)

# Merge allocation with geographic data
merged_gdf = wards_gdf.merge(allocation_df, on='NAME')

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
merged_gdf.plot(column='Officers', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('Optimized Police Officer Allocation per London Ward - XGBoost', fontsize=15)
ax.axis('off')
plt.show()

# Save the plot
fig.savefig("officer_allocation_forecast_map - XGBoost.png", dpi=300)

# Print some statistics
print("\nAllocation Statistics:")
print(f"Total officers allocated: {sum(optimisation.values())}")
print("\nTop 10 wards by officer allocation:")
for ward, officers in sorted(optimisation.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{ward}: {officers} officers")









df = pd.read_csv(csv_path)

    # Calculate average predictions
crime_counts = df.groupby("Ward")['Predicted'].mean().to_dict()
# Creating a DataFrame for analysis
analysis_df = pd.DataFrame({'Ward': list(crime_counts.keys()),'Crimes': list(crime_counts.values()),'Officers': [global_allocation[ward] for ward in crime_counts.keys()]})

# Calculateing statistics
analysis_df['Crime_Rate'] = analysis_df['Crimes'] / analysis_df['Crimes'].sum()
analysis_df['Officer_Rate'] = analysis_df['Officers'] / analysis_df['Officers'].sum()
r2 = r2_score(analysis_df['Crime_Rate'], analysis_df['Officer_Rate'])
rmse = np.sqrt(mean_squared_error(analysis_df['Crime_Rate'], analysis_df['Officer_Rate']))
correlation, p_value = stats.pearsonr(analysis_df['Crime_Rate'], analysis_df['Officer_Rate'])

# Printing the results 
print("\nModel Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Correlation coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

# Calculateing distribution statistics
print("\nDistribution Statistics:")
print("Officers per crime:")
print(analysis_df['Officers'].sum() / analysis_df['Crimes'].sum())
print("\nSummary statistics for officer allocation:")
print(analysis_df['Officers'].describe())

# Create a scatter plot of crime rate vs officer rate
plt.figure(figsize=(10, 6))
plt.scatter(analysis_df['Crime_Rate'], analysis_df['Officer_Rate'], alpha=0.5)
plt.plot([0, max(analysis_df['Crime_Rate'])], [0, max(analysis_df['Crime_Rate'])], 'r--', label='Perfect correlation')
plt.xlabel('Crime Rate')
plt.ylabel('Officer Rate')
plt.title('Crime Rate vs Officer Rate')
plt.legend()
plt.grid(True)
plt.savefig("correlation_plot.png", dpi=300)
plt.close()










# Calculating accuracy
def calculate_accuracy_metrics(actual, predicted, tolerances=[0.05, 0.1, 0.2]):
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    accuracy_scores = {}
    for tolerance in tolerances:
        within_tolerance = np.sum(np.abs((actual - predicted) / actual) <= tolerance)
        accuracy_scores[f"Accuracy within {tolerance*100}%"] = within_tolerance / len(actual) * 100
    
    return mape, accuracy_scores

# Calculate accuracy metrics
mape, accuracy_scores = calculate_accuracy_metrics(analysis_df['Crime_Rate'], analysis_df['Officer_Rate'])

# Printing accuracy
print("\nAccuracy Metrics:")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
for threshold, score in accuracy_scores.items():
    print(f"{threshold}: {score:.2f}%")

# Calculating relative allocation accuracy
over_allocated = len(analysis_df[analysis_df['Officer_Rate'] > analysis_df['Crime_Rate']])
under_allocated = len(analysis_df[analysis_df['Officer_Rate'] < analysis_df['Crime_Rate']])
perfectly_allocated = len(analysis_df[analysis_df['Officer_Rate'] == analysis_df['Crime_Rate']])

print("\nAllocation Analysis:")
print(f"Over-allocated wards: {over_allocated} ({over_allocated/len(analysis_df)*100:.1f}%)")
print(f"Under-allocated wards: {under_allocated} ({under_allocated/len(analysis_df)*100:.1f}%)")
print(f"Perfectly allocated wards: {perfectly_allocated} ({perfectly_allocated/len(analysis_df)*100:.1f}%)")


pd.set_option('display.max_rows', None)
print(analysis_df)


