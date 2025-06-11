import pulp
import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

global_allocation = {}

def ilp_optimisation_model_prophet(csv_path, year, month):
    """
    Optimisation model for allocating police officers to wards based on burglary forecasts.
    
    Parameters:
    csv_path (str): Path to the CSV file containing ward-level burglary forecasts.
    year (int): Year to filter for
    month (int): Month to filter for (1-12)
    
    Returns:
    dict: A dictionary with ward names as keys and the number of officers allocated as values.
    """
    global global_allocation
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert ds column to datetime if it's not already
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Filter data for specific year and month
    mask = (df['ds'].dt.year == year) & (df['ds'].dt.month == month)
    df_filtered = df[mask]
    
    if df_filtered.empty:
        raise ValueError(f"No data found for year {year} month {month}")

    # Calculate predictions using filtered data
    crime_counts = df_filtered.groupby("Ward")['yhat'].mean().to_dict()
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

    # Results - convert to DataFrame
    allocation_dict = {w: int(officers[w].varValue) for w in wards}
    allocation_df = pd.DataFrame.from_dict(allocation_dict, orient='index', columns=['Officers'])
    allocation_df.index.name = 'NAME'
    allocation_df.reset_index(inplace=True)
    
    return allocation_df
    # print(allocation)


current_dir = os.getcwd()
csv_filename = 'ward_level_burglary_forecasts.csv'
csv_path = os.path.join(current_dir, csv_filename)

# Get optimization results
optimisation_df = ilp_optimisation_model_prophet(csv_path, year=2023, month=10)

# Save directly to CSV
optimisation_df.to_csv("officer_allocation_forecast.csv", index=False)

# Create visualization
wards_gdf = gpd.read_file("London-wards-2018/London-wards-2018_ESRI/London_Ward.shp")

# Merge allocation with geographic data (optimisation_df is already a DataFrame)
merged_gdf = wards_gdf.merge(optimisation_df, on='NAME')

# Plot the results
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
merged_gdf.plot(column='Officers', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('Optimized Police Officer Allocation per London Ward - Prophet', fontsize=15)
ax.axis('off')
plt.show()

# Save the plot
fig.savefig("officer_allocation_forecast_map - Prophet.png", dpi=300)

# Print statistics using DataFrame
print("\nAllocation Statistics:")
print(f"Total officers allocated: {optimisation_df['Officers'].sum()}")
print("\nTop 10 wards by officer allocation:")
print(optimisation_df.nlargest(10, 'Officers')[['NAME', 'Officers']].to_string(index=False))

