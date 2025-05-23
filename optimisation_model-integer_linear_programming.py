import pulp
import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats


#Loading the data with the robust datapath
current_dir = os.getcwd()
csv_filename = 'london_crime_with_wards.csv'
csv_path = os.path.join(current_dir, csv_filename)
df = pd.read_csv(csv_path)

# Burglary counts per ward
crime_counts = df.groupby("NAME").size().to_dict()
wards = list(crime_counts.keys())

# Calculateing total crimes and crime proportions
total_crimes = sum(crime_counts.values())
crime_proportions = {w: count/total_crimes for w, count in crime_counts.items()}

# Constraints
max_officers_per_ward = 100
min_officers_per_ward = 0
burglary_hours_per_officer = 8  # 2h/day × 4 days
total_officer_limit = 100 * len(wards)  # Set a fixed limit for total officers

# Creating decision variables
officers = pulp.LpVariable.dicts("Officers", wards, lowBound=min_officers_per_ward, upBound=max_officers_per_ward, cat='Integer')

# Createing  the model
prob = pulp.LpProblem("Minimize_Officers", pulp.LpMinimize)

# Creating a balance between minimizing total officers and ensuring proportional distribution
prob += pulp.lpSum(officers[w] for w in wards), "Minimize_Total_Officers"

# Constraints: total officers and proportional distribution
prob += pulp.lpSum([officers[w] for w in wards]) <= total_officer_limit, "TotalOfficerLimit"

min_total_officers = max(sum(crime_counts.values()) / (burglary_hours_per_officer * 5), len(wards) * min_officers_per_ward)
for w in wards:
    target_officers = max(min_officers_per_ward, 
                         min(max_officers_per_ward, 
                             int(crime_proportions[w] * min_total_officers)))
    prob += officers[w] >= target_officers * 0.8, f"Min_Coverage_{w}"
    prob += officers[w] <= target_officers * 1.2, f"Max_Coverage_{w}"

#Solving
prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=120, gapRel=0.02))

#Printing the results
print("Optimization Status:", pulp.LpStatus[prob.status])
allocation = {w: int(officers[w].varValue) for w in wards}
for ward, n in sorted(allocation.items(), key=lambda x: -x[1]):
    print(f"{ward}: {n} officers")

print("Total officers used:", sum(allocation.values()))









#Loading the ward boundaries shapefile
wards_gdf = gpd.read_file("London-wards-2018/London-wards-2018_ESRI/London_Ward.shp")


#Ccreating an allocation DataFrame of wards in london (for the map)
allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Officers'])
allocation_df.index.name = 'NAME'
allocation_df.reset_index(inplace=True)

#Merging the allocation data with the geospatial data
merged_gdf = wards_gdf.merge(allocation_df, on='NAME')

#Making the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
merged_gdf.plot(column='Officers', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('Optimized Police Officer Allocation per London Ward - ILP', fontsize=15)
ax.axis('off')
plt.show()

#Saving the plot
fig.savefig("officer_allocation_map.png", dpi=300)











# Creating a DataFrame for analysis
analysis_df = pd.DataFrame({'Ward': list(crime_counts.keys()),'Crimes': list(crime_counts.values()),'Officers': [allocation[ward] for ward in crime_counts.keys()]})

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
