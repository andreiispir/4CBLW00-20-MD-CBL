import pulp
import pandas as pd
import os
import matplotlib.pyplot as plt
import geopandas as gpd


#loading the data with the robust datapath
current_dir = os.getcwd()
csv_filename = 'london_crime_with_wards.csv'
csv_path = os.path.join(current_dir, csv_filename)
df = pd.read_csv(csv_path)

# Burglary counts per ward
crime_counts = df.groupby("NAME").size().to_dict()
wards = list(crime_counts.keys())

# constraints
max_officers_per_ward = 100
burglary_hours_per_officer = 8  # 2h/day × 4 days
total_officer_limit = 4000  # set your actual total limit here

# create a linear programming problem
prob = pulp.LpProblem("Maximize_Safety_Given_Officer_Limit", pulp.LpMaximize)

# creating decision variables
officers = pulp.LpVariable.dicts("Officers", wards, lowBound=0, upBound=max_officers_per_ward, cat='Integer')
coverage_ratio = pulp.LpVariable("CoverageRatio", lowBound=0, cat='Continuous')

# maximise the hours of burglary coverage per crime --> the task
prob += coverage_ratio, "Maximize_Coverage_Per_Crime"

# constraints added to the model
prob += pulp.lpSum([officers[w] for w in wards]) <= total_officer_limit, "TotalOfficerLimit"

# burglary coverage must meet coverage_ratio × crimes
for w in wards:
    prob += burglary_hours_per_officer * officers[w] >= coverage_ratio * crime_counts[w], f"CoverageIn_{w}"

# Solveing command
prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=120, gapRel=0.02))


# printing the results
print("Optimization Status:", pulp.LpStatus[prob.status])
print(f"Max achievable coverage (hours per crime): {coverage_ratio.varValue:.2f}")
allocation = {w: int(officers[w].varValue) for w in wards}
for ward, n in sorted(allocation.items(), key=lambda x: -x[1]):
    print(f"{ward}: {n} officers")

print("Total officers used:", sum(allocation.values()))




# Loading the ward boundaries shapefile
wards_gdf = gpd.read_file("London-wards-2018/London-wards-2018_ESRI/London_Ward.shp")


# Ccreating an allocation DataFrame of wards in london (for the map)
allocation_df = pd.DataFrame.from_dict(allocation, orient='index', columns=['Officers'])
allocation_df.index.name = 'NAME'
allocation_df.reset_index(inplace=True)

# Merging the allocation data with the geospatial data
merged_gdf = wards_gdf.merge(allocation_df, on='NAME')

# making the plot
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
merged_gdf.plot(column='Officers', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('Optimized Police Officer Allocation per London Ward - ILP', fontsize=15)
ax.axis('off')
plt.show()

# saving the plot
fig.savefig("officer_allocation_map.png", dpi=300)
