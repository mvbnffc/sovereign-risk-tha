# This script prepares the analysis basins for the risk analysis.
# This is combines HydroATLAS basins (https://www.hydrosheds.org/hydroatlas) Levels 4 - 7 with Admin 1 data (from the flood_protection_costs.py script)
# Risk analysis will be run on this layer to create a Look-Up table for different types of analyses and risk aggregation approaches

import geopandas as gpd
from functools import reduce
from preperation_functions import union_gdfs

# Set dataset paths
# HydroATLAS basins
basin_04_path = r"D:\projects\sovereign-risk\Thailand\data\flood\basins\BA_THA_lev04.shp"
basin_05_path = r"D:\projects\sovereign-risk\Thailand\data\flood\basins\BA_THA_lev05.shp"
basin_06_path = r"D:\projects\sovereign-risk\Thailand\data\flood\basins\BA_THA_lev06.shp"
basin_07_path = r"D:\projects\sovereign-risk\Thailand\data\flood\basins\BA_THA_lev07.shp"
# Admin1 data from the flood_protection_costs.py scripts
flopros_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\flopros\final_admin_flopros.gpkg"
# Final analysis basin file output path
output_path = r"D:\projects\sovereign-risk\Thailand\data\flood\basins\analysis_basins.gpkg"

# Load the datasets
basin_04 = gpd.read_file(basin_04_path)
basin_05 = gpd.read_file(basin_05_path)
basin_06 = gpd.read_file(basin_06_path)
basin_07 = gpd.read_file(basin_07_path)
flopros = gpd.read_file(flopros_path)

# Rename the columns that we want to merge. 
basin_04 = basin_04.rename(columns={'HYBAS_ID': 'HYBAS_ID_04'})
basin_05 = basin_05.rename(columns={'HYBAS_ID': 'HYBAS_ID_05'})
basin_06 = basin_06.rename(columns={'HYBAS_ID': 'HYBAS_ID_06'})
basin_07 = basin_07.rename(columns={'HYBAS_ID': 'HYBAS_ID_07'})
flopros = flopros.rename(columns={'GID_1': 'flpr_gid_1'})

# Select the relevant columns
basin_04_selected = basin_04[['geometry', 'HYBAS_ID_04']]
basin_05_selected = basin_05[['geometry', 'HYBAS_ID_05']]
basin_06_selected = basin_06[['geometry', 'HYBAS_ID_06']]
basin_07_selected = basin_07[['geometry', 'HYBAS_ID_07']]
flopros_selected = flopros[['geometry', 'flpr_gid_1', 'NAME', 'MerL_Riv', 'r_lng_km', 'u_r_lng_km', 'Add_Pr', 'New_Pr_L', 'Add_Pr_c', 'Add_Pr_c_u']]

# Put datasets in list
datasets = [basin_04_selected, basin_05_selected, basin_06_selected, basin_07_selected, flopros_selected]

# Use reduce to sequentially apply union operation across all datasets
merged = reduce(union_gdfs, datasets)

# Clean dataset (remove all rows where there are no column values consistently accross the rows)
columns_to_check = ["HYBAS_ID_04", "HYBAS_ID_05", "HYBAS_ID_06", "HYBAS_ID_07", "flpr_gid_1"]
# Create a boolean mask for non-null and non-zero values
mask = (merged[columns_to_check].notnull() & (merged[columns_to_check] != 0)).all(axis=1)
# Filter the GeoDataFrame to keep only rows that meet the criteria
filtered_merged = merged.loc[mask]

# Save to file
filtered_merged.to_file(output_path)

