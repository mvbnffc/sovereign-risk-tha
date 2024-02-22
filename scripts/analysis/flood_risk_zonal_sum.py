# This script loops through all the risk rasters, summing risk within the elements of the input vector dataset.

import geopandas as gpd
from rasterstats import zonal_stats
import pandas as pd
from analysis_info import zonal_sum_rasters_info

# Set basin path
basin_path = r"D:\projects\sovereign-risk\Thailand\data\flood\basins\analysis_basins.gpkg"
# Set output path
output_path = r"D:\projects\sovereign-risk\Thailand\analysis\flood\risk_analysis\risk_basin_zonal_sum.csv"

# Load basins
basin_df = gpd.read_file(basin_path)

# Initialize an empty DataFrame to store zonal statistics from all rasters
results_df = pd.DataFrame()

# Loop through all rasters. Calculating zonal sum for each
for raster_info in zonal_sum_rasters_info:
    # Calculate zonal statistics for each raster
    zs = zonal_stats(basin_df, raster_info['file'], stats="sum", geojson_out=True)
    
    # Prepare a DataFrame from the zonal statistics. Store all the necessary column info from the original GeoDataFrame
    temp_df = pd.DataFrame({
        "FID": [feat['id'] for feat in zs],
        "GID_1": [feat['properties']['flpr_gid_1'] for feat in zs],
        "NAME": [feat['properties']['NAME'] for feat in zs],
        "HB_L4": [feat['properties']['HYBAS_ID_04'] for feat in zs],
        "HB_L5": [feat['properties']['HYBAS_ID_05'] for feat in zs],
        "HB_L6": [feat['properties']['HYBAS_ID_06'] for feat in zs],
        "HB_L7": [feat['properties']['HYBAS_ID_07'] for feat in zs],
        "Pr_L": [feat['properties']['MerL_Riv'] for feat in zs],
        "Add_Pr": [feat['properties']['Add_Pr'] for feat in zs],
        "New_Pr_L": [feat['properties']['New_Pr_L'] for feat in zs],
        "damages": [feat['properties']['sum'] for feat in zs]
    })
    
    # Add raster information to the DataFrame
    temp_df["epoch"] = raster_info['epoch']
    temp_df["adaptation_scenario"] = raster_info['adaptation_scenario']
    temp_df["RP"] = raster_info['RP']
    temp_df["urban_class"] = raster_info['urban_class']

    # Remove NaNs from damage column
    temp_df['damages'] = temp_df['damages'].fillna(0)
    
    # Concatenate the temporary DataFrame with the results DataFrame
    results_df = pd.concat([results_df, temp_df], ignore_index=True)

# Save results
results_df.to_csv(output_path)