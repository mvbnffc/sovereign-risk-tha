# Script to produce the exposure datasets used in the risk analysis.
# We will disaggregate residential and non-residential capital stock from GIRI to 
# GHSL residential and non-residential area grids.

import pandas as pd
import geopandas as gpd
import rasterio
from preperation_functions import disaggregate_building_values, write_raster

# Set the dataset paths
csv_file_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\GEM_Exposure_Summary_Adm1.csv"
admin_shapefile_path = r"D:\projects\sovereign-risk\Thailand\data\GADM\gadm36_THA_shp\gadm36_THA_1.shp"
residential_raster_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\GHSL_res_THA.tif"
non_residential_raster_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\GHSL_nres_THA.tif"
res_output_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\GHSL_res_val_THA.tif"
nres_output_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\GHSL_nres_val_THA.tif"

# Read CSV file
df = pd.read_csv(csv_file_path)

# Load Admin1 regions
admin_areas = gpd.read_file(admin_shapefile_path)

# Load raster datasets NOTE: at the moment we are using built up area datasets -> may want to test how results change with built up volume data
res_raster = rasterio.open(residential_raster_path)
nres_raster = rasterio.open(non_residential_raster_path)

# Disaggregate values
res_values = disaggregate_building_values(admin_areas, df, res_raster, 'Res')
com_values = disaggregate_building_values(admin_areas, df, nres_raster, 'Com')
ind_values = disaggregate_building_values(admin_areas, df, nres_raster, 'Ind')
# Combine commerical and industrial into non-residential
nres_values = com_values + ind_values

# Write rasters
write_raster(res_output_path, res_raster, res_values)
write_raster(nres_output_path, nres_raster, nres_values)
