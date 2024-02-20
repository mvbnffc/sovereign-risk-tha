# Script to calculate the cost of relocating buidlings. Approach follows Dottori et al (2023) https://www.nature.com/articles/s41558-022-01540-0#Sec8
# Calculates for range of return periods 2-1000 years. To combine with other protection measures will also do one with RP2 >1 m

# TODO: This is an awful untidy script. Need to simplify this when I get the chance

import rasterio
import numpy as np
from preperation_functions import calculate_reconstruction_value_exposed, load_raster, write_raster

# Set the dataset paths
residential_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\GHSL_res_reconstruction_THA.tif"
non_residential_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\GHSL_nres_reconstruction_THA.tif"
flood_2yr_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h2glob.tif"
flood_5yr_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h5glob.tif"
flood_10yr_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h10glob.tif"
flood_25yr_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h25glob.tif"
flood_50yr_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h50glob.tif"
flood_100yr_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h100glob.tif"
flood_200yr_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h200glob.tif"
flood_500yr_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h500glob.tif"
flood_1000yr_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h1000glob.tif"
res_reconstruction_2yr_1m_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_2yr_1m.tif"
nres_reconstruction_2yr_1m_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_2yr_1m.tif"
res_reconstruction_2yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_2yr.tif"
nres_reconstruction_2yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_2yr.tif"
res_reconstruction_5yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_5yr.tif"
nres_reconstruction_5yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_5yr.tif"
res_reconstruction_10yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_10yr.tif"
nres_reconstruction_10yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_10yr.tif"
res_reconstruction_25yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_25yr.tif"
nres_reconstruction_25yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_25yr.tif"
res_reconstruction_50yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_50yr.tif"
nres_reconstruction_50yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_50yr.tif"
res_reconstruction_100yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_100yr.tif"
nres_reconstruction_100yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_100yr.tif"
res_reconstruction_200yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_200yr.tif"
nres_reconstruction_200yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_200yr.tif"
res_reconstruction_500yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_500yr.tif"
nres_reconstruction_500yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_500yr.tif"
res_reconstruction_1000yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_1000yr.tif"
nres_reconstruction_1000yr_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_1000yr.tif"

# Load the data
residential_raster, meta = load_raster(residential_raster_path, save_info=True)
non_residential_raster = load_raster(non_residential_raster_path)
flood_2yr_raster = load_raster(flood_2yr_raster_path)
flood_5yr_raster = load_raster(flood_5yr_raster_path)
flood_10yr_raster = load_raster(flood_10yr_raster_path)
flood_25yr_raster = load_raster(flood_25yr_raster_path)
flood_50yr_raster = load_raster(flood_50yr_raster_path)
flood_100yr_raster = load_raster(flood_100yr_raster_path)
flood_200yr_raster = load_raster(flood_200yr_raster_path)
flood_500yr_raster = load_raster(flood_500yr_raster_path)
flood_1000yr_raster = load_raster(flood_1000yr_raster_path)

# Calculate the reconstruction costs exposed
res_reconstruction_2yr_1m = calculate_reconstruction_value_exposed(residential_raster, flood_2yr_raster, depth_threshold=100)
nres_reconstruction_2yr_1m = calculate_reconstruction_value_exposed(non_residential_raster, flood_2yr_raster, depth_threshold=100)
res_reconstruction_2yr = calculate_reconstruction_value_exposed(residential_raster, flood_2yr_raster)
nres_reconstruction_2yr = calculate_reconstruction_value_exposed(non_residential_raster, flood_2yr_raster)
res_reconstruction_5yr = calculate_reconstruction_value_exposed(residential_raster, flood_5yr_raster)
nres_reconstruction_5yr = calculate_reconstruction_value_exposed(non_residential_raster, flood_5yr_raster)
res_reconstruction_10yr = calculate_reconstruction_value_exposed(residential_raster, flood_10yr_raster)
nres_reconstruction_10yr = calculate_reconstruction_value_exposed(non_residential_raster, flood_10yr_raster)
res_reconstruction_25yr = calculate_reconstruction_value_exposed(residential_raster, flood_25yr_raster)
nres_reconstruction_25yr = calculate_reconstruction_value_exposed(non_residential_raster, flood_25yr_raster)
res_reconstruction_50yr = calculate_reconstruction_value_exposed(residential_raster, flood_50yr_raster)
nres_reconstruction_50yr = calculate_reconstruction_value_exposed(non_residential_raster, flood_50yr_raster)
res_reconstruction_100yr = calculate_reconstruction_value_exposed(residential_raster, flood_100yr_raster)
nres_reconstruction_100yr = calculate_reconstruction_value_exposed(non_residential_raster, flood_100yr_raster)
res_reconstruction_200yr = calculate_reconstruction_value_exposed(residential_raster, flood_200yr_raster)
nres_reconstruction_200yr = calculate_reconstruction_value_exposed(non_residential_raster, flood_200yr_raster)
res_reconstruction_500yr = calculate_reconstruction_value_exposed(residential_raster, flood_500yr_raster)
nres_reconstruction_500yr = calculate_reconstruction_value_exposed(non_residential_raster, flood_500yr_raster)
res_reconstruction_1000yr = calculate_reconstruction_value_exposed(residential_raster, flood_1000yr_raster)
nres_reconstruction_1000yr = calculate_reconstruction_value_exposed(non_residential_raster, flood_1000yr_raster)

# Save the rasters
print('Writing reconstruction cost rasters')
with rasterio.open(res_reconstruction_2yr_1m_path, 'w', **meta) as dst:
    dst.write(res_reconstruction_2yr_1m, 1)
with rasterio.open(nres_reconstruction_2yr_1m_path, 'w', **meta) as dst:
    dst.write(nres_reconstruction_2yr_1m, 1)
with rasterio.open(res_reconstruction_2yr_path, 'w', **meta) as dst:
    dst.write(res_reconstruction_2yr, 1)
with rasterio.open(nres_reconstruction_2yr_path, 'w', **meta) as dst:
    dst.write(nres_reconstruction_2yr, 1)
with rasterio.open(res_reconstruction_5yr_path, 'w', **meta) as dst:
    dst.write(res_reconstruction_5yr, 1)
with rasterio.open(nres_reconstruction_5yr_path, 'w', **meta) as dst:
    dst.write(nres_reconstruction_5yr, 1)
with rasterio.open(res_reconstruction_10yr_path, 'w', **meta) as dst:
    dst.write(res_reconstruction_10yr, 1)
with rasterio.open(nres_reconstruction_10yr_path, 'w', **meta) as dst:
    dst.write(nres_reconstruction_10yr, 1)
with rasterio.open(res_reconstruction_25yr_path, 'w', **meta) as dst:
    dst.write(res_reconstruction_25yr, 1)
with rasterio.open(nres_reconstruction_25yr_path, 'w', **meta) as dst:
    dst.write(nres_reconstruction_25yr, 1)
with rasterio.open(res_reconstruction_50yr_path, 'w', **meta) as dst:
    dst.write(res_reconstruction_50yr, 1)
with rasterio.open(nres_reconstruction_50yr_path, 'w', **meta) as dst:
    dst.write(nres_reconstruction_50yr, 1)
with rasterio.open(res_reconstruction_100yr_path, 'w', **meta) as dst:
    dst.write(res_reconstruction_100yr, 1)
with rasterio.open(nres_reconstruction_100yr_path, 'w', **meta) as dst:
    dst.write(nres_reconstruction_100yr, 1)
with rasterio.open(res_reconstruction_200yr_path, 'w', **meta) as dst:
    dst.write(res_reconstruction_200yr, 1)
with rasterio.open(nres_reconstruction_200yr_path, 'w', **meta) as dst:
    dst.write(nres_reconstruction_200yr, 1)
with rasterio.open(res_reconstruction_500yr_path, 'w', **meta) as dst:
    dst.write(res_reconstruction_500yr, 1)
with rasterio.open(nres_reconstruction_500yr_path, 'w', **meta) as dst:
    dst.write(nres_reconstruction_500yr, 1)
with rasterio.open(res_reconstruction_1000yr_path, 'w', **meta) as dst:
    dst.write(res_reconstruction_1000yr, 1)
with rasterio.open(nres_reconstruction_1000yr_path, 'w', **meta) as dst:
    dst.write(nres_reconstruction_1000yr, 1)

# What are the results? May want to add a % increase factor as in Dottori et al (2023)
print('RP2 > 1m relocation costs:', np.sum(res_reconstruction_2yr_1m) + np.sum(nres_reconstruction_2yr_1m))
print('RP2 relocation costs:', np.sum(res_reconstruction_2yr) + np.sum(nres_reconstruction_2yr))
print('RP5 relocation costs:', np.sum(res_reconstruction_5yr) + np.sum(nres_reconstruction_5yr))
print('RP10 relocation costs:', np.sum(res_reconstruction_10yr) + np.sum(nres_reconstruction_10yr))
print('RP25 relocation costs:', np.sum(res_reconstruction_25yr) + np.sum(nres_reconstruction_25yr))
print('RP50 relocation costs:', np.sum(res_reconstruction_50yr) + np.sum(nres_reconstruction_50yr))
print('RP100 relocation costs:', np.sum(res_reconstruction_100yr) + np.sum(nres_reconstruction_100yr))
print('RP200 relocation costs:', np.sum(res_reconstruction_200yr) + np.sum(nres_reconstruction_200yr))
print('RP500 relocation costs:', np.sum(res_reconstruction_500yr) + np.sum(nres_reconstruction_500yr))
print('RP1000 relocation costs:', np.sum(res_reconstruction_1000yr) + np.sum(nres_reconstruction_1000yr))













