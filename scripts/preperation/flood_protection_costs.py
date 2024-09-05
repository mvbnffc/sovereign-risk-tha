# Script calculates the cost of flood protection at the basin scale. Based on length of river and existing -> proposed protection levels.
import os
import geopandas as gpd
import rasterio
import pandas as pd
from preperation_functions import map_flopros_to_adm, calculate_river_length_per_admin, calculate_increased_protection, calculate_increased_protection_costs

# Set the data directory
data_dir = r"D:\projects\sovereign-risk\Thailand\data"

# Map the flopros information onto the admin 1 file for Thailand
print('Mapping FLOPROS to Admin Dataset')
flopros_database = gpd.read_file(os.path.join(data_dir, "flood/adaptation/flopros/flopros-THA.shp"))
adm_data = gpd.read_file(os.path.join(data_dir, "GADM/gadm36_THA_shp/gadm36_THA_1.shp"))
mapping_csv = pd.read_csv(os.path.join(data_dir, "flood/adaptation/flopros/flopros-adm1-map.csv"))
protection_levels = map_flopros_to_adm(mapping_csv, flopros_database, adm_data)

# Calculate river length within each basin
print('Calculate river length within each basin')
river_data = gpd.read_file(os.path.join(data_dir, "flood/river/hydroRIVERS_v10_THA.shp"))
urbanisation_data = gpd.read_file(os.path.join(data_dir, "flood/adaptation/ghsl_du/ghsl_du_gadm_THA.gpkg"))
urbanisation_level = 11
flopros_rivers = calculate_river_length_per_admin(protection_levels, river_data, 500, urbanisation_data, urbanisation_level) # 21 is peri-urban areas or denser. See GHSL report for details

# Calculate how much additional protection is needed to reach a target protection level (e.g. 100)
print('Calculate how much additional protection is needed')
flopros = calculate_increased_protection(flopros_rivers, 100)

# Calculate how much this additional protection will cost (using Boulange et al 2023 method)
print('Calculate cost of additional protection')
flopros = calculate_increased_protection_costs(flopros, 2399000) # $2.399 million per km unit cost from Boulange paper

# Write to final file
flopros.to_file(os.path.join(data_dir, 'flood/adaptation/flopros/final_admin_flopros_%s.gpkg' % urbanisation_level))

# ##### ADDITIONAL STEP FOR FUTURE RISK ANALYSIS Create a raster mask for the urban population protected by the flood infrastructure
# ref_raster = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h100glob.tif" # use a flood map as a reference for creating the mask rasters
# out_raster = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\flopros\urban_mask_%s.tif" % urbanisation_level
# filtered_urbanisation = urbanisation_data[urbanisation_data['L2'] >= urbanisation_level] # Same threshold as above

# # Open the reference raster to use its dimensions and CRS
# with rasterio.open(ref_raster) as ref:
#     # Create an empty mask with the same dimensions as the reference raster
#     mask = rasterio.features.rasterize(
#         ((geom, 1) for geom in filtered_urbanisation.geometry),
#         out_shape=ref.shape,
#         transform=ref.transform,
#         fill=0,  # Fill value for 'background'
#         all_touched=False,  # Only if centroids touch. 
#         dtype='uint8'
#     )

# # Save the mask raster
# with rasterio.open(
#     out_raster,
#     'w',
#     driver='GTiff',
#     height=ref.height,
#     width=ref.width,
#     count=1,
#     dtype='uint8',
#     crs=ref.crs,
#     transform=ref.transform,
# ) as dst:
#     dst.write(mask, 1)