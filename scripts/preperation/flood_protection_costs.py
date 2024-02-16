# Script calculates the cost of flood protection at the basin scale. Based on length of river and existing -> proposed protection levels.
import os
import geopandas as gpd
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
flopros_rivers = calculate_river_length_per_admin(protection_levels, river_data, 500, urbanisation_data, 21) # 21 is peri-urban areas or denser. See GHSL report for details

# Calculate how much additional protection is needed to reach a target protection level (e.g. 100)
print('Calculate how much additional protection is needed')
flopros = calculate_increased_protection(flopros_rivers, 100)

# Calculate how much this additional protection will cost (using Boulange et al 2023 method)
print('Calculate cost of additional protection')
flopros = calculate_increased_protection_costs(flopros, 2399000) # $2.399 million per km unit cost from Boulange paper
