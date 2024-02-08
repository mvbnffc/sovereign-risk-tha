# Script calculates the cost of flood protection at the basin scale. Based on length of river and existing -> proposed protection levels.
import os
import geopandas as gpd
from preperation_functions import calculate_river_length_per_admin, calculate_increased_protection, calculate_increased_protection_costs

# Load the data
data_dir = r"D:\projects\sovereign-risk\Thailand\data"
flopros_database = gpd.read_file(os.path.join(data_dir, "flood/adaptation/flopros/flopros-THA.shp"))
river_data = gpd.read_file(os.path.join(data_dir, "flood/river/hydroRIVERS_v10_THA.shp"))
urbanisation_data = gpd.read_file(os.path.join(data_dir, "flood/adaptation/ghsl_du/ghsl_du_gadm_THA.gpkg"))

# Calculate river length within each basin
flopros_rivers = calculate_river_length_per_admin(flopros_database, river_data, 500, urbanisation_data, 21) # 21 is peri-urban areas or denser. See GHSL report for details

# Calculate how much additional protection is needed to reach a target protection level (e.g. 100)
flopros = calculate_increased_protection(flopros_rivers, 100)

# Calculate how much this additional protection will cost (using Boulange et al 2023 method)
flopros = calculate_increased_protection_costs(flopros, 2399000) # $2.399 million per km unit cost from Boulange paper

print(flopros)