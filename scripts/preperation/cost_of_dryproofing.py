# Script to calculate the cost of dry-proofing buidlings. Approach follows Mortensen et al (2023) https://journals.open.tudelft.nl/jcrfr/article/view/6983/5771
# Buildings suitable for dry-proofing are those in 1-in-2 year flood zone with depth < 1 m and all RP inundated areas not excluded by the previous delineation.
# Houses are dry-proofed up to 1m. Costs are $1,300 m^2 for high/upper-middle income countries and $580 m^2 for lower-middle / low-income countries.

import os
import rasterio
import numpy as np
from preperation_functions import calculate_buidlings_for_dry_proofing, load_raster

# Set dataset paths
residential_raster_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\GHSL_res_THA.tif"
non_residential_raster_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\GHSL_nres_THA.tif"
flood_2yr_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h2glob.tif"
flood_1000yr_raster_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h1000glob.tif"
residential_dry_proofing_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\dry_proofing\residential_buildings_dry_proofing.tif"
non_residential_dry_proofing_path = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\dry_proofing\non_residential_buildings_dry_proofing.tif"

# What is dry proofing cost? Using USD 1300 m^2 as Thailand is upper-middle income country
dry_proof_cost = 1300

# Load the raster datasets
residential_raster, meta = load_raster(residential_raster_path, save_info=True)
non_residential_raster = load_raster(non_residential_raster_path)
flood_2yr_raster = load_raster(flood_2yr_raster_path)
flood_1000yr_raster = load_raster(flood_1000yr_raster_path)


# Calculate buildings for dry proofing
residential_dryproofing = calculate_buidlings_for_dry_proofing(residential_raster, flood_2yr_raster, flood_1000yr_raster)
non_residential_dryproofing = calculate_buidlings_for_dry_proofing(non_residential_raster, flood_2yr_raster, flood_1000yr_raster)


# Calculate cost of dry proofing
cost_residential_dryproofing = residential_dryproofing.astype('float64')*dry_proof_cost
cost_non_residential_dryproofing = non_residential_dryproofing.astype('float64')*dry_proof_cost

# Save binary raster of buidlings for dry proofing
binary_residential_dryproofing = np.where(residential_dryproofing>0, 1, 0)
binary_non_residential_dryproofing = np.where(non_residential_dryproofing>0, 1, 0)

print('Writing binary dry proofing rasters')
with rasterio.open(residential_dry_proofing_path, 'w', **meta) as res, rasterio.open(non_residential_dry_proofing_path, 'w', **meta) as nres:
    res.write(binary_residential_dryproofing, 1)
    nres.write(binary_non_residential_dryproofing, 1)

print('Total Cost of Dry Proofing: $US', np.sum(cost_residential_dryproofing) + np.sum(cost_non_residential_dryproofing))
print('Residential: $US', np.sum(cost_residential_dryproofing))
print('Non-Residential: $US', np.sum(cost_non_residential_dryproofing))

    
