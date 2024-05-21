# D1 code adjusted for precalculated exposure maps

import pandas as pd
import os
from analysis_functions import simple_risk_overlay
import rasterio
import numpy as np

# Set all dataset paths
# Flood
flood_map_path = r"D:\projects\sovereign-risk\Thailand\data\flood\maps"
flood_map_nested_dictionary = {'present':{2: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h2glob.tif",
                                          5: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h5glob.tif",
                                          10: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h10glob.tif",
                                          25: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h25glob.tif",
                                          50: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h50glob.tif",
                                          100: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h100glob.tif",
                                          200: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h200glob.tif",
                                          500: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h500glob.tif",
                                          1000: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h1000glob.tif"}}


# Depth adjustment (maps are in cm... need to convert to m) 
depth_adjuster = 100

# Exposure
res_exposure_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\DIGNAD\res_cap_stock_thb.tif"
com_exposure_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\DIGNAD\com_cap_stock_thb.tif"
ind_exposure_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\DIGNAD\ind_cap_stock_thb.tif"
inf_exposure_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\DIGNAD\inf_cap_stock_thb.tif"       

# Vulnerability
jrc_depth_damage = r"C:\Users\Mark.DESKTOP-UFHIN6T\Projects\sovereign-risk-THA\data\flood\vulnerability\jrc_depth_damage.csv"
# Extract vulnerability information
vuln_df = pd.read_csv(jrc_depth_damage)
v_heights = vuln_df['flood_depth'].to_list()
v_dp_heights = vuln_df['flood_depth_dryproof'].to_list() # dryproof functions
v_res = vuln_df['asia_residential'].to_list()
v_res_sd = vuln_df['asia_residential_sd'].to_list()
v_res_dp = vuln_df['asia_residential_dryproof'].to_list()
v_res_dp_sd = vuln_df['asia_residential_dryproof_sd'].to_list()
v_com = vuln_df['asia_commercial'].to_list()
v_com_sd = vuln_df['asia_commercial_sd'].to_list()
v_com_dp = vuln_df['asia_commercial_dryproof'].to_list()
v_com_dp_sd = vuln_df['asia_commercial_dryproof_sd'].to_list()
v_ind = vuln_df['asia_industrial'].to_list()
v_ind_sd = vuln_df['asia_industrial_sd'].to_list()
v_ind_dp = vuln_df['asia_industrial_dryproof'].to_list()
v_ind_dp_sd = vuln_df['asia_industrial_dryproof_sd'].to_list()
v_inf = vuln_df['asia_infrastructure'].to_list()
v_inf_sd = vuln_df['asia_infrastructure_sd'].to_list()

# Masks
flopros_100_mask = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\flopros\urban_mask.tif"
res_dryproofing_mask = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\dry_proofing\residential_buildings_dry_proofing.tif"
nres_dryproofing_mask = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\dry_proofing\non_residential_buildings_dry_proofing.tif"
res_relocation_mask = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_2yr_1m.tif"
nres_relocation_mask = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_2yr_1m.tif"

#### STEP 1: Baseline Damages ####
## Begin by calculating damages under baseline adaptation scenario (no adaptation)
baseline_results_dir = r"D:\projects\sovereign-risk\Thailand\analysis\flood\risk_maps\dignad\baseline"
if not os.path.exists(baseline_results_dir):
    os.makedirs(baseline_results_dir)
    print('Baseline directory does not exist...creating')

print('Working on Baseline Risk Maps')

for epoch in flood_map_nested_dictionary:
    for RP in flood_map_nested_dictionary[epoch]:
        res_output_path = os.path.join(baseline_results_dir, '%s_%s_res_damages.tif') % (epoch, RP)
        com_output_path = os.path.join(baseline_results_dir, '%s_%s_com_damages.tif') % (epoch, RP)
        ind_output_path = os.path.join(baseline_results_dir, '%s_%s_ind_damages.tif') % (epoch, RP)
        inf_output_path = os.path.join(baseline_results_dir, '%s_%s_inf_damages.tif') % (epoch, RP)
        res_sd_output_path = os.path.join(baseline_results_dir, '%s_%s_res_damages_sd.tif') % (epoch, RP)
        com_sd_output_path = os.path.join(baseline_results_dir, '%s_%s_com_damages_sd.tif') % (epoch, RP)
        ind_sd_output_path = os.path.join(baseline_results_dir, '%s_%s_ind_damages_sd.tif') % (epoch, RP)
        inf_sd_output_path = os.path.join(baseline_results_dir, '%s_%s_inf_damages_sd.tif') % (epoch, RP)
        # Skip if files already exist
        if os.path.exists(res_output_path) and os.path.exists(com_output_path) and os.path.exists(ind_output_path) and os.path.exists(inf_output_path):
            print('Dataset already exists: skipping...', res_output_path, com_output_path, ind_output_path, inf_output_path) 
            continue
        else:
            flood_path = flood_map_nested_dictionary[epoch][RP]
            simple_risk_overlay(flood_path, res_exposure_path, res_output_path, [v_heights, v_res], depth_adjuster)
            simple_risk_overlay(flood_path, com_exposure_path, com_output_path, [v_heights, v_com], depth_adjuster)
            simple_risk_overlay(flood_path, ind_exposure_path, ind_output_path, [v_heights, v_ind], depth_adjuster)
            simple_risk_overlay(flood_path, inf_exposure_path, inf_output_path, [v_heights, v_inf], depth_adjuster)
            
            # TODO: calculate 5th and 95th percentile maps.