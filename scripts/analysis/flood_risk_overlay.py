# This script does the overlay analysis of flood maps and exposure. Applying a damage function.
# Flood risk results are summed over a specified vector geometry, results are then stored in a dataframe for further analysis. 
import pandas as pd
from analysis_functions import simple_risk_overlay, flopros_risk_overlay, dryproofing_risk_overlay, relocation_risk_overlay
import os

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
                                          1000: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_pc_h1000glob.tif"},
                               'future_l':{2: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp26_h2glob.tif",
                                          5: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp26_h5glob.tif",
                                          10: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp26_h10glob.tif",
                                          25: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp26_h25glob.tif",
                                          50: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp26_h50glob.tif",
                                          100: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp26_h100glob.tif",
                                          200: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp26_h200glob.tif",
                                          500: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp26_h500glob.tif",
                                          1000: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp26_h1000glob.tif"},
                               'future_h':{2: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp85_h2glob.tif",
                                          5: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp85_h5glob.tif",
                                          10: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp85_h10glob.tif",
                                          25: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp85_h25glob.tif",
                                          50: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp85_h50glob.tif",
                                          100: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp85_h100glob.tif",
                                          200: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp85_h200glob.tif",
                                          500: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp85_h500glob.tif",
                                          1000: r"D:\projects\sovereign-risk\Thailand\data\flood\maps\THA_global_rcp85_h1000glob.tif"}}

# Depth adjustment (maps are in cm... need to convert to m) 
depth_adjuster = 100

# Exposure
res_exposure_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\GHSL_res_val_THA.tif"
nres_exposure_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\GHSL_nres_val_THA.tif"

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
# Calculate average of commercial and industrial functions to use as "non-residential" function
v_nres = [(x + y) / 2 for x, y in zip(v_com, v_ind)]
v_nres_dp = [(x + y) / 2 for x, y in zip(v_com_dp, v_ind_dp)]

# Masks
flopros_100_mask = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\flopros\urban_mask.tif"
res_dryproofing_mask = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\dry_proofing\residential_buildings_dry_proofing.tif"
nres_dryproofing_mask = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\dry_proofing\non_residential_buildings_dry_proofing.tif"
res_relocation_mask = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\res_reconstruction_2yr_1m.tif"
nres_relocation_mask = r"D:\projects\sovereign-risk\Thailand\data\flood\adaptation\relocation\nres_reconstruction_2yr_1m.tif"


#### STEP 1: Baseline Damages ####
## Begin by calculating damages under baseline adaptation scenario (no adaptation)
baseline_results_dir = r"D:\projects\sovereign-risk\Thailand\analysis\flood\risk_maps\baseline"
if not os.path.exists(baseline_results_dir):
    os.makedirs(baseline_results_dir)
    print('Baseline directory does not exist...creating')

print('Working on Baseline Risk Maps')

for epoch in flood_map_nested_dictionary:
    for RP in flood_map_nested_dictionary[epoch]:
        res_output_path = os.path.join(baseline_results_dir, '%s_%s_res_damages.tif') % (epoch, RP)
        nres_output_path = os.path.join(baseline_results_dir, '%s_%s_nres_damages.tif') % (epoch, RP)
        # Skip if files already exist
        if os.path.exists(res_output_path) and os.path.exists(nres_output_path):
            print('Dataset already exists: skipping...', res_output_path, nres_output_path) 
            continue
        else:
            flood_path = flood_map_nested_dictionary[epoch][RP]
            simple_risk_overlay(flood_path, res_exposure_path, res_output_path, [v_heights, v_res], depth_adjuster)
            simple_risk_overlay(flood_path, nres_exposure_path, nres_output_path, [v_heights, v_nres], depth_adjuster)

#### STEP 2: FLOPROS Urban Damages Masked
## Calculate damages as previously but mask urban damages below a given RP threshold.
flopros_urban_results_dir = r"D:\projects\sovereign-risk\Thailand\analysis\flood\risk_maps\infrastructure\urban_rp100"
if not os.path.exists(flopros_urban_results_dir):
    os.makedirs(flopros_urban_results_dir)
    print('FLOPROS Urban damage directory does not exist...creating')

print('Working on FLOPROS Risk Maps')

# Set RP threshold with which to apply mask
RP_threshold = 100
for epoch in flood_map_nested_dictionary:
    for RP in flood_map_nested_dictionary[epoch]:
        res_output_path = os.path.join(flopros_urban_results_dir, '%s_%s_res_damages.tif') % (epoch, RP)
        nres_output_path = os.path.join(flopros_urban_results_dir, '%s_%s_nres_damages.tif') % (epoch, RP)
        # Skip if files already exist
        if os.path.exists(res_output_path) and os.path.exists(nres_output_path):
            print('Dataset already exists: skipping...', res_output_path, nres_output_path) 
            continue
        else:
            flood_path = flood_map_nested_dictionary[epoch][RP]
            # Only mask the urban areas if it falls below the RP threshold. 
            if RP <= RP_threshold:
                flopros_risk_overlay(flood_path, res_exposure_path, res_output_path, flopros_100_mask, [v_heights, v_res], depth_adjuster)
                flopros_risk_overlay(flood_path, nres_exposure_path, nres_output_path, flopros_100_mask, [v_heights, v_nres], depth_adjuster)
            else:
                simple_risk_overlay(flood_path, res_exposure_path, res_output_path, [v_heights, v_res], depth_adjuster)
                simple_risk_overlay(flood_path, nres_exposure_path, nres_output_path, [v_heights, v_nres], depth_adjuster)

#### STEP 3: Dry Proofing Damages Masked
## Calculate damages with dry proofing applied. This will adjust the vulnerability function of all exposure within the mask to be resilient up to flood depths of 1 m
dry_proofing_results_dir = r"D:\projects\sovereign-risk\Thailand\analysis\flood\risk_maps\dry_proofing"

if not os.path.exists(dry_proofing_results_dir):
    os.makedirs(dry_proofing_results_dir)
    print('Dry proofing directory does not exist...creating')

print('Working on Dry-Proofing Risk Maps')

for epoch in flood_map_nested_dictionary:
    for RP in flood_map_nested_dictionary[epoch]:
        res_output_path = os.path.join(dry_proofing_results_dir, '%s_%s_res_damages.tif') % (epoch, RP)
        nres_output_path = os.path.join(dry_proofing_results_dir, '%s_%s_nres_damages.tif') % (epoch, RP)
        # Skip if files already exist
        if os.path.exists(res_output_path) and os.path.exists(nres_output_path):
            print('Dataset already exists: skipping...', res_output_path, nres_output_path) 
            continue
        else:
            flood_path = flood_map_nested_dictionary[epoch][RP]
            dryproofing_risk_overlay(flood_path, res_exposure_path, res_output_path, res_dryproofing_mask, [v_heights, v_res], [v_dp_heights, v_res_dp], depth_adjuster)
            dryproofing_risk_overlay(flood_path, nres_exposure_path, nres_output_path, nres_dryproofing_mask, [v_heights, v_nres], [v_dp_heights, v_nres_dp], depth_adjuster)
        
#### STEP 4: Relocation of Assets
## Calculate the updated damage if we assume that all assets in the 1in2 year flood zone w/ depth > 1m are relocated
relocation_results_dir = r"D:\projects\sovereign-risk\Thailand\analysis\flood\risk_maps\relocation"

if not os.path.exists(relocation_results_dir):
    os.makedirs(relocation_results_dir)
    print('Relocation directory does not exist...creating')

print('Working on Relocation Risk Maps')

for epoch in flood_map_nested_dictionary:
    for RP in flood_map_nested_dictionary[epoch]:
        res_output_path = os.path.join(relocation_results_dir, '%s_%s_res_damages.tif') % (epoch, RP)
        nres_output_path = os.path.join(relocation_results_dir, '%s_%s_nres_damages.tif') % (epoch, RP)
        # Skip if files already exist
        if os.path.exists(res_output_path) and os.path.exists(nres_output_path):
            print('Dataset already exists: skipping...', res_output_path, nres_output_path) 
            continue
        else:
            flood_path = flood_map_nested_dictionary[epoch][RP]
            relocation_risk_overlay(flood_path, res_exposure_path, res_output_path, res_relocation_mask, [v_heights, v_res], depth_adjuster)
            relocation_risk_overlay(flood_path, nres_exposure_path, nres_output_path, nres_relocation_mask, [v_heights, v_nres], depth_adjuster)



