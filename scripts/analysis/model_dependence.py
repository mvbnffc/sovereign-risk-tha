# This script calculates the dependence between basins using copulas following the approach in https://onlinelibrary.wiley.com/doi/epdf/10.1111/risa.12382?saml_referrer
# The river flow data for the dependence calculation is from GloFAS
# See jupyter notebook for iterative walk-through...

# Import Functions
import os
import xarray as xr
import pandas as pd
from analysis_functions import combine_glofas, perform_outlet_data_checks, extract_discharge_timeseries, fit_gumbel_distribution, calculate_uniform_marginals, calculate_basin_copula_pairs, minimax_ordering

def model_dependence():

    # Set relevant directories and filepaths
    glofas_dir = r"D:\projects\sovereign-risk\Thailand\data\flood\dependence\glofas" # where is the GloFAS data stored?
    basin_outlet_file = r"D:\projects\sovereign-risk\Thailand\data\flood\dependence\thailand-basins\lev06_outlets_final_clipped_Thailand_no_duplicates.csv" # CSV file with basin outlet points produced manually

    # Step 1: Load GloFAS river discharge data and upstream accumulating area data
    # Discharge data for producing GIRI maps is from 1979-2016
    start_year = 1979
    end_year = 2016
    area_filter = 500 # not considering rivers with upstream areas below 500 km^2
    glofas_data = combine_glofas(start_year, end_year, glofas_dir, area_filter)

    # Step 2: Load the basin outlet file, perform some data checks (to ensure we have valid discharge timeseries at each basin outlet point), and then extract discharge timeseries for each basin
    basin_outlets = pd.read_csv(basin_outlet_file)
    # Note to align the two datasets we need to make the following adjustment to lat lons (based on previous trial and error)
    basin_outlets['Latitude'] = basin_outlets['Latitude'] + 0.05/2
    basin_outlets['Longitude'] = basin_outlets['Longitude'] - 0.05/2
    # Perform data check
    outlet_data_check = perform_outlet_data_checks(basin_outlets, glofas_data)
    # Extract discharge timeseries
    basin_timeseries = extract_discharge_timeseries(basin_outlets, glofas_data)

    # Step 3: Fit extreme value distribution (Gumbel) to the discharge timeseries
    gumbel_params, fit_quality = fit_gumbel_distribution(basin_timeseries)

    # Step 4: Transform data to Uniform Marginals
    uniform_marginals = calculate_uniform_marginals(basin_timeseries, gumbel_params)

    # Step 5: Calculate pairwise basin dependence using the inverse Clayton copula
    clayton_copula_models, clayton_error_basins, dependence_matrix = calculate_basin_copula_pairs(uniform_marginals, plot_dependence_matrix=False)

    # Step 6: Follow the minimax structuring approach (described in the Timonina et al (2015) paper)
    basin_ids = list(uniform_marginals.keys()) # take the basin IDs from the uniform marginals dictionary
    ordered_basins = minimax_ordering(dependence_matrix, basin_ids)

    return clayton_copula_models, ordered_basins