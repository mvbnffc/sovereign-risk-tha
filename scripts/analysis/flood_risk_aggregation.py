# Aggregate flood risk from the analysis basin scale to the national scale using Monte Carlo simulation.
# Here we will also incorporate information on river basin flood spatial dependence.

import pandas as pd
import numpy as np
from model_dependence import model_dependence
from analysis_functions import monte_carlo_dependence_simulation, urban_monte_carlo_dependence_simulation

### Step 1: Load risk data and prepare for analysis
print('Step 1: Load risk data and prepare for analysis.')
risk_data_file = r"D:\projects\sovereign-risk\Thailand\analysis\flood\risk_analysis\risk_basin_zonal_sum.csv"
risk_data = pd.read_csv(risk_data_file)
# Add columne for annual exceedance probability
risk_data['AEP'] = 1 / risk_data['RP']
# Add a column converting current prorection level into AEP
risk_data['Pr_L_AEP'] = np.where(risk_data['Pr_L'] == 0, 0, 1 / risk_data['Pr_L']) # using numpy where avoids zero division errors
# Add row for each combination that sums residential and non-residential damages
grouped = risk_data.groupby(['FID', 'GID_1', 'NAME', 'HB_L4', 'HB_L5', 'HB_L6', 'HB_L7', 'Pr_L', 'Pr_L_AEP', 'Add_Pr', 'New_Pr_L', 'epoch', 'adaptation_scenario', 'RP', 'AEP'], as_index=False)['damages'].sum()
grouped['urban_class'] = 'Combined'  # Add a column for urban_class with value 'total'
risk_data = pd.concat([risk_data, grouped], ignore_index=True).sort_values(by=['FID', 'GID_1', 'NAME', 'HB_L4', 'HB_L5', 'HB_L6', 'HB_L7', 'Pr_L', 'Pr_L_AEP', 'Add_Pr', 'New_Pr_L', 'epoch', 'adaptation_scenario', 'RP', 'AEP'])
risk_data.reset_index(drop=True, inplace=True)

### Step 2: Calculate dependence between basins
print('Step 2: Calculate dependence between basins.')
clayton_copula_models, ordered_basins = model_dependence()

### Step 3: Run Monte Carlo simulations to aggregate losses
print('Step 3: Running Monte Carlo simulations.')
# Set up variables for the simulations
rps = [2, 5, 10, 25, 50, 100, 200, 500, 1000] # these are the RPs we are considering
n_simulations = 500
n_years = 1000
basin = 'HB_L6' # only looking at HydroBASIN level 6 basins
epochs = ['Today', 'Future_High_Emission', 'Future_Low_Emission']
scenarios = ['Baseline', 'Dry_Proofing', 'Relocation', 'Urban_Protection_RP100']
urban_classes = ['Residential', 'Non-Residential', 'Combined']
# Prepare an empty dataframe to store results
results_df = pd.DataFrame()
# Loop through the analysis
for epoch in epochs:
    for scenario in scenarios:
        for urban_class in urban_classes:
            # Print progress
            print(f'Working on epoch: {epoch}, scenario: {scenario}, urban class: {urban_class}')
            if scenario == 'Urban_Protection_RP100':
                loss_df = urban_monte_carlo_dependence_simulation(risk_data, rps, basin, epoch, scenario, urban_class, 0.5, n_years, ordered_basins, clayton_copula_models, n_simulations)
            else:
                loss_df = monte_carlo_dependence_simulation(risk_data, rps, basin, epoch, scenario, urban_class, 0.5, n_years, ordered_basins, clayton_copula_models, n_simulations)
            losses = loss_df.values.flatten() # flatten to get a single array of losses
            results_df[f'{epoch}_{scenario}_{urban_class}'] = losses

### Step 4: Save results to CSV
print('Step 4: Saving results to CSV')
results_file = r"C:\Users\Mark.DESKTOP-UFHIN6T\Projects\sovereign-risk-THA\results\%s_risk_%s_years.csv" % (basin, (n_simulations*n_years))
results_df.to_csv(results_file, index=True)


