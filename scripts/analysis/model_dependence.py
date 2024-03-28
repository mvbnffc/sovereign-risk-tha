# This script calculates the dependence between basins using copulas following the approach in https://onlinelibrary.wiley.com/doi/epdf/10.1111/risa.12382?saml_referrer
# The river flow data for the dependence calculation is from GloFAS
# See jupyter notebook for iterative walk-through

# Import Functions
import os
import xarray as xr
import netCDF4
import cdsapi

# Deal with warnings
# Disable warnings for data download via API
import urllib3 
urllib3.disable_warnings()
# Disable xarray runtime warnings
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

####### Step 1. Download river flow data for Thailand #######

# Set up CDS API key for downloading data

if os.path.isfile("C:/Users/Mark.DESKTOP-UFHIN6T/.cdsapirc"):
    cdsapi_kwargs = {}
else:
    URL = 'https://cds.climate.copernicus.eu/api/v2'
    KEY = '##################################'
    cdsapi_kwargs = {
        'url': URL,
        'key': KEY,
    }

# Where do we want to store the GLOFAS data? 
DATADIR = r"D:\projects\sovereign-risk\Thailand\data\flood\dependence\glofas"
os.makedirs(DATADIR, exist_ok=True)