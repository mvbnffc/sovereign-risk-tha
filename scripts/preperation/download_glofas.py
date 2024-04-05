# This script downloads the GloFAS data from the CDS API and stores it do specified disk
# We will download daily discharge data and river accumulation dataset. 

import cdsapi
import urllib3
urllib3.disable_warnings()
import os

# Need to specify CDS API key to be able to download the GloFAS data
cds_api_key = r"C:/Users/Mark.DESKTOP-UFHIN6T/.cdsapirc"

# Where do we want to save the data
data_dir = r"D:\projects\sovereign-risk\Thailand\data\flood\dependence\glofas"
os.makedirs(data_dir, exist_ok=True)

# What years do we want to download this for?
start_year = 1979
end_year = 2024

# What is the bounding box to download the data for? (slightly larger bounding box than Thailand)
bbox = [21, 97, 5, 106] # [maxy, minx, miny, maxx]

# Step 1: Set up CDS API key
if os.path.isfile(cds_api_key):
    cdsapi_kwargs = {}
else:
    URL = 'https://cds.climate.copernicus.eu/api/v2'
    KEY = '##################################'
    cdsapi_kwargs = {
        'url': URL,
        'key': KEY,
    }

# Step 2: Download river discharge data
c = cdsapi.Client(
    **cdsapi_kwargs
    )
for year in range(start_year, end_year+1):
    download_file = f"{data_dir}/glofas_THA_{year}.grib"
    if not os.path.isfile(download_file):
        request_params = {
            'system_version': 'version_4_0',
            'hydrological_model': 'lisflood',
            'product_type': 'consolidated',
            'variable': 'river_discharge_in_the_last_24_hours',
            'hyear': [f"{year}"],
            'hmonth': ['january', 'february', 'march', 'april', 'may', 'june', 
                       'july', 'august', 'september', 'october', 'november', 'december'],
            'hday': [f"{day:02d}" for day in range(1,31)],
            'format': 'grib',
            'area': bbox, 
        }
        c.retrieve('cems-glofas-historical', request_params).download(download_file)

# Step 3: Download the upstream accumulating area
# NOTE: issue downloading a valid netcdf in current script. Workaround at the moment is using file I've previously downloaded
upstream_area_fname = f"uparea_glofas_v4_0.nc"
upstream_area_file = os.path.join(data_dir, upstream_area_fname)
# If we have not already downloaded the data, download it.
if not os.path.isfile(upstream_area_file):
    u_version=2 # file version
    upstream_data_url = (
        f"https://confluence.ecmwf.int/download/attachments/242067380/{upstream_area_file}?"
        f"version{u_version}&modificationDate=1668604690076&api=v2&download=true"
    )
    import requests
    result = requests.get(upstream_data_url)
    with open(upstream_area_file, 'wb') as f:
        f.write(result.content)
