import geopandas as gpd
from shapely.geometry import LineString
from pyproj import Geod
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask

def map_flopros_to_adm(map_df, flopros, adm):
    '''
    This function maps the flopros map and protection values to the admin1 vector. Doing this because we will use the admin1 vector in further analysis and
    it's borders differ slightly to the FLOPROS layer
    '''
    # Merging mapping dataframe with admin file (to ensure we have ID<>ID mapping info)
    merged_df = adm.merge(map_df, how='left', left_on='GID_1', right_on='GID_1') # these are the basin IDs in Admin file
    # We only want to bring the merged protection layer column from the FLOPROS dataset
    new_adm = merged_df.merge(flopros[['OBJECTID', 'MerL_Riv']], how='left', left_on='OBJECTID', right_on='OBJECTID') # these are the basin IDs in FLOPROS file

    return new_adm

def calculate_geod_length(line):
    '''
    Function to caluclate the geodetic length of a LineString
    '''
    geod = Geod(ellps="WGS84") # our data is in WGS84 projection
    length = 0
    if isinstance(line, LineString):
        for i in range(len(line.coords)-1):
            lon1, lat1 = line.coords[i]
            lon2, lat2 = line.coords[i + 1]
            _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
            length += distance

    return length

def calculate_river_length_per_admin(admin, rivers, threshold, urbanisation, urbanisation_threshold):
    '''
    Function that caluclates the length of river in each admin area. The size of the river network
    is determined by an upsteam drainage area threshold (in sq km). Function is written to take
    FLOPROS as input for admin and hydroRIVERS as rivers. Function inputs should be geodataframes.
    Function outputs an admin dataset with an additional column of total river length within the 
    admin area. The function also calculates the length of rivers in highly populated areas using the
    GHSL Degree of Urbanisation Dataset
    ''' 
    # Filter the river lines based on the size threshold
    filtered_rivers = rivers[rivers['UPLAND_SKM'] > threshold] # NOTE: this column name is unique to HydroRIVERS
    # Filter urban areas based on the population density threshold
    urban_areas = urbanisation[urbanisation['L2'] >= urbanisation_threshold]
    # Intersect filtered rivers with dense urban areas
    urban_rivers = gpd.overlay(filtered_rivers, urban_areas, how='intersection')
    # Intersect river lines with admin areas, splitting them as necessary
    intersected_rivers = gpd.overlay(filtered_rivers, admin, how='intersection')
    intersected_urban_rivers = gpd.overlay(urban_rivers, admin, how='intersection')
    # Calculate the geodetic length of each river segment 
    intersected_rivers['r_lng_m'] = intersected_rivers['geometry'].apply(calculate_geod_length)
    intersected_urban_rivers['u_r_lng_m'] = intersected_urban_rivers['geometry'].apply(calculate_geod_length)
    # Groub by admin area and sum segment lengths
    river_lengths_by_area = intersected_rivers.groupby('OBJECTID')['r_lng_m'].sum().reset_index() # NOTE: this column name is unique to FLOPROS dataset
    urban_river_lengths_by_area = intersected_urban_rivers.groupby('OBJECTID')['u_r_lng_m'].sum().reset_index() # NOTE: this column name is unique to FLOPROS dataset
    # Add the total length to the admin areas DataFrame
    admin = admin.merge(river_lengths_by_area, how='left', left_on='OBJECTID', right_on='OBJECTID') # NOTE: this column name is unique to FLOPROS dataset
    admin = admin.merge(urban_river_lengths_by_area, how='left', left_on='OBJECTID', right_on='OBJECTID') # NOTE: this column name is unique to FLOPROS dataset
    admin['r_lng_km'] = admin['r_lng_m'].fillna(0) / 1000
    admin['u_r_lng_km'] = admin['u_r_lng_m'].fillna(0) / 1000
    admin.drop(columns=['r_lng_m'], inplace=True)
    admin.drop(columns=['u_r_lng_m'], inplace=True)
        
    return admin

def calculate_increased_protection(admin, protection_goal):
    '''
    Function calculates how much additional protection is needed to achieve a protection goal.
    Function requires as input the target protection level and the admin dataset (FLOPROS layer).
    Function ouptus an admin dataset with an additional column with the amount of additional 
    protection needed.
    '''
    # Calculate additional protection needed
    admin['Add_Pr'] = protection_goal - admin['MerL_Riv']
    # Ensure there are no negative values
    admin['Add_Pr'] = admin['Add_Pr'].clip(lower=0)
    # Store the new protection level
    admin['New_Pr_L'] = protection_goal

    return admin

def calculate_increased_protection_costs(admin, unit_cost):
    '''
    This function calculates the cost of additional protection given a unit cost (per km).
    The function calculates the unit cost based on the length of rivers within the admin area.
    The cost is applied using the formula unit_cost * log2(additional_protection) from Boulange et al 2023.
    https://link.springer.com/article/10.1007/s11069-023-06017-7 
    The function outputs an admin dataset with an additional column with cost of increasing protection
    to desired levels. Also calculates cost for only urban rivers.
    '''
    # Calculate additional costs of protection
    admin['Add_Pr_c'] = np.log2(admin['Add_Pr']) * unit_cost * admin['r_lng_km']
    admin['Add_Pr_c_u'] = np.log2(admin['Add_Pr']) * unit_cost * admin['u_r_lng_km']

    return admin


def disaggregate_building_values(admin_areas, df, raster, occupancy_type, column_val):
    '''
    This function disaggregates building values from the GEM exposure database (https://github.com/gem/global_exposure_model)
    at Admin1 level to gridded urban maps maps from GHSL
    '''

    # Initialize an empty array for the output raster values
    output_values = np.zeros_like(raster.read(1), dtype=np.float32)

    # Ensure relevant columns are being treated as numeric
    df['TOTAL_REPL_COST_USD'] = pd.to_numeric(df['TOTAL_REPL_COST_USD'], errors='coerce')
    
    for index, admin_area in admin_areas.iterrows():
        
        # Filter the DataFrame for the current admin area and the specified occupancy type
        filtered_df = df[(df['GID_1'] == admin_area['GID_1']) & (df['OCCUPANCY'] == occupancy_type)]
        
        # Aggregate the total value for the specified occupancy type within the admin area
        total_value = filtered_df['TOTAL_REPL_COST_USD'].sum()
        
        # Proceed if there's a meaningful total value to disaggregate
        if total_value > 0:
            # Create a mask for the admin area
            geom_mask = geometry_mask([admin_area.geometry], invert=True, transform=raster.transform, out_shape=raster.shape)
            
            # Calculate the total area of the admin area covered by buildings in the raster
            total_area = raster.read(1)[geom_mask].sum()
            
            if total_area > 0:  # Prevent division by zero
                # Disaggregate the total value across the raster, weighted by building area
                output_values[geom_mask] += (raster.read(1)[geom_mask] / total_area) * total_value
    
    return output_values


def write_raster(output_path, raster_template, data):
    '''
    Function to write raster datasets
    '''
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=raster_template.height,
        width=raster_template.width,
        count=1,
        dtype=data.dtype,
        crs=raster_template.crs,
        transform=raster_template.transform,
    ) as dst:
        dst.write(data, 1)

def calculate_buidlings_for_dry_proofing(building_area, flood_2, flood_1000):
    '''
    function calculates which buildings are elgible for dry proofing. The criteria is buidlings within the 1000 year flood zone that are not exposed to
    flood depths >1 m in the 2 year flood zone. Function returns building area raster array with buildings elgible for dry proofing.
    '''
    # isolate buidlings within the 1000-year flood zone
    buildings_in_1000_year_flood = np.where((flood_1000 > 0) & (building_area > 0), 1, 0)
    # exclude buidlings in 2-year flood zone with flood depth >1 m
    buildings_for_dry_proofing = np.where((flood_2 <= 100) | (flood_2 == 0), buildings_in_1000_year_flood, 0) # NOTE: GIRI data is in cm

    return buildings_for_dry_proofing

def load_raster(raster_path, save_info=False):
    '''
    Load raster and make sure we get rid of NaNs and negatives.
    '''
    
    raster = rasterio.open(raster_path)
    raster_array = raster.read(1)
    
    # Clean array (remove negatives and nans)
    raster_array[np.isnan(raster_array)] = 0
    raster_array = np.where(raster_array > 0, raster_array, 0)

    if save_info:
        return raster_array, raster.meta
    
    else:
        return raster_array

def calculate_reconstruction_value_exposed(reconstruction_value_grid, flood, depth_threshold=0):
    '''
    function calculates the reconstruction value of buildings within the flood extent and (if specified) a flood depth threshold.
    function retrurns an array of the flood exposed reconstruction value
    '''
    # Mask extent based on depth threshold
    flood_extent = np.where(flood>depth_threshold, flood, 0)
    # Calculate exposed reconstruction value
    value_exposed = np.where(flood_extent>0, reconstruction_value_grid, 0)

    return value_exposed


def union_gdfs(gdf1, gdf2):
    '''
    Function to apply union operation on two GeoDataFrames
    '''
    # Make sure both GeoDataFrames have the same CRS
    gdf2_copy = gdf2.copy().to_crs(gdf1.crs)
    # Perform the union operation
    return gpd.overlay(gdf1, gdf2_copy, how='union')