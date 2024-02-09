import geopandas as gpd
from shapely.geometry import LineString
from pyproj import Geod
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask

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
    intersected_rivers['river_segment_length_m'] = intersected_rivers['geometry'].apply(calculate_geod_length)
    intersected_urban_rivers['urban_river_segment_length_m'] = intersected_urban_rivers['geometry'].apply(calculate_geod_length)
    # Groub by admin area and sum segment lengths
    river_lengths_by_area = intersected_rivers.groupby('OBJECTID')['river_segment_length_m'].sum().reset_index() # NOTE: this column name is unique to FLOPROS
    urban_river_lengths_by_area = intersected_urban_rivers.groupby('OBJECTID')['urban_river_segment_length_m'].sum().reset_index() # NOTE: this column name is unique to FLOPROS
    # Add the total length to the admin areas DataFrame
    admin = admin.merge(river_lengths_by_area, how='left', left_on='OBJECTID', right_on='OBJECTID') # NOTE: this column name is unique to FLOPROS
    admin = admin.merge(urban_river_lengths_by_area, how='left', left_on='OBJECTID', right_on='OBJECTID') # NOTE: this column name is unique to FLOPROS
    admin['river_length_km'] = admin['river_segment_length_m'].fillna(0) / 1000
    admin['river_length_km_urban'] = admin['urban_river_segment_length_m'].fillna(0) / 1000
    admin.drop(columns=['river_segment_length_m'], inplace=True)
    admin.drop(columns=['urban_river_segment_length_m'], inplace=True)
        
    return admin

def calculate_increased_protection(admin, protection_goal):
    '''
    Function calculates how much additional protection is needed to achieve a protection goal.
    Function requires as input the target protection level and the admin dataset (FLOPROS layer).
    Function ouptus an admin dataset with an additional column with the amount of additional 
    protection needed.
    '''
    # Calculate additional protection needed
    admin['Additional_Protection'] = protection_goal - admin['MerL_Riv']
    # Ensure there are no negative values
    admin['Additional_Protection'] = admin['Additional_Protection'].clip(lower=0)
    # Store the new protection level
    admin['New_Protection_Level'] = protection_goal

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
    admin['Additional_Protection_Costs'] = np.log2(admin['Additional_Protection']) * unit_cost * admin['river_length_km']
    admin['Additional_Protection_Costs_Urban'] = np.log2(admin['Additional_Protection']) * unit_cost * admin['river_length_km_urban']

    return admin


def disaggregate_building_values(admin_areas, df, raster, occupancy_type):
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