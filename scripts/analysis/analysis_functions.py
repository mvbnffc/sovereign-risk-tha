# Script with common analysis functions that will be used throughout.

import numpy as np
import rasterio

def vectorized_damage(depth, value, heights, damage_percents):
    '''
    Vectorized damage function
    Apply damage function given a flood depth and exposure value.
    Function also needs as input the damage function heights > damage_percents
    '''
    # Use np.interp for vectorized linear interpolation
    damage_percentage = np.interp(depth, heights, damage_percents)
    return damage_percentage * value

def calculate_risk(flood, building_values, heights, damage_percents):
    '''
    Calculate flood risk given a flood map and a map of building values
    '''
    exposure = np.where(flood>0, building_values, 0)
    risk = vectorized_damage(flood, exposure, heights, damage_percents)

    return risk

def simple_risk_overlay(flood_path, exposure_path, output_path, damage_function):
    '''
    This function performs a simple risk overlay analysis.
    It takes as input a flood map, an exposure map, and a vulnerability curve.
    It outputs a risk raster
    '''
    # Load the rasters
    flood = rasterio.open(flood_path)
    exposure = rasterio.open(exposure_path)

    # Data info
    profile = flood.meta.copy()
    profile.update(dtype=rasterio.float32, compress='lzw', nodata=0)
    nodata = flood.nodata

    with rasterio.open(output_path, 'w', **profile) as dst:
        i = 0
        for ji, window in flood.block_windows(1):
            i += 1

            affine = rasterio.windows.transform(window, flood.transform)
            height, width = rasterio.windows.shape(window)
            bbox = rasterio.windows.bounds(window, flood.transform)

            profile.update({
                'height': height,
                'width': width,
                'affine': affine
            })

            flood_array = flood.read(1, window=window)
            exposure_array = exposure.read(1, window=window)
            flood_array = np.where(flood_array>0, flood_array, 0) # remove negative values
            flood_array = flood_array/100 # convert to m
            risk = calculate_risk(flood_array, exposure_array, damage_function[0], damage_function[1]) # depths index 0 and prp damage index 1

            dst.write(risk.astype(rasterio.float32), window=window, indexes=1)

def flopros_risk_overlay(flood_path, exposure_path, output_path, mask_path, damage_function):
    '''
    This function performs a risk overlay analysis. Before the risk analysis it masks all urban areas in the exposure dataset
    It takes as input a flood map, an exposure map, an urban area mask map, and a vulnerability curve.
    It outputs a risk raster
    '''
    # Load the rasters
    flood = rasterio.open(flood_path)
    exposure = rasterio.open(exposure_path)
    mask = rasterio.open(mask_path)

    # Data info
    profile = flood.meta.copy()
    profile.update(dtype=rasterio.float32, compress='lzw', nodata=0)
    nodata = flood.nodata

    with rasterio.open(output_path, 'w', **profile) as dst:
        i = 0
        for ji, window in flood.block_windows(1):
            i += 1

            affine = rasterio.windows.transform(window, flood.transform)
            height, width = rasterio.windows.shape(window)
            bbox = rasterio.windows.bounds(window, flood.transform)

            profile.update({
                'height': height,
                'width': width,
                'affine': affine
            })

            flood_array = flood.read(1, window=window)
            exposure_array = exposure.read(1, window=window)
            mask_array = mask.read(1, window=window)
            exposure_array = np.where(mask_array==1, 0, exposure_array) # wherever the urban mask equals 1, set to zero in exposure dataset
            flood_array = np.where(flood_array>0, flood_array, 0) # remove negative values
            flood_array = flood_array/100 # convert to m
            risk = calculate_risk(flood_array, exposure_array, damage_function[0], damage_function[1]) # depths index 0 and prp damage index 1

            dst.write(risk.astype(rasterio.float32), window=window, indexes=1)

def dryproofing_risk_overlay(flood_path, exposure_path, output_path, mask_path, damage_function, damage_function_dp):
    '''
    This function performs a risk overlay analysis. For masked dryproofed cells, the damage function is adjusted to return no damages for depths < 1m
    It takes as input a flood map, an exposure map, an urban area mask map, and two vulnerability curves.
    It outputs a risk raster
    '''
    # Load the rasters
    flood = rasterio.open(flood_path)
    exposure = rasterio.open(exposure_path)
    mask = rasterio.open(mask_path)

    # Data info
    profile = flood.meta.copy()
    profile.update(dtype=rasterio.float32, compress='lzw', nodata=0)
    nodata = flood.nodata

    with rasterio.open(output_path, 'w', **profile) as dst:
        i = 0
        for ji, window in flood.block_windows(1):
            i += 1

            affine = rasterio.windows.transform(window, flood.transform)
            height, width = rasterio.windows.shape(window)
            bbox = rasterio.windows.bounds(window, flood.transform)

            profile.update({
                'height': height,
                'width': width,
                'affine': affine
            })

            flood_array = flood.read(1, window=window)
            exposure_array = exposure.read(1, window=window)
            mask_array = mask.read(1, window=window)
            dry_proof_array = np.where(mask_array==1, exposure_array, 0) # cells for dry proofing are marked as 1
            non_dry_proof_array = np.where(mask_array==1, 0, exposure_array)
            flood_array = np.where(flood_array>0, flood_array, 0) # remove negative values
            flood_array = flood_array/100 # convert to m
            dry_proof_risk = calculate_risk(flood_array, dry_proof_array, damage_function_dp[0], damage_function_dp[1]) # pass new damage function for dry proof cells
            non_dry_proof_risk = calculate_risk(flood_array, non_dry_proof_array, damage_function[0], damage_function[1]) # normal damage function
            # Sum risk arrays
            risk = dry_proof_risk + non_dry_proof_risk

            dst.write(risk.astype(rasterio.float32), window=window, indexes=1)

def relocation_risk_overlay(flood_path, exposure_path, output_path, mask_path, damage_function):
    '''
    This function performs a risk overlay analysis. Urban cells in the 2 year flood plain exposed to flood depths > 1 m are removed from the analysis.
    It takes as input a flood map, an exposure map, an urban cell (2yr flood > 1 m depth) map, and a vulnerability curves.
    It outputs a risk raster
    '''
    # Load the rasters
    flood = rasterio.open(flood_path)
    exposure = rasterio.open(exposure_path)
    mask = rasterio.open(mask_path)

    # Data info
    profile = flood.meta.copy()
    profile.update(dtype=rasterio.float32, compress='lzw', nodata=0)
    nodata = flood.nodata

    with rasterio.open(output_path, 'w', **profile) as dst:
        i = 0
        for ji, window in flood.block_windows(1):
            i += 1

            affine = rasterio.windows.transform(window, flood.transform)
            height, width = rasterio.windows.shape(window)
            bbox = rasterio.windows.bounds(window, flood.transform)

            profile.update({
                'height': height,
                'width': width,
                'affine': affine
            })

            flood_array = flood.read(1, window=window)
            exposure_array = exposure.read(1, window=window)
            mask_array = mask.read(1, window=window)
            exposure_array = np.where(mask_array>0, 0, exposure_array) # Remove cells that will be relocated
            flood_array = np.where(flood_array>0, flood_array, 0) # remove negative values
            flood_array = flood_array/100 # convert to m
            risk = calculate_risk(flood_array, exposure_array, damage_function[0], damage_function[1])
            
            dst.write(risk.astype(rasterio.float32), window=window, indexes=1)