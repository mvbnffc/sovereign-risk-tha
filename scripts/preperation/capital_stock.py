# This script is an alternative approach for calculating exposure.
# It disaggregates Thailand capital stock (in Billion Bahts) from https://www.nesdc.go.th/nesdb_en/more_news.php?cid=159&filename=index
# onto GHSL residential and non-residential area grids as well as on gridded road and rail data extracted from OSM (TODO: need to write up scripts for this - currently rough in jupyter)
# The purpose of this script is to have risk assessment outputs that can be directly input into DIGND (e.g. private vs public and tradable vs non-tradable)

import pandas as pd

# Set the dataset paths
THA_capital_stock = "D:\projects\sovereign-risk\Thailand\data\exposure\THA_net_capital_stock_2022.csv"
residential_raster_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\GHSL_res_THA.tif"
non_residential_raster_path = r"D:\projects\sovereign-risk\Thailand\data\exposure\GHSL_nres_THA.tif"
road_raster_path = r"D:\projects\sovereign-risk\Thailand\data\infrastructure\gridded\final\road_lengths_m.tif"
rail_raster_path = r"D:\projects\sovereign-risk\Thailand\data\infrastructure\gridded\final\rail_lengths_m.tif"

df = pd.read_csv(THA_capital_stock)

print(df)