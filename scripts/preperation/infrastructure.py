# This script prepares the infrastructure layer for the risk assessment. It aggregates the flood maps to the resolution of the critical infrastructure
# map (90m -> 11km) conserving the flood volume in each grid cell. The script also disaggregates infrastructure values onto the critical infrastructure map
# weighting the distribution of infrastructure values by the CISI of each cell (CISI dataset -> https://www.nature.com/articles/s41597-022-01218-4)

