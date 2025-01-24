"""Example script for the Mobility Mapping Toolbox

This is an example script for the Mobility Mapping Toolbox
"""

import logging
import os
import pathlib

import numpy as np
import rasterio as rio
import yaml

from coloredlogs import ColoredFormatter

from moma.mapping import (AerialMap, ElevationMap, LandUseMap,
                          MoistureContentMap, NoGoMap, RasterMono, RasterRGBA,
                          SlopeMap, SoilTypeMap)
from moma.mobility import (DirectionalMobilityMap, MobilityAnalysis,
                           MobilityMap)
from moma.planning import find_shortest_path, make_speed_graph

base_path = pathlib.Path("./examples/wtd41")

RESAMPLING_FACTOR = 0.2
RESAMPLING_ALGORITHM = "bilinear"

DATABASE_PATH = base_path / "database/mobility.yaml"

LANDUSE_MAPPING = base_path / "mappings/wtd41_landuse_mapping.yaml"
LANDUSE_NOGO = base_path / "mappings/wtd41_landuse_nogo.yaml"
SOILTYPE_MAPPING = base_path / "mappings/wtd41_soiltype_mapping.yaml"
MOISTURECONTENT_MAPPING = base_path / "mappings/wtd41_moisturecontent_mapping.yaml"

#TODO Switch to automatic soil types detection
SOIL_TYPES = ["SC", "GW", "CL"]

# Raster locations
AERIAL_PATH = base_path / "rasters/aerial_1m_clipped.tif"
ELEVATION_PATH = base_path / "rasters/elevationmono_1m_clipped.tif"
LANDUSE_PATH = base_path / "rasters/landuse_1m_clipped.tif"
SOILTYPE_PATH = base_path / "rasters/soiltype30_1m_clipped.tif"
MOISTURECONTENT_PATH = base_path / "rasters/moisturecontent15_1m_clipped.tif"
EXPORT_PATH = base_path / "export/"
WAYPOINT_START = (400, 350)
WAYPOINT_END = (120, 50)

# Initialise logger
LOG_LEVEL = logging.INFO
LOGFORMAT = "  %(levelname)s %(message)s"

formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
log = logging.getLogger("MOMA")
log.setLevel(LOG_LEVEL)
log.addHandler(stream)
os.system("clear")

log.info("Initialising MOMA")

# ============
#  AERIAL MAP
# ============
# Import and process the aerial map raster
aerial_raster = RasterRGBA(AERIAL_PATH)

# Resample the aerial raster by the specified factor
aerial_raster.resample(RESAMPLING_FACTOR, resampling="average")

# Instantiate an aerial map object
aerial_map = AerialMap("WTD41", aerial_raster)

# Show the aerial map (and save plot to disk)
aerial_map.plot(show=False, export=EXPORT_PATH)

WAYPOINT_START = np.ravel_multi_index(WAYPOINT_START,
                                      aerial_raster.data[0].shape)
WAYPOINT_END = np.ravel_multi_index(WAYPOINT_END, aerial_raster.data[0].shape)

# ===============
#  ELEVATION MAP
# ===============
# Import and process the elevation map raster
elevation_raster = RasterMono(ELEVATION_PATH)

# Resample the elevation raster by the specified factor
elevation_raster.resample(RESAMPLING_FACTOR, resampling=RESAMPLING_ALGORITHM)

# Instantiate an elevation map object
elevation_map = ElevationMap("WTD-41", elevation_raster)

# Show the elevation map
elevation_map.plot(show=False, export=EXPORT_PATH)

# ========================
#  DIRECTIONAL SLOPE MAPS
# ========================
#TODO Switch directional slope maps generation to user-initiated
# Instantiate a slope map object (this will automatically compute directional slope maps!)
slope_map = SlopeMap("WTD-41", elevation_map)

# Show the directional slope maps
slope_map.plot(show=False, export=EXPORT_PATH)

# ==============
#  LAND USE MAP
# ==============

# Load land use raster
landuse_raster = RasterRGBA(LANDUSE_PATH)

# Resample the land use raster by the specified factor
landuse_raster.resample(RESAMPLING_FACTOR, resampling="nearest")

# Import land use values mappings
#TODO Get rid of the hard-coded land use value mappings
with open(LANDUSE_MAPPING, encoding="utf8") as landuse_valuemap_file:
    landuse_valuemap = yaml.load(landuse_valuemap_file, Loader=yaml.SafeLoader)

# Import land use no-go mappings
#TODO Get rid of the hard-coded land use no-go mappings
with open(LANDUSE_NOGO, encoding="utf8") as landuse_nogo_file:
    landuse_nogo = yaml.load(landuse_nogo_file, Loader=yaml.SafeLoader)

# Instantiate a land use map object
landuse_map = LandUseMap("WTD-41", landuse_raster, landuse_valuemap,
                         landuse_nogo)

# Show the land use map
landuse_map.plot(show=False, export=EXPORT_PATH)

# ===============
#  SOIL TYPE MAP
# ===============

# Generate soil type map
soiltype_raster = RasterRGBA(SOILTYPE_PATH)

# Resample the soil type raster by the specified factor
soiltype_raster.resample(RESAMPLING_FACTOR, resampling="nearest")

with open(SOILTYPE_MAPPING, encoding="utf8") as soiltype_mapping_file:
    soiltype_valuemap = yaml.load(soiltype_mapping_file, Loader=yaml.SafeLoader)

# Instantiate a soil type map object
soiltype_map = SoilTypeMap("WTD-41", soiltype_raster, soiltype_valuemap)

# Show the soil type map
soiltype_map.plot(show=False, export=EXPORT_PATH)

# ======================
#  MOISTURE CONTENT MAP
# ======================

# Generate moisture content map
moisturecontent_raster = RasterRGBA(MOISTURECONTENT_PATH)

# Resample the moisture content raster by the specified factor
moisturecontent_raster.resample(RESAMPLING_FACTOR, resampling="nearest")

with open(MOISTURECONTENT_MAPPING,
          encoding="utf8") as moisturecontent_valuemap_file:
    moisturecontent_valuemap = yaml.load(moisturecontent_valuemap_file,
                                         Loader=yaml.SafeLoader)

moisturecontent_map = MoistureContentMap("WTD-41", moisturecontent_raster,
                                         moisturecontent_valuemap)

# Show the moisture content map
moisturecontent_map.plot(show=False, export=EXPORT_PATH)

# ===========
#  NO-GO MAP
# ===========

# Generate no-go map, use aerial map for visualisation
nogo_map = NoGoMap("WTD-41", aerial_map)

# Add no-go zones
nogo_map.add_zone(100, 150, 310, 340, "A dangerous zone")
nogo_map.add_zone(300, 400, 380, 400, "Another dangerous zone")
nogo_map.add_zone(175, 300, 200, 250, "Yet another dangerous zone")

# Show the no-go map
nogo_map.plot(show=False, export=EXPORT_PATH)

# ===========================
#  DIRECTIONAL MOBILITY MAPS
# ===========================

# Generate mobility tensors
soil_parameters = ["moisture_content"]
mobility_analysis = MobilityAnalysis(DATABASE_PATH, "mungo", SOIL_TYPES,
                                     soil_parameters)
mobility_analysis.generate_mobility_tensors("slope")

# Generate mobility maps
mobility_maps = {}
for direction in ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]:
    log.info(f"Generating mobility map in the {direction} direction.")
    mobility_maps[direction] = MobilityMap(direction, slope_map, landuse_map,
                                           nogo_map, moisturecontent_map,
                                           soiltype_map,
                                           mobility_analysis.mobility_tensors)

    # Show the mobility map
    mobility_maps[direction].plot(show=False, export=EXPORT_PATH)

# ==============
#  PATH PLANNER
# ==============

# Find optimal path
directional_mobility_map = DirectionalMobilityMap(mobility_maps)

# Generate directional mobility graph
directional_mobility_graph = make_speed_graph(
    directional_mobility_map.mobility_maps)

# Find the shortest path between two waypoints
shortpath = find_shortest_path(directional_mobility_graph, WAYPOINT_START,
                               WAYPOINT_END)
shortpath.print_path(raster=elevation_map.raster)

#
shortpath.smoothen_spline(mobility_maps["N"].data.shape)

# Import and process the elevation map raster
elevation_raster_back = RasterMono(ELEVATION_PATH)

# Resample the elevation raster by the specified factor
elevation_raster_back.resample(RESAMPLING_FACTOR,
                               resampling=RESAMPLING_ALGORITHM)

# Show optimal path on the aerial map
shortpath.plot(overlay=elevation_raster_back, show=False, export=EXPORT_PATH)
