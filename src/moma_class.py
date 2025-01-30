"""class collecting the diffent moma objects and adding generlized functionallyty to them
"""

import logging
import os
import pathlib

import numpy as np
import rasterio as rio
import yaml

from coloredlogs import ColoredFormatter

from coloredlogs import ColoredFormatter

from moma.mapping import (AerialMap, ElevationMap, LandUseMap,
                          MoistureContentMap, NoGoMap, RasterMono, RasterRGBA,
                          SlopeMap, SoilTypeMap)
from moma.mobility import (DirectionalMobilityMap, MobilityAnalysis,
                           MobilityMap)
from moma.planning import find_shortest_path, make_speed_graph

class moma():
    def __init__(self, path):
        """
        path to the site folder with the following structure:
        site_name/
            database/
                mobility.yaml
            mappings/
                landuse_mapping.yaml
                landuse_nogo.yaml
                soiltype_mapping.yaml
                moisturecontent_mapping.yaml
            rasters/
                aerial_1m_clipped.tif
                elevationmono_1m_clipped.tif
                landuse_1m_clipped.tif
                soiltype30_1m_clipped.tif
                moisturecontent15_1m_clipped.tif
        """

        base_path = pathlib.Path(path)
        self.base_path = base_path
        site_name = path.split("/")[-1]
        self.site_name = site_name

        self.RESAMPLING_FACTOR = 0.2
        self.RESAMPLING_ALGORITHM = "bilinear"
        
        self.DATABASE_PATH = base_path / "database/mobility.yaml"

        self.LANDUSE_MAPPING         = base_path / "mappings/landuse_mapping.yaml"
        self.LANDUSE_NOGO            = base_path / "mappings/landuse_nogo.yaml"
        self.SOILTYPE_MAPPING        = base_path / "mappings/soiltype_mapping.yaml"
        self.MOISTURECONTENT_MAPPING = base_path / "mappings/moisturecontent_mapping.yaml"

        #TODO Switch to automatic soil types detection
        self.SOIL_TYPES = ["SC", "GW", "CL"]
        
        # Raster locations
        self.AERIAL_PATH          = base_path / "rasters/aerial_1m_clipped.tif"
        self.ELEVATION_PATH       = base_path / "rasters/elevationmono_1m_clipped.tif"
        self.LANDUSE_PATH         = base_path / "rasters/landuse_1m_clipped.tif"
        self.SOILTYPE_PATH        = base_path / "rasters/soiltype30_1m_clipped.tif"
        self.MOISTURECONTENT_PATH = base_path / "rasters/moisturecontent15_1m_clipped.tif"
        self.EXPORT_PATH          = base_path / "export/"

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
    def aerial_map(self, show=False):
        # Import and process the aerial map raster
        aerial_raster = RasterRGBA(self.AERIAL_PATH)

        # Resample the aerial raster by the specified factor
        aerial_raster.resample(self.RESAMPLING_FACTOR, resampling="average")

        # Saves the aerial raster to the class for later use
        self.aerial_raster = aerial_raster

        # Instantiate an aerial map object
        aerial_map = AerialMap(self.site_name, aerial_raster)

        # Show the aerial map (and save plot to disk)
        aerial_map.plot(show=show, export=self.EXPORT_PATH)

    # ===============
    #  ELEVATION MAP
    # ===============
    def elevation_map(self):
        # Import and process the elevation map raster
        elevation_raster = RasterMono(ELEVATION_PATH)
    
    # ========================
    #  DIRECTIONAL SLOPE MAPS
    # ========================
    def directional_slope_maps(self):
        pass

    # ==============
    #  LAND USE MAP
    # ==============
    def land_use_map(self):
        pass
    
    # ===============
    #  SOIL TYPE MAP
    # ===============
    def soil_type_map(self):
        pass
    
    # ======================
    #  MOISTURE CONTENT MAP
    # ======================
    def moisture_content_map(self):
        pass

    # ===========
    #  NO-GO MAP
    # ===========
    def no_go_map(self):
        pass
    
    # ===========================
    #  DIRECTIONAL MOBILITY MAPS
    # ===========================
    def directional_mobility_maps(self):
        pass

    # ==============
    #  PATH PLANNER
    # ==============
    def path_planner(self, waypoint_start, waypoint_end):
        # Flattens the start and end waypoints
        self.waypoint_start = np.ravel_multi_index(waypoint_start,
                                      self.aerial_raster.data[0].shape)
        self.waypoint_end = np.ravel_multi_index(waypoint_end, self.aerial_raster.data[0].shape)
