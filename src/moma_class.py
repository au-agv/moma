"""class collecting the diffent moma objects and adding generlized functionallyty to them
"""

import logging
import os

import numpy as np
import rasterio as rio
import yaml

from coloredlogs import ColoredFormatter

from coloredlogs import ColoredFormatter

from moma.mapping import (AerialMap, ElevationMap, LandUseMap,
                          MoistureContentMap, NoGoMap,
                          SlopeMap, SoilTypeMap)
from moma.mobility import (DirectionalMobilityMap, MobilityAnalysis,
                           MobilityMap)
from moma.planning import find_shortest_path, make_speed_graph

from moma.dataloader import get_dataloader

class MOMA():
    def __init__(self, site_path, vehicle_path, site_name, vehicle_name):
        # Initialise logger
        LOG_LEVEL = logging.INFO
        LOGFORMAT = "  %(levelname)s %(message)s"

        formatter = ColoredFormatter(LOGFORMAT)
        stream = logging.StreamHandler()
        stream.setLevel(LOG_LEVEL)
        stream.setFormatter(formatter)
        self.log = logging.getLogger("MOMA")
        self.log.setLevel(LOG_LEVEL)
        self.log.addHandler(stream)
        os.system("clear")

        base_path = site_path / site_name
        self.base_path = base_path
        self.site_name = site_name

        # Database for vehicle mobility and landuse_nogo
        self.vehicle_path = vehicle_path
        self.vehicle = vehicle_name

        self.log.info(f"Initialising MOMA for \n\tsite: {self.site_name} \n\tvehicle: {self.vehicle}")

        with open(base_path / "site.yaml") as site_yaml:
            site_yaml = yaml.load(site_yaml, Loader=yaml.SafeLoader)
        self.datasource  = site_yaml["DataSource"]
        self.coordinates = site_yaml["Coordinates"]
        self.nogo_zones  = site_yaml["NoGoZones"]
        
        self.dataloader = get_dataloader(base_path, self.datasource, self.coordinates)
        self.log.info(f"Data source: {self.dataloader.source}")

        self.RESAMPLING_FACTOR = 0.2
        self.RESAMPLING_ALGORITHM = "bilinear"

        #TODO Switch to automatic soil types detection
        self.SOIL_TYPES = ["SC", "GW", "CL"]
        
        # Additional mapping imformation
        self.SOILTYPE_MAPPING        = base_path / "mappings/soiltype_mapping.yaml"
        self.MOISTURECONTENT_MAPPING = base_path / "mappings/moisturecontent_mapping.yaml"

        self.EXPORT_PATH          = base_path / "export/"

    # ============
    #  AERIAL MAP
    # ============
    def generate_aerial_map(self, show=False):
        # Import and process the aerial map raster
        self.aerial_raster = self.dataloader.get_aerial_raster()

        # Resample the aerial raster by the specified factor
        self.aerial_raster.resample(self.RESAMPLING_FACTOR, resampling="average")

        # Instantiate an aerial map object
        self.aerial_map = AerialMap(self.site_name, self.aerial_raster)

        # Show the aerial map (and save plot to disk)
        self.aerial_map.plot(show=show, export=self.EXPORT_PATH)

    # ===============
    #  ELEVATION MAP
    # ===============
    def generate_elevation_map(self, show=False):
        # Import and process the elevation map raster
        elevation_raster = self.dataloader.get_elevation_raster()

        # Resample the elevation raster by the specified factor
        elevation_raster.resample(self.RESAMPLING_FACTOR, resampling=self.RESAMPLING_ALGORITHM)

        # Instantiate an elevation map object
        self.elevation_map = ElevationMap(self.site_name, elevation_raster)

        # Show the elevation map
        self.elevation_map.plot(show=show, export=self.EXPORT_PATH)
    
    # ========================
    #  DIRECTIONAL SLOPE MAPS
    # ========================
    def generate_directional_slope_maps(self, show=False, pool_type=None, pool_weight=0.5):
        # TODO Switch directional slope maps generation to user-initiated
        # Instantiate a slope map object (this will automatically compute directional slope maps!)
        self.slope_map = SlopeMap(self.site_name, self.elevation_map, pool=pool_type, pool_weight=pool_weight)

        # Show the directional slope maps
        self.slope_map.plot(show=show, export=self.EXPORT_PATH)

    # ==============
    #  LAND USE MAP
    # ==============
    def generate_land_use_map(self, show=False):
        # Load land use raster and vlaue mappings
        landuse_raster, landuse_valuemap = self.dataloader.get_landuse_raster()

        # Resample the land use raster by the specified factor
        landuse_raster.resample(self.RESAMPLING_FACTOR, resampling="nearest")
    
        # Import land use no-go mappings
        #TODO Get rid of the hard-coded land use no-go mappings
        with open(self.vehicle_path / "landuse_nogo.yaml", encoding="utf8") as landuse_nogo_file:
            landuse_nogo = yaml.load(landuse_nogo_file, Loader=yaml.SafeLoader)
            landuse_nogo = landuse_nogo[self.vehicle]

        # Instantiate a land use map object
        self.landuse_map = LandUseMap(self.site_name, landuse_raster, landuse_valuemap,
                                      landuse_nogo)

        # Show the land use map
        self.landuse_map.plot(show=show, export=self.EXPORT_PATH)

    # ===============
    #  SOIL TYPE MAP
    # ===============
    def generate_soil_type_map(self, show=False):
        # Generate soil type map
        soiltype_raster = self.dataloader.get_soiltype_raster()

        # Resample the soil type raster by the specified factor
        soiltype_raster.resample(self.RESAMPLING_FACTOR, resampling="nearest")

        with open(self.SOILTYPE_MAPPING, encoding="utf8") as soiltype_mapping_file:
            soiltype_valuemap = yaml.load(soiltype_mapping_file, Loader=yaml.SafeLoader)

        # Instantiate a soil type map object
        self.soiltype_map = SoilTypeMap(self.site_name, soiltype_raster, soiltype_valuemap)

        # Show the soil type map
        self.soiltype_map.plot(show=show, export=self.EXPORT_PATH)
    
    # ======================
    #  MOISTURE CONTENT MAP
    # ======================
    def generate_moisture_content_map(self, show=False):
        # Generate moisture content map
        moisturecontent_raster = self.dataloader.get_moisturecontent_raster()

        # Resample the moisture content raster by the specified factor
        moisturecontent_raster.resample(self.RESAMPLING_FACTOR, resampling="nearest")

        with open(self.MOISTURECONTENT_MAPPING, encoding="utf8") as moisturecontent_valuemap_file:
            moisturecontent_valuemap = yaml.load(moisturecontent_valuemap_file, Loader=yaml.SafeLoader)

        self.moisturecontent_map = MoistureContentMap(self.site_name, moisturecontent_raster, moisturecontent_valuemap)

        # Show the moisture content map
        self.moisturecontent_map.plot(show=show, export=self.EXPORT_PATH)

    # ===========
    #  NO-GO MAP
    # ===========
    def generate_no_go_map(self, show=False):
        # Generate no-go map, use aerial map for visualization
        self.nogo_map = NoGoMap(self.site_name, self.vehicle, self.aerial_map, self.landuse_map)

        # Add no-go zones from a list of zones with the format: [[x1, x2, y1, y2, "description"], ...]
        if self.nogo_zones is not None:
            for zone in self.nogo_zones:
                if type(zone[-1]) == str:
                    self.nogo_map.add_zone(zone[0:-1], zone[-1])
                else:
                    self.nogo_map.add_zone(zone)
        
        # Show the no-go map
        self.nogo_map.plot(show=show, export=self.EXPORT_PATH)

    
    # ===========================
    #  DIRECTIONAL MOBILITY MAPS
    # ===========================
    def generate_directional_mobility_maps(self, directions="all", show=False):
        if directions == "all":
            directions = ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]
        # Generate mobility tensors
        soil_parameter = ["moisture_content"]
        mobility_analysis = MobilityAnalysis(self.vehicle_path / "mobility.yaml", self.vehicle, self.SOIL_TYPES, soil_parameter)
        mobility_analysis.generate_mobility_tensors("slope")

        # Generate mobility maps
        self.mobility_maps = {}
        for direction in directions:
            self.log.info(f"Generating mobility map in the {direction} direction.")
            self.mobility_maps[direction] = MobilityMap(direction, self.slope_map, self.landuse_map, 
                                                        self.nogo_map, self.moisturecontent_map, 
                                                        self.soiltype_map, mobility_analysis.mobility_tensors)

            # Show the mobility map
            self.mobility_maps[direction].plot(show=show, export=self.EXPORT_PATH)

    # ==============
    #  PATH PLANNER
    # ==============
    def path_planner(self, waypoint_start, waypoint_end, show=False):
        # Flattens the start and end waypoints
        self.waypoint_start = np.ravel_multi_index(waypoint_start,
                                      self.aerial_raster.data[0].shape)
        self.waypoint_end = np.ravel_multi_index(waypoint_end, self.aerial_raster.data[0].shape)

        # Find optimal path
        directional_mobility_map = DirectionalMobilityMap(self.mobility_maps)

        # Generate directional mobility graph
        directional_mobility_graph = make_speed_graph(
            directional_mobility_map.mobility_maps)
        
        # Find the shortest path between two waypoints
        shortpath = find_shortest_path(directional_mobility_graph, 
                                       self.waypoint_start, self.waypoint_end)
        shortpath.print_path(raster=self.elevation_map.raster)

        shortpath.smoothen_spline(self.mobility_maps["N"].data.shape)

        # Import and process the elevation map raster
        elevation_raster_back = self.dataloader.get_elevation_raster

        # Resample the elevation raster by the specified factor
        elevation_raster_back.resample(self.RESAMPLING_FACTOR, resampling=self.RESAMPLING_ALGORITHM)

        # Show optimal path on the aerial map
        shortpath.plot(overlay=elevation_raster_back, show=show, export=self.EXPORT_PATH)