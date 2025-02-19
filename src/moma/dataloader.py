import requests
import urllib.request
from collections import namedtuple
from urllib.parse import urlencode, urlunparse

import os.path
import yaml

import io
import rasterio
from rasterio.transform import Affine
from rasterio.transform import rowcol
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from PIL import Image

import logging

from moma.mapping import RasterRGBA, RasterMono

log = logging.getLogger("MOMA")

# Factory generator that initializes the correct dataloader based on the data source
def get_dataloader(site_path, data_source, coordinates):
    if data_source == "Trier_raster":
        return DataloaderTrier(site_path)
    elif data_source == "Dataforsyningen":
        return DataloaderDataforsyningen(site_path, coordinates)
    else:
        raise ValueError(f"Unknown data source: {data_source}")

"""
Data loader should return the following rasters in a consistent grid.:
- Aerial
- Elevation
- Landuse
- Soiltype
- Moisturecontent
"""

def create_url_with_parameters(netloc, query_params):
    Components = namedtuple(
        typename='Components', 
        field_names=['scheme', 'netloc', 'url', 'path', 'query', 'fragment']
    )
    url = urlunparse(
        Components(
            scheme='https',
            netloc=netloc,
            query=urlencode(query_params),
            path='',
            url='',
            fragment=''
        )
    )
    return url


class DataloaderTrier:
    def __init__(self, site_path):
        self.site_path = site_path
        self.source = "Dataloader for pregenerated Trier"

        # Raster locations
        self.AERIAL_PATH          = site_path / "rasters/aerial_1m_clipped.tif"
        self.ELEVATION_PATH       = site_path / "rasters/elevationmono_1m_clipped.tif"
        self.LANDUSE_PATH         = site_path / "rasters/landuse_1m_clipped.tif"
        self.SOILTYPE_PATH        = site_path / "rasters/soiltype30_1m_clipped.tif"
        self.MOISTURECONTENT_PATH = site_path / "rasters/moisturecontent15_1m_clipped.tif"

        # Additional mapping imformation
        self.LANDUSE_MAPPING         = site_path / "mappings/landuse_mapping.yaml"
        self.SOILTYPE_MAPPING        = site_path / "mappings/soiltype_mapping.yaml"
        self.MOISTURECONTENT_MAPPING = site_path / "mappings/moisturecontent_mapping.yaml"
        
    def get_aerial_raster(self):
        return RasterRGBA(self.AERIAL_PATH)

    def get_elevation_raster(self):
        return RasterMono(self.ELEVATION_PATH)
    
    def get_landuse_raster(self):
        #TODO Get rid of the hard-coded land use value mappings
        with open(self.LANDUSE_MAPPING, encoding="utf8") as landuse_valuemap_file:
            landuse_valuemap = yaml.load(landuse_valuemap_file, Loader=yaml.SafeLoader)

        return RasterRGBA(self.LANDUSE_PATH), landuse_valuemap
    
    def get_soiltype_raster(self):
        with open(self.SOILTYPE_MAPPING, encoding="utf8") as soiltype_mapping_file:
            soiltype_valuemap = yaml.load(soiltype_mapping_file, Loader=yaml.SafeLoader)
    
        return RasterRGBA(self.SOILTYPE_PATH), soiltype_valuemap
    
    def get_moisturecontent_raster(self):
        with open(self.MOISTURECONTENT_MAPPING, encoding="utf8") as moisturecontent_valuemap_file:
            moisturecontent_valuemap = yaml.load(moisturecontent_valuemap_file, Loader=yaml.SafeLoader)

        return RasterRGBA(self.MOISTURECONTENT_PATH), moisturecontent_valuemap


class DataloaderDataforsyningen:
    def __init__(self, site_path, coordinates):
        self.site_path = site_path
        self.coordinates = coordinates
        self.source = "Dataforsyningen, Datafordeleren, Geus and BaseMap04"
        file_subname = f"_{coordinates[0]}_{coordinates[1]}_{coordinates[2]}_{coordinates[3]}"
        
        if not os.path.exists(site_path / "rasters"):
            os.makedirs(site_path / "rasters")

        if not os.path.exists(site_path / "export"):
            os.makedirs(site_path / "export")

        self.basemap_location = "C:/Users/nikol/OneDrive - Aarhus universitet/Master project/Map data/Basemap04_2021/lu_agg_2021.tif"
        self.jordartskort_location = "C:/Users/nikol/OneDrive - Aarhus universitet/Master project/Map data/Jordlag/QGIS/Jordart_25000_v7.shp"
        self.basemap_to_landuse_label = {
            110000: "Building",                # Building
            121000: "Built-up area",           # Build-up area with low buildings
            121110: "Building",                # Low building
            122000: "Built-up area",           # Build-up area with high buildings
            122110: "Building",                # High building
            123000: "Built-up area",           # City center
            123110: "Building",                # Building in city center
            124000: "Built-up area",           # Other build-up area
            124110: "Building",                # Building in other build-up area
            125000: "Built-up area",           # Industry area
            125110: "Building",                # Building in indutry area

            126000: "Airport",                 # Airport area
            126110: "Building",                # Airport building

            130000: "Recreational",            # Recreational area / sport facility
            130110: "Building",                # Building in recreational area / sport facility

            141000: "Road",                    # Road, fortified
            142000: "Road",                    # Road

            150000: "Railway",                 # Railway
            150110: "Building",                # Railway building

            160000: "Mining",                  # Mining

            211000: "Farming",                 # Farming intensively, temporary crops
            212000: "Farming",                 # Farming intensively, permanent crops
            220000: "Farming",                 # Farming extensive
            230000: "Farming",                 # Farming no classified

            311000: "Forrest",                 # Forrest
            312000: "Forrest",                 # Forrest, wet

            321000: "Nature",                  # Nature, dry
            321220: "Nature",                  # Nature, dry, farming extensive
            322000: "Nature",                  # Nature, wet
            322220: "Nature",                  # Nature, wet, farming extensive

            411000: "Lake",                    # Lake
            412000: "Stream",                  # Strem
            420000: "Sea",                     # Sea

            800000: "Non classified",          # Non classified
            999999: "Germany",                 # Germany
        }
        self.landuse_label_to_color = {
            "Building":                [255,   0,   0],
            "Built-up area":           [127,   0,   0],
            "Road":                    [  0,   0,   0],
            "Railway":                 [127, 127, 127],
            "Recreational":            [100, 200, 127],
            "Mining":                  [127,  50,   0],
            "Airport":                 [191,   0,   0],
            "Farming":                 [227, 230,  70],
            "Forrest":                 [  0, 127,  20],
            "Nature":                  [127, 255, 127],
            "Lake":                    [  0, 200, 255],
            "Stream":                  [  0, 120, 255],
            "Sea":                     [  0,  40, 255],
            "Non classified":          [255, 105, 180],
            "Germany":                 [255, 106,   0],
}
        self.jordart_to_soil_number = {
            'BK - Danian bryozoan og corallian limestone'       : 1,  # limestone
            'BY - Town'                                         : 15, # town
            'DG - Meltwater gravel'                             : 2,  # gravel
            'DI - Meltwater silt'                               : 3,  # silt
            'DL - Meltwater clay'                               : 4,  # clay
            'DS - Meltwater sand'                               : 5,  # sand
            'DV - Alternating thin meltwater beds'              : 11, # mixed layer
            'ED - Eocene diatomite'                             : 12, # diatomite
            'EE - Eocene volcanic ash'                          : 13, # volcanic ash
            'EK - Aeolian dune sand'                            : 5,  # sand
            'ES - Aeolian sand'                                 : 5,  # sand
            'FG - Freshwater gravel'                            : 2,  # gravel
            'FHG - Delta gravel'                                : 2,  # gravel
            'FHL - Delta clay'                                  : 4,  # clay
            'FHS - Delta sand'                                  : 5,  # sand
            'FI - Freshwater silt'                              : 3,  # silt
            'FJ - Ocher and bog iron'                           : 14, # ore                     
            'FK - Tufa, bog-and lake marl'                      : 6,  # chalk
            'FL - Freshwater clay'                              : 4,  # clay
            'FP - Freshwater gyttja'                            : 7,  # gyttja
            'FS - Freshwater sand'                              : 5,  # sand
            'FT - Freshwater peat'                              : 8,  # peat
            'FV - Alternating thin freshwater beds'             : 11, # mixed layer
            'G - Gravel / sand and gravel'                      : 2,  # gravel
            'GC - Oligocene/Miocene/Pliocene brown coal'        : 9,  # brown coal
            'GL - Oligocene/Miocene/Pliocene mica clay'         : 4,  # clay
            'GS - Oligocene/Miocene/Pliocene mica sand'         : 5,  # sand
            'GV - Oligocæn/Miocæn/Pliocæn alternating layers'   : 11, # mixed layer
            'HG - Saltwater gravel'                             : 2,  # gravel
            'HI - Saltwater silt'                               : 3,  # silt
            'HL - Saltwater clay'                               : 4,  # clay
            'HP - Saltwater gyttja'                             : 7,  # gyttja
            'HS - Saltwater sand'                               : 5,  # sand
            'HSG - Saltwater shell gravel'                      : 2,  # gravel
            'HT - Saltwater peat'                               : 8,  # peat
            'HV - Alternating thin saltwater beds, marsh'       : 10, # marsh
            'IA - No access'                                    : 17, # non classified
            'IT - Freshwater peat'                              : 8,  # peat
            'K - Chalk and limestone'                           : 1,  # limestone
            'KMG - Limey till, gravelly'                        : 2,  # gravel
            'KML - Limey till, clayey'                          : 4,  # clay
            'KMS - Limey till, sandy'                           : 5,  # sand
            'KS - Miocene quartz sand'                          : 5,  # sand
            'LL - Eocene clay, plastic clay'                    : 4,  # clay
            'LRÅ - Abandoned pit'                               : 16, # other
            'LSL - Landslide'                                   : 16, # other
            'MG - Gravelly till'                                : 2,  # gravel
            'MI - Silty till'                                   : 3,  # silt
            'ML - Clayey till'                                  : 4,  # clay
            'MS - Sandy till'                                   : 5,  # sand
            'MV - Alternating thin till beds'                   : 11, # mixed layer
            'O - Land fill'                                     : 16, # other
            'OL - Oligocene clay'                               : 4,  # clay
            'PKV - Pre-Quaternary layers'                       : 11, # mixed layer
            'PL - Selandian clay, Paleocene clay'               : 4,  # clay
            'PS - Selandian sand, Paleocene greensand'          : 5,  # sand
            'QG - Saltwater gravel'                             : 2,  # gravel
            'QL - Saltwater clay'                               : 4,  # clay
            'QS - Saltwater sand'                               : 5,  # sand
            'RL - Eocene Røsnæs clay'                           : 4,  # clay
            'RÅ - Pit'                                          : 16, # other
            'S - Sand'                                          : 5,  # sand
            'SK - Campanian-Maastrichtian chalk'                : 6,  # chalk
            'SL - Eocene Svind marl'                            : 6,  # chalk
            'SØ - Freshwater'                                   : 16, # other
            'TA - Technical and artificial construction'        : 16, # other
            'TG - Meltwater gravel'                             : 2,  # gravel
            'TI - Meltwater silt'                               : 3,  # silt
            'TL - Meltwater clay'                               : 4,  # clay
            'TS - Meltwater sand'                               : 5,  # sand
            'TV - Alternating thin meltwater beds'              : 11, # mixed layer
            'X - Bed unkown, no infomation'                     : 16, # other
            'YG - Saltwater gravel'                             : 2,  # gravel
            'YL - Saltwater clay'                               : 4,  # clay
            'YP - Saltwater gyttja'                             : 7,  # gyttja
            'YS - Saltwater sand'                               : 5,  # sand
            'ZG - Glaciolacustrine gravel'                      : 2,  # gravel
            'ZK - Danian chalk / chalk and flint'               : 6,  # chalk
            'ZL - Glaciolacustrine clay'                        : 4,  # clay
            'ZS - Glaciolacustrine sand'                        : 5,  # sand
            'ZV - Alternating thin glaciolacustrine beds'       : 11, # mixed layer
        }
        self.soil_number_to_label = {
            1: "limestone",
            2: "gravel",
            3: "silt",
            4: "clay",
            5: "sand",
            6: "chalk",
            7: "gyttja",
            8: "peat",
            9: "brown coal",
            10: "marsh",
            11: "mixed layer",
            12: "diatomite",
            13: "volcanic ash",
            14: "ore",
            15: "town",
            16: "other",
            17: "non classified",
        }
        self.soil_label_to_color = {
            "limestone"      : [210, 200, 170],  # Beige
            "gravel"         : [160, 160, 160],  # Light medium gray
            "silt"           : [130, 150, 110],  # Light brown-gray
            "clay"           : [190, 100,  10],  # Reddish brown
            "sand"           : [245, 210, 170],  # Warm, saturated tan
            "chalk"          : [250, 240, 230],  # Whiter beige
            "gyttja"         : [120, 100,  40],  # Distinct brownish olive
            "peat"           : [ 70,  50,  30],  # Dark brown
            "brown coal"     : [160, 100,  60],  # Rich warm brown
            "marsh"          : [  0, 100,   0],  # Dark green
            "mixed layer"    : [255,   0, 255],  # Vibrant magenta
            "diatomite"      : [245, 245, 220],  # Off-white / beige
            "volcanic ash"   : [100, 100, 100],  # Cooler medium gray
            "ore"            : [100,  90,  80],  # Metallic dark gray with warm tint
            "town"           : [ 50,  50,  50],  # Very dark gray
            "other"          : [186,  85, 211],  # Vivid purple
            "non classified" : [255, 105, 180],  # Hot pink
        }

        self.query_params = {
                "username": "DNHZAPRXQZ",
                "password": "njp2025_AU_",
                "BBOX":   f"{self.coordinates[0]},{self.coordinates[1]},{self.coordinates[2]},{self.coordinates[3]}",
                "WIDTH":  f"{self.coordinates[2] - self.coordinates[0]}",
                "HEIGHT": f"{self.coordinates[3] - self.coordinates[1]}"
        }

        # Raster locations
        self.AERIAL_PATH          = site_path / ("rasters/aerial_1m_clipped" + file_subname +".tif")
        self.ELEVATION_PATH       = site_path / ("rasters/elevationmono_1m_clipped" + file_subname +".tif")
        self.LANDUSE_PATH         = site_path / ("rasters/landuse_1m_clipped" + file_subname +".tif")
        self.SOILTYPE_PATH        = site_path / ("rasters/soiltype_1m_clipped" + file_subname +".tif")
        self.MOISTURECONTENT_PATH = site_path / ("rasters/moisturecontent_1m_clipped" + file_subname +".tif")

    def get_aerial_raster(self):
        """ Source: https://datafordeler.dk/dataoversigt/geodanmark-ortofoto/ortofoto-foraar-wms/
        """
        if os.path.isfile(self.AERIAL_PATH):
            log.info("Aerial data is already stored locally and will not be downloaded again")
        else:
            local_query_params = dict(self.query_params)
            local_query_params.update(
                {
                    "SERVICE": "WMS",
                    "REQUEST": "GetMap",
                    "VERSION": "1.3.0",
                    "STYLES": "",
                    "FORMAT": "image/png",
                    "CRS": "EPSG:25832",
                    "DPI": "96",
                    "MAP_RESOLUTION": "96",
                    "FORMAT_OPTIONS": "dpi:96",
                    "Layers": "orto_foraar"
                }
            )
            url = create_url_with_parameters("services.datafordeler.dk/GeoDanmarkOrto/orto_foraar/1.0.0/WMS", local_query_params)

            log.info(f"Downloading data from datafordeler, url: \n{url}")
            res = requests.get(url)
            if res.status_code == 200:
                img = Image.open(io.BytesIO(res.content))
                arr = np.array(img)
                arr = np.moveaxis(arr, 2, 0)

                # Adding alpha layer
                arr = np.concatenate((arr, 255*np.ones((1, arr.shape[1], arr.shape[2]), dtype=int)))
                
                transform = Affine(1, 0, self.coordinates[0], 0, -1, self.coordinates[3])

                new_dataset = rasterio.open(
                    self.AERIAL_PATH,
                    'w',
                    driver='GTiff',
                    height=arr.shape[1],
                    width=arr.shape[2],
                    count=4,
                    dtype=arr.dtype,
                    crs=rasterio.crs.CRS.from_epsg(25832),
                    transform=transform,
                )
                new_dataset.write(arr)
                new_dataset.close()
            else:
                log.info("Data could not be downloaded from source")

        return RasterRGBA(self.AERIAL_PATH)

    def get_elevation_raster(self):
        """ Source: https://datafordeler.dk/dataoversigt/danmarks-hoejdemodel-historisk-dhm/dhm-2007-historik-wcs/
        """
        if os.path.isfile(self.ELEVATION_PATH):
            log.info("Elevation data is already stored locally and will not be downloaded again")
        else:
            local_query_params = dict(self.query_params)
            local_query_params.update(
                {
                    "COVERAGE": "dhm2007_terraen",
                    "SERVICE": "WCS",
                    "REQUEST": "GetCoverage",
                    "VERSION": "1.0.0",
                    "STYLE": "default",
                    "FORMAT": "GTiff",
                    "CRS": "EPSG:25832",
                    "RESPONSE_CRS": "EPSG:25832",
                }
            )
            url = create_url_with_parameters("services.datafordeler.dk/DHMterraenBro2007/dhm_historik/1.0.0/WCS", local_query_params)
            
            log.info(f"Downloading data from datafordeler, url: \n{url}")
            urllib.request.urlretrieve(url, self.ELEVATION_PATH)
            
        return RasterMono(self.ELEVATION_PATH)
    
    def get_landuse_raster(self):
        """ Source: https://envs.au.dk/om-instituttet-1/faglige-omraader/samfund-miljoe-og-ressourcer/arealanvendelse-og-gis/basemap/basemap03-til-download-1
        """
        if os.path.isfile(self.LANDUSE_PATH):
            log.info("Landuse data is already stored locally and will not be downloaded again")
        else:
            log.info("Creating data from BaseMap04")

            with rasterio.open(self.basemap_location) as raster:
                # Find area of interrest in basemap
                t = raster.transform
                input = [572955, 6229509, 576252, 6231353] # [W, S, E, N]
                y1, x1 = rowcol(raster.transform, input[0], input[3])
                y2, x2 = rowcol(raster.transform, input[2], input[1])

                # Widen area to ensure that need area is not cut off in rounding
                x1 -= 1
                x2 += 1
                y1 -= 1
                y2 += 1

                # Creates window and read data
                window = Window(x1, y1, x2 - x1, y2 - y1)
                data = raster.read(1, window=window)
                win_transform = raster.window_transform(window)

            # Initiates map with color corresponding to non classified
            arr = np.zeros((data.shape[0], data.shape[1], 3), dtype=int)
            arr[:, :, 0] = self.landuse_label_to_color["Non classified"][0]
            arr[:, :, 1] = self.landuse_label_to_color["Non classified"][1]
            arr[:, :, 2] = self.landuse_label_to_color["Non classified"][2]

            # Fill in colors for landuse
            for base_map, label in self.basemap_to_landuse_label.items():
                arr[data == base_map] = self.landuse_label_to_color[label]

            # Upscale to 1 meter grid
            arr = arr.repeat(10, 0).repeat(10, 1)
            transform = Affine(1, 0, win_transform[2], 0, -1, win_transform[5])

            # Get correct location on the upscaled grid
            y1, x1 = rowcol(transform, input[0], input[3])
            y2, x2 = rowcol(transform, input[2], input[1])
            arr = arr[y1:y2, x1+1:x2+1, :]
            transform = Affine(1, 0, input[0], 0, -1, input[3])

            # Move bands to be the first dimension
            arr = np.moveaxis(arr, 2, 0)
            # Adding alpha layer
            arr = np.concatenate((arr, 255*np.ones((1, arr.shape[1], arr.shape[2]), dtype=int)))

            # Saves the cutted dataset                
            new_dataset = rasterio.open(
                self.LANDUSE_PATH,
                'w',
                driver='GTiff',
                height=arr.shape[1],
                width=arr.shape[2],
                count=4,
                dtype=arr.dtype,
                crs=rasterio.crs.CRS.from_epsg(25832),
                transform=transform,
            )
            new_dataset.write(arr)
            new_dataset.close()
        landuse_valuemap = {}

        for label, color in self.landuse_label_to_color.items():
            landuse_valuemap[f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"] = label

        return RasterRGBA(self.LANDUSE_PATH), landuse_valuemap
    
    def get_soiltype_raster(self):
        """ Source: https://www.geus.dk/produkter-ydelser-og-faciliteter/data-og-kort/danske-kort/download-jordartskort
        """
        if os.path.isfile(self.SOILTYPE_PATH):
            log.info("Soil type data is already stored locally and will not be downloaded again")
        else:
            log.info("Creating data from Jordartskort")
            log.info("Importing data from local file")
            
            gdf = gpd.read_file(self.jordartskort_location)
            transform = Affine(1, 0, self.coordinates[0], 0, -1, self.coordinates[3])
            
            for jordart, number in self.jordart_to_soil_number.items():
                gdf.loc[gdf.sediment == jordart, "sediment"] = number

            shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.sediment))
            
            log.info("Rasterizing data")
            rasterized = rasterize(shapes, out_shape=(int(self.query_params["HEIGHT"]), int(self.query_params["WIDTH"])), transform=transform, fill=0, dtype=np.int32)

            # Initiates map with color corresponding to non classified
            arr = np.zeros((rasterized.shape[0], rasterized.shape[1], 3), dtype=int)
            arr[:, :, 0] = self.soil_label_to_color["non classified"][0]
            arr[:, :, 1] = self.soil_label_to_color["non classified"][1]
            arr[:, :, 2] = self.soil_label_to_color["non classified"][2]

            # Fill in colors for soil type
            for number, label in self.soil_number_to_label.items():
                arr[rasterized == number] = self.soil_label_to_color[label]

            # Move bands to be the first dimension
            arr = np.moveaxis(arr, 2, 0)
            # Adding alpha layer
            arr = np.concatenate((arr, 255*np.ones((1, arr.shape[1], arr.shape[2]), dtype=int)))

            # Save the raster
            new_dataset = rasterio.open(
                self.SOILTYPE_PATH,
                'w',
                driver='GTiff',
                height=arr.shape[1],
                width=arr.shape[2],
                count=4,
                dtype=rasterized.dtype,
                crs=rasterio.crs.CRS.from_epsg(25832),
                transform=transform,
            )
            new_dataset.write(arr)
            new_dataset.close()

        soiltype_valuemap = {}

        for label, color in self.soil_label_to_color.items():
            soiltype_valuemap[f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"] = label

        return RasterRGBA(self.SOILTYPE_PATH), soiltype_valuemap
    
    def get_moisturecontent_raster(self):
        return None