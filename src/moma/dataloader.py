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
        return RasterRGBA(self.SOILTYPE_PATH)
    
    def get_moisturecontent_raster(self):
        return RasterRGBA(self.MOISTURECONTENT_PATH)


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
        self.basemap_to_label = {
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
        self.label_to_color = {
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

        self.query_params = {
                "username": "DNHZAPRXQZ",
                "password": "njp2025_AU_",
                "BBOX": f"{self.coordinates[0]},{self.coordinates[1]},{self.coordinates[2]},{self.coordinates[3]}",
                "WIDTH": f"{self.coordinates[2]-self.coordinates[0]}",
                "HEIGHT": f"{self.coordinates[3]-self.coordinates[1]}"
        }

        # Raster locations
        self.AERIAL_PATH          = site_path / ("rasters/aerial_1m_clipped" + file_subname +".tif")
        self.ELEVATION_PATH       = site_path / ("rasters/elevationmono_1m_clipped" + file_subname +".tif")
        self.LANDUSE_PATH         = site_path / ("rasters/landuse_1m_clipped" + file_subname +".tif")
        self.SOILTYPE_PATH        = site_path / ("rasters/soiltype30_1m_clipped" + file_subname +".tif")
        self.MOISTURECONTENT_PATH = site_path / ("rasters/moisturecontent15_1m_clipped" + file_subname +".tif")

    def get_aerial_raster(self):
        """ Source: https://datafordeler.dk/dataoversigt/geodanmark-ortofoto/ortofoto-foraar-wms/
        """
        if os.path.isfile(self.AERIAL_PATH):
            log.info("Data is already stored locally and will not be downloaded again")
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
            log.info("Data is already stored locally and will not be downloaded again")
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
            log.info("Data is already stored locally and will not be downloaded again")
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

            # Initiates map with non clissified color
            arr = np.zeros((data.shape[0], data.shape[1], 3), dtype=int)
            arr[:, :, 0] = self.label_to_color["Non classified"][0]
            arr[:, :, 1] = self.label_to_color["Non classified"][1]
            arr[:, :, 2] = self.label_to_color["Non classified"][2]

            # Fill in colors for landuse
            for base_map, label in self.basemap_to_label.items():
                arr[data == base_map] = self.label_to_color[label]

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

        for label, color in self.label_to_color.items():
            landuse_valuemap[f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"] = label

        return RasterRGBA(self.LANDUSE_PATH), landuse_valuemap
    
    def get_soiltype_raster(self):
        return None
    
    def get_moisturecontent_raster(self):
        return None