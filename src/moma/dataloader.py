import urllib.request
from collections import namedtuple
from urllib.parse import urlencode, urlunparse
import os.path

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

    def get_aerial_raster(self):
        return RasterRGBA(self.AERIAL_PATH)

    def get_elevation_raster(self):
        return RasterMono(self.ELEVATION_PATH)
    
    def get_landuse_raster(self):
        return RasterRGBA(self.LANDUSE_PATH)
    
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
        if os.path.isfile(self.AERIAL_PATH):
            log.info("Data is already stored locally and will not be downloaded again")
        else:
            local_query_params = dict(self.query_params)
            local_query_params.update(
                {
                    "SERVICE": "WMS",
                    "REQUEST": "GetMap",
                    "VERSION": "1.0.0",
                    "STYLE": "",
                    "FORMAT": "image/png",
                    "CRS": "EPSG:25832",
                    "DPI": "96",
                    "MAP_RESOLUTION": "96",
                    "FORMAT_OPTIONS": "dpi:96",
                    "Layers": "orto_foraar"
                }
            )
            url = create_url_with_parameters("services.datafordeler.dk/GeoDanmarkOrto/orto_foraar/1.0.0/WMS", local_query_params)
            
            log.info(f"Downloading data from dataforder, url: \n{url}")
            urllib.request.urlretrieve(url, self.ELEVATION_PATH)

        return RasterRGBA(self.AERIAL_PATH)

    def get_elevation_raster(self):
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
            
            log.info(f"Downloading data from dataforder, url: \n{url}")
            urllib.request.urlretrieve(url, self.ELEVATION_PATH)
            
        return RasterRGBA(self.ELEVATION_PATH)
    
    def get_landuse_raster(self):
        return None
    
    def get_soiltype_raster(self):
        return None
    
    def get_moisturecontent_raster(self):
        return None
    