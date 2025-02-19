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

def save_new_raster(array, path, transform):
    new_dataset = rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=array.shape[1],
        width=array.shape[2],
        count=4,
        dtype=array.dtype,
        crs=rasterio.crs.CRS.from_epsg(25832),
        transform=transform,
    )
    new_dataset.write(array)
    new_dataset.close()

"""
Data loader should return the following rasters in a consistent grid along with some colormappings for them:
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
        self.AERIAL_PATH          = site_path / "rasters" / "aerial_1m_clipped.tif"
        self.ELEVATION_PATH       = site_path / "rasters" / "elevationmono_1m_clipped.tif"
        self.LANDUSE_PATH         = site_path / "rasters" / "landuse_1m_clipped.tif"
        self.SOILTYPE_PATH        = site_path / "rasters" / "soiltype30_1m_clipped.tif"
        self.MOISTURECONTENT_PATH = site_path / "rasters" / "moisturecontent15_1m_clipped.tif"

        # Additional mapping imformation
        self.LANDUSE_MAPPING         = site_path.parents[1] / "Dataloaders" / "Trier value maps" / "landuse_mapping.yaml"
        self.SOILTYPE_MAPPING        = site_path.parents[1] / "Dataloaders" / "Trier value maps" / "soiltype_mapping.yaml"
        self.MOISTURECONTENT_MAPPING = site_path.parents[1] / "Dataloaders" / "Trier value maps" / "moisturecontent_mapping.yaml"
        
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

        # Additional mapping imformation
        self.LANDUSE_MAPPING         = site_path.parents[1] / "Dataloaders" / "Dataforsyningen value maps" / "landuse_mapping.yaml"
        self.SOILTYPE_MAPPING        = site_path.parents[1] / "Dataloaders" / "Dataforsyningen value maps" / "soiltype_mapping.yaml"
        
    def get_aerial_raster(self):
        """ Source: https://datafordeler.dk/dataoversigt/geodanmark-ortofoto/ortofoto-foraar-wms/
        """
        if os.path.isfile(self.AERIAL_PATH):
            log.info("Aerial raster is already stored locally and will not be downloaded again")
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

            log.info(f"Downloading aerial raster from datafordeler, url: \n{url}")
            res = requests.get(url)
            if res.status_code == 200:
                img = Image.open(io.BytesIO(res.content))
                arr = np.array(img)

                # Move bands to be the first dimension
                arr = np.moveaxis(arr, 2, 0)
                # Adding alpha layer
                arr = np.concatenate((arr, 255*np.ones((1, arr.shape[1], arr.shape[2]), dtype=int)))
                
                transform = Affine(1, 0, self.coordinates[0], 0, -1, self.coordinates[3])

                save_new_raster(arr, self.AERIAL_PATH, transform)

            else:
                log.info("Data could not be downloaded from source")

        return RasterRGBA(self.AERIAL_PATH)

    def get_elevation_raster(self):
        """ Source: https://datafordeler.dk/dataoversigt/danmarks-hoejdemodel-historisk-dhm/dhm-2007-historik-wcs/
        """
        if os.path.isfile(self.ELEVATION_PATH):
            log.info("Elevation raster is already stored locally and will not be downloaded again")
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

            #TODO fix problem with downloading data taht occurs sometimes, quickfix delete created raster and try again            
            log.info(f"Downloading elevation raster from datafordeler, url: \n{url}")
            urllib.request.urlretrieve(url, self.ELEVATION_PATH)
            
        return RasterMono(self.ELEVATION_PATH)
    
    def get_landuse_raster(self):
        """ Source: https://envs.au.dk/om-instituttet-1/faglige-omraader/samfund-miljoe-og-ressourcer/arealanvendelse-og-gis/basemap/basemap03-til-download-1
        """
        
        with open(self.LANDUSE_MAPPING, encoding="utf8") as landuse_valuemap_file:
            loaded_data = yaml.load(landuse_valuemap_file, Loader=yaml.SafeLoader) 
            basemap_to_landuse_label = loaded_data["basemap_to_landuse_label"]
            landuse_label_to_color   = loaded_data["label_to_color"]
        
        if os.path.isfile(self.LANDUSE_PATH):
            log.info("Landuse raster is already stored locally and will not be downloaded again")
        else:
            log.info("Creating landuse raster from BaseMap04")

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
            arr[:, :, 0] = landuse_label_to_color["Non classified"][0]
            arr[:, :, 1] = landuse_label_to_color["Non classified"][1]
            arr[:, :, 2] = landuse_label_to_color["Non classified"][2]

            # Fill in colors for landuse
            for base_map, label in basemap_to_landuse_label.items():
                arr[data == base_map] = landuse_label_to_color[label]

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

            save_new_raster(arr, self.LANDUSE_PATH, transform)

        landuse_valuemap = {}

        for label, color in landuse_label_to_color.items():
            landuse_valuemap[f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"] = label

        return RasterRGBA(self.LANDUSE_PATH), landuse_valuemap
    
    def get_soiltype_raster(self):
        """ Source: https://www.geus.dk/produkter-ydelser-og-faciliteter/data-og-kort/danske-kort/download-jordartskort
        """
        
        with open(self.SOILTYPE_MAPPING, encoding="utf8") as soiltype_valuemap_file:
            loaded_data = yaml.load(soiltype_valuemap_file, Loader=yaml.SafeLoader) 
            jordart_to_soil_number = loaded_data["jordart_to_soil_number"]
            soil_number_to_label   = loaded_data["number_to_label"]
            soil_label_to_color    = loaded_data["label_to_color"]
        
        if os.path.isfile(self.SOILTYPE_PATH):
            log.info("Soil type raster is already stored locally and will not be downloaded again")
        else:
            log.info("Creating soil type raster from Jordartskort")
            log.info("Importing data from local file")
            
            gdf = gpd.read_file(self.jordartskort_location)
            transform = Affine(1, 0, self.coordinates[0], 0, -1, self.coordinates[3])
            
            for jordart, number in jordart_to_soil_number.items():
                gdf.loc[gdf.sediment == jordart, "sediment"] = number

            shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.sediment))
            
            log.info("Rasterizing data")
            rasterized = rasterize(shapes, out_shape=(int(self.query_params["HEIGHT"]), int(self.query_params["WIDTH"])), transform=transform, fill=0, dtype=np.int32)

            # Initiates map with color corresponding to non classified
            arr = np.zeros((rasterized.shape[0], rasterized.shape[1], 3), dtype=int)
            arr[:, :, 0] = soil_label_to_color["non classified"][0]
            arr[:, :, 1] = soil_label_to_color["non classified"][1]
            arr[:, :, 2] = soil_label_to_color["non classified"][2]

            # Fill in colors for soil type
            for number, label in soil_number_to_label.items():
                arr[rasterized == number] = soil_label_to_color[label]

            # Move bands to be the first dimension
            arr = np.moveaxis(arr, 2, 0)
            # Adding alpha layer
            arr = np.concatenate((arr, 255*np.ones((1, arr.shape[1], arr.shape[2]), dtype=int)))

            save_new_raster(arr, self.SOILTYPE_PATH, transform)

        soiltype_valuemap = {}

        for label, color in soil_label_to_color.items():
            soiltype_valuemap[f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"] = label

        return RasterRGBA(self.SOILTYPE_PATH), soiltype_valuemap
    
    def get_moisturecontent_raster(self):
        return None