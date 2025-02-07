from moma.mapping import RasterRGBA, RasterMono

# Factory generator that initializes the correct dataloader based on the data source
def get_dataloader(site_path, data_source, coordinates):
    if data_source == "Trier_raster":
        return DataloaderTrier(site_path)
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
    