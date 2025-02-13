import logging
from abc import ABC
from tkinter import font

import colorama
import cv2  # Used for the application of convolution kernels
import matplotlib as mpl  # Used for color maps definition
import matplotlib.pyplot as plt  # Used for generating raster plots
import numpy as np  # Used for raster data array operations
import rasterio  # Used for raster data import
import shapely.geometry as shpgeo  # Used for no-go zones definition
from matplotlib.colors import \
    ListedColormap  # Used for the definition of discrete color maps
from rasterio.enums import Resampling  # Used for rasters resampling operations

import jax.numpy as jnp
from jax import jit, vmap

log = logging.getLogger("MOMA")


class Raster(ABC):
    """Raster
    """

    COLORMAPS = ["RdYlBu"]

    colormap = "RdYlBu"

    RESAMPLING_METHODS = {
        "average": Resampling.average,
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest
    }

    def set_colormap(self, key):
        """Set raster colormap

        Args:
            key (string): Colormap name. Possible values are "RdYlBu". Defaults to "RdYlBu".
        """

        self.colormap = getattr(mpl.cm, self.COLORMAPS.index(
            key)) if key in self.COLORMAPS else self.COLORMAP_DEFAULT


class RasterMono(Raster):
    """Single-band raster
    """

    def __init__(self, path):

        log.info(f"Initialising new single-channel raster from file {path}.")

        # Set raster path
        self.path = path

        # Create RasterIO raster handle
        raster = rasterio.open(path, "r")

        # Get raster data and data type
        self.data = raster.read(1)
        self.dtype = raster.dtypes

        # Get raster dimensions
        self.width = raster.width
        self.height = raster.height

        # Get raster resolution
        self.resolution_x = raster.transform[0]
        self.resolution_y = -raster.transform[4]

        # Get raster boundaries
        self.x_min, self.y_min = raster.transform * (0, 0)
        self.x_max, self.y_max = raster.transform * (raster.width,
                                                     -raster.height)

        # Get coordinate reference system
        self.crs = raster.crs
        self.transform = raster.transform

        # Close Rasterio raster handle
        raster.close()

    def plot(self, show=True, export=False):
        """Show raster
        """

        # Open new figure
        (figure, axes) = plt.subplots(1, 1)

        # Set and invert colormap
        cmap = mpl.cm.get_cmap(self.colormap + "_r")

        # Normalise colormap range. Defaults to a range of three standard deviations around the mean value.
        norm = mpl.colors.Normalize(
            vmin=np.mean(self.data) - 1.25 * np.std(self.data),
            vmax=np.mean(self.data) + 1.25 * np.std(self.data))

        # Draw raster data
        image = axes.imshow(
            self.data,
            extent=[self.x_min, self.x_max, self.y_min, self.y_max],
            cmap=cmap,
            norm=norm)

        # Disable scientific notation for ticks
        axes.ticklabel_format(style="plain")

        # Add axes labels
        axes.set_xlabel("Longitude")
        axes.set_ylabel("Latitude")

        # Add color_bar
        color_bar = figure.colorbar(image, ax=axes)
        color_bar.ax.get_yaxis().labelpad = 25
        color_bar.ax.set_ylabel("Raster value", rotation=270)
        color_bar.minorticks_on()

        if export:
            plt.savefig(export / "Raster.pdf", format="pdf")

        if show:
            plt.show()

        if export or show:
            plt.close()
        else:
            return (figure, axes, color_bar, image)

    def resample(self, factor, resampling="average"):
        """Resample raster

        Args:
            factor (float): Scaling factor
            resampling (str, optional): Resampling algorithm. Valid options are "average", "bilinear", "cubic", "lanczos" or "nearest". Defaults to "average".
        """

        # Compute new raster size and shape
        self.height = int(self.height * factor)
        self.width = int(self.width * factor)
        shape = (self.height, self.width)

        #TODO Consider using "with open"
        # Open RasterIO raster handle
        raster = rasterio.open(self.path, "r")

        # Resample original raster data
        self.data = raster.read(
            1,
            out_shape=shape,
            resampling=self.RESAMPLING_METHODS.get(resampling))

        # Update transform
        self.transform = self.transform * raster.transform.scale(
            (self.width / shape[-1]), (self.height / shape[-2]))

        # Get new raster resolution
        self.resolution_x = self.transform[0]
        self.resolution_y = -self.transform[4]

        # Close RasterIO raster handle
        raster.close()

    def add_cylinder(self, coordinate_x: float, coordinate_y: float,
                     radius: float, height: float):
        """Add cylindrical feature

        Args:
            coordinate_x (float): Cylinder center x coordinate
            coordinate_y (float): Cylinder center y coordinate
            radius (float): Cylinder radius
            height (float): Cylinder height
        """

        circle = shpgeo.Point(coordinate_x, coordinate_y).buffer(radius)

        iterator = np.nditer(self.data, flags=["multi_index"])
        for _ in iterator:
            i, j = iterator.multi_index
            coordinates = self.coordinates(i, j)
            if circle.contains(shpgeo.Point(coordinates)):
                self.data[i, j] = height

    def coordinates(self, i, j):
        """Retrieve raster coordinates for index pair (i, j).

        Args:
            i (int): X coordinate index
            j (int): Y coordinate index

        Returns:
            double: Raster value at cell (i, j)
        """

        return self.transform * (i, j)

    def save(self, path):
        """Save raster to disk
        """

        # Open new raster handle
        raster = rasterio.open(path,
                               "w",
                               driver="GTiff",
                               width=self.width,
                               height=self.height,
                               count=1,
                               dtype=self.dtype,
                               crs=self.crs,
                               transform=self.transform)

        # Export raster and close handle
        raster.write(self.data)
        raster.close()


class RasterRGBA(Raster):
    """4-band raster
    """

    def __init__(self, path):

        log.info(f"Initialising new four-channel raster from file {path}.")

        # Set raster path
        self.path = path

        # Create RasterIO raster handle
        raster = rasterio.open(path, "r")

        # Get raster data and data type
        self.data = [None] * 4
        self.data[0] = raster.read(1)
        self.data[1] = raster.read(2)
        self.data[2] = raster.read(3)
        self.data[3] = raster.read(4)
        self.dtype = raster.dtypes

        # Get raster dimensions
        self.width = raster.width
        self.height = raster.height

        # Get raster resolution
        self.resolution_x = raster.transform[0]
        self.resolution_y = -raster.transform[4]

        # Get coordinate reference system
        self.crs = raster.crs
        self.transform = raster.transform

        # Close RasterIO raster handle
        raster.close()

    def resample(self, factor, resampling="average"):
        """Resample raster

        Args:
            factor (float): Scaling factor
            resampling (str, optional): Resampling algorithm. Valid options are "average", "bilinear", "cubic", "lanczos" or "nearest". Defaults to "average".
        """

        log.info(
            f"Resampling raster by a factor {factor} using resampling algorithm {resampling}."
        )

        # Compute new raster size and shape
        height = int(self.height * factor)
        width = int(self.width * factor)
        shape = (height, width)

        # Update raster information
        self.height = height
        self.width = width

        #TODO Consider using "with open"
        # Create RasterIO raster handle
        raster = rasterio.open(self.path, "r")

        #TODO Consider converting this code block to a list comprehension.
        # Resample RGBA data
        self.data[0] = raster.read(
            1,
            out_shape=shape,
            resampling=self.RESAMPLING_METHODS.get(resampling))

        self.data[1] = raster.read(
            2,
            out_shape=shape,
            resampling=self.RESAMPLING_METHODS.get(resampling))

        self.data[2] = raster.read(
            3,
            out_shape=shape,
            resampling=self.RESAMPLING_METHODS.get(resampling))

        self.data[3] = raster.read(
            4,
            out_shape=shape,
            resampling=self.RESAMPLING_METHODS.get(resampling))

        # Update transform
        self.transform = self.transform * raster.transform.scale(
            (width / shape[-1]), (height / shape[-2]))

        # Close RasterIO raster handle
        raster.close()

    def get_indices(self, lon, lat):

        #TODO Consider using "with open"
        # Create RasterIO raster handle
        raster = rasterio.open(self.path, "r")

        (py, px) = raster.index(lon, lat)
        
        # Close RasterIO raster handle
        raster.close()
        
        return (py, -px)

    def save(self, path):
        """Save RGBA raster to disk
        """

        # Open new raster handle
        raster = rasterio.open(path,
                               "w",
                               driver="GTiff",
                               width=self.width,
                               height=self.height,
                               count=4,
                               dtype=self.dtype,
                               crs=self.crs,
                               transform=self.transform)

        # Export raster and close handle
        raster.write(self.data)
        raster.close()


class Map(ABC):
    """Map
    """

    pass


class AerialMap(Map):
    """Aerial map
    """

    def __init__(self, name, raster):

        log.info(f"Initialising new aerial map for site {name}.")

        self.name = name
        self.raster = raster

    def plot(self, show=False, export=False):

        # Open new figure
        (figure, axes) = plt.subplots(num=self.name)

        rgb = np.dstack(
            (self.raster.data[0], self.raster.data[1], self.raster.data[2]))
        # Set nodata values
        rgb[rgb == 0] = 255

        # Draw raster data
        image = axes.imshow(rgb)
        #plt.axis("off")
        #axes.xaxis.set_visible(False)
        #axes.yaxis.set_visible(False)

        axes.grid("both")

        # Set horizontal and vertical labels
        plt.tight_layout()

        if export:
            plt.savefig(export / "Aerial view.pdf", format="pdf")

        if show:
            plt.show()

        if export or show:
            plt.close()
        else:
            return (figure, axes)


class ElevationMap(Map):
    """Elevation map
    """

    def __init__(self, name, raster):

        log.info(f"Initialising new elevation map for site {name}.")

        self.name = name
        self.raster = raster

    def plot(self, show=False, export=False):

        log.info(f"Plotting elevation map.")

        (figure, axes, color_bar, _) = self.raster.plot(show=False,
                                                        export=False)

        color_bar.ax.set_ylabel("Elevation (m)", rotation=270)

        if export:
            log.info(f"Saving elevation map to disk under {export}.")
            plt.savefig(export / "Elevation map.pdf", format="pdf")

        if show:
            plt.show()

        if export or show:
            plt.close()
        else:
            return (figure, axes)


class DirectionalMap(Map):

    directional_rasters = {}

    DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

    DIRECTIONAL_KERNELS = {
        "N": np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]]),
        "S": np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]),
        "E": np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]),
        "W": np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]]),
        "NW": np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]),
        "SE": np.array([[-1, 0, 0], [0, 0, 0], [0, 0, 1]]),
        "NE": np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]),
        "SW": np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]])
    }

    def plot(self):

        for key, value in self.directional_rasters.items():

            value.plot(f"Direction {key}\n{value.path}")


class SlopeMap(DirectionalMap):

    def __init__(self, name, map_elevation):

        log.info(f"Initialising slope map from elevation data.")

        self.name = name
        self.raster = map_elevation.raster

        # Compute directional slopes
        for key, _ in self.DIRECTIONAL_KERNELS.items():
            self.directional_rasters[key] = RasterMono(self.raster.path)
            self.directional_rasters[key].data = self.directional_slope(key)

    def directional_slope(self, direction="N"):

        log.info(f"Computing slope map in direction {direction}.")

        # Retrieve kernel for the selected direction
        kernel = self.DIRECTIONAL_KERNELS[direction]

        raster = cv2.filter2D(self.raster.data, -1, kernel)

        raster = raster / (2 * self.raster.resolution_x)

        raster = np.rad2deg(np.arctan(raster))

        log.info(
            f"Slope max in direction {direction}: {colorama.Fore.BLUE}Mean {np.mean(raster):0.2g} deg{colorama.Fore.RESET}, Std. deviation: {np.std(raster):0.2g} deg."
        )

        return raster

    def plot(self, show=False, export=False):

        for direction, raster in self.directional_rasters.items():

            # Open new figure
            (figure, axes, color_bar, image) = raster.plot(show=False,
                                                           export=False)

            # Add color_bar
            image.set_clim(-20, 20)
            color_bar.ax.set_ylabel("Slope (deg)", rotation=270)

            if export:
                plt.savefig(export / f"Slope map ({direction}).pdf",
                            format="pdf")

            if show:
                plt.show()

            plt.close()


class ParameterMap(Map):

    pass


class LandUseMap(ParameterMap):
    """Land use map coded through an RGB raster mapped to ECDS classification codes. Used to delineate no-go zones and hard surfaces on the map.
    """

    # More information on land use codes at the following link:
    # https://www.sedris.org/sdk_4.1/src/lib/edcs/docs/dictionary/ECC_C.htm
    LANDUSE_CODES = {
        "AL020": {
            "edcs_classification_label": "BUILT_UP_REGION",
            "edcs_classification_symbolic_constant": "ECC_BUILT_UP_REGION",
            "edcs_classification_code": 177,
            "definition":
            "A <TRACT> containing a concentration of <BUILDING>s and other <STRUCTURE>s; a built-up region.",
            "edcs_group_labels": ["SHELTER"],
            "reference_type": "IR",
            "reference": "S57A, Object: BUAARE"
        },
        "EA010": {
            "edcs_classification_label":
            "CROP_LAND",
            "edcs_classification_symbolic_constant":
            "ECC_CROP_LAND",
            "edcs_classification_code":
            295,
            "definition":
            "A <TRACT> that has been tilled for the planting of crops.",
            "edcs_group_labels":
            ["AGRICULTURE", "LIVING_ORGANISM", "PLANT", "USAGE_REGION"],
            "reference_type":
            "IR",
            "reference":
            "FACC, Annex A: EA010"
        },
        "EC030": {
            "edcs_classification_label": "TREED_TRACT",
            "edcs_classification_symbolic_constant": "ECC_TREED_TRACT",
            "edcs_classification_code": 1282,
            "definition":
            "A tract covered by a non empty set of trees; a treed tract. Examples: copse, thicket, stand of trees.",
            "edcs_group_labels": ["LIVING_ORGANISM", "PHYSIOGRAPHY", "PLANT"],
            "reference_type": "IR",
            "reference": "FACC, Annex A: EC030"
        }
    }

    def __init__(self, name, raster, value_map, nogo):
        """Instantiate a land use map through a given raster and color mapping.

        Args:
            raster (aumap.RasterRGBA): RGBA land use raster
            value_map (dict): Dictionary of {Hex color}: {EDCS code} mappings
        """

        log.info(f"Initialising new land use map for site {name}.")

        self.name = name
        self.raster = raster
        self.value_map = value_map

        log.info(f"Provided value map: {value_map}")
        log.info(f"Provided nogo map: {nogo}")

        # Map colors from RGB integer values to corresponding map quantity (e.g. "255: "EC030")
        self.col_dict = {hex_to_rgb(k): v for k, v in self.value_map.items()}
        
        self.rgb = jnp.dstack(
            (self.raster.data[0], self.raster.data[1], self.raster.data[2]))

        # List of landuse RGB that is nogo
        self.rgb_nogo = np.zeros(256**3)

        for rgb, landuse in self.col_dict.items():
            if nogo[landuse]:
                self.rgb_nogo[rgb] = 1 
        self.rgb_nogo = jnp.array(self.rgb_nogo)

        @jit
        def is_nogo(index_x, index_y):
            rgb_value = self.rgb[index_x, index_y, :].astype(int)
            combined_rgb = rgb_value[0] * 256 * 256 + rgb_value[1] * 256 + rgb_value[2]
            return self.rgb_nogo[combined_rgb]

        self.vmap_is_nogo = vmap(is_nogo)

        index_x, index_y = jnp.unravel_index(np.arange(0, self.raster.data[0].size), self.raster.data[0].shape)
        self.nogo = self.vmap_is_nogo(index_x, index_y)
        self.nogo = np.array(self.nogo).reshape(self.raster.data[0].shape)

    def plot(self, show=False, export=False):

        # Generate a colormap based on available colors
        discrete_colormap = ListedColormap(
            [k for k, _ in self.value_map.items()])

        # Define a description for each colour category - order matters!
        labels = [str(value) for value in self.col_dict.values()]

        # Generate normalization bins
        normalization_bins = np.sort([*self.col_dict.keys()]) + 0.5
        normalization_bins = np.insert(normalization_bins, 0,
                                       np.min(normalization_bins) - 1.0)

        # Set up normalizer and formatter
        normalization = mpl.colors.BoundaryNorm(normalization_bins,
                                                len(labels),
                                                clip=True)
        colorbar_formatter = mpl.ticker.FuncFormatter(
            lambda x, pos: labels[normalization(x)])

        # Plot our figure
        figure, axes = plt.subplots()

        image = axes.imshow(self.rgb, cmap=discrete_colormap, norm=normalization)

        # Set spacing between color_bar entries
        spacing = normalization_bins[1:] - normalization_bins[:-1]
        ticks = normalization_bins[:-1] + spacing / 2

        # Add color_bar
        figure.colorbar(image, format=colorbar_formatter, ticks=ticks)

        if export:
            plt.savefig(export / "Land use.pdf", format="pdf")

        if show:
            plt.show()

        plt.close()


class SoilTypeMap(ParameterMap):
    """Soil type map
    """

    SOIL_LIST = {
        "GW": "Well-graded gravel, fine to coarse gravel",
        "GP": "Poorly graded gravel",
        "GM": "Silty gravel",
        "GC": "Clayey gravel",
        "SW": "Well-graded sand, fine to coarse sand",
        "SP": "Poorly graded sand",
        "SM": "Silty sand",
        "SC": "Clayed sand",
        "ML": "Silt",
        "CL": "Lean clay",
        "OL": "Organic silt, organic clay",
        "MH": "Elastic silt",
        "CH": "Fat clay",
        "OH": "Organic clay, organic silt",
        "PT": "Peat"
    }

    def __init__(self, name, raster, value_map):

        log.info(f"Initialising new soil type map for site {name}")

        self.name = name
        self.raster = raster
        self.value_map = value_map

        log.info(f"Registered soil types:\n\t{value_map}")

        self.make_rgb_int_map()
        self.make_soil_array()

    def make_rgb_int_map(self):

        self.rgb_map = {}
        for key, value in self.value_map.items():
            self.rgb_map[hex_to_rgb(key)] = value

    def make_soil_array(self):

        log.info("Converting soil type raster from integers to soil labels.")

        self.soilarray = np.zeros(self.raster.data[0].shape, dtype=object)

        for i in range(0, self.raster.data[0].size):
            index = np.unravel_index(i, self.raster.data[0].shape)

            myvalue = self.raster.data[0][index] * 256 * 256 + self.raster.data[
                1][index] * 256 + self.raster.data[2][index]

            self.soilarray[index] = self.rgb_map[myvalue]

    def plot(self, show=False, export=False):

        # Map colors from RGB integer values to corresponding map quantity (e.g. "255: "EC030")
        col_dict = {hex_to_rgb(k): v for k, v in self.value_map.items()}

        # Generate a colormap based on available colors
        discrete_colormap = ListedColormap(
            [k for k, _ in self.value_map.items()])

        # Define a description for each colour category - order matters!
        labels = [str(value) for value in col_dict.values()]

        # Generate normalization bins
        normalization_bins = np.sort([*col_dict.keys()]) + 0.5
        normalization_bins = np.insert(normalization_bins, 0,
                                       np.min(normalization_bins) - 1.0)

        # Set up normalizer and formatter
        normalization = mpl.colors.BoundaryNorm(normalization_bins,
                                                len(labels),
                                                clip=True)
        colorbar_formatter = mpl.ticker.FuncFormatter(
            lambda x, pos: labels[normalization(x)])

        # Plot our figure
        figure, axes = plt.subplots()
        rgb = np.dstack(
            (self.raster.data[0], self.raster.data[1], self.raster.data[2]))
        image = axes.imshow(rgb, cmap=discrete_colormap, norm=normalization)

        # Set spacing between color_bar entries
        spacing = normalization_bins[1:] - normalization_bins[:-1]
        ticks = normalization_bins[:-1] + spacing / 2

        # Add color_bar
        figure.colorbar(image, format=colorbar_formatter, ticks=ticks)

        if export:
            plt.savefig(export / f"Soil map.pdf", format="pdf")

        if show:
            plt.show()

        plt.close()


class MoistureContentMap(ParameterMap):
    """Moisture content map
    """

    def __init__(self, name, raster, value_map):

        log.info(f"Initialising new moisture content map for site {name}.")

        self.name = name
        self.raster = raster
        self.value_map = value_map
        self.make_rgb_int_map()
        self.make_moisture_array()

        log.info(f"Registered moisture content values:\n\t{value_map}")

    def make_rgb_int_map(self):

        self.rgb_map = {}
        for key, value in self.value_map.items():
            self.rgb_map[hex_to_rgb(key)] = value

    def make_moisture_array(self):

        self.moisturearray = np.zeros(self.raster.data[0].shape, dtype=object)

        for i in range(0, self.raster.data[0].size):
            index = np.unravel_index(i, self.raster.data[0].shape)

            myvalue = self.raster.data[0][index] * 256 * 256 + self.raster.data[
                1][index] * 256 + self.raster.data[2][index]

            self.moisturearray[index] = self.rgb_map[myvalue]

    def plot(self, show=False, export=False):

        # Map colors from RGB integer values to corresponding map quantity (e.g. "255: "EC030")
        col_dict = {hex_to_rgb(k): v for k, v in self.value_map.items()}

        # Generate a colormap based on available colors
        discrete_colormap = ListedColormap(
            [k for k, _ in self.value_map.items()])

        # Define a description for each colour category - order matters!
        labels = [str(value) for value in col_dict.values()]

        # Generate normalization bins
        normalization_bins = np.sort([*col_dict.keys()]) + 0.5
        normalization_bins = np.insert(normalization_bins, 0,
                                       np.min(normalization_bins) - 1.0)

        # Set up normalizer and formatter
        normalization = mpl.colors.BoundaryNorm(normalization_bins,
                                                len(labels),
                                                clip=True)
        colorbar_formatter = mpl.ticker.FuncFormatter(
            lambda x, pos: labels[normalization(x)])

        # Plot our figure
        figure, axes = plt.subplots()
        rgb = np.dstack(
            (self.raster.data[0], self.raster.data[1], self.raster.data[2]))
        image = axes.imshow(rgb, cmap=discrete_colormap, norm=normalization)

        # Set spacing between color_bar entries
        spacing = normalization_bins[1:] - normalization_bins[:-1]
        ticks = normalization_bins[:-1] + spacing / 2

        # Add color_bar
        figure.colorbar(image, format=colorbar_formatter, ticks=ticks)

        if export:
            plt.savefig(export / f"Moisture map.pdf", format="pdf")

        if show:
            plt.show()

        plt.close()


class NoGoMap(Map):
    """No-go map
    """

    def __init__(self, name, vehicle, aerial, landuse):

        log.info(f"Initialising no-go zones map site {name}, vehicle {vehicle}.")

        self.vehicle = vehicle
        self.aerial = aerial
        self.data = np.zeros(
            aerial.raster.data[0].shape
        )  # We are using the aerial red channel to deduct shape
        self.landuse = landuse
        self.zones = []

    def add_zone(self, x_min, x_max, y_min, y_max, description=None):

        log.info(
            f"Adding zone \"{description}\" with bounding box [{x_min}, {x_max}, {y_min}, {y_max}]"
        )

        if description is None:
            description = f"Zone {len(self.zones) + 1}"

        self.zones.append({})
        self.zones[-1]["description"] = description
        self.zones[-1]["vertices"] = (x_min, x_max, y_min, y_max)

        self.data[y_min:y_max, x_min:x_max] = 1

        log.info(f"Total number of zones updated to: {len(self.zones)}")

    def plot(self, show=False, export=False):
        """Plot no-go zones
        """

        # Open new figure
        (figure, axes) = plt.subplots(1, 1)

        rgb = np.dstack((self.aerial.raster.data[0], self.aerial.raster.data[1],
                         self.aerial.raster.data[2]))
        # Set nodata values
        rgb[rgb == 0] = 255

        # Draw raster data
        image = axes.imshow(rgb)
        #plt.axis("off")
        #axes.xaxis.set_visible(False)
        #axes.yaxis.set_visible(False)

        axes.grid("both")

        # Set horizontal and vertical labels
        plt.tight_layout()

        # Set style for the no-go zone labels
        label_style = dict(boxstyle="round", facecolor="indianred", alpha=0.9)
        font_style = dict(fontsize="x-small", color="white")

        # Create custom colormap
        colormap1 = mpl.colors.ListedColormap(["none", "darkorange"])
        colormap2 = mpl.colors.ListedColormap(["none", "crimson"])

        # Add no-go from land use
        axes.imshow(self.landuse.nogo, vmin=0, vmax=1, cmap=colormap1, alpha=0.7)

        # Plot no-go zones
        axes.imshow(self.data, vmin=0, vmax=1, cmap=colormap2, alpha=0.7)

        # Add no-go zone labels
        for entry in self.zones:
            # Get no-go zone label position
            x_min, x_max, y_min, y_max = entry["vertices"]
            x = (x_max + x_min) / 2
            y = (y_max + y_min) / 2
            offset = 10
            axes.text(x,
                      y_max + offset,
                      entry["description"],
                      bbox=label_style,
                      fontdict=font_style,
                      ha="center",
                      va="center")

        if export:
            plt.savefig(export / f"{self.vehicle} No-go map.pdf", format="pdf")

        if show:
            plt.show()

        plt.close()


class MapEditor:

    def __init__(self):

        pass

    @staticmethod
    def convert_geohornet(raster_rgba, value_null=0):
        """
        GeoHornet 32-bit RGBA raster to 8-bit single-channel raster

        Convert the 32-bit RGBA elevation raster from the GeoHornet server source to an 8-bit single-channel raster suitable for processing in the MOMA.

        Args:
            raster_rgba (rasterio.raster): 32-bit RGBA elevation raster
            value_null (int, optional): Value to assign to cells without data. Defaults to 0.

        Returns:
            numpy.array: 8-bit single-channel elevation raster
        """

        # Assemble elevation information
        # - Alpha channel (0: no data, 255: data available)
        # - Red channel (bits 17 to 24)
        # - Green channel (bits 9 to 16)
        # - Blue channel (bits 1 to 8)
        h_alpha = raster_rgba[3].astype("float")
        h_red = raster_rgba[0].astype("float") * 255 * 255
        h_green = raster_rgba[1].astype("float") * 255
        h_blue = raster_rgba[2].astype("float")
        h_final = h_red + h_green + h_blue

        # Return elevation value only where available
        h_mono = np.where(h_alpha == 255, h_final, value_null)

        # Convert the elevation values from mm to m
        h_mono = h_mono * 1e-3

        return h_mono


def hex_to_rgb(hex_color):
    """Convert hexadecimal color to RGB integer

    Args:
        hex_color (string): Hexadecimal color value, including "#"

    Returns:
        int: RGB integer
    """

    # Convert hex value to RGB triplet
    rgb_triplet = mpl.colors.hex2color(hex_color)

    # Map RGB triplet from [0, 1] to [0, 255]
    rgb_triplet = tuple([int(255 * x) for x in rgb_triplet])

    # Convert RGB triplet to an integer
    rgb_integer = rgb_triplet[0] * 256 * 256 + rgb_triplet[
        1] * 256 + rgb_triplet[2]

    return rgb_integer
