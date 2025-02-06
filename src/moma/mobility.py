"""Mobility module
"""

import logging
from abc import ABC

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi
import yaml

from moma.mapping import DirectionalMap

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.interpolate import RegularGridInterpolator

log = logging.getLogger("MOMA")


class MobilityAnalysis():
    """Mobility analysis object class.

    A mobility analysis object encapsulates a vehicle and its corresponding simulation results database subset, an arbitrary number of soil types and an arbitrary number of soil parameters for the soil types set. On the basis of the provided data, it allows for the generation of mobility tensors, one for each soil type, allowing for interpolation of simulated data over the n-dimensional hyperplane defined by the chosen parameters.
    """

    # List of parameters for each mobility test
    TESTS_PARAMETERS_LIST = {"slope": ["slope_angle"]}

    def __init__(self, database_path, vehicle, soil_types, soil_parameters):
        """Mobility analysis object constructor

        Args:
            vehicle (string): Vehicle name.
            soil_types (list): _description_
            soil_parameters (list): _description_
        """

        log.info(
            f"Starting new mobility analysis for:\n\tVehicle {vehicle}\n\tSoil parameters: {soil_parameters}\n\tSoil types: {soil_types}"
        )

        # Initialise vehicle simulation results database subset
        self.database = self.load_vehicle_metrics(database_path, vehicle)

        # Initialise soil types list
        self.soil_types = soil_types

        # Initialise soil parameters list
        self.soil_parameters = soil_parameters

        # Pre-allocate mobility tensors dictionary
        self.mobility_tensors = {}

    def load_vehicle_metrics(self, database_path, vehicle):
        """Load simulation results database subset for a given vehicle.

        Args:
            vehicle (string): Vehicle name.

        Returns:
            dict: Simulation results database subset.
        """

        # Load mobility database
        with open(database_path) as database:
            mobility_database = yaml.load(database, Loader=yaml.SafeLoader)

        # Return simulation results database subset for the selected vehicle
        return mobility_database[vehicle]

    def generate_mobility_tensors(self, test):

        for soil_type in self.soil_types:

            valid_entries = []

            # Get all entries for the given soil type
            for entry in self.database:

                if entry["soil"]["type"] == soil_type:
                    soil_parameters = []
                    test_parameters = []
                    for soil_parameter, test_parameter in zip(
                            self.soil_parameters,
                            self.TESTS_PARAMETERS_LIST[test]):
                        soil_parameters.append(entry["soil"][soil_parameter])
                        test_parameters.append(entry["test"][test_parameter])
                        max_speed = entry["value"]
                    database_entry = (*soil_parameters, *test_parameters,
                                      max_speed)

                    valid_entries.append(database_entry)

            valid_params_list = list(zip(*valid_entries))
            parameters = valid_params_list[0:-1]
            values = valid_params_list[-1]

            # Find unique entries
            parameters = [np.unique(parameter) for parameter in parameters]

            values = np.array(values).reshape(np.flip(np.array([len(parameter) for parameter in parameters]))).T

            log.info(
                f"Initialising new mobility tensor for soil {soil_type} with settings:\n\tParameters: {parameters}, Values: {values}"
            )

            self.mobility_tensors[soil_type] = MobilityTensor(
                soil_type, parameters, values)


class MobilityTensor():
    """Mobility tensor object class.

    A mobility tensor object is an n-dimensional interpolator defined for a given soil type and an arbitrarily long set of parameter for the given soil. When called with an n-tuple of parameters as argument, it returns the interpolator evaluation for the given soil parameters coordinate.
    """

    def __init__(self, soil_type, parameters, values):
        """Mobility tensor object constructor.

        Args:
            soil_type (string): Soil type to be associated with the mobility tensor.
            parameters (list): List of parameters (with each element being a np.array) associated with the soil type.
            values (list): List of objective function value associated with each grid point of the parameters grid.
        """

        self.soil_type = soil_type
        self.parameters = parameters

        log.info(
            f"Initialising new RegularGridInterpolator for soil {soil_type}, using extrapolation for points outside the grid"
        )

        #TODO: Handle cases where the grid is not regular, could be using a different interpolator to fill the missing grid points
        #TODO: Standardize the structure of the parameters and values in the database

        # Initialise linear radial-basis function interpolator 
        self.interpolator = RegularGridInterpolator(self.parameters,
                                                    values,
                                                    method="linear", 
                                                    fill_value=None)

    def __call__(self, call):
        """Evaluate radial-basis function interpolator at the provided n-tuple

        Args:
            call (np.array): n-tuple for which the objective function is evaluated.

        Returns:
            double: Objective function value
        """

        return self.interpolator(call.reshape(1, len(self.parameters)))[0]


class Map(ABC):

    def overlay(self, axis, map):

        cmap2 = matplotlib.colors.ListedColormap(["none", "red"])

        axis.imshow(map, alpha=1.0, cmap=cmap2)


class DirectionalMobilityMap(DirectionalMap):

    def __init__(self, mobility_maps):

        log.info(f"Initialising directional mobility map container.")

        self.mobility_maps = mobility_maps

    def show_overview(self, directions="N"):

        axes = []

        _, axis = plt.subplots(len(directions))

        for direction in directions:

            axis.title(f"Directional mobility map\n{direction} direction")
            axis.set_xlabel("X coordinate (m)")
            axis.set_ylabel("Y coordinate (m)")

            axes.append(axis)

        return axes

    def show_separate(self, directions="N"):

        axes = []

        for direction in directions:
            _, axis = plt.subplots(1, 1)
            axis.title(f"Directional mobility map\n{direction} direction")
            axis.set_xlabel("X coordinate (m)")
            axis.set_ylabel("Y coordinate (m)")

            axes.append(axis)

        return axes


class MobilityMap(Map):

    def __init__(self, direction, slope_map, land_use_map, no_go_map,
                 moisture_content_map, soil_type_map, mobility_tensors):

        log.info("Initialising new mobility map.")

        # The mobility map is initialised with zero-speeds, using land use map for shape reference
        self.data = jnp.zeros(no_go_map.data.shape)

        # Set direction associated with mobility map
        self.direction = direction

        #
        self.slope_map = slope_map
        self.land_use_map = land_use_map
        self.soil_type_map = soil_type_map
        self.moisture_content_map = moisture_content_map
        self.no_go_map = no_go_map

        #TODO: Maybe convert soil type map ealier in the process to be more efficient

        # Converts soil type map to a map of integers and mobility tensors to a list
        self.mobility_tensors = []
        self.soil_dict = {}
        for soil_type, mobility_tensor in mobility_tensors.items():
            self.mobility_tensors.append(mobility_tensor)
            self.soil_type_map.soilarray[self.soil_type_map.soilarray == soil_type] = len(self.mobility_tensors) - 1
            self.soil_dict[len(self.mobility_tensors) - 1] = soil_type

        @jit
        def evaluate_mobility_tensor(soil, moisture, slope):
            params = jnp.array([moisture, slope])
            return jax.lax.switch(soil, self.mobility_tensors, params)
        self.vmap_evaluate_mobility_tensor = jax.vmap(evaluate_mobility_tensor)

        #TODO: You should probably convert RasterMono to a factory method and provide an empty raster constructor.
        # Initialise a raster object based on the slope map
        self.raster = slope_map.raster

        log.info("Populating mobility map.")

        self.populate_map()

    def __call__(self):

        return np.array(self.data)

    def populate_map(self):

        index_x, index_y = np.unravel_index(np.arange(0, self.no_go_map.data.size), self.data.shape)

        no_go_mask = self.no_go_map.data[index_x, index_y] == 1

        # Vectors with index for go and no-go zones
        no_go_index_x = index_x[ no_go_mask]
        no_go_index_y = index_y[ no_go_mask]
        go_index_x    = index_x[~no_go_mask]
        go_index_y    = index_y[~no_go_mask]

        #TODO: Implement land-use based nogo.
        # Check if cell is a no-go zone based on land use
        #if self.land_use_map.is_nogo(index):
        #pass#self.data[index] = 0

        # Set no-go zones to zero
        self.data = self.data.at[no_go_index_x, no_go_index_y].set(0)

        # Get soil type, moisture content and directional slope for go zones
        soil_type = jnp.array(self.soil_type_map.soilarray[go_index_x, go_index_y], dtype=jnp.int32)
        moisture_content = jnp.array(self.moisture_content_map.moisturearray[go_index_x, go_index_y], dtype=jnp.float32)
        directional_slope = jnp.array(self.slope_map.directional_rasters[self.direction].data[go_index_x, go_index_y])

        low_slope_mask = directional_slope < 1

        # Take care of negative slopes and filter out for small slopes
        self.data = self.data.at[go_index_x[low_slope_mask], go_index_y[low_slope_mask]].set(
            self.vmap_evaluate_mobility_tensor(soil_type[low_slope_mask], moisture_content[low_slope_mask], jnp.zeros_like(moisture_content[low_slope_mask])))

        # Get corrosponding tensor speed
        self.data = self.data.at[go_index_x[~low_slope_mask], go_index_y[~low_slope_mask]].set(
            self.vmap_evaluate_mobility_tensor(soil_type[~low_slope_mask], moisture_content[~low_slope_mask], directional_slope[~low_slope_mask]))

        # Update raster
        self.raster.data = self.data

    def get_values(self, slope_dataset, slope_mobility):

        # Initialise interpolator for speed values
        speed_interpolator = spi.interp1d(list(slope_mobility.keys()),
                                          list(slope_mobility.values()),
                                          kind="linear")

        for i in range(0, slope_dataset.size):
            index = np.unravel_index(i, self.data.shape)
            self.data[index] = speed_interpolator(slope_dataset[index])

    def plot(self, show=True, export=False):
        
        _, axes = plt.subplots(1, 1)

        (_, axes, color_bar, _) = self.raster.plot(show=False, export=False)

        color_bar.ax.set_ylabel("Speed (m/s)", rotation=270)

        if export:
            plt.savefig(export / f"Mobility map ({self.direction}).pdf",
                        format="pdf")

        #TODO: for some reasons it opens two windows only one with content
        if show:
            plt.show()

        plt.close()
