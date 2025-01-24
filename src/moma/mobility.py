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

        # Parameters is a list of NumPy arrays
        parameters_grid = np.meshgrid(*parameters)

        # Define a grid for evaluation of the soil parameters
        points = np.stack([parameter.ravel() for parameter in parameters_grid],
                          -1)

        log.info(
            f"Initialising new RBF interpolator for soil {soil_type} using 2 neighbours and kernel \"linear\""
        )

        # Initialise linear radial-basis function interpolator
        self.interpolator = spi.RBFInterpolator(points,
                                                values,
                                                smoothing=0,
                                                neighbors=3,
                                                kernel="linear")

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

        # The mobility map is initialised with zero-speeds, using land use map for shape reference
        self.data = np.zeros(no_go_map.data.shape)

        # Set direction associated with mobility map
        self.direction = direction

        #
        self.slope_map = slope_map
        self.land_use_map = land_use_map
        self.soil_type_map = soil_type_map
        self.moisture_content_map = moisture_content_map
        self.no_go_map = no_go_map

        #
        self.mobility_tensors = mobility_tensors

        #TODO: You should probably convert RasterMono to a factory method and provide an empty raster constructor.
        # Initialise a raster object based on the slope map
        self.raster = slope_map.raster

        #
        self.populate_map()

    def __call__(self):

        return self.data

    def populate_map(self):

        for i in range(0, self.no_go_map.data.size):
            index = np.unravel_index(i, self.data.shape)

            #TODO: Implement land-use based nogo.
            # Check if cell is a no-go zone based on land use
            #if self.land_use_map.is_nogo(index):
            #pass#self.data[index] = 0

            # Check if a cell is a no-go zone based on user-specified no-go zones
            if self.no_go_map.data[index] == 1:
                self.data[index] = 0
                continue

            # Get cell soil type
            soil_type = self.soil_type_map.soilarray[index]

            # Get cell moisture content
            moisture_content = self.moisture_content_map.moisturearray[index]

            # Get cell directional slope
            directional_slope = self.slope_map.directional_rasters[
                self.direction].data[index]

            # Take care of negative slopes and filter out for small slopes
            if (directional_slope < 1):
                speed = self.mobility_tensors[soil_type](np.array(
                    [moisture_content, 0]))
            else:

                # Get corresponding tensor speed
                speed = self.mobility_tensors[soil_type](np.array(
                    [moisture_content, directional_slope]))

            # Assign speed value
            self.data[index] = speed

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

        if show:
            plt.show()

        plt.close()
