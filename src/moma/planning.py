"""Planning module
"""

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.interpolate as spi
import rasterio as rio

log = logging.getLogger("MOMA")


class Path:
    """Path
    """

    def __init__(self, waypoints):
        """Path object constructor. Defines a path using a provided set of waypoints.

        Args:
            waypoints (list): Iterable of waypoints defined as 2-element tuples.
        """

        self.waypoints = waypoints

    #def get_optimal_path_raster(self, shape):
    #    """Optimal path raster generator
    #
    #    Args:
    #        shape (tuple): Output raster shape
    #    """
    #
    #    # Pre-allocate shortest path raster
    #    optimal_path_raster = np.zeros(shape)
    #
    #    # Highlight points which are part of the shortest path
    #    for index, _ in enumerate(optimal_path_raster.flat):
    #        if index in self.waypoints:
    #            optimal_path_raster[np.unravel_index(index, shape)] = 1
    #        else:
    #            optimal_path_raster[np.unravel_index(index, shape)] = 0
    #
    #    self.raster = optimal_path_raster

    def plot(self, overlay, show=False, axis=None, export=False):
        import matplotlib as mpl  # Used for color maps definition

        #if axis is None:
        #    _, axis = plt.subplots(1, 1)

        #rgb = np.dstack((overlay.raster.data[0], overlay.raster.data[1], overlay.raster.data[2]))
        # Set nodata values
        #rgb[rgb == 0] = 255

        # Open new figure
        (figure, axes) = plt.subplots(1, 1)

        # Set and invert colormap
        #cmap = mpl.cm.get_cmap(overlay.raster.colormap + "_r")

        # Normalise colormap range. Defaults to a range of three standard deviations around the mean value.
        norm = mpl.colors.Normalize(
            vmin=np.mean(overlay.data) - 1.25 * np.std(overlay.data),
            vmax=np.mean(overlay.data) + 1.25 * np.std(overlay.data))

        # Draw raster data
        image2 = axes.imshow(overlay.data, norm=norm, cmap="viridis")

        # Disable scientific notation for ticks
        axes.ticklabel_format(style="plain")

        # Add axes labels
        axes.set_xlabel("Map CRS X coordinate")
        axes.set_ylabel("Map CRS Y coordinate")

        # Add color_bar
        #color_bar = figure.colorbar(image2, ax=axes)
        #color_bar.ax.get_yaxis().labelpad = 25
        #color_bar.ax.set_ylabel("Elevation (m)", rotation=270)
        #color_bar.minorticks_on()

        #figure, axis = overlay.plot(show=False, export=False)
        #axis.imshow(overlay.raster.data)
        #rastermasked = np.ma.masked_where(self.raster == 0, self.raster)
        #axis.imshow(rastermasked, interpolation="none")

        axes.scatter(self.xint,
                     self.yint,
                     color="r",
                     marker="o",
                     s=1.25,
                     cmap="seismic")
        axes.scatter(self.xint[0], self.yint[0], color="b", marker="s", s=7.5)
        axes.scatter(self.xint[-1], self.yint[-1], color="g", marker="X", s=7.5)

        if export:
            plt.savefig(export / "Optimal path.pdf", format="pdf")

        if show:
            plt.show()

        plt.close()

    #TODO Review spline smoothing algorithm
    def smoothen_spline(self, shape, points=400):

        logging.info(
            f"Densifying and smoothing path using a cubic spline interpolator over {points} points."
        )

        y, x = np.unravel_index(self.waypoints, shape)
        f, u = spi.splprep([x, y], k=1, s=0.15)
        self.xint, self.yint = spi.splev(np.linspace(0, 1, points), f)

    def print_path(self, raster):
        log.info(f"Shortest path found! Printing cell list to console.")

        # Print the shortest path to console
        #if verbose:
        #    print(f"{best_path[0]}", end="")
        #    for node in best_path[1:-1]:
        #        print(f"-> {node} ", end="")
        #    print(f"-> {best_path[-1]}")
        for node in self.waypoints[0:-1]:
            indices = np.unravel_index(node, raster.data.shape)
            print(
                f"{node} -> {indices} -> {rio.transform.xy(raster.transform, indices[0], indices[1])}"
            )
        print(
            f"{self.waypoints[-1]} -> {np.unravel_index(node, raster.data.shape)}",
            end="")

    #TODO Define Savgol filter algorithm and compare with the spline algorithm
    #def smoothen_savgol(self):
    #
    #    #yint = savgol_filter(y, 51, 3) # window size 51, polynomial order 3
    #    #shortpath = np.ravel_multi_index((xint, yint), (125,125))

    #TODO Define an AUVAS specification for importing coordinates from a planner.
    #def convert_path_to_coordinates(xmin, xmax, ymin, ymax, xres, yres, path):
    #
    #    pass
    #
    #def convert_to_vehicle_origin(waypoints, vehiclex, vehicley):
    #
    #    new_waypoints = []
    #
    #    for waypointx, waypointy in waypoints:
    #        new_waypoints.append(waypointx+vehiclex)


class MobilityPlanner():
    """Mobility planner object
    """

    def __init__(self, site):

        self.site = site


def make_speed_graph(mobility_map):
    """Returns the weighted digraph for a given mobility map

    Returns the weighted digraph for a directional mobility map by defining a node for each mobility map cell and a set of at least 3 and at most 8 weighted edges connecting each cell to its neighbours through an appropriate speed weight as specified by the mobility map layer in the given direction.

    Args:
        mobility_map (dict): Directional mobility map with cardinal and intercardinal direction as keys

    Returns:
       networkx.digraph: Directional mobility map weighted digraph
    """

    log.info(f"Generating directional mobility graph from mapped data.")

    # Initialise empty graph
    DG = nx.DiGraph()

    # Add weighted edges
    we = define_weighted_edges(mobility_map)
    DG.add_edges_from(we)

    return DG


def find_all_neighbours(i, j, arr):
    """Return all neighbour points for a matrix element

    Returns a dictionary with keys corresponding to the cardinal an intercardinal directions with tuples (i, j) as values corresponding to the C-indexed coordinates of the neighbour point in the chosen direction. If no neighbour are present for a given direction, the dictionary holds a None value for the entry for that direction. The current implementation relies on an iterative approach.

    Args:
        i (int): Selected element C-style row index
        j (int): Selected element C-style column index
        arr (numpy.ndarray): Two-dimensional NumPy array

    Returns:
        dict: Neighbour elements dictionary with cardinal and intercardinal direction keys
    """

    neighbors = {
        "N": None,
        "NE": None,
        "E": None,
        "SE": None,
        "S": None,
        "SW": None,
        "W": None,
        "NW": None,
    }

    # Listing neighbors for boundary nodes
    if i == 0 or i == arr.shape[0] - 1 or j == 0 or j == arr.shape[1] - 1:
        if i != 0 and j != 0:
            neighbors["NW"] = (i - 1, j - 1)
        if i != 0:
            neighbors["N"] = (i - 1, j)
        if i != 0 and j != (arr.shape[1] - 1):
            neighbors["NE"] = (i - 1, j + 1)
        if j != arr.shape[1] - 1:
            neighbors["E"] = (i, j + 1)
        if i != (arr.shape[0] - 1) and j != arr.shape[1] - 1:
            neighbors["SE"] = (i + 1, j + 1)
        if i != (arr.shape[0] - 1):
            neighbors["S"] = (i + 1, j)
        if i != (arr.shape[0] - 1) and j != 0:
            neighbors["SW"] = (i + 1, j - 1)
        if j != 0:
            neighbors["NW"] = (i, j - 1)

    # Listing neighbors for internal nodes
    else:
        neighbors["N"] = (i - 1, j)
        neighbors["NE"] = (i - 1, j + 1)
        neighbors["E"] = (i, j + 1)
        neighbors["SE"] = (i + 1, j + 1)
        neighbors["S"] = (i + 1, j)
        neighbors["SW"] = (i + 1, j - 1)
        neighbors["W"] = (i, j - 1)
        neighbors["NW"] = (i - 1, j - 1)

    return neighbors


def define_weighted_edges(mobility_map):

    log.info(f"Computing mobility graph edge weights across the map.")

    weighted_edges_list = []

    for index, _ in enumerate(mobility_map["N"].data.flat):

        (i, j) = np.unravel_index(index, mobility_map["N"].data.shape)

        neighbors = find_all_neighbours(i, j, mobility_map["N"].data)
        for direction, neighbor in neighbors.items():

            # If there is no neighbor, skip item
            if neighbor is None:
                continue

            neighbor_index = np.ravel_multi_index(
                neighbor, mobility_map[direction].data.shape)

            neighbor_weight = 1 / mobility_map[direction].data[
                neighbor] if mobility_map[direction].data[neighbor] != 0 else 0

            weighted_edges_list.append((index, neighbor_index, {
                "weight": neighbor_weight
            }))

    return weighted_edges_list


def find_shortest_path(mobility_digraph, source, target, verbose=True):
    """Returns the shortest path between a source and target nodes in a directional mobility digraph

    Returns the shortest path between a source and target nodes in a directional mobility digraph.

    Args:
        mobility_digraph (_type_): _description_
        source (_type_): _description_
        target (_type_): _description_
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    log.info(f"Finding shortest path between {source} and {target}.")

    best_path = nx.shortest_path(mobility_digraph,
                                 source=source,
                                 target=target,
                                 weight="weight")

    return Path(best_path)


#TODO Implement weighted edges search through a convolution kernel
#def define_weighted_edges_kernel(mobility_maps):
#
#    weighted_edges_list = []
#
#    kernels = {
#        "N":  np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
#        "S":  np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
#        "E":  np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
#        "W":  np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
#        "NW": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
#        "SW": np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
#        "NE": np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
#        "SE": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
#    }
#    dirs = ["N", "S", "E", "W", "NW", "SW", "NE", "SE"]
#
#    shape = mobility_maps["N"].data.shape
#    indices = np.linspace(0, mobility_maps["N"].data.size, mobility_maps["N"].data.size).reshape(shape)
#
#
#
#    # Compute all neighbors
#    neighbors = {}
#    for dire in dirs:
#        neighbors[dire] = cv2.filter2D(indices, -1, kernels[dire])
#        neighbors[dire] = neighbors[dire].astype(int)
#    indices = indices.astype(int)
#    arrays = [indices, *neighbors.values()]
#    merged_arrays = {}
#    #merged_arrays = np.stack(arrays, axis=2).reshape(indices.size, 9)
#
#    merged_arrays["N"] = np.stack([indices, neighbors["N"]], axis=2).reshape(indices.size, 2).astype(int).tolist()
#    merged_arrays["N"] = [(*element, {"weight": 1}) for element in merged_arrays["N"]]
#    #for item in merged_arrays:
#    #    #TODO fix the weights
#    #    [weighted_edges_list.append((item[0], myitem, {"weight": 1})) for myitem in item[1:-1]]
#
#    return merged_arrays["N"]
