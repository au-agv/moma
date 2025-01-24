"""Database module
"""

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import yaml


class Database():
    """Mobility database
    """

    path = "moma.db"
    entries = {}
    tags = []

    def __init__(self):

        pass

    def add_entry(self, speed, tags):

        logging.info(f"Adding mobility entry: speed {speed} m/s, tags: {tags}.")

        missing_tags = {tag: tag for tag in tags if tag not in self.tags}

        for tag in missing_tags:

            logging.info(
                f"Tag {tag} not previously in database - adding it now.")

            # Add missing tag
            self.tags.append(tags)

        self.entries[vehicle].append(DatabaseEntry(speed, tags))

    def get_entry(self, tags):

        for entry in self.entries:

            matching_tags = {tag: tag for tag in entry.tags if tag in self.tags}

            if len(matching_tags) == len(tags):

                speed = entry.speed

                logging.info(
                    f"Found match: {matching_tags}, speed {speed} m/s.")

                return speed

    def update_tags(self):

        logging.info(f"Updating tags database.")

        for entry in self.entries:

            missing_tags = {
                tag: tag
                for tag in entry.tags if tag not in self.tags
            }

            for tag in missing_tags:

                logging.info(f"Adding tag {tag} to database.")

    def save(self):

        with open("data.json", "w") as f:

            json.dump(self.path, f)

    def load(self):

        with open(self.path, "w") as f:

            self.entries = json.load(self.path, f)

        self.update_tags()

    def load_speed(self, path):
        """Load maximum traversal speed for a vehicle test
        """

        # Read simulation import configuration
        with open(path, "r") as config_file:

            config = yaml.safe_load(config_file)

        # Find all slope and mu entries
        alphas = []
        mus = []
        for key, entry in config.items():

            if entry.get("skip", False):

                logging.info(f"Skipping entry {key}.")

                continue

            logging.info(f"Processing entry {key}...")

            # Append entry slope angle and friction coefficient to the parameters lists
            tags = entry.get("tags")
            alphas.append(entry["attributes"]["slope"])
            mus.append(entry["attributes"]["mu"])

        # Sort entries and remove duplicates
        alphas = list(set(sorted(alphas)))
        mus = list(set(sorted(mus)))

        logging.info(
            f"Found {len(alphas)} slopes and {len(mus)} friction coefficient entries."
        )

    def get_entries_count(self, tags):

        tag_count = {tag: set([]) for tag in tags}

        for entry in self.entries:

            if all(tag in entry.tags for tag in tags):

                for tag in matching_tags:

                    tag_count[tag].add(tag)

        return tag_count

    def get_speed(self, path):

        for key, simulation in config.items():

            if simulation.get("skip", False):

                logging.info(f"Skipping entry {key}.")

                continue

            logging.info(f"Now processing entry {key}.")

            # Get IVRESS result DAT file path
            ivress_results_path = os.path.join(os.path.dirname(path),
                                               simulation["path"])

            _, speed = IVRESSResult.ivress2numpy(ivress_results_path,
                                                 metrics="Speed")

            # Get database metadata
            tags = simulation["tags"]

            # Get processing information
            t_step = tags["time_step"]
            start = simulation["filtering"]["start"]
            end = simulation["filtering"]["end"]
            min_mu = simulation["filtering"]["min_mu"]
            max_sigma = simulation["filtering"]["max_sigma"]

            # Initialize result
            result = Result(speed,
                            t_step,
                            window_range=(start, end),
                            min_mu=min_mu,
                            max_sigma=max_sigma)

            # Get speed
            result.get_signal_stats()

            v = result.mu

            self.addentry(v, tags)

    def add_to_mobility_tensor(self, tensor):

        # Initialise a hard terrain mobility tensor
        mobility_tensor = HardMobilityTensor(alphas, mus)

        mobility_tensor.add(tags["slope"], tags["mu"], v)
        mobility_tensor.update_interpolator()


class DatabaseAgent():

    def __init__(self, database, vehicle):

        self.vehicle = vehicle
        self.database = database

    def get_parameter_limits(self, terrain, parameters):

        parameter_span = set([])

        matching_entries = self.database.entries.match(
            [self.vehicle, terrain, parameters])

        for entry in matching_entries:

            parameter_span.add(entry[parameter])

        return parameter_span


class DatabaseEntry():

    speed = 0
    tags = {}

    def __init__(self, speed, tags):

        self.speed = speed
        self.tags = tags


class Result():
    """Speed result
    """

    def __init__(self,
                 y,
                 t_step,
                 window_range=(0, 100),
                 min_mu=0.1,
                 max_sigma=1e-1,
                 unit="time"):
        """Speed result constructor

        Args:
            y (numpy.array): Time series signal
            t_step (float): Time series time step
            window_range (tuple, optional): Start and env times for the analysis window. Defaults to (0, 100).
            min_mu (float, optional): Minimum average speed value before result is ignored. Defaults to 0.1.
            max_sigma (float, optional): Maximum standard deviation before result is ignored. Defaults to 1e-1.
            unit (str, optional): Whether the window is specified in sample number ("sample") or time ("time"). Defaults to "time".
        """

        self.y = y
        self.t_step = t_step
        self.window_range = window_range
        self.min_mu = min_mu
        self.max_sigma = max_sigma
        self.unit = unit

    def get_range(self):

        if self.unit == "sample":

            start = self.window_range[0]
            end = self.window_range[1]

        elif self.unit == "time":

            start = int(self.window_range[0] / self.t_step)
            end = int(self.window_range[1] / self.t_step)

        else:

            raise ValueError

        return (start, end)

    def get_window(self):
        """Time-series window

        Returns a slice of the time series defined by a window start and end point.
        """

        (start, end) = self.get_range()

        return self.y[start:end]

    def get_signal_stats(self):

        # Compute window
        window = self.get_window()

        # Compute average speed
        self.mu = np.mean(window)

        # Check if average speed is higher than minimum threshold
        if self.mu < self.min_mu:

            logging.warning(
                f"Minimum mean speed is lower than specified threshold (V = {self.mu:0.3g} m/s)"
            )

            # Ignore the result by assigning zero speed
            self.mu = 0

        # Compute speed standard deviation
        self.sigma = np.std(window)

        # Check if standard deviation is higher than minimum threshold
        if self.sigma > self.max_sigma:

            logging.warning(
                f"Standard deviation is larger than specified threshold (V_std = {self.sigma:0.3g} m/s)"
            )

            # Ignore the result by assigning zero speed
            self.mu = 0

        if (self.mu < self.min_mu) or (self.sigma > self.max_sigma):

            # Plot signal and save plot for debugging
            self.plot_warning()

    def plot_warning(self, show=True, save=True):
        """Export plot of a signal which raised a warning
        """

        # Get range
        (start, end) = self.get_range()
        start = self.t_step * start
        end = self.t_step * end

        x = np.linspace(0, self.y.size * self.t_step, self.y.size)

        # Create new plot
        _, axes = plt.subplots(1, 1)

        # Plot speed time series
        axes.plot(x, self.y)

        # Plot time series slice start and end points
        axes.axvline(x=start)
        axes.axvline(x=end)

        # Show signal
        if show:
            plt.show()

        # Export figure to file
        #if save:
        #    plt.savefig("last.png")
