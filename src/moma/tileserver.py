"""Tile server interface module

Provides a barebones interface to the Terracotta tile server.
"""

import subprocess


class TileServer():
    """Simple interface to Terracotta tile server
    """

    def __init__(self, input_path, output_path, database_path):
        """Tile server object constructor. Defines a workspace with an input raster folder path and output folders for the processed rasters and the tile server database. At the current time it relies striclty on file names to label rasters.

        Args:
            input_path (str): Path to the directory holding the site rasters.
            output_path (str): Path to the directory where processed rasters will be stored.
            database_path (str): Path to the Terracotta SQLite database.
        """

        # Define paths
        self.input_path = input_path
        self.output_path = output_path
        self.database_path = database_path

        # Define file path mask for raster files (select all *.tif)
        self.rasters_mask = input_path + "/*.tif"

        # Define mask to identify rasters in the database . Currently uses the entire file name.
        self.optimized_rasters_mask = output_path + "/{name}.tif"

    def update(self):
        """Update the tile server database with optimised versions of the rasters
        """

        # Run raster optimization process
        subprocess.run([
            "terracotta", "optimize-rasters", "--output-folder",
            f"{self.output_path}", "--overwrite", f"{self.rasters_mask}"
        ],
                       check=True)

        # Run database generation/update process
        subprocess.run([
            "terracotta", "ingest", "--output-file", "terracotta.sqlite",
            f"{self.optimized_rasters_mask}"
        ],
                       check=True)

    def serve(self):
        """Run the Terracotta XYZ tile server
        """

        # Run Terracotta XYZ tile server process
        subprocess.run([
            "terracotta", "serve", "--database",
            f"{self.database_path}/aumap.sqlite", "--port", "5100"
        ],
                       check=True)

    def connect(self):
        """Run the Terracotta XYZ tile client
        """

        # Run Terracotta XYZ tile client process
        subprocess.run(["terracotta", "connect", "localhost", "--port", "5100"],
                       check=True)
