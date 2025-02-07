""" This file runs the moma_class.py with the given arguments

Example calls from command line:
    python src/use_moma_class.py --show_plots True --vehicle mungo --site wtd41_copy
"""

import pathlib
import argparse

from moma_class import MOMA

# ============================================
# Settings for parsing command line arguments 
# ============================================
parser = argparse.ArgumentParser()
# Show plots
parser.add_argument("--show_plots", help="show popup with plots", type=bool, default=False)
# Vehicle
parser.add_argument("--vehicle", help="name of the vehicle to generate plots for", type=str, default="mungo")
# Path to the site folder
parser.add_argument("--site", help="path to the site folder", type=str, default="wtd41_copy")
args = parser.parse_args()

# Placment of sites and vehicle mobilities on the specific computer
sites_path    = pathlib.Path("./examples/sites/")
vehicles_path = pathlib.Path("./examples/vehicles")

# ===============
# Map generation
# ===============
moma = MOMA(sites_path, vehicles_path, args.site, args.vehicle)

moma.generate_aerial_map(show=args.show_plots)
moma.generate_elevation_map(show=args.show_plots)
moma.generate_directional_slope_maps(show=args.show_plots)
moma.generate_land_use_map(show=args.show_plots)
moma.generate_soil_type_map(show=args.show_plots)
moma.generate_moisture_content_map(show=args.show_plots)
moma.generate_no_go_map(show=args.show_plots)
moma.generate_directional_mobility_maps(directions="all", show=args.show_plots)

# Setting map generation parameters
waypoint_start = (400, 350)
waypoint_end   = (120,  50)
#moma.path_planner(waypoint_start, waypoint_end, show=args.show_plots)

