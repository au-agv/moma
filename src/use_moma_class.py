""" This file runs the moma_class.py with the given arguments

Example calls from command line:
    python src/use_moma_class.py --show_plots True --vehicle mungo --site_path ./examples/wtd41_copy
"""

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
parser.add_argument("--site_path", help="path to the site folder", type=str, default="./examples/wtd41_copy")
args = parser.parse_args()

# ======
# Setup
# ======

# no-go zones
#[[x1, x2, y1, y2, "description"], ...]
nogo_zones = [
    [100, 150, 310, 340, "A dangerous zone"],
    [300, 400, 380, 400, "Another dangerous zone"],
    [175, 300, 200, 250, "Yet another dangerous zone"]
]

# Setting map generation parameters
waypoint_start = (400, 350)
waypoint_end   = (120,  50)

# ===============
# Map generation
# ===============
moma = MOMA(args.site_path)

moma.generate_aerial_map(show=args.show_plots)
moma.generate_elevation_map(show=args.show_plots)
moma.generate_directional_slope_maps(show=args.show_plots)
moma.generate_land_use_map(show=args.show_plots)
moma.generate_soil_type_map(show=args.show_plots)
moma.generate_moisture_content_map(show=args.show_plots)
moma.generate_no_go_map(nogo_zones, show=args.show_plots)
moma.generate_directional_mobility_maps(args.vehicle, directions="all", show=args.show_plots)
#moma.path_planner(waypoint_start, waypoint_end, show=args.show_plots)

