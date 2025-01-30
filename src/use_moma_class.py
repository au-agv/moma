""" This demostaes how to use the moma_class 
"""

from moma_class import moma

show_plots = True

# Path to the site folder
site_path = "./examples/wtd41_copy"

# Setting map generation parameters
waypoint_start = (400, 350)
waypoint_end   = (120,  50)

# Running MOMA class
MOMA = moma(site_path)

MOMA.aerial_map(show=show_plots)

MOMA.path_planner(waypoint_start, waypoint_end)

