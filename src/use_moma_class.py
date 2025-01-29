""" This demostaes how to use the moma_class 
"""

from moma_class import moma

# Path to the site folder
site_path = "./examples/wtd41"

# Setting map generation parameters
waypoint_start = (400, 350)
waypoint_end   = (120,  50)

# Running mapping and planning
MOMA = moma(site_path)

MOMA.path_planner(waypoint_start, waypoint_end)

