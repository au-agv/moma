""" This demostaes how to use the moma_class 
"""

from moma_class import MOMA

# ======
# Setup
# ======
show_plots = False

# Path to the site folder
site_path = "./examples/wtd41_copy"

# no-go zones
#[[x1, x2, y1, y2, "description"], ...]
nogo_zones = [
    [100, 150, 310, 340, "A dangerous zone"],
    [300, 400, 380, 400, "Another dangerous zone"],
    [175, 300, 200, 250, "Yet another dangerous zone"]
]

# Vehicle
vehicle = "mungo"

# Setting map generation parameters
waypoint_start = (400, 350)
waypoint_end   = (120,  50)

# ===============
# Map generation
# ===============
moma = MOMA(site_path)

moma.generate_aerial_map(show=show_plots)
moma.generate_elevation_map(show=show_plots)
moma.generate_directional_slope_maps(show=show_plots)
moma.generate_land_use_map(show=show_plots)
moma.generate_soil_type_map(show=show_plots)
moma.generate_moisture_content_map(show=show_plots)
moma.generate_no_go_map(nogo_zones, show=show_plots)
moma.generate_directional_mobility_maps(vehicle, directions="all", show=show_plots)
#moma.path_planner(waypoint_start, waypoint_end, show=show_plots)

