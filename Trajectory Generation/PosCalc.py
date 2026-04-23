"""
Export stage coordinates (positions.ini) for Bruker ESPRIT SEM control software.

Stage coordinates are in µm and follow the microscope convention: origin at
the top-right corner of the map area. grid_x is NOT flipped here (unlike the
visualisation in methods.place_tiles) because the microscope software already
uses a top-right origin.

Each position is the centre of the corresponding tile, computed as:
    stage = (grid_index + 0.5) * tile_size_px * pixel_size_nm / 1000 + origin_µm

NOTE — constants below are sample-specific and must be edited per acquisition.
The defaults shown are for a test run and do not correspond to either dataset
reported in the manuscript. Manuscript values:
    Schist:  p=53, q=38, pixel_size=500 nm
    Diorite: p=57, q=40, pixel_size=500 nm
Stage origin (initial_x, initial_y) and z depend on the physical position of
the sample in the SEM and must be set by the operator for each session.
"""
from methods import *
import configparser

min_tile_size, max_tile_size = 768, 1024
p, q = 45, 64
pixel_size = 430  # In nm

# Set initial stage coordinates in um - Use top right corner as reference!
initial_x = 48000
initial_y = 52000
z = 36357

# Sanity check
map_size = [min_tile_size * p, max_tile_size * q]
tile_width, tile_height, num_tiles_x, num_tiles_y, p, q = find_valid_tile_size(
    map_size, min_tile_size, max_tile_size, p, q, maximize_reflections=False
)

print('Tile width & height:', tile_width, tile_height)
print('p & q:', p, q)

# Compute time steps for tile alignment
t_vals = compute_time_steps(num_tiles_x, num_tiles_y, p, q)

config = configparser.ConfigParser()
coordinates = []

for t in t_vals:
    x, y = billiard_knot(t, p, q)
    grid_x = int(x * num_tiles_x)
    grid_y = int(y * num_tiles_y)
    stage_x = ((grid_x + 0.5) * tile_width * pixel_size / 1000) + initial_x
    stage_y = ((grid_y + 0.5) * tile_height * pixel_size / 1000) + initial_y
    coordinates.append((stage_x, stage_y, z))

config["Global"] = {"PositionCount": str(len(coordinates))}

for i, (x, y, z) in enumerate(coordinates, start=1):
    section = f"Position_{i}"
    config[section] = {
        "X": f"{x:.2f}",
        "Y": f"{y:.2f}",
        "Z": f"{z:.2f}"
    }

# Save to .ini file
with open("positions.ini", "w") as configfile:
    config.write(configfile)
