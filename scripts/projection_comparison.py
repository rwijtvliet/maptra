#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create an image of Europe, showing the distortion caused by projection.

@author: Ruud Wijtvliet, rwijtvliet@gmail.com
"""

# %% Imports.

from maptra import Map, CreateLocations, Location

# Create.
m = Map(Location((53, 10)))
filtr = CreateLocations.geofilter()
m.add_locations(CreateLocations.on_hexagonal_grid(m.start, 100_000, [1_500_000], geofilter=filtr))


# %% Visualize.

from maptra import Visualization
viz = Visualization(m, 'EPSG:5243') #) #, '3395, 4326, 5243, 25832

# Background map:
is_detailed = False
#   a) Background map: world/europe scale.
scale = ('50m', '10m')[is_detailed]
viz.add_background_fromfile(f'data/world/naturalearth/{scale}_cultural/ne_{scale}_admin_0_countries_lakes.shp', color='grey', alpha=0.3, edgecolor='black')

# Content: 
m.spoof()
viz.add_endpoints(True, marker='o', color='black', alpha=0.8)
viz.showfig(0.03)

# %% Save file.

from pathlib import Path
folderpath = 'output/projections/' #with trailing /
filename = 'epsg5243' #no extension

Path(folderpath).mkdir(parents=True, exist_ok=True)
sizes = {'small': 400, 'medium': 2000, 'large': 5000}
for size, pix in sizes.items():
    viz.savefig(f'{folderpath}{filename}_{size}.png', minwidth=pix, minheight=pix)
viz.savefig(f'{folderpath}{filename}.svg')