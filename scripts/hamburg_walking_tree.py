#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create an image of Hamburg, showing how to get where on foot.

@author: Ruud Wijtvliet, rwijtvliet@gmail.com
"""

# %% Create map.

from maptra import Map, Location
import maptra.locations as ml

with open('apikey.txt') as f:
    apikey = f.read().strip()
    Map.set_gmaps_api_key(apikey)

# Create.
m = Map.from_address('Stresemannstrasse 320, Hamburg', 'walking')
filtr = ml.geofilter()
m.add_ends(ml.on_circular_grid(m.start, [10_000], 6_000, geofilter=filtr))

# Save.
m.to_pickle("pickle/hamburg_walking_1000_10000.pkl")
# Load.
m = Map.from_pickle("pickle/hamburg_walking_1000_10000.pkl")


# %% Visualize.

from maptra import MapViz
import geopandas as gpd

viz = MapViz(m, 'EPSG:5243') #) #, '3395, 4326, 5243, 25832

# Background map:
is_detailed = False
#   a) Background map: world/europe scale.
scale = ('50m', '10m')[is_detailed]
viz.add_background_fromfile(f'data/world/naturalearth/{scale}_cultural/ne_{scale}_admin_0_countries_lakes.shp', color='grey', alpha=0.3, edgecolor='black')
# viz.add_background_fromfile(f'data/world/naturalearth/{scale}_physical/ne_{scale}_land.shp', color='grey', alpha=0.2)
# viz.add_background_fromfile(f'data/experimental/eurogeographics/DATA/Countries/DE/RoadL.shp', color='red', alpha=0.2)

#   b) Background map: Hamburg scale.
gdf = gpd.read_file('/home/ruud/syncNone/Shapefiles/hh_30kmaround_wgs-84_2020-03-31_shapefile_simplified.shp/landuse_a.shp')
colors = {('orchar', 'farm', 'plant_nursery', 'meadow',):'#e1f4cb',
         ('residential',): '#f1bf98',
         ('industrial','grass'):'#756d',
         ('park',):'#9bbf80',
         ('forest', 'nature_reserve'): '#bacba9'}
# for types, color in colors.items():
#     viz.add_background(gdf[gdf['type'].isin(types)], color=color, alpha=0.3)

base_path = 'data/hamburg/osmaxx/' + ('simplified/', 'detailed/')[is_detailed]
# # viz.add_background_fromfile(base_path + 'road_l.shp', color='#290022', alpha=0.4)
# viz.add_background_fromfile(base_path + 'railway_l.shp', color='#330306', alpha=0.2)
# viz.add_background_fromfile(base_path + 'water_a.shp', color='lightblue', alpha=0.8)

# viz.add_background_fromfile(base_path + 'road_l.shp', color='white', alpha=1, linewidth=0.3)
# viz.add_background_fromfile(base_path + 'railway_l.shp', color='grey', alpha=0.7, linewidth=0.1)
viz.add_background_fromfile(base_path + 'water_a.shp', color='white', alpha=1, zorder=10)


# Content: 
viz.add_routes(alpha=0.9, minimum_width=1, color='black', var_width='lin', zorder=12)
viz.add_voronoi('speed', alpha=0.9) #edgecolor='black')
viz.add_startpoint(alpha=1, color='grey', markersize=90)
viz.add_endpoints(marker='o', color='grey', markersize=10, alpha=0.87)
# viz.add_quiver(cmap='brg')
viz.showfig(0.03)

# %% Save file.

from pathlib import Path
folderpath = 'output/temp/' #must have trailing /
filename = 'walking'        #no file extension

Path(folderpath).mkdir(parents=True, exist_ok=True)
sizes = {'small': 400, 'medium': 2000, 'large': 5000}
for size, pix in sizes.items():
    viz.savefig(f'{folderpath}{filename}_{size}.png', minwidth=pix, minheight=pix)
viz.savefig(f'{folderpath}{filename}.svg')