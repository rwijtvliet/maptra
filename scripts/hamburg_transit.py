#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create an image of Hamburg, showing how to get where on by transit.

@author: Ruud Wijtvliet, rwijtvliet@gmail.com
"""

# %% Create map.

from maptra import Map, Location
import maptra.locations as locas

with open('apikey.txt') as f:
    apikey = f.read().strip()
    Map.set_gmaps_api_key(apikey)

# Create.
m = Map.from_address('stresemannstrasse 320, hamburg', 'transit')
filtr = locas.geofilter()
m.add_locations(locas.on_hexagonal_grid(m.start, 5_000, [7_000], geofilter=filtr))

# Save.
# m.to_pickle("pickle/hamburg_transit_5000_10000.pkl")
# Load
# m = Map.from_pickle("pickle/hamburg_transit_5000_10000.pkl")


# %% Visualize.

from maptra import Visualization
import geopandas as gpd

viz = Visualization(m)# 'EPSG:25832') #) #, 'epsg:4326')#, 

# Background map:
is_detailed = False
#   a) Background map: world/europe scale.
scale = ('50m', '10m')[is_detailed]
viz.add_background_fromfile(f'data/world/naturalearth/{scale}_cultural/ne_{scale}_admin_0_countries_lakes.shp', color='grey', alpha=0.2)
# viz.add_background_fromfile(f'data/world/naturalearth/{scale}_physical/ne_{scale}_land.shp', color='grey', alpha=0.2)
# viz.add_background_fromfile(f'data/experimental/eurogeographics/DATA/Countries/DE/RoadL.shp', color='red', alpha=0.2)
### viz.addckground_fromfile(f'data/eurogeographics/DATA/Countries/DE/RailrdL.shp', color='green', alpha=0.2)

#   b) Background map: Hamburg scale.
gdf = gpd.read_file('/home/ruud/syncNone/Shapefiles/hh_30kmaround_wgs-84_2020-03-31_shapefile_simplified.shp/landuse_a.shp')
colors = {('orchar', 'farm', 'plant_nursery', 'meadow',):'#e1f4cb',
         ('residential',): '#f1bf98',
         ('industrial','grass'):'#70756d',
         ('park',):'#9bbf80',
         ('forest', 'nature_reserve'): '#bacba9'}
# for types, color in colors.items():
#     viz.add_background(gdf[gdf['type'].isin(types)], color=color, alpha=0.3)

base_path = 'data/hamburg/osmaxx/' + ('simplified/', 'detailed/')[is_detailed]
# viz.add_background_fromfile(base_path + 'road_l.shp', color='#290022', alpha=0.4)
# viz.add_background_fromfile(base_path + 'railway_l.shp', color='#330306', alpha=0.2)
viz.add_background_fromfile(base_path + 'water_a.shp', color='lightblue', alpha=0.8)

# viz.add_background_fromfile(base_path + 'road_l.shp', color='grey', alpha=1, linewidth=0.1)
# viz.add_background_fromfile(base_path + 'railway_l.shp', color='grey', alpha=0.7, linewidth=0.1)
# viz.add_background_fromfile(base_path + 'water_a.shp', color='black', alpha=0.22)


# Content: 
viz.add_voronoi('duration', 0.05, alpha=0.3)
viz.add_lines(alpha=1, minimum_width=0.7)
viz.add_startpoint(alpha=1, color='blue', markersize=90)
viz.add_endpoints(marker='o', color='green', markersize=50)
viz.add_quiver(cmap='brg') #cmap='RdYlGn_r',
viz.showfig(0.1)

# %% Save file.

from pathlib import Path
folderpath = 'output/temp/' #with trailing /
filename = 'trippy' #no extension

Path(folderpath).mkdir(parents=True, exist_ok=True)
sizes = {'small': 400, 'medium': 2000, 'large': 5000}
for size, pix in sizes.items():
    viz.savefig(f'{folderpath}{filename}_{size}.png', minwidth=pix, minheight=pix)
viz.savefig(f'{folderpath}{filename}.svg')