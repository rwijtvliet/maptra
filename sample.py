#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create an image of Hamburg, showing how to get where on foot.

@author: Ruud Wijtvliet, rwijtvliet@gmail.com
"""

# %% Create map.

from maptra import Map, CreateLocations

Map.set_gmaps_api_key("apikey") #Put in your own api key string.

# Create.
m = Map.from_address('Stresemannstrasse 320, Hamburg', 'walking') #not my actual address ;)
filtr = CreateLocations.geofilter()
locas = CreateLocations.on_circular_grid(m.start, 1_000, [10_000], geofilter=filtr)
m.add_locations(locas)

# Save. (Can be loaded at later point in time with Map.from_pickle(...).)
m.to_pickle("hamburg_walking_1000_10000.pkl") # To save on api-calls.

# %% Visualize.

from maptra import Visualization
import geopandas as gpd

viz = Visualization(m, crs='epsg:3395') #a good crs for Germany.

# Background map:
    
#   a) Background map: world/europe scale.
viz.add_background_fromfile(f'data/world/naturalearth/10m_cultural/ne_10m_admin_0_countries_lakes.shp', color='grey', alpha=0.2)

#   b) Background map: Hamburg scale.
folder = 'data/hamburg/osmaxx/simplified'
viz.add_background_fromfile(f'{folder}/road_l.shp', color='#290022', alpha=0.4)
viz.add_background_fromfile(f'{folder}/railway_l.shp', color='#330306', alpha=0.2)
viz.add_background_fromfile(f'{folder}/water_a.shp', color='lightblue', alpha=0.8)

gdf = gpd.read_file(f'{folder}/landuse_a.shp')
areacolors = {('orchar', 'farm', 'plant_nursery', 'meadow',):'#e1f4cb',
              ('residential',): '#f1bf98',
              ('industrial','grass'):'#70756d',
              ('park',):'#9bbf80',
              ('forest', 'nature_reserve'): '#bacba9'}
for types, color in areacolors.items():
    viz.add_background(gdf[gdf['type'].isin(types)], color=color, alpha=0.3)
    
    
# Content: 
    
# viz.add_voronoi('duration', 0, alpha=0.9)
viz.add_lines(alpha=1, minimum_width=0.7, color='white')
viz.add_startpoint(alpha=1, color='grey', markersize=90)
viz.add_endpoints(marker='o', color='grey')
# viz.add_quiver(cmap='brg')
viz.showfig(0.03)

# %% Save file.

viz.savefig(f'examples/hamburg_walking.png', minwidth=800, minheight=800)
viz.savefig(f'examples/hamburg_walking.svg')