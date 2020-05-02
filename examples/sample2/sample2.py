#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:56:11 2020

@author: Ruud Wijtvliet, rwijtvliet@gmail.com
"""

from maptra import Map, CreateLocations, Visualization, Location

Map.set_gmaps_api_key("your key here") #Put in your own api key string.

m = Map(Location([53.563, 9.928]), 'transit')
locas = CreateLocations.on_hexagonal_grid(m.start, 1000, [4000])
m.add_locations(locas)

viz = Visualization(m)
folder = 'data/hamburg/osmaxx/simplified'
viz.add_background_fromfile(f'{folder}/road_l.shp', color='#290022', alpha=0.8)
viz.add_background_fromfile(f'{folder}/railway_l.shp', color='#330306', alpha=0.2)
viz.add_background_fromfile(f'{folder}/water_a.shp', color='lightblue', alpha=0.8)
                            
viz.add_lines(minimum_width=1, alpha=1)
viz.add_startpoint()
viz.add_endpoints()
viz.showfig(0.5)

viz.savefig(f'map_transit.png', minwidth=800, minheight=800)
