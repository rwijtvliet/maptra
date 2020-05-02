#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:56:11 2020

@author: Ruud Wijtvliet, rwijtvliet@gmail.com
"""

from maptra import Map, CreateLocations, Visualization

Map.set_gmaps_api_key("your key here") #Put in your own api key string.

m = Map((53.563, 9.928), 'walking')
locas = CreateLocations.on_hexagonal_grid(m.start, 1000, [4000])
m.add_locations(locas)

viz = Visualization(m)
folder = 'data/hamburg/osmaxx/simplified'
viz.add_background_fromfile(f'{folder}/road_l.shp', color='#290022', alpha=0.8)
viz.add_background_fromfile(f'{folder}/railway_l.shp', color='#330306', alpha=0.2)
viz.add_background_fromfile(f'{folder}/water_a.shp', color='lightblue', alpha=0.8)
                            
viz.add_lines()
viz.add_startpoint()
viz.showfig()

viz.savefig(f'hamburg_walking.png', minwidth=800, minheight=800)
