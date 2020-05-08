#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:47:20 2020

@author: ruud
"""

#%% Imports.

import os
os.chdir(os.path.dirname(__file__))

from rwmap.maps import Map, CreateLocations
from rwmap.components import Location
import datetime

#%% First example:


with open('apikey.txt') as f:
    apikey = f.read().strip()
    Map.set_gmaps_api_key(apikey)

#Create.
m = Map.from_address("Stresemannstrasse 326, Hamburg", 
                     'walking', departure_time=datetime.date.today()+datetime.timedelta(days=1, hours=12))
addressbook= {
            "Do & Kaethe": "Marktstrasse 93, 20357 Hamburg",
            "Oerjan & Petra": "Stresemannstrasse 326, 22761 Hamburg",
            "Nils & Sabi": "Schulterblatt 88, 20357 Hamburg",
            "Detlef & Gabi": "Eppendorfer Weg 119, 20259 Hamburg",
            "Merle": "Schützenstraße 64, 22761 Hamburg",
            "Karin & Arno": "Andreas-Knack-Ring 22d, 22307 Hamburg",
            "Bjoern & Lucy": "Eichenstrasse 22, 20259 Hamburg",
            "Kerstin Mahnke": "Pagenfelder Str. 17, 22111 Hamburg",
            "Laura Laskos": "Hagenau 25, 22089 Hamburg", 
            "Lukas & Steffi": "Langbehnstrasse 13E, 22761 Hamburg",
            "Stefan Dunkhorst": "Jürgen-Töpfer-Straße 115, 22763 Hamburg",
            "Tim & Caro": "Tekenbarg 13, 21224 Klekken",
            "Valeska": "Agathenstraße 1, 20357 Hamburg",
            "Volker Luebbers": "Farmser Zoll 6, Hamburg",
            "Doris": "Rothestrasse 55, 22765 Hamburg",
            "Tina": "Hofweg 20, 22085 Hamburg",
            "Julian Buschenhenke": "Ruhrstraße 19, 22761 Hamburg"
            }
m.add_locations(CreateLocations.from_address_dict(addressbook))
#Save.
m.to_pickle("pickle/friends.pkl")
#Load.
m = Map.from_pickle("pickle/friends.pkl")



#%%
m = Map(Location([53.551086, 9.993682]), 'transit', departure_time=datetime.date.today()+datetime.timedelta(days=1, hours=12))
# m.add_locations(CreateLocations.on_square_grid(Location([48.14, 11.56]), 400_000, 10))
filtr = CreateLocations.geofilter()
t = datetime.datetime.now()
m.add_locations(CreateLocations.on_rectangular_grid(Location([53.5, 10]), 20_000, 450_000, 500_000, 800_000, geofilter=filtr))
print((datetime.datetime.now()-t).total_seconds())
pois_coords = {"Germany": [(52.5250839, 13.369402),
                            (53.5529259, 10.0066045),
                            (48.1402669, 11.559998),
                            (50.9432141, 6.9586017),
                            (50.10652899999999, 8.6621618)]}
for center in pois_coords["Germany"]:
    m.add_locations(CreateLocations.on_circular_grid(Location(center), 20_000, 2, geofilter=filtr))

m2 = Map.from_map(m, 'walking')
m2.to_pickle("pickle/friends_walking.pkl")

#%% Visualize.

from rwmap.visualize import Visualization
viz = Visualization(m, 'EPSG:25832')

#Background map:
show_detail = True
#   a) Background map: world/europe scale.
scale = ('50m', '10m')[show_detail]
# viz.add_background(f'data/world/naturalearth/{scale}_cultural/ne_{scale}_admin_0_countries_lakes.shp')
viz.add_background(f'data/world/naturalearth/{scale}_physical/ne_{scale}_land.shp',
                   color='grey', alpha=0.1)
### viz.add_background(f'data/eurogeographics/DATA/Countries/DE/RoadL.shp', color='red', alpha=0.2)
### viz.add_background(f'data/eurogeographics/DATA/Countries/DE/RailrdL.shp', color='green', alpha=0.2)
#   b) Background map: Hamburg scale.
base_path = 'data/hamburg/osmaxx/' + ('simplified/', 'detailed/')[show_detail]
viz.add_background(base_path + 'road_l.shp', color='#290022', alpha=0.08)
# viz.add_background(base_path + 'railway_l.shp', color='#330306', alpha=0.04)
viz.add_background(base_path + 'water_a.shp', color='lightblue', alpha=0.6)

#Content:
# viz.add_voronoi()
# viz.add_lines(minimum_width=2)
viz.add_startpoint(markersize=500, alpha=0.5)
viz.add_endpoints(False, markersize=30)
viz.add_quiver(cmap='brg') #cmap='RdYlGn_r',
viz.showfig()
viz.savefig('testwarp.png', minwidth=500, minheight=500)
# viz.savefig('test1.png', minwidth=2000, minheight=2000)
# viz.savefig('output/transit_grid800c16_6_speedmap.png')

#%% get voronoi inside country


from scipy.spatial import SphericalVoronoi
from rwmap.voronoi_utility import convert_spherical_array_to_cartesian_array as csatcs
import numpy as np



pois_strings = {"Germany": ["Berlin Central Station", 
                            "Hamburg Central Station", 
                            "Munich (München) Central Station", 
                            "Cologne (Köln) Central Station", 
                            "Frankfurt am Main Central Station"]}
pois_coords = {"Germany": [(52.5250839, 13.369402),
                           (53.5529259, 10.0066045),
                           (48.1402669, 11.559998),
                           (50.9432141, 6.9586017),
                           (50.1065290, 8.6621618)]}

xyz = csatcs(np.array([[1, 90-lat, lon] for lat, lon in pois_coords["Germany"]]), 'degrees')

sv = SphericalVoronoi(xyz) 

