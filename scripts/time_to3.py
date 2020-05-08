#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:47:20 2020

@author: ruud
"""

#%% Imports.

import os
os.chdir(os.path.dirname(__file__))

from rwmap.maps import Map
import datetime


with open('apikey.txt') as f:
    apikey = f.read()
if apikey[-1] == '\n': apikey = apikey[:-1]

#%% Create and save map.

m = Map("Stresemannstrasse 326, Hamburg", apikey, 
        'transit', departure_time=datetime.date.today()+datetime.timedelta(days=1, hours=12))

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
            # "Tim & Caro": "Tekenbarg 13, 21224 Klekken",
            "Valeska": "Agathenstraße 1, 20357 Hamburg",
            "Volker Luebber's": "Farmser Zoll 6, Hamburg",
            "Doris": "Rothestrasse 55, 22765 Hamburg",
            "Tina": "Hofweg 20, 22085 Hamburg",
            "Julian": "Ruhrstraße 19, 22761 Hamburg"}
m.add_locations_from_object(addressbook)

m.add_locations_from_grid(500000, 'c', 2)


#%% Save.

m.save('pickle/walking_grid800c8_6.pkl')
m.save('pickle/walking_grid500000c2.pkl')
import pickle
with open('pickle/walking_grid500000c2.pkl.dict', 'wb') as f: 
    pickle.dump(m.__dict__, f)
#%% Load.

m = Map.load_obj('pickle/transit_grid8000s8.pkl')
m = Map.load_obj('pickle/walking_grid800c16.pkl')

#%% Visualize.

from rwmap.visualize import Visualization
viz = Visualization(m)
viz.add_background('o', 'wa')
viz.add_voronoi()
viz.add_lines(2, 2)
viz.add_startpoint()
viz.add_endpoints()
viz.showfig()
viz.savefig('test2.png', minwidth=200, minheight=200)
viz.savefig('test1.png', minwidth=2000, minheight=2000)
# viz.savefig('output/transit_grid800c16_6_speedmap.png')


#%%

import geopandas as gpd
from shapely.geometry import Point

crs_lonlat = 'epsg:4326'
crs_new = 'epsg:3395'
gdf = gpd.GeoDataFrame(crs=crs_lonlat)      #all geometries are entered in this crs.
gdf['geom1'] = [Point(9,53), Point(9,54)]
gdf['geom2'] = [Point(8,63), Point(8,64)]

#Working: setting geometry and reprojecting for the first time.
gdf = gdf.set_geometry('geom1')
gdf = gdf.to_crs(crs_new)   #geom1 column is reprojected to crs_new, other column still in crs_lonlat
gdf
gdf.crs
#...so far, so good.

#Not working: setting geometry and reprojecting for second time.
gdf = gdf.set_geometry('geom2') #in crs_lonlat...
gdf.crs #... but this still says crs_new!...
gdf = gdf.to_crs(crs_new) #...so this doesn't work!
gdf

#%%

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

crs_lonlat = 'epsg:4326'
crs_new = 'epsg:3395'
gdf = gpd.GeoDataFrame()
gdf['geom1'] = gpd.GeoSeries([Point(9,53), Point(9,54)], crs=crs_lonlat)
gdf['geom2'] = gpd.GeoSeries([Point(8,63), Point(8,64)], crs=crs_lonlat)
# gdf.to_crs(crs_new) #fails with AttributeError: No geometry data set yet (expected in column 'geometry'.

#Working: setting geometry and reprojecting for the first time.
gdf = gdf.set_geometry('geom1')
gdf = gdf.to_crs(crs_new) #geometry column is reprojected to crs_new, other columns still in crs_lonlat
gdf
gdf.crs
#...so far, so good.

#Not working: setting geometry and reprojecting for second time.
gdf = gdf.set_geometry('geom2') #in crs_lonlat...
gdf.crs #... but this still says crs_new!...
gdf.to_crs(crs_new) #...so this doesn't work!

#%%

crs_lonlat = 'epsg:4326'
crs_new = 'epsg:3395'
gdf = pd.DataFrame()
gdf['geom1'] = gpd.GeoSeries([Point(9,53), Point(9,54)], crs=crs_lonlat).to_crs(crs_new)
gdf['geom2'] = gpd.GeoSeries([Point(8,63), Point(8,64)], crs=crs_lonlat).to_crs(crs_new)
# gdf.to_crs(crs_new) #fails with AttributeError: No geometry data set yet (expected in column 'geometry'.

#Working: setting geometry and reprojecting for the first time.
gdf = gdf.set_geometry('geom1')
gdf = gdf.to_crs(crs_new) #geometry column is reprojected to crs_new, other columns still in crs_lonlat
gdf
gdf.crs
#...so far, so good.

#Not working: setting geometry and reprojecting for second time.
gdf = gdf.set_geometry('geom2') #in crs_lonlat...
gdf.crs #... but this still says crs_new!...
gdf.to_crs(crs_new) #...so this doesn't work!

#%%
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
gdf = gpd.read_file('data/world/naturalearth/ne_50m_admin_0_countries_lakes/ne_50m_admin_0_countries_lakes.shp')


#%% TEST

m = Map("Stresemannstrasse 326, Hamburg", apikey, 
        'walking', departure_time=datetime.date.today()+datetime.timedelta(days=1, hours=12))
locs = ["Rome, Italy", "Bologna, Italy", "Hannover, Germany", "Barcelona, Spain"]
m.add_locations_from_object(locs)

m.tree_route