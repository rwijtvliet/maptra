#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:25:42 2020

@author: ruud
"""
import os
os.chdir(os.path.dirname(__file__))
from rwmap import Map, ForestStruct
import pandas as pd

m = Map.from_pickle('pickle/debugging_copy.pkl')

m.to_pickle('pickle/debugging_copy_to_be_overwritten.pkl')

labels = m.df.location.apply(lambda x: x.label).str\
    .replace('Location on circular grid, on circle ', '')\
    .str.replace('bearing ', '').str.replace(' deg.', '')\
    .str.replace('(',"").str.replace(')','').str

cb = pd.DataFrame(labels.split(', ').tolist(), columns=['c','b'])
cb.c = cb.c.apply(float)    
cb.b = cb.b.apply(float)
keep = labels.contains('53.56, 9') \
    | labels.contains('2,30.0') \
    | labels.contains('3,0.0') \
    | labels.contains('3,20.0') \
    | labels.contains('3,40.0') \
    | labels.contains('4,0.0') \
    | labels.contains('4,15.0') \
    | labels.contains('4,30.0') \
    | labels.contains('4,45.0') \
    | labels.contains('4,')

keep = cb.c.eq(53.56) | cb.c.eq(4) & cb.b.between(55, 115)

m._df = m.df[keep]
#%%
m = Map.from_pickle('pickle/debugging_copy_reduced.pkl')



import geopandas as gpd
import shapely.geometry as sg

points = [(9.95166, 53.56601),
          (9.95171, 53.56603),
          (9.9517,  53.56601),
          (9.9517,  53.56602)]
mask = sg.Polygon.from_bounds(8,50,12,55) #all points within mask

gdf1 = GeoDataFrame(geometry=[sg.LineString(points)])
gdf2 = gpd.clip(gdf1, clipping_mask)

p1 = gdf1.geometry[0]
p2 = gdf2.geometry[0]

for o1 in p2:
    print(o1)


#%%

import geopandas as gpd
import shapely.geometry as sg

gdf1 = gpd.GeoDataFrame([{'geometry':sg.MultiLineString([[[0,0],[0,1]],[[0,1],[1,1]]]),
                         'linewidth':8, 'color':'red', 'alpha':0.3},
                        {'geometry':sg.LineString([[0,0],[2,1]]),
                         'linewidth':2, 'color':'blue', 'alpha':0.8}])
gdf1.plot(linewidth=3, color=(.7,.2,.3))

gdf1.plot(linewidth=4 ,color='red')

gdf2 = gpd.GeoDataFrame([{'geometry':sg.MultiPolygon([ [([2,0],[2,0.5],[1.5,0]),[]],[([1.5,0],[1.5,0.5],[1,0]),[]]]),
                         'linewidth':2, 'facecolor':'green', 'edgecolor':'purple'},
                        {'geometry': sg.Polygon([[2,0],[2,1],[3,0]]),
                         'linewidth':4, 'facecolor':'yellow', 'edgecolor':'orange'}])
#, alpha=gdf1['alpha'])
gdf2.plot(linewidth=gdf2['linewidth'], alpha=0.5, facecolor=gdf2['facecolor'], edgecolors=gdf2['edgecolor'])

gdf3 = gpd.GeoDataFrame([{'geometry':sg.MultiPoint([[2,0],[2,0.5],[1.5,0]]),
                         'linewidth':2, 'facecolor':'green', 'edgecolor':'purple'},
                        {'geometry': sg.Point([3,3]),
                         'linewidth':4, 'facecolor':'yellow', 'edgecolor':'orange'}])
#, alpha=gdf1['alpha'])
gdf3.plot(linewidth=gdf2['linewidth'], marker='>', color='green', edgecolors=gdf3['edgecolor'], alpha=0.3)

# %%
import shapely.geometry as sg
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely.geometry as sg
df = gpd.GeoDataFrame([{'geometry':sg.MultiPolygon([ [([2,0],[2,0.5],[1.5,0]),[]],[([1.5,0],[1.5,0.5],[1,0]),[]]]),
                         'linewidth':2, 'facecolor':'green', 'edgecolor':'purple'},
                        {'geometry': sg.Polygon([[2,0],[2,1],[3,0]]),
                         'linewidth':4, 'facecolor':'yellow', 'edgecolor':'orange'}])
df.geometry = df.geometry.buffer(-0.1)
df.plot(linewidth=df['linewidth'], alpha=0.5, facecolor=df['facecolor'], edgecolors=df['edgecolor'])
