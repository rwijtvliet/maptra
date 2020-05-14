#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 01:47:25 2020

@author: ruud

Update pickle to new version by manually extracting the api result and creating new
Location and Directions objects from them. 
"""

from maptra import Location, Directions, Map

m = Map.from_pickle("pickle/hamburg_walking_1000_10000.pkl")

locas = m.df.location
dirs = []
for d in m.df.directions:
    dire = Directions(d.start, d.end, **d._gmapsparameters)
    dire._full_api_result = d._api_result
    dirs.append(dire)

m2 = Map.from_map(m)
m2._df = pd.DataFrame({'location': locas, 'directions':dirs})