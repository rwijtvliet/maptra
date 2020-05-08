#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:59:09 2020

@author: Ruud Wijtvliet, rwijtvliet@gmail.com

Playing around with overpass to see, which queries return the best results.
In parallel, some things are checked visually on https://overpass-turbo.eu/
"""

import overpy

api = overpy.Overpass()
result1 = api.query('node["public_transport"="station"](53.5, 9.8, 53.7, 10.1);out;')
result2 = api.query('node["highway"="bus_stop"](53.5, 9.8, 53.7, 10.1);out;')


#%%

import pandas as pd
from collections import Counter
import numpy as np
import json


def transit_points(query, bbox):
    if rail_or_bus == 'rail':
        query = '"public_transport"="station"'
    else:
        query = '"highway"="bus_stop"'
        query = '"public_transport"="platform"]["bus"="yes"'
    result = api.query(f'node[{query}]{bbox};out;')
    c = Counter()
    data = []
    for n in result.nodes:
        data.append(n.tags)
        c.update(n.tags.keys())
    c = {k: np.round(v/len(result.nodes),2) for k, v in c.most_common()}
    c['datapoints']=len(result.nodes)
    df = pd.DataFrame([{k:v for k, v in dic.items() if c[k]>0.25} for dic in data])
    return c, df

def poi_rail(bbox):
    return transit_points('rail', bbox)
  
def poi_bus(bbox):
    return transit_points('bus', bbox)


with open('source_verification/country-capitals.json', 'r') as f: #source: http://techslides.com/demos/country-capitals.json
    countries = json.load(f)
cities = {d['CapitalName']:(float(d['CapitalLatitude']), float(d['CapitalLongitude'])) for d in countries}
bboxes = {k:(v[0]-0.1,v[1]-0.1,v[0]+0.1,v[1]+0.1) for k, v in cities.items()}
df = pd.DataFrame([poi_bus(bbox)[0] for bbox in bboxes.values()], index=bboxes.keys())
columns = df.isnull().sum().sort_values().index.tolist()
df = df[columns]

#%% How to get best / most results for bus stops.

import pandas as pd
import overpy

# Get largest 2 cities for each country having at least 1 city with > 200k population.
df = pd.read_csv('source_verification/worldcities.csv')
keep = df[df.population > 200_000].country.unique()
df = df[df.country.isin(keep)].groupby('country').apply(lambda d: d[:2])
df = df.reset_index()
df = df.drop('level_1', axis=1)
# Classify by continent.
def continent(lat, lon):
    if lat > 36 and -26 < lon < 30:
        return 'europe'
    if lat < 36 and -26 < lon < 53:
        return 'africa'
    if lat > 40 and lon > 30:
        return 'russia'
    if lat < -10 and lon > 109:
        return 'oceania'
    if lat > 26 and lon < -26:
        return 'northamerica'
    if lat < 26 and lon < -26:
        return 'latinamerica'
    return 'asia'
df['continent'] = df.apply(lambda row: continent(row.lat, row.lng), axis=1)

api = overpy.Overpass()
df['bbox'] = df.apply(lambda row: (row.lat - 0.1, row.lng - 0.1, row.lat + 0.1, row.lng + 0.1), axis=1)
df['A'] = df.bbox.apply(lambda bbox: api.query(f'node["highway"="bus_stop"]{bbox};out;').nodes)
df['B'] = df.bbox.apply(lambda bbox: api.query(f'node["public_transport"="platform"]["bus"="yes"]{bbox};out;').nodes)
df['countA'] = df.A.apply(len)
df['countB'] = df.B.apply(len)
df['countA-B'] = df.countA - df.countB
df_bak = df

# Result:
# . For all but 3 cities, A returns at least as many results as B.
# . Upon inspection on overpass turbo, these additional results are indeed correct/useful, at least whereever checked.
# . Also, many bus stops come in small groups (one for each direction); we should somehow eliminate them.
# . However, there are still many cities, for which both return 0 or very few results.
# Conclusion:
# Use A, but be aware data might still be missing.

#%% Ways to eliminate bus stop groups.

#%% Option 1: by name. Check what's the maximum number of results that share the same name.

def none_and_max_counts(stops):
    c = Counter()
    for stop in stops:
        c.update([stop.tags.get('name', None)])
    if len(stops):
        nonefrac = np.round(c[None]/len(stops), 3)
    else:
        nonefrac = np.nan
    if c[None]:
        c.pop(None)
    try:
        maxsharedname, maxsharedcount = c.most_common(1)[0]
    except:
        maxsharedname, maxsharedcount = '', np.nan
    return nonefrac, maxsharedcount, maxsharedname
    
df[['nonamefraction','maxsharedcount', 'maxsharedname']] = df['A'].apply(none_and_max_counts).apply(pd.Series)

x = df[['city', 'country' , 'continent', 'countA', 'nonamefraction', 'maxsharedcount', 'maxsharedname']]
x.groupby('continent').describe().T

# Result:
# . In 75% of European cities, at most 8.7% of bus stops has no name. In other
#   continents, this fraction (of unnamed bus stops) is higher; 32% for North-
#   america, and >50% for cities on other continents.
# . For most cities, not too many bus stops share the same name. 
# . Therefore, for European cities, it is generally OK to use the name of a 
#   bus stop to find the bus stops close to it.
# Still, let's see if we have other options.

#%% Option 2: by distance. Find all bus stops, define a minimum distance, and
# start eliminating bus stops until no minimum distance requirements are violated.
# We will do this only for Hamburg, as the idea is the same in each setting.

import pandas as pd
import overpy
from geopy.distance import great_circle

api = overpy.Overpass()
stops = api.query('node["highway"="bus_stop"](53.5, 9.8, 53.7, 10.1);out;').nodes
df = pd.DataFrame({'stop': stops})
df['coords'] = df.stop.apply(lambda stop: (float(stop.lat), float(stop.lon)))

def tooclose_function(min_dist, latrange):
    deltalatlim = np.rad2deg(min_dist / 6356000)
    deltalonlim = deltalatlim / np.cos(np.deg2rad(np.max(np.abs(latrange))))
    def f(c1, c2):
        if abs(c1[0]-c2[0]) > deltalatlim:
            return False
        if abs(c1[1]-c2[1]) > deltalonlim:
            return False
        if (great_circle(c1, c2).m) > min_dist:
            return False
        return True
    return f

tooclose = tooclose_function(200, (53.5, 53.7))
matrix = np.zeros((len(df), len(df)), bool)
for i1, c1 in enumerate(df.coords):
    for i2, c2 in enumerate(df.coords):
        if i2 < i1: 
            matrix[i1, i2] = matrix[i2, i1]
        elif i2 > i1:
            matrix[i1, i2] = tooclose(c1, c2)

while True:
    conflicts = sum(matrix>0)
    worst = max(conflicts)
    if worst == 0:
        break
    idx = np.where(conflicts == worst)[0][0]
    #delete conflicting nodes.
    df = df.drop(index=df.index[idx])
    matrix = np.delete(matrix, idx, axis=0)
    matrix = np.delete(matrix, idx, axis=1)