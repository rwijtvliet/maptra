#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:30:56 2020

@author: Ruud Wijtvliet, rwijtvliet@gmail.com

Script to test a way to reduce the number of google maps api calls,
by seeing if the time to intermediate points on a long route can be 
calculated from that longer route. 
"""

from maptra import Directions, Location
from itertools import accumulate
from typing import List, Tuple
import googlemaps
from datetime import datetime
from googlemaps.convert import decode_polyline

with open('apikey.txt') as f:
    apikey = f.read().strip()
    client = googlemaps.Client(key=apikey)
    Location.set_gmaps_client(client)
    Directions.set_gmaps_client(client)
                               
d = Directions(Location.from_address('Stresemannstrasse 326, Hamburg'), 
               Location.from_address('berlin kitkat club'), mode='transit',
               departure_time=datetime(2020, 5, 20, 12))

res = d.api_result()
steps = d.steps()


#%%

def cums(api_result) -> Tuple[List, List]:
    result = api_result[0]['legs'][0]
    # For each step, get value for cumulative duration and distance at start of that step. 
    dura_cum, duration_cum = 0, []  #duration since start of route, EXCL current step.
    dist_cum, distance_cum = 0, []
    for step in result['steps']:
        #If initial departure time, and departure time of this step, are known:
        #calculate dura_cum from this. Otherwise, take previously calculated one. 
        try:
            dura_cum = step['transit_details']['departure_time']['value'] - result['departure_time']['value']
        except (KeyError, TypeError):
            pass
        duration_cum.append(dura_cum)
        distance_cum.append(dist_cum)
        dura_cum += step['duration']['value']
        dist_cum += step['distance']['value']
    return distance_cum, duration_cum


from geopy.distance import great_circle    
class Step():
    #https://developers.google.com/maps/documentation/directions/intro
    
    TRAVEL_MODE_GROUPS = {
        ('RAIL', 'METRO_RAIL', 'SUBWAY', 'TRAM', 'MONORAIL', 'HEAVY_RAIL', 'COMMUTER_TRAIN'): 'LOCAL_RAIL', 
        ('HIGH_SPEED_TRAIN', ): 'HIGH_SPEED_TRAIN',
        ('LONG_DISTANCE_TRAIN', ): 'LONG_DISTANCE_TRAIN',
        ('BUS', 'INTERCITY_BUS', 'TROLLEYBUS', 'SHARE_TAXI'): 'ROAD', 
        ('FERRY', ): 'WATER',
        ('CABLE_CAR', 'GONDOLA_LIFT', 'FUNICULAR'): 'STEEP',
        ('OTHER', ): 'OTHER'}
    
    def __init__(self, api_result_step, distance_at_start, duration_at_start):
        self._api_result_step = api_result_step
        self._distance_at_start = distance_at_start
        self._duration_at_start = duration_at_start
        lats, lons = zip(*self.route())
        self._bbox = (min(lats), min(lons), max(lats), max(lons)) #TODO: verify order
    @property
    def start_coords(self):
        coords = self._api_result_step['start_location']
        return (coords['lat'], coords['lng'])
    @property
    def end_coords(self):
        coords = self._api_result_step['end_location']
        return (coords['lat'], coords['lng'])
    @property
    def distance(self):
        return self._api_result_step['distance']['value']
    @property
    def distance_at_start(self):
        return self._distance_at_start
    @property
    def distance_at_end(self):
        return self.distance_at_start + self.distance
    @property
    def duration(self):
        return self._api_result_step['duration']['value']
    @property
    def duration_at_start(self):
        return self._duration_at_start
    @property
    def duration_at_end(self):
        return self.duration_at_start + self.duration
    @property
    def mode(self):
        """Return one of {'WALKING', 'DRIVING', 'BICYCLING', 'TRANSIT'}."""
        return self._api_result_step['travel_mode']
    @property
    def vehicle_type(self):
        """If mode == 'TRANSIT', return one of {RAIL, METRO_RAIL, SUBWAY, TRAM,
        MONORAIL, HEAVY_RAIL, COMMUTER_TRAIN, HIGH_SPEED_TRAIN, LONG_DISTANCE_TRAIN,
        BUS, INTERCITY_BUS, TROLLEYBUS, SHARE_TAXI, FERRY, CABLE_CAR, GONDOLA_LIFT,
        FUNICULAR, OTHER}. Return None otherwise."""
        if self.mode == 'TRANSIT':
            return self._api_result_step['transit_details']['line']['vehicle']['type'] 
    @property
    def means(self):
        """Return mode or a reduced version of the vehicle type. Return one of
        {WALKING, DRIVING, BICYCLING, LOCAL_RAIL, LONG_DISTANCE_RAIL, HIGH_SPEED_RAIL,
        ROAD, WATER, STEEP, OTHER}"""
        if (m := self.mode) != 'TRANSIT':
            return m
        vt = self.vehicle_type
        for group, name in TRAVEL_MODE_GROUPS.items():
            if vt in group:
                return name
        return 'OTHER'
    def set_routeend(self, coords):
        self._routeend = coords
    def route(self):
        coordlist = [(p['lat'], p['lng']) for p in 
                     decode_polyline(self._api_result_step['polyline']['points'])]
        try:
            coordlist.append(self._routeend)
        except AttributeError:
            pass
        return coordlist
    def on_route(self, coords, max_dist:float=10):
        """If point with 'coords' lies on route, return fraction (0..1) between
        start and end it's found at. Fraction calculated from aerial distance to
        both points. 'Lie on route' defined as: one of route points lies in square 
        with half-sidelength 'max_side' around the point. Return False otherwise."""
        #Fast check: inside bounding box.
        if not (self._bbox[0] <= coords[0] <= self._bbox[2]
            and self._bbox[1] <= coords[1] <= self._bbox[3]):
            return False
        #Slower check: within lat-lon rect around point.
        deltalatlim = np.rad2deg(max_dist / 6356000)
        deltalonlim = deltalatlim / np.cos(np.deg2rad(coords[0]))
        for p in self.route():
            if not (-deltalatlim < coords[0] - p[0] < deltalatlim
                and -deltalonlim < coords[1] - p[1] < deltalonlim):
                continue
            #Found point. Return fraction.
            dist_to_start = great_circle(coords, self.start_coords).m
            dist_to_end = great_circle(coords, self.end_coords).m
            return dist_to_start / (dist_to_start + dist_to_end)
        return False
        
def steps(api_result) -> List[Step]:
    """Parse api_result and return list of Step-objects contained within."""
    result = api_result()[0]['legs'][0]
    dura_cum = dist_cum = 0  #duration since start of route, EXCL current step.
    steps = []
    for step in result['steps']:
        #If initial departure time, and departure time of this step, are known:
        #calculate dura_cum from this. Otherwise, take previously calculated one. 
        try:
            dura_cum = step['transit_details']['departure_time']['value'] - result['departure_time']['value']
        except (KeyError, TypeError):
            pass
        steps.append(Step(step, dist_cum, dura_cum))
        dura_cum += step['duration']['value']
        dist_cum += step['distance']['value']
    
    #Make sure the entire route (across steps) is gapless.    
    for s0, s1 in zip(steps[:-1], steps[1:]):
        if not np.isclose(s0.end_coords, s1.start_coords, atol=1e-5).all():
            s0.set_routeend(s1.start_coords)
    return steps
    

ss = steps(d.api_result)
        
        
#%%

# Calculate the sublocations, and their characteristics.

def sub_directions(main_directions) -> List[Directions]:
    result = main_directions.api_result[0]['legs'][0]
    directions_list = []
    for step in result['steps']:
        end = Location((step['end_location']['lat'], step['end_location']['lng'])
        d = Directions(main_directions.start, end) 