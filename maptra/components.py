#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 1 18:05:18 2020

@author: rwijtvliet@gmail.com
"""

from maptra.memoize import memoize_mutable, memoize_immutable

from typing import Tuple, List, Iterable, Dict, Set, Union
import random
import numpy as np
import pygeodesy.sphericalTrigonometry as st
import shapely.geometry as sg
from googlemaps.convert import decode_polyline, encode_polyline

class Location:
    """
    Class of objects that contain information about a location on Earth.
    Information belonging to point:
    address   <-->   geocode api-result   <-->   coords (lat, lon)   <-->   LatLon-object
    Internally, only the geocode-result (nested dict-list object) and the coordinates 
    (2-tuple) are stored.
    The others (address and LatLon-object) are looked-up/created when needed.
    To initialize the location, its (lat, lon) coordinates must be supplied, 
    but other initialization methods (e.g. .from_address) exist.
    If an address is used for initialisation, it's immediately geocoded to get coordinates.
    If coordinates are used, they are only (reverse) geocoded when address is wanted.
    """
    _gmaps = None

    @classmethod
    def set_gmaps_client(cls, client) -> None:
        cls._gmaps = client
    
    #Various constructors.
    
    @classmethod
    def from_address(cls, address:str):
        """Create instance from address string. String is immediately geocoded."""
        if cls._gmaps is None:
            raise ValueError("A gmaps client is needed; set with 'set_gmaps_client'.")
        api_result = cls._gmaps.geocode(address)
        if not api_result:
            raise ValueError("Geocoding failed: " + address)
        location = api_result[0]['geometry']['location']
        coords = (location['lat'], location['lng']) #coordinates of first found location
        instance = cls(coords)
        instance._api_result = api_result
        return instance
    
    @classmethod
    def from_latlon(cls, ll:st.LatLon):
        """Create instance from a pygeodesy.sphericalTrigonomitry.LatLon-object."""
        instance = cls(tuple(ll.latlon))
        return instance

    #Instance methods.
    
    def __init__(self, coords:Iterable[float]):
        self._coords = tuple(coords)
        self._api_result = None #Finding address belonging to coordinates: 
                                #None: not yet tried, []: failed, [...]: success
        self._label = None

    def __get_api_result(self):
        """Geocoded information about the location, as returned from google api.
        Returns None if reverse-geocoding failed."""
        if self._api_result is None:    #No attempt made to reverse-geocode coordinates yet.
            if self._gmaps is None:
                raise ValueError("A gmaps client is needed; set with 'set_gmaps_client'.")
            self._api_result = self._gmaps.reverse_geocode(self._coords)  # Try to find address.
        if len(self._api_result) == 0:  #Attempt was made, but unsuccessful.
            return None
        else:                           #Attempt was made and successful.
            return self._api_result

    @property
    def api_result(self):
        """Return the unaltered api-result."""
        return self.__get_api_result()

    @property
    def address(self) -> str:
        """Address of the location."""
        if (api_result := self.__get_api_result()) is not None:
            return api_result['formatted_address']  # return geolocated address instead of given address
    @property
    def coords(self) -> Tuple[float, float]:
        """Location's coordinates (lat, lon) in degrees."""
        return self._coords
    @property
    def ll(self) -> st.LatLon:
        """Instance of LatLon-object (of pygeodesy's sphericalTrigonometry 
        module) for the location."""
        return st.LatLon(*self.coords)
    @property
    @memoize_immutable
    def point(self) -> sg.Point:
        """Instance of Point-object (of shapely's geometry module) for the location."""
        return sg.Point(self.coords[::-1]) #needs (lon, lat) instead of (lat, lon) order, so swapped

    @property
    def label(self) -> str:
        """Return label for this location."""
        if self._label is None:
            return "({:.2f}, {:.2f})".format(*self.coords)
        else:
            return self._label
    
    @label.setter
    def label(self, val:str):
        """Set label for this location."""
        self._label = val
    
    @staticmethod
    def changerepr(current):
        def new(*args, **kwargs):
            return current(*args, **kwargs).replace('rwap.components', '')
        return new
    
    def __repr__(self):
        return self.__class__.__name__ + ' object at ' + hex(id(self))


class Directions:
    """
    Class of objects that contain information about directions from one to another location on Earth.
    Information belonging to directions:
    start location, end location   -->   directions    -->   duration, distance, route, steps
    Internally, only the locations (Location objects) and directions (nested dist-
    list object) are saved.
    The others (duration, distance, route, steps) are looked up and created when needed.
    To define/initialize the directions object, the 2 locations must be supplied.
    They are only passed to google maps (to get the directions) when information 
    about the directions is needed.
    
    Terminology: 
        * route: list of points from start to end location
        * steps: route, split up into steps; each step having a single travel 
            mode. List of {'mode': str, 'route': []}-dictionaries.
    """
    _gmaps = None

    @classmethod
    def set_gmaps_client(cls, client):
        cls._gmaps = client
        
    def __init__(self, start:Location, end:Location, **gmapsparameters):
        self._start = start
        self._end = end
        self._gmapsparameters = gmapsparameters
        self._api_result = None #Finding directions: 
                                #None: not yet tried, []: failed, [...]: success

    @property
    def start(self) -> Location:
        """Return start Location."""
        return self._start

    @property
    def end(self) -> Location:
        """Return end Location."""
        return self._end

    @property
    def distance_by_air(self) -> float:
        """Return distance as-the-crow-flies, in meters."""
        return self.start.ll.distanceTo(self.end.ll)

    @property
    def bearing_by_air(self) -> float:
        """Return (initial) bearing as-the-crow flies, in degrees."""
        return self.start.ll.initialBearingTo(self.end.ll)

    @property
    def mode(self) -> str:
        """Return transportation mode ('walking', 'bicycling', 'driving', or 
        'transit') to get from start to end."""
        return self._gmapsparameters['mode']
    
    def spoof_api_result(self):
        """Make up random api-result with a few steps, so that no api-calls need 
        to be made. (debugging purposes only) To undo, set ._api_result = None."""
        lldict = lambda coords: {k: v for k, v in zip(('lat', 'lng'), coords)}
        start_coords = np.array(self.start.coords)
        end_coords = np.array(self.end.coords)
        delta_coords = end_coords - start_coords
        stepcount = random.randint(3, 9)
        middle_coords = lambda i: start_coords + delta_coords * (i / (stepcount-1) + np.random.normal(0, 0.1, 2))
        points = [start_coords] + [middle_coords(i) for i in range(1, stepcount)] + [end_coords]
        steps = [[s, e] for s, e in zip(points[:-1], points[1:])]
        steps = [{'distance': {'text': '11 m', 'value': 11}, 'duration': {'text': '1 min', 'value': 11},
                  'end_location': lldict(e), 'start_location': lldict(s),
                  'polyline': {'points': encode_polyline([lldict(s), lldict(e)])},
                  'html_instructions': 'Move-it move-it', 'travel_mode': 'WALKING'} for (s, e) in steps]
        lats =  [l for s in steps for l in (s['end_location']['lat'], s['start_location']['lat'])]
        lngs =  [l for s in steps for l in (s['end_location']['lng'], s['start_location']['lng'])]        
        self._api_result = [
            {'bounds': {'northeast': {'lat': max(lats), 'lng': max(lngs)},
                        'southwest': {'lat': min(lats), 'lng': min(lngs)}},
             'copyrights': '11 monkeys',
             'legs': [
                 {'distance': {'text': '11 km', 'value': 11111 + np.random.normal(0, 1000)}, 'duration': {'text': '11 mins', 'value': 11 + np.random.normal(0,1)},
                  'end_location': steps[-1]['end_location'], 'start_location': steps[0]['start_location'],      
                  'steps': steps,
                  'traffic_speed_entry': [], 'via_waypoint': []}],
             'overview_polyline': {'points': encode_polyline([steps[0]['start_location']] + [s['end_location'] for s in steps])},
             'summary': 'Summary111', 'waypoint_order': [], 'warnings': ['Spoofed directions to avoid calling api.']}]
        
    def __get_api_result(self):
        """Information about the directions, as object returned by google api."""
        if self._api_result is None:    #No attempt made to find directions yet.
            if self._gmaps is None:
                raise ValueError("A gmaps client is needed; set with 'set_gmaps_client'.")
            self._api_result = self._gmaps.directions(self.start.coords, 
                                    self.end.coords, **self._gmapsparameters)  # Try to find directions.
        if len(self._api_result) == 0:  #Attempt was made, but unsuccessful.
            return None
        else:                           #Attempt was made and successful.
            return self._api_result
        
    @property
    def api_result(self):
        """Return the unaltered api-result."""
        return self.__get_api_result()
    
    @property
    def distance(self) -> float:
        """Distance over-the-road, in meters."""
        if (api_result := self.__get_api_result()) is not None:
            return api_result[0]['legs'][0]['distance']['value']

    @property
    def duration(self) -> float:
        """Duration of trip over-the-road, in seconds."""
        if (api_result := self.__get_api_result()) is not None:
            return api_result[0]['legs'][0]['duration']['value']
    
    """
    api_result[0]['legs'][0]['steps'] is quite different based on the transportation mode.
    Here an example when going from Hamburg to Rome.
    If mode in ['walking', 'bicycling', 'driving']:
      * All steps have the same structure, with keys 'travel_mode', 'start_location', 
        'end_location', 's', 'distance', 'duration'. (All also have 
        'html_instructions', and some have 'maneuver', but these are of no 
        interest.). All steps have the same value for the key 'travel_mode':
        namely 'WALKING', 'BICYCLING', or 'DRIVING'.
      * For walking and bicycling, there are many steps: ~1300 in this case.
        For driving, there are much less: 53 in this case. At the level of the
        individual steps, we know the distance and duration.
      * Decoding the step['polyline']['points'] for each step, and concatenating,
        we get 43_000 (driving) to ~73_000 (walking, bicycling) individual 
        (lat, lon)-points.
    If mode == 'transit':
      * Steps have the following structure: all have keys 'travel_mode', 
        'start_location', 'end_location', 'polyline', 'distance', 'duration' 
        (and 'html_instructions'), plus one more key. The value for 'travel_mode'
        is either 'WALKING' or 'TRANSIT'. 
        If 'travel_mode' == 'WALKING', there is also the (nested) key 'steps', 
        which in itself is a stucture equal to that described above. We don't need 
        to access the nested steps if we're only interested in the cumulative
        walking route; the step['polyline']['point'] contains the route in full
        resolution.
        If 'travel_mode' == 'TRANSIT', there is also the key 'transit_details', 
        which is a nested dictionary and its most important keys are 
        step['transit_details']['departure_stop']['location'],
        step['transit_details']['arrival_stop']['location'], and 
        step['transit_details']['line']['vehicle']['type'] (which can be 'BUS',
        'HEAVY_RAIL', etc.). 
      * There are very few steps: 9 in this case. Again, at the level of the 
        individual steps, we know the distance and duration.
      * Decoding the step['polyline']['points'] for each step, and concatenating
        (possibly split by vehicle type if this distinction is wanted), we get 
        ~38_000 individual (lat, lon)-points.
    The best way to reduce the number of points per route to a manageble level
    is by starting with the decoded polylines, and then reducing the number of
    points with an algorithm, e.g. Douglas-Peucker.
    """
    @property
    @memoize_immutable    #Can use memoize because route is calculated once and then doesn't change anymore (same for steps and modes)
    def route(self) -> List[Tuple]:
        """Return route of the trip, as list of (lat, lon)-Tuples. Use lowres=True
        to get one point per step; may be necessary for longer routes (due to 
        performance issues when calculating trees)."""
        if (api_result := self.__get_api_result()) is not None:
            #We don't need to look at nested steps (see self.steps), because the top-level step polyline already contains the complete route.
            route = [(p['lat'], p['lng']) for s in api_result[0]['legs'][0]['steps']
                     for p in decode_polyline(s['polyline']['points'])]
            if len(route) > 1: #don't return route if it has only one point
                return route
        return None
    
    @property
    @memoize_immutable
    def steps(self) -> List[Dict]:
        """Route of the trip, broken up into steps of a single travel mode. List
        of {'mode': str, 'route': List[Tuple]}-dictionaries. NB: the google api-
        result also has 'steps' property, which is used (and condensed) in this 
        function."""
        def getdict(step:Dict) -> Dict:
            mode = step['transit_details']['line']['vehicle']['type'] \
                if ('transit_details' in step) else step['travel_mode']
            route = [(p['lat'], p['lng']) for p in decode_polyline(step['polyline']['points'])]
            return {'mode': mode, 'route': route}
        
        api_result = self.__get_api_result()
        if api_result is not None:
            #First: one dictionary per step...
            steps_all = [getdict(step) for step in api_result[0]['legs'][0]['steps']]
            #...then: concatenate routes for steps with same mode...
            steps = []
            last_step = None
            for s in steps_all:
                if last_step is not None and last_step['mode'] == s['mode']:
                    last_step['route'] += s['route'][1:]
                else:
                    last_step = s
                    steps.append(last_step)
            #...then: ensure no gap between steps (even when switching travel_mode)...
            for s0, s1 in zip(steps[:-1], steps[1:]): 
                if  s0['route'][-1] != s1['route'][0]:
                    s0['route'].append(s1['route'][0])
            #...finally: ensure no 1-point 'routes'.
            return [s for s in steps if len(s['route'])>1]
        return []

    @property
    def modes(self) -> Set[str]:
        """Return set of all unique travel modes in these directions."""
        return set([rs['mode'] for rs in self.steps])
        
    @property
    def factor_distance_vs_air(self) -> float:
        """Ratio of distance-over-the-road to distance-by-air."""
        if self.distance is not None:
            if self.distance_by_air == 0:
                return 1 #if distance by air == 0, start and end must be same, so no distance.
            else:
                return self.distance / self.distance_by_air
    
    @property
    def distancing_speed(self) -> float:
        """How fast the (over-the-road-)route is increasing the (air-)distance 
        from the start, in meters per second."""
        if self.duration is not None and self.duration > 0:
            return self.distance_by_air / self.duration

    def corrected_end(self, av_distancing_speed: float) -> Location:
        """Where end location would be, if air-distance of all locations were 
        proportional to the time it takes to get there from the start location,
        i.e., if route for all points had an identical distancing speed. This
        location is in the same compass direction (bearing) as the original."""
        if self.distance is not None:
            corrected_distance_by_air = av_distancing_speed * self.duration
            return Location.from_latlon(self.start.ll.destination(corrected_distance_by_air, self.bearing_by_air))
        
    def __repr__(self):
        return self.__class__.__name__ + ' object at ' + hex(id(self))