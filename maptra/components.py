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
from geopy.distance import great_circle    
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
    
    @classmethod
    def from_dict(cls, latlondict):
        """Create instance from a dictionary with keys 'lat' and ('lon' or 'lng')."""
        try:
            return cls((latlondict['lat'], latlondict['lon']))
        except KeyError:
            return cls((latlondict['lat'], latlondict['lng']))

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


class Directions1:
    """
    Class of objects that contain information about directions from one to another location on Earth.
    Information belonging to directions:
    start location, end location   -->   directions    -->   duration, distance, route, steps
    Internally, only the locations (Location objects) and directions (nested dist-
    list object) are saved.
    The others (duration, distance, route, steps) are looked up and created when needed.
    To define/initialize the directions object, the 2 locations must be supplied.
    They are only passed to google maps once (to get the directions), when this
    information about the directions is needed.
    
    Terminology: 
        * route: list of points from start to end location
        * steps: route, split up into steps; each step having a single travel 
            mode. List of {'mode': str, 'route': []}-dictionaries.
    """
        
 
  
              
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


    
class Step1:
    #https://developers.google.com/maps/documentation/directions/intro
    
    TRAVEL_MODE_GROUPS = {
        ('RAIL', 'METRO_RAIL', 'SUBWAY', 'TRAM', 'MONORAIL', 'HEAVY_RAIL', 'COMMUTER_TRAIN'): 'LOCAL_RAIL', 
        ('LONG_DISTANCE_TRAIN', ): 'LONG_DISTANCE_RAIL',
        ('HIGH_SPEED_TRAIN', ): 'HIGH_SPEED_RAIL',
        ('BUS', 'INTERCITY_BUS', 'TROLLEYBUS', 'SHARE_TAXI'): 'ROAD', 
        ('FERRY', ): 'WATER',
        ('CABLE_CAR', 'GONDOLA_LIFT', 'FUNICULAR'): 'STEEP',
        ('OTHER', ): 'OTHER'}
    
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
        
    
class Movement:
    """
    Base class for the Directions and the Step class.
    Cannot be used on its own.
    Child classes must implement following properties (possibly as attribute):
        ._origin, ._start, ._end (Locations)
        ._start_distance, ._start_duration (floats)
        .bbox()
        .api_result()
        .route()
    Additionally, child classes can override any properties/methods of this class.
    Many methods have parameter 'cumul', which is used to choose between calcu-
    lating from the start (cumul == False) or the origin (cumul == True, default),
    with the start being the start point of the movement, and origin the start
    of all preceding movements. Distinction is only relevant for Steps, not for
    Directions.        
    """
    
    def start(self, cumul:bool=True) -> Location:
        """Return start Location."""
        return self._origin if cumul else self._start

    def end(self) -> Location:
        """Return end Location."""
        return self._end
    
    def distance(self, cumul:bool=True, fraction=1) -> float:
        """Distance of this movement over-the-road, in meters. 'fraction' ==
        0 (distance at start), 1 (at end), 0..1 (in between)."""
        return (self._start_distance if cumul else 0) + \
            fraction * self.api_result()['distance']['value']
            
    def duration(self, cumul:bool=True, fraction=1) -> float:
        """Duration of this movement over-the-road, in seconds. 'fraction' ==
        0 (duration at start), 1 (at end), 0..1 (in between)."""
        return (self._start_duration if cumul else 0) + \
            fraction * self.api_result()['duration']['value']
    
    def on_route(self, coords, max_dist:float=10) -> float:
        """If point with 'coords' lies on route, return fraction (0..1) between
        start and end it's found at. Fraction calculated from aerial distance to
        both points. 'Lie on route' defined as: one of route points lies in square 
        with half-sidelength 'max_side' around the point. Return False otherwise.
        Fraction only usable if movement is single step."""
        #Fast check: inside bounding box.
        bbox = self.bbox()
        if not (bbox[0] <= coords[0] <= bbox[2]
            and bbox[1] <= coords[1] <= bbox[3]):
            return False
        #Slower check: within lat-lon rect around point.
        deltalatlim = np.rad2deg(max_dist / 6356000)
        deltalonlim = deltalatlim / np.cos(np.deg2rad(coords[0]))
        for p in self.route():
            if not (-deltalatlim < coords[0] - p[0] < deltalatlim
                and -deltalonlim < coords[1] - p[1] < deltalonlim):
                continue
            #Found point. Return fraction.
            dist_to_start = great_circle(coords, self.start(False).coords).m
            dist_to_end = great_circle(coords, self.end().coords).m
            return dist_to_start / (dist_to_start + dist_to_end)
        return False
            
    def crow_distance(self, cumul:bool=True) -> float:
        """Return distance as-the-crow-flies, in meters."""
        return self.start(cumul).ll.distanceTo(self.end().ll)

    def crow_bearing(self, cumul:bool=True) -> float:
        """Return (initial) bearing as-the-crow flies, in degrees."""
        return self.start(cumul).ll.initialBearingTo(self.end().ll)
    
    def crow_speed(self, cumul:bool=True) -> float:
        """How fast the movement is increasing the (air-)distance from start,
        in relation to duration (in meters per second)."""
        if (dura := self.duration(cumul)) is not None and dura > 0:
            return self.crow_distance(cumul) / dura
        
    def distance_factor(self, cumul:bool=True) -> float:
        """How fast the movement is increasing the (air-)distance from start,
        as fraction of actual (over-the-road-)distance."""
        if (road := self.distance(cumul)) is not None:
            if (crow := self.crow_distance(cumul)) == 0:
                return 1 #if distance by air == 0, start and end must be same, so no distance.
            else:
                return crow / road
            
    def end_durationcorrected(self, crow_speed:float, cumul:bool=True) -> Location:
        """Where end location would be, if movement had provided (instead of 
        actual) 'crow speed', while keeping its duration. Location in same 
        compass direction (bearing) as the original."""
        corrected_crow_distance = crow_speed * self.duration(cumul)
        strt = self.start(cumul).ll
        return Location.from_latlon(strt.destination(corrected_crow_distance, strt.initialBearingTo(self.end().ll)))
  
    def end_distancecorrected(self, distance_factor:float, cumul:bool=True) -> Location:
        """Where end location would be, if movement had provided (instead of
        actual) 'distance factor', while keeping its (over-the-road-)distance."""
        corrected_crow_distance = distance_factor * self.distance(cumul)
        strt = self.start(cumul).ll
        return Location.from_latlon(strt.destination(corrected_crow_distance, strt.ll.initialBearingTo(self.end().ll)))

    
class Step(Movement):
    def __init__(self, api_result_step:str, start_duration:float, start_distance:float,
                 origin: Location):
        self._api_result_step = api_result_step
        self._start = Location.from_dict(api_result_step['start_location'])
        self._end = Location.from_dict(api_result_step['end_location'])
        self._origin = origin #initial start point, i.e., of first movement preceding this one.
        self._start_duration = start_duration
        self._start_distance = start_distance
        lats, lons = zip(*self.route())
        self._bbox = (min(lats), min(lons), max(lats), max(lons)) #TODO: verify order

    def api_result(self) -> str:
        """Return the unaltered api-result of this step."""
        return self._api_result_step
    
    def travel_mode(self) -> str:
        """Return one of {'WALKING', 'DRIVING', 'BICYCLING', 'TRANSIT'}."""
        return self.api_result()['travel_mode']
    
    def vehicle_type(self) -> str:
        """If travel_mode == 'TRANSIT', return one of {RAIL, METRO_RAIL, SUBWAY, TRAM,
        MONORAIL, HEAVY_RAIL, COMMUTER_TRAIN, HIGH_SPEED_TRAIN, LONG_DISTANCE_TRAIN,
        BUS, INTERCITY_BUS, TROLLEYBUS, SHARE_TAXI, FERRY, CABLE_CAR, GONDOLA_LIFT,
        FUNICULAR, OTHER}. Return None otherwise."""
        if self.travel_mode() == 'TRANSIT':
            return self.api_result()['transit_details']['line']['vehicle']['type'] 
    
    def carrier(self) -> str:
        """Return travel_mode (if it is one of {'WALKING', 'BICYCLING', 'DRIVING'}) 
        or, if it is 'TRANSIT', return vehicle_type.""" 
        if (travel_mode := self.travel_mode()) != 'TRANSIT':
            return travel_mode
        else:
            return self.vehicle_type()
    
    def extend_route(self, coords) -> None:
        self._routeend = coords
    
    def route(self) -> List[Tuple]:
        """Return route of movement as list of (lat, lon)-tuples."""
        coordlist = [(p['lat'], p['lng']) for p in 
                     decode_polyline(self.api_result()['polyline']['points'])]
        try:
            return coordlist + [self._routeend]
        except AttributeError:
            return coordlist
        
    def bbox(self) -> Tuple[float]:
        """Return bounding box of points in route."""
        return self._bbox
    
class Directions(Movement):
    
    _gmaps = None
    @classmethod
    def set_gmaps_client(cls, client) -> None:
        cls._gmaps = client
        
    def __init__(self, start:Location, end:Location, **gmapsparameters):
        self._full_api_result = None #Finding directions: None: to-try, []: failed, [...]: success
        self._start = self._origin = start
        self._end = end
        self._gmapsparameters = gmapsparameters
        self._start_duration = 0
        self._start_distance = 0

    def mode(self) -> str: #Overrides .mode of Movement class
         return self._gmapsparameters['mode'].upper()

    def travel_modes(self) -> Set[str]:
        """Return set of all unique travel modes in these directions."""
        return set([s.travel_mode() for s in self.steps()])
    
    def vehicle_types(self) -> Set[str]:
        """Return set of all unique vehicle types in these directions."""
        return set([s.vehicle_type() for s in self.steps()])
    
    def carriers(self) -> Set[str]:
        """Return set of all unique carriers in these directions."""
        return set([s.carrier() for s in self.steps()])
        
    def spoof(self, spoof:bool=True):
        """Make up random api-result with a few steps, so that no api-calls need 
        to be made. (debugging purposes only) To undo, set ._full_api_result = None."""
        if not spoof: 
            self._full_api_result = None
            return
        lldict = lambda coords: {k: v for k, v in zip(('lat', 'lng'), coords)}
        start_coords = np.array(self.start().coords)
        end_coords = np.array(self.end().coords)
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
        self._full_api_result = [
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
    
    def __get_full_api_result(self) -> str:
        """Information about the directions, as object returned by google api."""
        if self._full_api_result is None:    #No attempt made to find directions yet.
            if self._gmaps is None:
                raise ValueError("A gmaps client is needed; set with 'set_gmaps_client'.")
            self._full_api_result = self._gmaps.directions(self.start().coords, 
                                    self.end().coords, **self._gmapsparameters)  # Try to find directions.
        if not self._full_api_result:        #Attempt was made, but unsuccessful.
            return None
        else:                                #Attempt was made and successful.
            return self._full_api_result
        
    def api_result(self) -> str:    
        return self.__get_full_api_result()[0]['legs'][0]
 
    def bbox(self) -> Tuple[float]:
        """Return bounding box of points in route."""
        bounds = self.__get_full_api_result()[0]['bounds']
        return (bounds['southwest']['lat'], bounds['southwest']['lng'], 
                bounds['northeast']['lat'], bounds['northeast']['lng'])
    
    def route(self) -> List[Tuple]:
        """Return route of movement as list of (lat, lon)-tuples."""
        coordlist = [p for s in self.steps() for p in s.route()]
        if len(coordlist) > 1: #don't return route if it has only one point
            return coordlist
        else:
            return [self.start().coords, self.start().coords]
       

    @memoize_immutable
    def steps(self) -> List[Step]:
        """Parse api_result and return list of Step-objects contained within."""
        ar = self.api_result()
        dura_cum = dist_cum = 0  #duration since start of route, EXCL current step.
        steps = []
        for ar_step in ar['steps']:
            try: #If known, calculate dura_cum from departure times, in order to possible include waiting time.
                dura_cum = ar_step['transit_details']['departure_time']['value'] \
                              - ar['departure_time']['value']
            except (KeyError, TypeError):
                pass
            steps.append(Step(ar_step, dist_cum, dura_cum, self.start()))
            dura_cum += ar_step['duration']['value']
            dist_cum += ar_step['distance']['value']
        #...then, make sure the entire route (across steps) is gapless...   
        for s0, s1 in zip(steps[:-1], steps[1:]):
            if not np.isclose(s0.end().coords, strt:=s1.start(False).coords, atol=1e-5).all():
                s0.extend_route(strt)
        #...finally: only steps with actual routes.
        return [s for s in steps if len(s.route())>1]