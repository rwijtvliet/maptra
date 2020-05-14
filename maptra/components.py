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
    Parent class for the Directions, Step, and Hop classes.
    
    In order to use without overriding any methods or attributes, child classes
    must implement following methods/attributes:
        ._start, ._routestart, ._end (Locations)
        ._prior_distance, ._prior_duration (floats)
        .bbox
        .route
        .api_result()

    Additionally, child classes can override any properties/methods of this class.
    """
    
    @property
    def start(self) -> Location:
        """Return start Location."""
        return self._start
    
    @property
    def routestart(self) -> Location:
        """Return Location at start of the section for which a route is available."""
        return self._routestart
    
    @property
    def end(self) -> Location:
        """Return end Location."""
        return self._end
    
    
    # @property
    # def distance(self) -> float:
    #     """Cumulative distance over-the-road at end of this movement, in meters."""
    #     return self._prior_distance + self.api_result()['distance']['value']

    def get_distance(self, cumul:bool=True, fraction=1) -> float:
        """Distance of this movement over-the-road, in meters. 'fraction' ==
        0 (distance at start), 1 (at end), 0..1 (in between), linear interp."""
        return (self._prior_distance if cumul else 0) + \
            fraction * self.api_result()['distance']['value']
    distance = property(get_distance)
    # @property
    # def duration(self) -> float:
    #     """Cumulative duration at the end of this movement, in seconds."""
    #     return self._prior_duration + self.api_result()['duration']['value']
     
    def get_duration(self, cumul:bool=True, fraction=1) -> float:
        """Duration of this movement over-the-road, in seconds. 'fraction' ==
        0 (duration at start), 1 (at end), 0..1 (in between), linear interp."""
        return (self._prior_duration if cumul else 0) + \
            fraction * self.api_result()['duration']['value']
    duration = property(get_duration)
            
    def __start(self, cumul:bool=True) -> Location:
        return self.start if cumul else self.routestart
    
    def get_crow_distance(self, cumul:bool=True) -> float:
        """Distance as-the-crow-flies, in meters."""
        return self.__start(cumul).ll.distanceTo(self.end.ll)
    crow_distance = property(get_crow_distance)

    def get_crow_bearing(self, cumul:bool=True) -> float:
        """Bearing as-the-crow-flies during movement, in degrees."""
        return self.__start(cumul).ll.initialBearingTo(self.end.ll)
    crow_bearing = property(get_crow_bearing)
    
    @property
    def crow_speed(self) -> float:
        """Speed as-the-crow-flies from start to end of this movement, in meters per second."""
        if (dura := self.duration) is not None and dura > 0:
            return self.crow_distance / dura    
        
    @property  
    def distance_factor(self) -> float:
        """Ratio between as-the-crow-flies distance and over-the-road distance, 
        from origin to end of this movement."""
        if (road := self.distance) is not None:
            if (crow := self.crow_distance) == 0:
                return 1 #if distance by air == 0, start and end must be same, so no distance.
            else:
                return crow / road          
    
    def on_route(self, coords, max_dist:float=10) -> Union[bool, Tuple[float]]:
        """If point with 'coords' lies on route, return estimate for cumulative
        distance and duration when reaching the point. Return False otherwise."""
        #Fast check: inside bounding box.
        bbox = self.bbox
        if not (bbox[0] <= coords[0] <= bbox[2]
            and bbox[1] <= coords[1] <= bbox[3]):
            return False
        #Slower check: within lat-lon rect around point.
        deltalatlim = np.rad2deg(max_dist / 6356000)
        deltalonlim = deltalatlim / np.cos(np.deg2rad(coords[0]))
        for p in self.route:
            if (-deltalatlim < coords[0] - p[0] < deltalatlim
                and -deltalonlim < coords[1] - p[1] < deltalonlim):
                #Found point. Return values.
                dist_to_rstart = great_circle(coords, self.routestart.coords).m
                dist_to_end = great_circle(coords, self.end.coords).m
                f = dist_to_rstart / (dist_to_rstart + dist_to_end)
                return (self.get_distance(True, f), self.get_duration(True, f))
        return False
    
    def end_durationcorrected(self, crow_speed:float) -> Location:
        """Where end location would be, if cumulative movement had given
        (instead of actual) 'crow speed', while keeping its duration. 
        Location in same compass direction (bearing) as the original."""
        dist, bear = crow_speed * self.duration, self.crow_bearing
        return Location.from_latlon(self.start.ll.destination(dist, bear))
  
    def end_distancecorrected(self, distance_factor:float) -> Location:
        """Where end location would be, if cumularive movement had given
        (instead of actual) 'distance factor', while keeping its (over-the-road-)
        distance. Location in same compass direction (bearing) as original."""
        dist, bear = distance_factor * self.distance, self.crow_bearing
        return Location.from_latlon(self.start.ll.destination(dist, bear))  


class Hop(Movement):
    """
    Type of Movement that has very limited information about getting from start to end:
    * duration, distance - across all transportation means;
    * no route.
    """
    
    def __init__(self, start:Location, end:Location, distance, duration):
        self._start = start
        self._routestart = None
        self._end = end
        self._distance = distance
        self._duration = duration
    
    # def get_distance(self, cumul:bool=True, fraction=1) -> float:
    #     """Distance of this movement over-the-road, in meters. 'fraction' ==
    #     0 (distance at start), 1 (at end), 0..1 (in between), linear interp."""
    #     return fraction * self._distance
            
    @property
    def distance(self) -> float:
        return self._distance
    
    @property
    def duration(self) -> float:
        return self._duration
    
    @property
    def on_route(self, coords, max_dist) -> bool:
        return False
    
    
class Step(Movement):
    """
    Type of Movement that has partial information about getting from start to end:
    * duration, distance - across all transportation means;
    * route - for SINGLE transportation means (i.e., last one necessary to reach end).
    """
    
    def __init__(self, api_result_step:str, prior_duration:float, prior_distance:float,
                 start:Location):
        self._api_result_step = api_result_step
        self._start = start
        self._routestart = Location.from_dict(api_result_step['start_location'])
        self._end = Location.from_dict(api_result_step['end_location'])
        self._prior_duration = prior_duration
        self._prior_distance = prior_distance
        lats, lons = zip(*self.route)
        self._bbox = (min(lats), min(lons), max(lats), max(lons)) #TODO: verify order

    def api_result(self) -> str:
        """Return the unaltered api-result of this step."""
        return self._api_result_step
    
    @property
    def travel_mode(self) -> str:
        """Return one of {'WALKING', 'DRIVING', 'BICYCLING', 'TRANSIT'}."""
        return self.api_result()['travel_mode']
    @property
    def vehicle_type(self) -> str:
        """If travel_mode == 'TRANSIT', return one of {RAIL, METRO_RAIL, SUBWAY, TRAM,
        MONORAIL, HEAVY_RAIL, COMMUTER_TRAIN, HIGH_SPEED_TRAIN, LONG_DISTANCE_TRAIN,
        BUS, INTERCITY_BUS, TROLLEYBUS, SHARE_TAXI, FERRY, CABLE_CAR, GONDOLA_LIFT,
        FUNICULAR, OTHER}. Return None otherwise."""
        if self.travel_mode == 'TRANSIT':
            return self.api_result()['transit_details']['line']['vehicle']['type'] 
    @property
    def carrier(self) -> str:
        """Return travel_mode (if it is one of {'WALKING', 'BICYCLING', 'DRIVING'}) 
        or, if it is 'TRANSIT', return vehicle_type.""" 
        if (travel_mode := self.travel_mode) != 'TRANSIT':
            return travel_mode
        else:
            return self.vehicle_type
        
    @property
    def bbox(self) -> Tuple[float]:
        """Bounding box of points on route."""
        return self._bbox
    
    def extend_route(self, coords) -> None:
        self._routeend = coords
    
    @property
    def route(self) -> List[Tuple]:
        """Route of this movement as list of (lat, lon)-tuples."""
        coordlist = [(p['lat'], p['lng']) for p in 
                     decode_polyline(self.api_result()['polyline']['points'])]
        try:
            return coordlist + [self._routeend]
        except AttributeError:
            return coordlist
    

    
class Directions(Movement):
    """
    Type of Movement that has full information about getting from start to end:
    * duration, distance - across all transportation means;
    * route - across all transportation means;
    * individual Steps.
    """
    
    _gmaps = None
    @classmethod
    def set_gmaps_client(cls, client) -> None:
        cls._gmaps = client
        
    def __init__(self, start:Location, end:Location, **gmapsparameters):
        self._full_api_result = None #Finding directions: None: to-try, []: failed, [...]: success
        self._start = self._routestart = start
        self._end = end
        self._gmapsparameters = gmapsparameters
        self._prior_duration = 0
        self._prior_distance = 0

    def mode(self) -> str: #Overrides .mode of Movement class
         return self._gmapsparameters['mode'].upper()

    @property 
    def travel_modes(self) -> Set[str]:
        """Return set of all unique travel modes in these directions."""
        return set([s.travel_mode for s in self.steps()])
    @property
    def vehicle_types(self) -> Set[str]:
        """Return set of all unique vehicle types in these directions."""
        return set([s.vehicle_type for s in self.steps()])
    @property
    def carriers(self) -> Set[str]:
        """Return set of all unique carriers in these directions."""
        return set([s.carrier for s in self.steps()])
        
    def spoof(self, spoof:bool=True):
        """Make up random api-result with a few steps, so that no api-calls need 
        to be made. (debugging purposes only) To undo, set ._full_api_result = None."""
        if not spoof: 
            self._full_api_result = None
            return
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
            self._full_api_result = self._gmaps.directions(self.start.coords, 
                                    self.end.coords, **self._gmapsparameters)  # Try to find directions.
        if not self._full_api_result:        #Attempt was made, but unsuccessful.
            return None
        else:                                #Attempt was made and successful.
            return self._full_api_result
    
    def api_result(self) -> str:    
        return self.__get_full_api_result()[0]['legs'][0]
 
    @property
    def bbox(self) -> Tuple[float]:
        """Bounding box of points in route."""
        bounds = self.__get_full_api_result()[0]['bounds']
        return (bounds['southwest']['lat'], bounds['southwest']['lng'], 
                bounds['northeast']['lat'], bounds['northeast']['lng'])
    
    @property
    def route(self) -> List[Tuple]:
        """Return route of movement as list of (lat, lon)-tuples."""
        return [p for s in self.steps() for p in s.route]    
        coordlist = [p for s in self.steps() for p in s.route] #TODO:delete    
        if len(coordlist) > 1: #don't return route if it has only one point
            return coordlist
        else:
            return [self.start.coords, self.start.coords]

    #Overwrite original method in Movement class for something better.
    def on_route(self, coords, max_dist:float=10) -> Union[bool, Tuple[float]]:
        """If point with 'coords' lies on route, return estimate for cumulative
        distance and duration when reaching the point. Return False otherwise."""
        #Fast check: inside bounding box of directions.
        bbox = self.bbox
        if not (bbox[0] <= coords[0] <= bbox[2]
            and bbox[1] <= coords[1] <= bbox[3]):
            return False
        #More precise: check per step.
        for step in self.steps():
            if (result := step.on_route(coords, max_dist)):
                return result
        #Not found on any of the steps.            
        return False
    
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
            steps.append(Step(ar_step, dist_cum, dura_cum, self.start))
            dura_cum += ar_step['duration']['value']
            dist_cum += ar_step['distance']['value']
        #...then, make sure the entire route (across steps) is gapless...   
        for s0, s1 in zip(steps[:-1], steps[1:]):
            if not np.isclose(s0.end.coords, strt:=s1.routestart.coords, atol=1e-5).all():
                s0.extend_route(strt)
        #...finally: only steps with actual routes.
        return [s for s in steps if len(s.route)>1]
#%%

class Test:
    
    def __init__(self, *x):
        self._x = list(x)
    
    def get_val(self, i=0):
        return self._x[i]
    
    def set_val(self, val, i=0):
        self._x[i] = val
        
    first = property(get_val, set_val)
    second = property(lambda t: t.get_val(1), lambda t, val: t.set_val(val, 1))
    third = property(lambda t: t.get_val(2), lambda t, val: t.set_val(val, 2))
    
t = Test('one', True, 3.14, 34)
t.get_val()   #'one'
t.first       #'one'
t.get_val(1)  #True 
t.second      #True
t.get_val(2)  #3.14
t.third       #3.14
