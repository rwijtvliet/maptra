"""
Classes for movement, in various detail levels.
"""

from maptra.locations import Location
from maptra.memoize import memoize_immutable
import random
import geopandas as gpd
import numpy as np
from typing import Dict, Tuple, List, Union, Set
from geopy.distance import great_circle
from googlemaps.convert import decode_polyline, encode_polyline

# class Directions1:
#     """
#     Class of objects that contain information about directions from one to another location on Earth.
#     Information belonging to directions:
#     start location, end location   -->   directions    -->   duration, distance, route, steps
#     Internally, only the locations (Location objects) and directions (nested dist-
#     list object) are saved.
#     The others (duration, distance, route, steps) are looked up and created when needed.
#     To define/initialize the directions object, the 2 locations must be supplied.
#     They are only passed to google maps once (to get the directions), when this
#     information about the directions is needed.
    
#     Terminology: 
#         * route: list of points from start to end location
#         * steps: route, split up into steps; each step having a single travel 
#             mode. List of {'mode': str, 'route': []}-dictionaries.
#     """
        
 
  
              
#     """
#     api_result[0]['legs'][0]['steps'] is quite different based on the transportation mode.
#     Here an example when going from Hamburg to Rome.
#     If mode in ['walking', 'bicycling', 'driving']:
#       * All steps have the same structure, with keys 'travel_mode', 'start_location', 
#         'end_location', 's', 'distance', 'duration'. (All also have 
#         'html_instructions', and some have 'maneuver', but these are of no 
#         interest.). All steps have the same value for the key 'travel_mode':
#         namely 'WALKING', 'BICYCLING', or 'DRIVING'.
#       * For walking and bicycling, there are many steps: ~1300 in this case.
#         For driving, there are much less: 53 in this case. At the level of the
#         individual steps, we know the distance and duration.
#       * Decoding the step['polyline']['points'] for each step, and concatenating,
#         we get 43_000 (driving) to ~73_000 (walking, bicycling) individual 
#         (lat, lon)-points.
#     If mode == 'transit':
#       * Steps have the following structure: all have keys 'travel_mode', 
#         'start_location', 'end_location', 'polyline', 'distance', 'duration' 
#         (and 'html_instructions'), plus one more key. The value for 'travel_mode'
#         is either 'WALKING' or 'TRANSIT'. 
#         If 'travel_mode' == 'WALKING', there is also the (nested) key 'steps', 
#         which in itself is a stucture equal to that described above. We don't need 
#         to access the nested steps if we're only interested in the cumulative
#         walking route; the step['polyline']['point'] contains the route in full
#         resolution.
#         If 'travel_mode' == 'TRANSIT', there is also the key 'transit_details', 
#         which is a nested dictionary and its most important keys are 
#         step['transit_details']['departure_stop']['location'],
#         step['transit_details']['arrival_stop']['location'], and 
#         step['transit_details']['line']['vehicle']['type'] (which can be 'BUS',
#         'HEAVY_RAIL', etc.). 
#       * There are very few steps: 9 in this case. Again, at the level of the 
#         individual steps, we know the distance and duration.
#       * Decoding the step['polyline']['points'] for each step, and concatenating
#         (possibly split by vehicle type if this distinction is wanted), we get 
#         ~38_000 individual (lat, lon)-points.
#     The best way to reduce the number of points per route to a manageble level
#     is by starting with the decoded polylines, and then reducing the number of
#     points with an algorithm, e.g. Douglas-Peucker.
#     """




class _Base:
    """
    Parent class for movement.
    
    In order to use without overriding any methods or attributes, descendent 
    classes must implement following attributes or property methods (setter not
    required):
        ._start (Location)
        ._end (Location), 
        ._mode (string)
        .distance (float), 
        .duration (float)

    Additionally, child classes can override any properties/methods of this class.
    """

    @property
    def start(self) -> Location:
        """Return start Location."""
        return self._start
        
    @property
    def end(self) -> Location:
        """Return end Location."""
        return self._end

    @property
    def mode(self) -> str:
        """Return transportation mode."""
        return self._mode       
    
    @property
    def crow_distance(self) -> float:
        """Distance as-the-crow-flies, in meters."""
        return self.start.ll.distanceTo(self.end.ll)

    @property
    def crow_bearing(self) -> float:
        """Bearing as-the-crow-flies during movement, in degrees."""
        return self.start.ll.initialBearingTo(self.end.ll)
    
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


class Step(_Base):
    """
    Type of Movement with partial information about getting from start to end:
    * .distance and .duration apply to the whole movement from start;
    * All other properties (.travel_mode, .vehicletype, .carrier, .bbox, .route)
      apply to SINGLE step (i.e., last one necessary to reach end, starting at
      .routestart).
    """ 

    def __init__(self, start:Location, api_result_step:Dict, 
                 prior_duration:float, prior_distance:float):
        self._start = start
        self._api_result_step = api_result_step
        self._routestart = Location.from_dict(api_result_step['start_location'])
        self._end = Location.from_dict(api_result_step['end_location'])
        self._prior_duration = prior_duration
        self._prior_distance = prior_distance
        lats, lons = zip(*self.route)
        self._bbox = (min(lats), min(lons), max(lats), max(lons))

    def distance_to_midpoint(self, fraction:float=1) -> float:
        """Distance over-the-road, in meters. Measured from start. 'fraction' 
        == 0 (to routestart), 1 (to end), 0..1 (in between, estimated)."""
        return self._prior_distance + fraction * self._api_result_step['distance']['value']
    distance = property(
        lambda self: self.distance_to_midpoint(1),
        doc="Distance over-the-road, from start to end, in meters.")
    distance_routeonly = property(
        lambda self: self.distance_to_midpoint(1) - self.distance_to_midpoint(0),
        doc="Distance over-the-road, from routestart to end, in meters.")
    
    def duration_to_midpoint(self, fraction:float=1) -> float:
        """Duration over-the-road, in seconds. Measured from start. 'fraction' 
        == 0 (to routestart), 1 (to end), 0..1 (in between, estimated)."""
        return self._prior_duration + fraction * self._api_result_step['duration']['value']
    duration = property(
        lambda self: self.duration_to_midpoint(1),
        doc="Duration over-the-road, from start to end, in seconds.")
    duration_routeonly = property(
        lambda self: self.duration_to_midpoint(1) - self.duration_to_midpoint(0),
        doc="Duration over-the-road, from routestart to end, in seconds.")

    @property
    def routestart(self) -> Location:
        return self._routestart

    @property
    def travel_mode(self) -> str:
        """Return one of {'WALKING', 'DRIVING', 'BICYCLING', 'TRANSIT'}."""
        return self._api_result_step['travel_mode']
    
    @property
    def vehicletype(self) -> str:
        """If travel_mode == 'TRANSIT', return one of {RAIL, METRO_RAIL, SUBWAY, TRAM,
        MONORAIL, HEAVY_RAIL, COMMUTER_TRAIN, HIGH_SPEED_TRAIN, LONG_DISTANCE_TRAIN,
        BUS, INTERCITY_BUS, TROLLEYBUS, SHARE_TAXI, FERRY, CABLE_CAR, GONDOLA_LIFT,
        FUNICULAR, OTHER}. Return None otherwise."""
        if self.travel_mode == 'TRANSIT':
            return self._api_result_step['transit_details']['line']['vehicle']['type'] 
    @property
    def carrier(self) -> str:
        """Return travel_mode (if it is one of {'WALKING', 'BICYCLING', 'DRIVING'}) 
        or, if it is 'TRANSIT', return vehicletype.""" 
        if (travel_mode := self.travel_mode) != 'TRANSIT':
            return travel_mode
        else:
            return self.vehicletype
        
    def extend_route(self, coords) -> None:
        """Save additional point to add to the route."""
        self._routeend = coords
    
    @property
    def route(self) -> List[Tuple]:
        """Route of this step as list of (lat, lon)-tuples."""
        coordlist = [
            (p['lat'], p['lng']) for p in 
            decode_polyline(self._api_result_step['polyline']['points'])
            ]
        try:
            return coordlist + [self._routeend]
        except AttributeError:
            return coordlist
    
    def within_bbox(self, loca:Location) -> bool:
        """Return True if location lies within bounding box of route; False otherwise."""
        bbox = self._bbox
        coords = loca.coords
        return (bbox[0] <= coords[0] <= bbox[2] and
                bbox[1] <= coords[1] <= bbox[3])
    
    def on_route(self, loca:Location, max_dist:float=25) -> Union[bool, Dict]:
        """If location 'loca' lies on route (i.e., has a distance below 
        'max_dist' to a point on the route), return dictionary with estimate for 
        cumulative distance ('distance') and duration ('duration') when reaching 
        the location, point on route that is closest to location ('routepoint'),
        distance to routestart or end ('nearest_anchor') and vehicle (if 
        any) used to get there ('vehicletype'). Return False otherwise."""
        #Fast check: inside bounding box.
        if not self.within_bbox(loca):
            return False
        #Slower check: within lat-lon square around location.
        coords = loca.coords
        deltalatlim = np.rad2deg(max_dist / 6356000)
        deltalonlim = deltalatlim / np.cos(np.deg2rad(coords[0]))
        candidates = []
        for p in self.route:
            if (-deltalatlim < coords[0] - p[0] < deltalatlim
                and -deltalonlim < coords[1] - p[1] < deltalonlim):
                candidates.append(p)
        if not candidates:
            return False
        #Slowest check: great circle distance
        dists = np.array([great_circle(coords, p).m for p in candidates])
        i = dists.argmin()
        if dists[i] > max_dist:
            return False
        #Found point. Return values.
        dist_to_rstart = great_circle(coords, self.routestart.coords).m
        dist_to_end = great_circle(coords, self.end.coords).m
        f = dist_to_rstart / (dist_to_rstart + dist_to_end)
        return {'distance': self.distance_to_midpoint(f), 
                'duration': self.duration_to_midpoint(f),
                'stepfraction': f,
                'nearest_anchor': min([dist_to_end, dist_to_rstart]),
                'routepoint': candidates[i],
                'vehicletype': self.vehicletype}
    

class PartialStep(Step):
    """Partial Step, i.e., step (as returned by the Directions object) but with
    an arbitrary (new) end point along its route."""
    def __init__(self, step:Step, estimate:Dict):
        self.__dict__ = step.__dict__
        self._estimate = estimate
    
    def distance_to_midpoint(self, fraction:float=1) -> float:
        f = fraction * self._estimate['stepfraction'] #Fraction along entire (instead of partial) step
        return super().distance_to_midpoint(f)
    def duration_to_midpoint(self, fraction:float=1) -> float:
        f = fraction * self._estimate['stepfraction'] #Fraction along entire (instead of partial) step
        return super().duration_to_midpoint(f)
        
    @property
    def route(self):
        fullroute = super().route
        for i, p in enumerate(fullroute):
            if np.allclose(p, self._estimate['routepoint']):
                return fullroute[:i+1]
        raise ValueError(f"Cannot find point {self._estimate['routepoint']} on route of step.")
        
    def on_route(self, loca:Location, **kwargs) -> bool:
        return False #always False, because there is a better (entire) step somewhere to check for this.


class Directions(_Base):
    """
    Type of Movement with information about getting from start to end:
    * .distance and .duration apply to the whole movement from start;
    * Same goes for all other properties (.travel_modes, .vehicletypes, 
      .carriers, .bbox, .route)
    * individual Steps.
    
    Partial information (duration and distance) can be provided from externally,
    either during or after initialisation. In that case, an api-call is only 
    made if other information is needed. The .state property gives information 
    about this.
    """

    _gmaps = None
    @classmethod
    def set_gmaps_client(cls, client) -> None:
        cls._gmaps = client

    def __init__(self, start:Location, end:Location, mode:str='walking', 
                 **gmaps_kwargs):
        self._start = start
        self._end = end
        self._mode = mode
        self._gmaps_kwargs = gmaps_kwargs
        self._estimate = None
        self._spoofed = False
        self._full_api_result = None #Finding directions: None: to-try, []: failed, [...]: success

    @property
    def distance(self) -> float:
        """"Distance over-the-road, from start to end, in meters."""
        if self._full_api_result is None and self._estimate is not None:
            return self._estimate['distance']
        return self._get_full_api_result()[0]['legs'][0]['distance']['value']
    
    @property
    def duration(self) -> float:
        """Duration over-the-road, from start to end, in seconds."""
        if self._full_api_result is None and self._estimate is not None:
            return self._estimate['duration']
        return self._get_full_api_result()[0]['legs'][0]['duration']['value']
   
    def check_estimate(self, other:_Base) -> bool:
        """Check if information in 'other' can give an estimate for duration and
        distance. (Or improve on the current estimate.)"""
        if self.state == 2:
            return False # don't check estimate if maximum information already known
        estimate = other.on_route(self.end)
        if not estimate:
            return False # not on route
        if not compatibility_vehicletype_transittype(
                estimate['vehicletype'],
                getattr(self.end, 'transittype', None)):
            return False # on route, but route passes in incorrect vehicle
        if self._estimate is not None and self._estimate['nearest_anchor'] < estimate['nearest_anchor']:
            return False # estimate worse than currently present estimate
        self._estimate = estimate
        return True
    
    @property
    def state(self) -> int:
        """Decribes how much is known about getting from start to end.
        -1: no information was found. 
        0: no information was queried yet. 
        1: duration and distance only ('hop').
        2: also route, bbox, etc."""
        if self._full_api_result is not None and self._full_api_result != []:
            return 2
        if self._estimate is not None:
            return 1
        if self._full_api_result == []:
            return -1
        return 0
    
    def _get_full_api_result(self) -> str:
        """Information about the directions, as object returned by google api."""
        if self._full_api_result is None:    #No attempt made to find directions yet.
            if self._gmaps is None:
                raise ValueError("A gmaps client is needed; set with 'set_gmaps_client'.")
            self._full_api_result = self._gmaps.directions(self.start.coords, 
                self.end.coords, **{'mode': self.mode, **self._gmaps_kwargs})  # Try to find directions.
        if self._full_api_result == []:      #Attempt was made, but unsuccessful.
            raise ValueError('No directions from start to end has been found.')
        return self._full_api_result         #Attempt was made and successful.
    
    def spoof_apicall(self, do_spoof:bool=True) -> None:
        """Make up random api-result with a few steps, so that no api-calls need 
        to be made. (debugging purposes only) To undo, call with do_spoof=False."""
        if not do_spoof:
            if self._spoofed:
                self._full_api_result = None
                self._spoofed = False
            return
        #Do spoofing...
        if self._full_api_result is not None:
            return #...but only if legitimate api result doesn't exist.
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
        self._spoofed = True
    
    @memoize_immutable
    def steps(self) -> List[Step]:
        """Parse api_result and return list of Step-objects contained within."""
        if self.state == -1:
            return []
        if self.state == 1:
            i = self._estimate['source_stepindex']
            steps = self._estimate['source_directions'].steps()
            return steps[:i] + [PartialStep(steps[i], self._estimate)]
        # Use google api. 
        ar = self._get_full_api_result()[0]['legs'][0]
        dura_cum = dist_cum = 0  #duration since start of route, EXCL current step.
        steps = []
        for ar_step in ar['steps']:
            try: #If known, calculate dura_cum from departure times, in order to possible include waiting time.
                dura_cum = ar_step['transit_details']['departure_time']['value'] \
                              - ar['departure_time']['value']
            except (KeyError, TypeError):
                pass
            steps.append(Step(self.start, ar_step, dura_cum, dist_cum))
            dura_cum += ar_step['duration']['value']
            dist_cum += ar_step['distance']['value']
        #...then, make sure the entire route (across steps) is gapless...
        for s0, s1 in zip(steps[:-1], steps[1:]):
            if not np.isclose(s0.end.coords, strt:=s1.routestart.coords, atol=1e-5).all():
                s0.extend_route(strt) #(by repeating points)
        #...finally: only steps with actual routes.
        return [s for s in steps if len(s.route)>1]
    
    @property 
    def travel_modes(self) -> Set[str]:
        """Return list of all unique travel modes in these directions (in order of use)."""
        return [s.travel_mode for s in self.steps()]
    @property
    def vehicletypes(self) -> Set[str]:
        """Return list of all unique vehicle types in these directions (in order of use)."""
        return [s.vehicletype for s in self.steps()]
    @property
    def carriers(self) -> List[str]:
        """Return list of all carriers in these directions (in order of use)."""
        return [s.carrier for s in self.steps()]
   
    @property
    def route(self) -> List[Tuple]:
        """Return route of movement as list of (lat, lon)-tuples."""
        return [p for s in self.steps() for p in s.route]  

    def within_bbox(self, loca:Location) -> bool:
        """Return True if location lies within bounding box of route; False otherwise."""
        bounds = self._get_full_api_result()[0]['bounds']
        bbox = (bounds['southwest']['lat'], bounds['southwest']['lng'], 
                bounds['northeast']['lat'], bounds['northeast']['lng'])
        coords = loca.coords
        return (bbox[0] <= coords[0] <= bbox[2] and
                bbox[1] <= coords[1] <= bbox[3])        

    def on_route(self, loca:Location, max_dist:float=25) -> Union[bool, Dict]:
        """If location 'loca' lies on route, return estimate for cumulative
        distance and duration when reaching the point. Return False otherwise."""
        #Fast check: inside bounding box of directions.
        if not self.within_bbox(loca):
            return False
        #More precise: check per step.
        for i, step in enumerate(self.steps()):
            if (estimate := step.on_route(loca, max_dist)):
                estimate['source_stepindex'] = i
                estimate['source_directions'] = self
                return estimate
        #Not found on any of the steps.            
        return False   

def compatibility_vehicletype_transittype(vehicletype, transittype) -> bool:
    """Evaluates compatibility of vehicle type, used in a section (step) of a
    directions object, with transit type of a location."""
    if vehicletype is None: 
        #Step is not using public transport, so any location on the route can be reached.
        return True
    if vehicletype.upper() in ('BUS', 'TROLLEYBUS') \
        and transittype.lower() == 'bus':
        return True
    if vehicletype.upper() in ('RAIL', 'METRO_RAIL', 'SUBWAY', 'TRAM', 'MONORAIL', 
        'HEAVY_RAIL', 'COMMUTER_TRAIN', 'LONG_DISTANCE_TRAIN', 'HIGH_SPEED_TRAIN') \
        and transittype.lower() == 'rail':
        return True
    return False

    
#%%

class Upper:
    @property
    def add(self):
        return self._a + self.b
            
class Middle(Upper):
    def __init__(self, a, b):
        self._a = a
        self._b = b    
    @property 
    def b(self):
        return self._b

class Lower(Middle):
    def __init__(self, m, f):
        self.__dict__ = m.__dict__
        self._f = f
    @property
    def b(self):
        return super().b * self._f
    

#%%

class Step1:
    """
    Type of Movement with partial information about getting from start to end:
    * .distance and .duration apply to the whole movement from start;
    * All other properties (.travel_mode, .vehicletype, .carrier, .bbox, .route)
      apply to SINGLE step (i.e., last one necessary to reach end, starting at
      .routestart).
    """ 

    def __init__(self):
        self._prior_duration = 100
        self._prior_distance = 100
        self._dist = 80
        self._dur = 80


    def distance_to_midpoint(self, fraction:float=1) -> float:
        """Distance over-the-road, in meters. Measured from start. 'fraction' 
        == 0 (to routestart), 1 (to end), 0..1 (in between, estimated)."""
        return self._prior_distance + fraction * self._dist
    distance = property(
        lambda self: self.distance_to_midpoint(1),
        doc="Distance over-the-road, from start to end, in meters.")
    distance_routeonly = property(
        lambda self: self.distance_to_midpoint(1) - self.distance_to_midpoint(0),
        doc="Distance over-the-road, from routestart to end, in meters.")
    
    
class PartialStep1(Step1):
    """Partial Step, i.e., step (as returned by the Directions object) but with
    an arbitrary (new) end point along its route."""
    def __init__(self, step:Step1, f):
        self.__dict__ = step.__dict__
        self._f = f
    
    def distance_to_midpoint(self, fraction:float=1) -> float:
        f = fraction * self._f #Fraction along entire (instead of partial) step
        return super().distance_to_midpoint(f)
    def duration_to_midpoint(self, fraction:float=1) -> float:
        f = fraction * self._f #Fraction along entire (instead of partial) step
        return super().duration_to_midpoint(f)