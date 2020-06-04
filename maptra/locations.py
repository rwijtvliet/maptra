#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes and functions to create locations, i.e., points on earth.
"""

from maptra.memoize import memoize_mutable, memoize_immutable

from typing import Tuple, List, Iterable, Dict, Set, Union, Callable
import overpy
import random
import numpy as np
import geopandas as gpd
import pygeodesy.sphericalTrigonometry as st
import shapely.geometry as sg
from geopy.distance import great_circle


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
    
    def __init__(self, coords:Iterable[float], ):
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
    
    # @property 
    # def hop(self) -> float:
    #     """Return hop to this location."""
    #     pass
    
    # @hop.setter
    # def hop(self, val):
    #     """Set hop for this location."""
    #     self._hop = val
    
    @staticmethod
    def changerepr(current):
        def new(*args, **kwargs):
            return current(*args, **kwargs).replace('rwap.components', '')
        return new
    
    def __repr__(self):
        return self.__class__.__name__ + ' object at ' + hex(id(self))


# %% Filter.

CRS_LONLAT = 'epsg:4326'
PATH_LANDONLY = "maptra/shp/ne_10m_land.shp"
PATH_COUNTRIES = "maptra/shp/ne_10m_admin_0_countries_lakes.shp" 

def _geodf(*sovs:str) -> gpd.GeoDataFrame:
    """Return geodataframe with polygons of the world's land mass (if called
    without arguments) or the countries whose name contains one of the arguments."""
    if len(sovs) == 0:
        gdf = gpd.read_file(PATH_LANDONLY)
        print("GeoDataFrame: any locations on land.")
    else:
        filtr = False
        gdf = gpd.read_file(PATH_COUNTRIES)
        for sov in sovs:
            ftr = gdf['SOVEREIGNT'].str.contains(sov, case=False)
            if not ftr.any():
                raise ValueError(f"No sovereignty with a name containing '{sov}' was found.")
            else:
                filtr |= ftr
                print("GeoDataFrame: include sovereignties: " + ', '.join(gdf[ftr]['SOVEREIGNT'].unique()))                  
        gdf = gdf[filtr]
    return gdf.to_crs(CRS_LONLAT)
    
def geofilter(*sovs:str) -> Callable[[Iterable[Location]], List[Location]]:
    """Return a geofilter function. If no arguments are passed: filters for 
    locations on land (i.e., exclude those at sea). Otherwise: filters for 
    locations within countries/sovereignties whose names contain any of the 
    provided strings. The geofilter function accepts and returns a collection 
    of Location objects."""
    land = _geodf(*sovs)
    
    def filterlocations(locas:Iterable[Location]) -> List[Location]:
        l = gpd.GeoDataFrame({'location': locas, 'geometry':[l.point for l in locas]}, crs=CRS_LONLAT)
        l_in_land = gpd.sjoin(l, land, op='within')
        return l_in_land['location'].tolist()

    return filterlocations

def clip_to_geofilter(locas:Iterable[Location], *sovs:str) -> List[Location]:
    """Filter list of locations. If no sovs are supplied, there is no filter.
    If it is '' (empty string, default): filters for locations on land. 
    Otherwise: filters for locations within countries/sovereignties whose 
    names contains any of the provided strings."""
    filtr = geofilter(*sovs)
    return filtr(locas)

# %% Helper functions to define the perimeter.
                   
def expand_extent(extent: Iterable[float]) -> np.array:
    if not isinstance(extent, Iterable):
        extent = [extent]
    extent = np.array(extent)
    if len(extent) == 1: extent = np.append(extent, extent[0]) #NSWE -> NS, WE
    if len(extent) == 2: extent = np.append(extent, extent[0]) #NS, WE -> N, WE, S
    if len(extent) == 3: extent = np.insert(extent, 1, extent[1]) #N, WE, S -> N, W, E, S
    if len(extent) != 4: raise ValueError("Function must be called with at least 1 and and most 4 values for argument 'extent'.")
    return extent

def perimeterpoints(center:Location, extent:Iterable[float]) -> Dict[str, Location]:
    """
    Find 4 cornerpoints and 4 midpoints between them, for a rectangular area.
    Three points (e.g. SW, S, SE) always lie on a great circle.
    center: reference location of area.
    extent: how far in each direction (N, S, W, E) the area extends (in m).
        One/Two/Three/Four value(s): NSWE  /  NS, WE  /  N, WE, S   /   N, W, E, S.
    """
    extent = expand_extent(extent)
        
    nwes = {k: (point := Location.from_latlon(center.ll.destination(e, b)), 
                point.ll.initialBearingTo(center.ll))
            for k, e, b in zip("NWES", extent, [0, 270, 90, 180])}      
    
    corners = {}
    for key, swap in (('NE', False), ('SE', True), ('SW', False), ('NW', True)):
        a, b = key[::-1] if swap else key 
        (pointA, bearA), (pointB, bearB) = nwes[a], nwes[b]
        corners[key] = Location.from_latlon(pointA.ll.intersection(bearA-90, pointB.ll, bearB+90))
        
    return {**corners, **{k:v[0] for k,v in nwes.items()}}

def perimeterpolygon(center:Location, extent:Iterable[float]) -> sg.Polygon:
    """
    Polygon covering the area around the center. 
    center: reference location of area.
    extent: how far in each direction (N, S, W, E) the area extends (in m).
        One/Two/Three/Four value(s): NSWE  /  NS, WE  /  N, WE, S   /   N, W, E, S.
    """
    # Because rectangle on sphere is not rectangle in lat-lon space, more
    # intermediate points might be needed for large areas or those near the poles.
    locas = perimeterpoints(center, extent)
    keys = ('NW', 'W', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW')
    linestring = [locas[k].point for k in keys]
    return sg.Polygon(linestring)

# Helper function to thin out.

def thin_out(locas:Iterable[Location], min_dist:float=100) -> Iterable[Location]:
    """
    Return a subset of location-collection 'locas', in which all locations
    are further apart than 'min_dist'.
    """
    # Define function to determine if too close.
    def tooclose_function(maxabslat) -> Callable:
        deltalatlim = np.rad2deg(min_dist / 6356000)
        deltalonlim = deltalatlim / np.cos(np.deg2rad(maxabslat))
        def f(c1: Iterable[float], c2: Iterable[float]) -> bool:
            if abs(c1[0]-c2[0]) > deltalatlim:
                return False
            if abs(c1[1]-c2[1]) > deltalonlim:
                return False
            if (great_circle(c1, c2).m) > min_dist:
                return False
            return True
        return f
    # Prepare matrix to hold conflict pairs.
    locas = np.array(locas)
    lats = [l.coords[0] for l in locas]
    tooclose = tooclose_function(np.max(np.abs(lats)))
    matrix = np.zeros((len(locas), len(locas)), bool)
    for i1, l1 in enumerate(locas):
        for i2, l2 in enumerate(locas):
            if i2 < i1: 
                matrix[i1, i2] = matrix[i2, i1]
            elif i2 > i1:
                matrix[i1, i2] = tooclose(l1.coords, l2.coords)
    # Remove locations until no conflicts remain.
    while True:
        conflicts = sum(matrix>0)
        worst = max(conflicts)
        if worst == 0:
            break
        idx = np.where(conflicts == worst)[0][0]
        #delete conflicting locations.
        locas = np.delete(locas, idx)
        matrix = np.delete(matrix, idx, axis=0)
        matrix = np.delete(matrix, idx, axis=1)        
    return locas

# %% Create locations.

def from_address_dict(addresses:Dict, *, 
                      geofilter:Callable[[Location], bool]=None) -> List[Location]:
    """Create list of Locations from a {label: address}-dictionary, with the
    address a string to be geocoded. geofilter: see class's geofilter method."""
    locas = []
    for label, address in addresses.items():
        loca = Location.from_address(address)
        loca.label = label
        locas.append(loca)
    if geofilter is not None:
        locas = [loca for loca in locas if geofilter(loca)]
    return locas

def from_address_list(addresses:List, *, 
                      geofilter:Callable[[Location], bool]=None) -> List[Location]:
    """Create list of Locations from a list of addresses, with each address
    a string to be geocoded. geofilter: see class's geofilter method."""
    locas = [Location.from_address(address) for address in addresses]
    if geofilter is not None:
        locas = [loca for loca in locas if geofilter(loca)]
    return locas

def on_rectangular_grid(center:Location, spacing:float, extent:Iterable[float], *,
                        geofilter:Callable[[Iterable[Location]], bool]=None) -> List[Location]:
    """Create a rectangular grid of points around a central location. 
    center: location at center of grid.
    spacing: north-south and east-west spacing of points (in m), at least for locations
    near center and for those on main north-south and east-west lines from center.
    extent: how far in each direction (N, S, W, E) the grid must extend (in m).
        One/Two/Three/Four value(s): NSWE  /  NS, WE  /  N, WE, S   /   N, W, E, S.
    geofilter: see class's geofilter method."""
    #Get number of points in each compass direction.
    extent = expand_extent(extent)
    num = {k: v for k, v in zip('NWES', (extent/spacing).astype(int))}
    #Create temporary dataframe to calculate locations.
    grid = pd.DataFrame(data=None, columns=range(-num["W"], num["E"]+1), 
                        index=range(num["N"], -num["S"]-1, -1))
    #Find points on main (north/south and east/west) axes.
    for s_east in grid.columns:
        if s_east != 0: 
            grid.loc[0, s_east] = Location.from_latlon(center.ll.destination(s_east*spacing, 90))
        else:
            grid.loc[0, 0] = center
    for s_nrth in grid.index:
        if s_nrth != 0:
            grid.loc[s_nrth, 0] = Location.from_latlon(center.ll.destination(s_nrth*spacing, 0))
    #Find points away from main axes. Which are crossing of a point A on the east/west, and a point B on the north/south axis.
    for s_east in grid.columns:
        if s_east == 0: continue #point is on main axis.
        pointA = grid.loc[0, s_east].ll
        bearAtoStart = pointA.initialBearingTo(center.ll) #looking towards start point: 'eastish' (westish) for points west (east) of start point (on main east/west axis).
        
        for s_nrth in grid.index:
            if s_nrth == 0: continue #point is on main axis.
            pointB = grid.loc[s_nrth, 0].ll
            bearBtoStart = 0 if s_nrth > 0 else 180 #looking toward start point: south (north) for points north (south) of start point (on main north/south axis).
            
            bearA = bearAtoStart + 90 * np.sign(s_nrth) * np.sign(s_east) #turn 90 degrees to point into correct quadrant.
            bearB = bearBtoStart - 90 * np.sign(s_nrth) * np.sign(s_east) #turn 90 degrees to point into correct quadrant.
            # Find crossing between point A and B.
            grid.loc[s_nrth, s_east] = Location.from_latlon(pointA.intersection(bearA, pointB, bearB))
    #Add label.
    for s_east in grid.columns:
        for s_nrth in grid.index:
            grid.loc[s_nrth, s_east].label = f'Location on rectangular grid spot with index ({s_nrth}, {s_east}).'
    #Filter if wanted.
    locas = grid.values.flatten().tolist()
    if geofilter is not None:
        locas = geofilter(locas)
    #Return list.
    if len(locas) > 300:
        print(f"Created many ({len(locas)}) locations!")
    return locas

def on_circular_grid(center:Location, spacing:float, extent:Iterable[float], geofilter:Callable[[Iterable[Location]], bool]=None) -> List[Location]:
    """Create a grid of points in concentric circles, around a central location. 
    center: location at center of grid.
    spacing: value (in m) with which radius increases with each consecutive 
        circle. Also approximate spacing between location and 6 nearest neighbors.
    extent: how far in each direction (N, S, W, E) the grid must extend (in m).
        One/Two/Three/Four value(s): NSWE  /  NS, WE  /  N, WE, S   /   N, W, E, S.
    geofilter: see class's geofilter method."""
    #Get extent in each compass direction.
    extent = expand_extent(extent)
    #polygon the locations must lie within.
    perimeter = perimeterpolygon(center, extent)
    #Create list with locations.
    locas = [center]
    n = 0 #circle
    while True:
        n += 1
        radius = n * spacing
        check = (radius > min(extent))
            
        addedsome = False
        for bearing in np.linspace(0, 360, n*6, False): #points on the circle
            loca = Location.from_latlon(center.ll.destination(radius, bearing))
            loca.label = f'Location on circular grid, on circle {n}, bearing {bearing:.1f} deg.'
            if not check or perimeter.contains(loca.point):
                addedsome = True
                locas.append(loca)
        if not addedsome:
            break
    #Filter if wanted.
    if geofilter is not None:
        locas = geofilter(locas)
    #Return list.
    if len(locas) > 300:
        print(f"Created many ({len(locas)}) locations!")
    return locas

def on_hexagonal_grid(center:Location, spacing:float, extent:Iterable[float], geofilter:Callable[[Iterable[Location]], bool]=None) -> List[Location]:
    """Create a grid of points on hexagonal grid, around a central location. 
    center: location at center of grid.
    spacing: approximate distance (in m) between location and 6 nearest neighbors.
    extent: how far in each direction (N, S, W, E) the grid must extend (in m).
        One/Two/Three/Four value(s): NSWE  /  NS, WE  /  N, WE, S   /   N, W, E, S.
    geofilter: see class's geofilter method."""
    #Get extent in each compass direction.
    extent = expand_extent(extent)
    #polygon the locations must lie within.
    perimeter = perimeterpolygon(center, extent)
    #Create list with locations.
    locas = [center]
    n = 0 #hexagonal layer
    while True:
        n += 1
            
        addedsome = False
        for m in range(n): #points along one of the 6 sides of the layer
            steps = np.sqrt(n**2 + m**2 - n*m)
            radius = steps * spacing
            check = (radius > min(extent))
            bearing0 = np.rad2deg(np.arcsin(0.5*np.sqrt(3)*m/steps))
            
            for bearing in bearing0 + np.linspace(0, 360, 6, False): #the 6 sides of the layer
                loca = Location.from_latlon(center.ll.destination(radius, bearing))
                loca.label = f'Location on hexagonal grid, on layer {n}, bearing {bearing:.1f} deg.'
                if not check or perimeter.contains(loca.point):
                    addedsome = True
                    locas.append(loca)
        if not addedsome:
            break
    #Filter if wanted.
    if geofilter is not None:
        locas = geofilter(locas)
    #Return list.
    if len(locas) > 300:
        print(f"Created many ({len(locas)}) locations!")
    return locas

# %% Openstreetmap

def _get_nodes(center:Location, extent:Iterable[float], nodequery:str) -> Iterable[Location]:
    api = overpy.Overpass()
    perimeter = perimeterpoints(center, extent)
    lats, lons = zip(*[p.coords for p in perimeter.values()])
    bbox = (min(lats), min(lons), max(lats), max(lons))
    result = api.query(f'node[{nodequery}]{bbox};out;')
    return result.nodes if result else []
    
def busstops(center:Location, extent:Iterable[float], min_spacing=50) -> Iterable[Location]:
    """Find bus stops in certain area.
    center: location at center.
    extent: how far in each direction (N, S, W, E) bus stops must extend (in m).
        One/Two/Three/Four value(s): NSWE  /  NS, WE  /  N, WE, S   /   N, W, E, S.
    min_spacing: only include bus stops that are at least this far apart."""
    nodes = _get_nodes(center, extent, '"highway"="bus_stop"')
    locas = [Location((n.lat, n.lon)) for n in nodes]
    for l, n in zip(locas, nodes):
        l.transittype = 'bus_stop' #TODO: add as property to location class
        l.tags = n.tags
    if min_spacing > 0:
        locas = thin_out(locas, min_spacing)
    return locas

def railstops(center:Location, extent:Iterable[float], min_spacing=50) -> Iterable[Location]:       
    """Find railway stops in certain area; see .busstops method for more information."""
    nodes = _get_nodes(center, extent, '"public_transport"="station"')
    locas = [Location((n.lat, n.lon)) for n in nodes]
    for l, n in zip(locas, nodes):
        l.transittype = 'station' #TODO: add as property to location class
        l.tags = n.tags
    if min_spacing > 0:
        locas = thin_out(locas, min_spacing)
    return locas 

