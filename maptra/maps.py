#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:37:02 2020

@author: ruud wijtvliet, rwijtvliet@gmail.com

Create a map that shows the surroundings of a certain central point. The 
surroundings are shown by somehow taking into account how the various points
on the map can be reached.

"""

from .components import Location, Directions
from .forest import ForestStruct
from .memoize import memoize_immutable, memoize_mutable

from typing import Iterable, List, Tuple, Dict, Set, Any, Union, Callable
import numpy as np
import geopandas as gpd
import googlemaps
import pandas as pd
import pickle


CRS_LONLAT = 'epsg:4326'

class Map:
    """Map that shows the surroundings of a certain central point. The 
    surroundings are shown by somehow taking into account how the various points
    on the map can be reached.
    
    start_where: the central location on the map; either a Location object or a
        valid value of its 'where' initialization method.
    path: filepath to save pickle to.
    mode: 'bicycling', 'walking', 'transit', or 'driving'.
    kwargs: that are passed on to the googlemaps.directions function.
    """
    
    @staticmethod
    def set_gmaps_api_key(apikey:str) -> None :
        """Set the api key and create the gmaps client used for the geocoding."""
        client = googlemaps.Client(key=apikey)
        Location.set_gmaps_client(client)
        Directions.set_gmaps_client(client)
        
    #Instance initialization.
    
    @classmethod    
    def from_coords(cls, start_coords:Iterable, mode:str='walking', **kwargs):
        """Create instance of Map, with start location as a (lat, lon)-Tuple."""
        start = Location(start_coords)
        return cls(start, mode, **kwargs)
        
    @classmethod    
    def from_address(cls, start_address:str, mode:str='walking', **kwargs):
        """Create instance of Map, with start location as an address string."""
        start = Location.from_address(start_address)
        return cls(start, mode, **kwargs)
    
    @classmethod
    def from_pickle(cls, filepath:str):
        """Create instance of Map, from a pickled class dictionary."""
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
            instance.__dict__['_filepath'] = filepath #File might have been moved since it was pickled.
        return instance

    @classmethod
    def from_map(cls, mapp, mode:str='walking', **kwargs):
        """Create instance of Map, from another instance of Map. Start and 
        Locations are taken from it; transport parameters are supplied in-
        dividually for this new instance."""
        instance = cls(mapp.start, mode, **kwargs)
        instance.add_locations(mapp._df['location'])
        return instance

    def __init__(self, start:Location, mode:str='walking', **kwargs):
        self._start = start
        self._gmapsparameters = {'mode': mode, **kwargs}
        self._filepath = None
        self._df = pd.DataFrame(columns=['location', 'directions']) #Dataframe with all 'location' and 'directions' objects.

    #Save and load pickle to file.
        
    @property 
    def filepath(self) -> str:
        """Return path of file that object was stored to, during the previous save."""
        return self._filepath
    def to_pickle(self, filepath:str=None, *, ignore_err=True) -> Union[bool,str]:
        """Save object to provided path, if specified. If not specified, save
        to path that was used during previous save. Returns file path."""
        if filepath is not None:
            self._filepath = filepath            
        if self._filepath is None:
            if not ignore_err:
                raise ValueError("No file path has been set for the pickle file.")
            else:
                return False
        with open(self._filepath, 'wb') as f:  
            pickle.dump(self, f)
        return filepath
    def save(self) -> None:
        if self._filepath is not None:
            self.to_pickle()
    
    #Manipulating Locations and Directions.
    
    @property
    def start(self) -> Location:
        """Return Location object for central point in the map."""
        return self._start
    
    def add_locations(self, locas:Iterable[Location]) -> None:
        """Add locations to the map."""
        self.__add_locas_to_df(locas)
    
    def clear_locations(self) -> None:
        """Clear list of locations and directons (by putting them on 'inactive')."""
        self._df['in_df'] = False
    
    def __add_locas_to_df(self, locas:Iterable[Location]) -> None:
        """In list of locations, first check if they can be replaced by any that
        are already in the archive, to save on api-calls. If not, add them to 
        the archive."""
        #Check which locations are already in the dataframe.
        locas_toadd = []
        already = notyet = 0
        for loca in locas:
            found = False
            if not self._df.empty: 
                mask_same = self._df['location'].apply(lambda l: np.allclose(l.coords, loca.coords))
                if mask_same.any():
                    found = True
                    already += 1
            if not found:
                locas_toadd.append(loca)
                notyet += 1
        print(f"{already} locations found in archive; {notyet} locations are new.")
        #Add those which aren't there yet.
        if notyet:
            idx0 = 0 if self._df.empty else self._df.index.max() + 1
            data = [{'location': loca, 'directions': self.__Directions(loca)} for loca in locas_toadd]
            df_toadd = pd.DataFrame(data, index=range(idx0, idx0+len(data)))
            self._df = self._df.append(df_toadd)
    
    def __Directions(self, end:Location) -> Directions:
        """Return Directions object to end location, with all other
        parameters taken from standard class variables."""
        return Directions(self._start, end, **self._gmapsparameters)

    def spoof_directions(self):
        """Make up random route to each location, so that no api-calls need to 
        be made. (debugging purposes only)"""
        for d in self._df['directions']:
            d.spoof_api_result()

    @property
    def df(self) -> pd.DataFrame:
        """Return dataframe with locations and directions."""
        return self._df
    @property 
    def df_success(self):
        """Return dataframe with locations and directions, but with only those 
        that a route has been found to."""
        mask = self.df['directions'].apply(lambda x: bool(x.route))
        return self.df[mask]
    @property 
    def df_failure(self):
        """Return dataframe with locations and directions, but with only those 
        that a route has NOT been found to."""
        mask = self.df['directions'].apply(lambda x: not bool(x.route))
        return self.df[mask]
    
    #Routes and Modes.
    
    def __route_forest_structure(self) -> ForestStruct:
        fs = ForestStruct()
        paths = [d.route for d in self.df['directions']]
        fs.add_path(*paths)
        return fs
    @property 
    def route_forest(self) -> List:
        """Return current route forest, regardless of transportation mode, as 
        tree-list. Whereever forking occurs, a list is inserted. As all paths
        start at the same position, there should only be a single tree in this 
        forest."""
        return self.__route_forest_structure().forest
    @property
    def route_subpaths(self) -> Dict[int, List]:
        """Return non-forking subpaths of current route forest, in a {count: paths}-
        dictionary, with the key being the degeneracy (i.e., number of start-to-end 
        routes that include that subpath)."""               
        return self.__route_forest_structure().subpaths
    
    def __modes_forest_structure(self) -> Dict[str, ForestStruct]:
        modes_paths = {}
        for d in self.df['directions']:
            for step in d.steps:
                modes_paths.setdefault(step['mode'], []).append(step['route'])
        modes_fs = {}
        for mode, paths in modes_paths.items():
            fs = ForestStruct()
            fs.add_path(*paths)
            modes_fs[mode] = fs
        return modes_fs
    @property
    def modes_forest(self) -> Dict[str, List]:
        """Return current route forest, per transportation mode, as {mode: tree-
        list}-dictionary. Wherever forking occurs, a list is inserted. NB: when
        transportation modes are mixed, there is no longer just a single tree in 
        each forest."""
        return {mode: fs.forest for mode, fs in self.__modes_forest_structure().items()}
    @property
    def modes_subpaths(self) -> Dict[str, Dict[int, List]]:
        """Return non-forking subpaths of current route forest, in {mode: 
        {count: paths}}-dictionary, with first key being the transportation mode, 
        and second key the degeneracy (i.e., number of start-to-end routes that 
        include that subpath)."""
        return {mode: fs.subpaths for mode, fs in self.__modes_forest_structure().items()}
    @property
    def modes(self) -> Set[str]:
        """Return set of all unique travel modes in this map's directions."""
        return set.union(*[d.modes for d in self.df['directions']])

     
class CreateLocations:
    """Class that groups functions that return lists of locations."""    

    #Filter.

    PATH_LANDONLY = "maptra/shp/ne_10m_land.shp"
    PATH_COUNTRIES = "maptra/shp/ne_10m_admin_0_countries_lakes.shp" 
    
    @classmethod
    @memoize_immutable
    def _geodf(cls, *sovs:str) -> gpd.GeoDataFrame:
        """Return geodataframe with polygons of the world's land mass (if called
        without arguments) or the countries whose name contains one of the arguments."""
        if len(sovs) == 0:
            gdf = gpd.read_file(cls.PATH_LANDONLY)
            print("GeoDataFrame: any locations on land.")
        else:
            filtr = False
            gdf = gpd.read_file(cls.PATH_COUNTRIES)
            for sov in sovs:
                ftr = gdf['SOVEREIGNT'].str.contains(sov, case=False)
                if not ftr.any():
                    raise ValueError(f"No sovereignty with a name containing '{sov}' was found.")
                else:
                    filtr |= ftr
                    print("GeoDataFrame: include sovereignties: " + ', '.join(gdf[ftr]['SOVEREIGNT'].unique()))                  
            gdf = gdf[filtr]
        return gdf.to_crs(CRS_LONLAT)
        
    @classmethod
    def geofilter(cls, *sovs:str) -> Callable[[Iterable[Location]], List[Location]]:
        """Return a geofilter function. If no arguments are passed: filters for 
        locations on land (i.e., exclude those at sea). Otherwise: filters for 
        locations within countries/sovereignties whose names contain any of the 
        provided strings. The geofilter function accepts and returns a collection 
        of Location objects."""
        land = cls._geodf(*sovs)
        
        def filterlocations(locas:Iterable[Location]) -> List[Location]:
            l = gpd.GeoDataFrame({'location': locas, 'geometry':[l.point for l in locas]}, crs=CRS_LONLAT)
            l_in_land = gpd.sjoin(l, land, op='within')
            return l_in_land['location'].tolist()

        return filterlocations
    
    @classmethod
    def clip_to_geofilter(cls, locas:Iterable[Location], *sovs:str) -> List[Location]:
        """Filter list of locations. If no sovs are supplied, there is no filter.
        If it is '' (empty string, default): filters for locations on land. 
        Otherwise: filters for locations within countries/sovereignties whose 
        names contains any of the provided strings."""
        filtr = cls.geofilter(*sovs)
        return filtr(locas)

    #Static methods.
    
    @staticmethod
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

    @staticmethod
    def from_address_list(addresses:List, *, 
                          geofilter:Callable[[Location], bool]=None) -> List[Location]:
        """Create list of Locations from a list of addresses, with each address
        a string to be geocoded. geofilter: see class's geofilter method."""
        locas = [Location.from_address(address) for address in addresses]
        if geofilter is not None:
            locas = [loca for loca in locas if geofilter(loca)]
        return locas
    
    @staticmethod
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
        nums = (np.array(extent)/spacing).astype(int)
        if len(nums) == 1: nums = np.append(nums, nums[0]) #NSWE -> NS, WE
        if len(nums) == 2: nums = np.append(nums, nums[0]) #NS, WE -> N, WE, S
        if len(nums) == 3: nums = np.insert(nums, 1, nums[1]) #N, WE, S -> N, W, E, S
        if len(nums) != 4: raise ValueError("Function must be called with at least 1 and and most 4 values for argument 'extent'.")
        num = {k: v for k, v in zip("NWES", nums)}
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

    @staticmethod
    def on_circular_grid(center:Location, spacing:float, extent:Iterable[float], geofilter:Callable[[Iterable[Location]], bool]=None) -> List[Location]:
        """Create a grid of points in concentric circles, around a central location. 
        center: location at center of grid.
        spacing: value (in m) with which radius increases with each consecutive 
            circle. Also approximate spacing between location and 6 nearest neighbors.
        extent: how far in each direction (N, S, W, E) the grid must extend (in m).
            One/Two/Three/Four value(s): NSWE  /  NS, WE  /  N, WE, S   /   N, W, E, S.
        geofilter: see class's geofilter method."""
        #Get extent in each compass direction.
        extent = np.array(extent)
        if len(extent) == 1: extent = np.append(extent, extent[0]) #NSWE -> NS, WE
        if len(extent) == 2: extent = np.append(extent, extent[0]) #NS, WE -> N, WE, S
        if len(extent) == 3: extent = np.insert(extent, 1, extent[1]) #N, WE, S -> N, W, E, S
        if len(extent) != 4: raise ValueError("Function must be called with at least 1 and and most 4 values for argument 'extent'.")
        #Get extreme points, as well as the min and max compass directions, that the bearing from them to any locations must be in.
        extremes = {k: (Location.from_latlon(center.ll.destination(e, b)), (b+90, b+270)) 
                    for k, e, b in zip("NWES", extent, [0,270,90,180])}
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
                if check:
                    for extreme, (bmin, bmax) in extremes.values():
                        bear = extreme.ll.initialBearingTo(loca.ll)
                        while bear < bmin:
                            bear += 360
                        if bear > bmax:
                            break
                    else:
                        addedsome = True
                        locas.append(loca)
                else:
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
    
    @staticmethod
    def on_hexagonal_grid(center:Location, spacing:float, extent:Iterable[float], geofilter:Callable[[Iterable[Location]], bool]=None) -> List[Location]:
        """Create a grid of points on hexagonal grid, around a central location. 
        center: location at center of grid.
        spacing: approximate distance (in m) between location and 6 nearest neighbors.
        extent: how far in each direction (N, S, W, E) the grid must extend (in m).
            One/Two/Three/Four value(s): NSWE  /  NS, WE  /  N, WE, S   /   N, W, E, S.
        geofilter: see class's geofilter method."""
        #Get extent in each compass direction.
        extent = np.array(extent)
        if len(extent) == 1: extent = np.append(extent, extent[0]) #NSWE -> NS, WE
        if len(extent) == 2: extent = np.append(extent, extent[0]) #NS, WE -> N, WE, S
        if len(extent) == 3: extent = np.insert(extent, 1, extent[1]) #N, WE, S -> N, W, E, S
        if len(extent) != 4: raise ValueError("Function must be called with at least 1 and and most 4 values for argument 'extent'.")
        #Get extreme points, as well as the min and max compass directions, that the bearing from them to any locations must be in.
        extremes = {k: (Location.from_latlon(center.ll.destination(e, b)), (b+90, b+270)) 
                    for k, e, b in zip("NWES", extent, [0,270,90,180])}
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
                    if check:
                        for extreme, (bmin, bmax) in extremes.values():
                            bear = extreme.ll.initialBearingTo(loca.ll)
                            while bear < bmin:
                                bear += 360
                            if bear > bmax:
                                break
                        else:
                            addedsome = True
                            locas.append(loca)
                    else:
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

#%%