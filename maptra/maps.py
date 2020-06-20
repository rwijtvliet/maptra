#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:37:02 2020

@author: ruud wijtvliet, rwijtvliet@gmail.com

Create a map that shows the surroundings of a certain central point. The 
surroundings are shown by somehow taking into account how the various points
on the map can be reached.

"""

from maptra.locations import Location
from maptra.movements import Directions
from maptra.forest import ForestStruct
# from maptra.visualize import MapViz, MultimapViz

from typing import Iterable, List, Tuple, Dict, Set, Union, Callable
import numpy as np
import googlemaps
import pandas as pd
import pickle

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
        
    # Instance initialization.
    
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
        instance.add_locations(mapp.directions_all().apply(lambda d: d.end))
        return instance

    def __init__(self, start:Location, mode:str='walking', **kwargs):
        self._start = start
        self._gmapsparameters = {'mode': mode, **kwargs}
        self._filepath = None
        self._directions = pd.Series(name='directions')

    # Save and load pickle to file.
        
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
    
    # Manipulating Locations and Directions.
    
    @property
    def start(self) -> Location:
        """Return Location object for central point in the map."""
        return self._start
    
    def add_ends(self, locas:Iterable[Location]) -> None:
        """Add locations to the map as end points."""
        self.__add_locas(locas)

    def __add_locas(self, locas:Iterable[Location]) -> None:
        """In list of locations, first check if they can be replaced by any that
        are already in the archive, to save on api-calls. If not, add them to 
        the archive."""
        #Check which locations are already in the dataframe.
        locas_toadd = []
        already = notyet = 0
        for loca in locas:
            for l in np.concatenate((self.ends, [self.start])):
                if np.allclose(loca.coords, l.coords):
                    already += 1
                    break
            else:
                locas_toadd.append(loca)
                notyet += 1
        print(f"{already} locations found in archive; {notyet} locations are new.")
        #Add those which aren't there yet.
        if locas_toadd:
            s = pd.Series([Directions(self.start, loca, **self._gmapsparameters)
                           for loca in locas_toadd])
            self._directions = self._directions.append(s, ignore_index=True)
    
    def directions_all(self, *states) -> pd.Series:
        """Return series with directions without making api-calls. Specify
        states to get subset of directions having one of those states."""
        if not states:
            return self._directions
        else:
            mask = self._directions.apply(lambda d: d.state in states)
            return self._directions[mask]
    @property 
    def directions(self) -> pd.Series:
        """Return series with directions that a route has been found to. (NB:
        possibly forces many api-calls if these haven't been made yet.)"""
        self.make_apicalls()
        mask = self._directions.apply(lambda d: d.state > -1)
        return self._directions[mask]
    
    @property
    def ends(self) -> pd.Series:
        """Return series with end locations (no api-calls are made)."""
        return self._directions.apply(lambda d: d.end)

    # Get results.
    
    def make_apicalls(self, do_estimate:bool=True):
        """Make all API-calls at once (instead of later, once they are needed).
        Makes it possible to estimate the distance and duration of some direc-
        tions (when their end location happens to be on the route), use 
        'do_estimate' = False to disable."""
        while True:
            s = self.directions_all(0)
            if not len(s): #All done
                break
            # Find the furthest location of which distance and duration are unknown.
            i = s.apply(lambda d: d.crow_distance).idxmax()
            d = s.loc[i]
            _ = d._get_full_api_result() #Force API-call
            # See if estimate for other directions can be derived from this one.
            if do_estimate:
                for d2 in self.directions_all(0, 1):
                    d2.check_estimate(d)
    @property
    def apistats(self):
        """Return short stats on api-calls."""
        states = np.array([d.state for d in self.directions_all()])
        total = len(states)
        fail = sum(states == -1)
        success = sum(states > 0)
        est = sum(states == 1)
        return f'Number of directions: {total}\n' \
            + f'. Succeeded:   {success}\n' \
            + f'. . Called:    {success-est}\n' \
            + f'. . Estimated: {est} ({est/success:.0%})\n' \
            + f'. Failure:     {fail}\n' \
            + f'. Still open:  {total-success-fail}'

    def spoof_apicalls(self, do_spoof:bool=True):
        """Make up random directions to each location, so that no api-calls need to 
        be made. (debugging purposes only) """
        for d in self.directions_all(0, 2):
            d.spoof_apicall(do_spoof)   
            
    # Routes and Modes.
    
    def __route_forest_structure(self) -> ForestStruct:
        fs = ForestStruct()
        paths = [d.route for d in self.directions]
        fs.add_paths(*paths)
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
    
    def __carriers_forest_structure(self) -> Dict[str, ForestStruct]:
        carrier_paths = {}
        for d in self.directions:
            for step in d.steps():
                carrier_paths.setdefault(step.carrier, []).append(step.route)
        carrier_fs = {}
        for carrier, paths in carrier_paths.items():
            fs = ForestStruct()
            fs.add_paths(*paths)
            carrier_fs[carrier] = fs
        return carrier_fs
    @property
    def carriers_forest(self) -> Dict[str, List]:
        """Return current route forest, per carrier, as {carrier: tree-list}-
        dictionary. Wherever forking occurs, a list is inserted. NB: when
        carriers are mixed, there is no longer just a single tree in each forest."""
        return {carrier: fs.forest for carrier, fs in self.__carriers_forest_structure().items()}
    @property
    def carriers_subpaths(self) -> Dict[str, Dict[int, List]]:
        """Return non-forking subpaths of current route forest, in {carrier: 
        {count: paths}}-dictionary, with first key being the transport carrier, 
        and second key the degeneracy (i.e., number of start-to-end routes that 
        include that subpath)."""
        return {carrier: fs.subpaths for carrier, fs in self.__carriers_forest_structure().items()}
    @property
    def carriers(self) -> Set[str]:
        """Return set of all unique transport carriers in this map's directions."""
        return set([c for d in self.directions for c in d.carriers()])

    # Movements. 
    
    def steps(self, min_dist:float=0) -> pd.Series:
        """Returns individual steps of all directions in map, if route_distance
        of the step is at least 'min_dist'."""
        s_mov = pd.Series([step for steps in self.directions.apply(
            lambda d: d.steps()) for step in steps])
        # Remove steps with too small route distance.
        mask = (s_mov.apply(lambda m: m.distance_routeonly) > min_dist)
        s_mov = s_mov[mask]
        # Remove steps with duplicate end point.
        mask = s_mov.apply(lambda m: m.end.coords).duplicated()
        s_mov = s_mov[~mask]  #keep only one movement per end point.
        return s_mov


class Multimap:
    
        
    @staticmethod
    def set_gmaps_api_key(apikey:str) -> None :
        """Set the api key and create the gmaps client used for the geocoding."""
        client = googlemaps.Client(key=apikey)
        Location.set_gmaps_client(client)
        Directions.set_gmaps_client(client)
        
    # Instance initialization.
    
    @classmethod    
    def from_coords(cls, start_coords:Iterable, 
                    modes:List[str]=['transit', 'driving'], **kwargs):
        """Create instance of MultiMap, with start location as a (lat, lon)-Tuple."""
        start = Location(start_coords)
        return cls(start, modes, **kwargs)
        
    @classmethod    
    def from_address(cls, start_address:str, 
                     modes:List[str]=['transit', 'driving'], **kwargs):
        """Create instance of MultiMap, with start location as an address string."""
        start = Location.from_address(start_address)
        return cls(start, modes, **kwargs)
    
    @classmethod
    def from_pickle(cls, filepath:str):
        """Create instance of MultiMap, from a pickled class dictionary."""
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
            instance.__dict__['_filepath'] = filepath #File might have been moved since it was pickled.
        return instance

    @classmethod
    def from_map(cls, mapp, 
                 modes:List[str]=['transit', 'driving'], **kwargs):
        """Create instance of MultiMap, from an instance of Map. Start and 
        Locations are taken from it; transport parameters are supplied in-
        dividually for this new instance."""
        instance = cls(mapp.start, modes, **kwargs)
        instance.add_locations(mapp.directions_all().apply(lambda d: d.end))
        return instance
    
    def __init__(self, start:Location, modes:List[str]=['transit', 'driving'], **kwargs):
        self._start = start
        self._gmapsparameters = kwargs
        self._modes = modes
        self._maps = {mode: Map(start, mode, **kwargs) for mode in modes}
        
    # Save and load pickle to file.
        
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
           
    # Manipulating Locations and Maps.
     
    @property
    def start(self) -> Location:
        """Return Location object for central point in the map."""
        return self._start
    
    def add_ends(self, locas:Iterable[Location]) -> None:
        """Add locations to the map as end points."""
        for ma in self._maps.values():
            ma.add_ends(locas)
            
    @property
    def ends(self) -> pd.Series:
        # Return the end points from any of the maps, as they are all the same.
        return next(iter(self._maps.values()))._directions.apply(lambda d: d.end)

    @property
    def maps(self):
        return self._maps
    
    # Get results.
    
    def make_apicalls(self, do_estimate:bool=True):
        for ma in self._maps.values():
            ma.make_apicalls(do_estimate)
            
    @property
    def apistats(self):
        return "\n".join([f'{mo}:\n{ma.apistats}' for mo, ma in self._maps.items()])
    
    def spoof_apicalls(self, do_spoof:bool=True):
        for ma in self._maps.values():
            ma.spoof_apicalls(do_spoof)
            
    