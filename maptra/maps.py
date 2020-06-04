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

from typing import Iterable, List, Tuple, Dict, Set, Any, Union, Callable
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

    def spoof(self, do_spoof:bool=True):
        """Make up random directions to each location, so that no api-calls need to 
        be made. (debugging purposes only) """
        for d in self._df['directions']:
            d.spoof(do_spoof)

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
    
    def __carriers_forest_structure(self) -> Dict[str, ForestStruct]:
        carrier_paths = {}
        for d in self.df['directions']:
            for step in d.steps():
                carrier_paths.setdefault(step.carrier, []).append(step.route)
        carrier_fs = {}
        for carrier, paths in carrier_paths.items():
            fs = ForestStruct()
            fs.add_path(*paths)
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
        return set.union(*[d.carriers() for d in self.df['directions']])

    # Movements        
    
    # def directions(self, min_dist=40) -> List[Directions]:
    #     data = []
    #     s_dirs = self.df['directions']
    #     s_ends = s_dirs.apply(lambda d: d.end)
        
    def steps(self, min_dist = 40) -> pd.Series:
        """Returns individual steps of all directions in map, if route_distance
        of the step is at least 'min_dist'."""
        s_dirs = self.df_success['directions']
        df = pd.DataFrame({'step': [step for steps in s_dirs.apply(lambda d: d.steps())
                                   for step in steps]})
        # Remove steps with duplicate end point.
        df['coords'] = df.step.apply(lambda s: s.end.coords)
        df.drop_duplicates('coords', inplace=True)
        # Remove steps with too small route distance.
        df['routedist'] = df.step.apply(lambda s: s.distance_routeonly)
        mask = (df.routedist > min_dist)
        return df.step[mask]
        
    
