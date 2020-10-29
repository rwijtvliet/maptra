#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:07:20 2020

@author: ruud wijtvliet, rwijtvliet@gmail.com

Sources of ESRI Shappe files for the background map:

    * OSMAXX:
        Range: worldwide
        Detail: Simplified or full.
        Usage: select the wanted geographic area, the detail level, and click 'export'.
            The data will be processed and is ready to be exported in ~1h.
        The geographic area that can be selected for one download is about 200x200 km²
        The zip-files contain many files; keep the .shp, .shx, .prj, and .dbf-files of
            the layers you want to display.
            For information on the individual layers, see https://github.com/geometalab/osmaxx/blob/master/docs/osmaxx_data_schema.md
    
    * Natural earth
        www.naturalearthdata.com/    
        Range: worldwide
        Detail: 1:10_000_000 (most detailed), 1:50_000_000 and 1:110_000_000 (least detailed)
        Usage: download the correct dataset(s) in the correct detail level. 
        The geographic area always covers the entire world. 
        The zip-files contain many files; keep the .shp, .shx, .prj, and .dbf-files of
            the layers you want to display.
            
"""


#Ideas:
    #Map showing if bicycle, car, or public transport is fastest (and by how much)


from maptra.locations import Location
import maptra.locations as ml
from maptra.maps import Map, Multimap
from maptra.external.point2color import ColorMap2, ColorMap3
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shapely.geometry as sg
import pyproj
import warnings
from scipy.spatial import Voronoi
from maptra.voronoi_finite_polygons_2d import voronoi_finite_polygons_2d
from typing import Dict, List, Union, Iterable, Tuple, Optional, Callable


CRS_LONLAT = 'epsg:4326'


class Styles:
    """Return style for lines, based on carrier (i.e., transportation mode or 
    vehicle type)."""
    def style(carrier):
        carrier = carrier.upper()
        default = {'color': '#646613', 'linestyle': '-'}
        # These can be modes:
        if carrier == 'TRANSIT':
            return {**default, 'color': '#D87A11'} #yellow/orangeish
        if carrier == 'WALKING':
            return {**default, 'color': '#FF004D', 'linestyle': '--'} #pink/redish
        if carrier == 'DRIVING':
            return {**default, 'color': '#6C2FB1'} #purple
        if carrier == 'BICYCLING':
            return {**default, 'color': '#3C9537'} #green
        # These cannot be modes (i.e., always vehicle types when chosing mode TRANSIT):
        if carrier == 'SUBWAY':
            return {**default, 'color': '#136622'}
        if carrier == 'COMMUTER_TRAIN':
            return {**default, 'color': '#135766'}
        if carrier == 'HEAVY_RAIL' or carrier == 'LONG_DISTANCE_TRAIN':
            return {**default, 'color': '#5c6613'}
        if carrier == 'BUS':
            return {**default, 'color': '#131d66'}
        if carrier == 'FERRY':
            return {**default, 'color': '#331366'}
        return default
    def color(carrier):
        return Styles.style(carrier)['color']
    def linestyle(carrier):
        return Styles.style(carrier)['linestyle']


def adjust_lightness(color, amount=0.5):
    """Change color lightness by multiplication with specified amount. If 
    amount > 0, make lighter. If amount < 0, make darker.
    Source: https://stackoverflow.com/a/49601444/2302262"""
    import matplotlib.colors as mc
    import colorsys
    factor = (1 + amount)
    try:
        c = mc.cnames[color]
    except:
        c = color
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(h, max(0, min(1, factor * l)), s)


class _Base:
    """
    Parent class for visualisation.
    
    In order to use without overriding any methods or attributes, descendent 
    classes must implement following attributes or property methods (setter not
    required):
        ._crs (str) (before calling __init__ method of this class)

    Additionally, child classes can override any properties/methods of this class.
    """
    
    def __init__(self, gdf, clipping_margin):
        self._fig = self._ax = None
       
        #Check: locations inside area-of-use of crs.
        outside_aou = (~gdf.within(sg.box(*self._crs.area_of_use.bounds))).sum()
        if outside_aou:
            warnings.warn(f"{outside_aou} (of {len(gdf)}) locations are not within the CRS's intended area of use (which is {self._crs.area_of_use}).")
        #Calculate a generous mask around the locations (in (lat, lon)), to discard 
        #   anything that won't make it onto the map EVEN in the intended crs. (This 
        #   must be done in (lat, lon) as most material is supplied in this crs, and 
        #   clipping is best done before reprojecting.) 
        clipping_mask = GeoDataFrame(geometry=[self.box(gdf, clipping_margin)], 
                                     crs=gdf.crs)
        self._clip_and_reproject = self.clip_and_reproject_function(clipping_mask, self._crs)
        #Save bounds of locations in units of visualization's crs.
        self._locbounds = gdf.to_crs(self._crs).total_bounds

    # Functions that return objects in the same crs as that of the parameters.

    @staticmethod
    def box(gdf:Union[GeoDataFrame, gpd.GeoSeries], margin:float=0.3) -> sg.box:
        """Return numpy array with the bounds of the elements in the dataframe
        or series, in its current crs, in order (minx, miny, maxx, maxy)."""
        minx, miny, maxx, maxy = gdf.total_bounds
        dx, dy = margin * (maxx - minx), margin * (maxy - miny)
        return sg.box(minx-dx, miny-dy, maxx+dx, maxy+dy)

    # Functions that return objects in the visualization's intended crs.
    
    @staticmethod
    def clip_and_reproject_function(mask:GeoDataFrame, dest_crs:pyproj.CRS) -> Callable:
        """Return a function that takes a geodataframe or geoseries as input, 
        and outputs it clipped and reprojected. It is clipped to the mask, and
        reprojeced to the dest_crs."""
        #Reproject mask to standard crs, and clip to its bounds, if necessary, before using it.
        if mask.crs != CRS_LONLAT:
            mask = mask.to_crs(CRS_LONLAT)
        mask = gpd.clip(mask, sg.box(-180,-90,180,90)) #Clip mask if it extends beyond lon=-180..180, lat=-90..90
        #Check: area of use of destination crs should lie completely within mask, otherwise there *might* be problems --> Warn.
        aou = GeoDataFrame(geometry=[sg.box(*dest_crs.area_of_use.bounds)], crs=CRS_LONLAT)
        if not gpd.overlay(mask, aou, how='difference').empty:
            warnings.warn("The provided mask extends beyond the 'area of use' of the destination crs. If this" \
                        + " leads to problems, consider changing the visualization's crs or its clipping_margin.")
        def clip_and_reproject(gdf:Union[GeoDataFrame, gpd.GeoSeries]):
            """Clip gdf to previously supplied mask, and reproject to previously
            supplied crs."""
            geo = gdf.to_crs(CRS_LONLAT) #standard crs, to be able to clip to area_of_use of dest_crs.
            clipped = gpd.clip(geo, mask)
            reprojected = clipped.to_crs(dest_crs)
            return reprojected
        return clip_and_reproject
    
    def gdf_point_fromlocations(self, locas:Iterable[Location]) -> GeoDataFrame:
        """Return GeoDataFrame with shapely Points: one row for each Location
        in the provided iterable. Also clip, and reproject to wanted crs."""
        gdf = GeoDataFrame(geometry=[l.point for l in locas], crs=CRS_LONLAT)
        return self._clip_and_reproject(gdf)
    
    def gdf_point_fromcoordlist(self, coordlist:Iterable[Iterable[float]]) -> GeoDataFrame:
        """Return GeoDataFrame with shapely Points: one row for each (lat, lon)-
        Tuple in the provided iterable. Also clip, and reproject to wanted crs."""
        gdf = GeoDataFrame(geometry=[sg.Point(p[::-1]) for p in coordlist], crs=CRS_LONLAT)
        return self._clip_and_reproject(gdf)
    
    def gdf_linestring_fromroutes(self, routes:Iterable[Iterable[Iterable[float]]]
                                  ) -> GeoDataFrame:
        """Return GeoDataFrame with shapely Linestrings: one row for each
        route (= list of (lat, lon)-tuples) in the provided iterable. Also 
        clip, and reproject to wanted crs."""
        gdf = GeoDataFrame(geometry=[sg.LineString([p[::-1] for p in route]) 
                                         for route in routes], crs=CRS_LONLAT)
        return self._clip_and_reproject(gdf)
    
    #Figure and axis.

    def fig_and_ax(self, margin:float=0.2, ax_vis:bool=False, 
                   figsize:Iterable=(8.27, 11.69)) -> Tuple[plt.Figure, plt.Axes]:
        """Create (or adjust) and return the map's Figure and its Axes object."""
        if self._fig is None:
            self._fig, self._ax = plt.subplots()
            self._fig.tight_layout()
        if (self._fig.get_size_inches() != figsize).any():
            self._fig.set_size_inches(figsize)
        minx, miny, maxx, maxy = self._locbounds
        dx, dy = margin * (maxx - minx), margin * (maxy - miny)
        self._ax.set_xlim(minx-dx, maxx+dx)
        self._ax.set_ylim(miny-dy, maxy+dy)
        self._ax.get_xaxis().set_visible(ax_vis)
        self._ax.get_yaxis().set_visible(ax_vis)        
        return self._fig, self._ax
    
    def showfig(self, margin:float=0.2) -> None:
        """Show figure, with a margin around the locations."""
        fig, _ = self.fig_and_ax(margin)
        fig.show()
    
    def savefig(self, filepath:str='map.png', minwidth:int=2000, minheight:int=1000) -> None:
        """Save figure to disk. minwidth and minheight are number of pixels 
        the image should at least have."""
        #Get filename.        
        fil, ext = os.path.splitext(filepath)
        num = 0
        while os.path.isfile(filepath):
            filepath = fil + f' - ({num})' + ext
            num += 1
        #Resize figure.
        xlim, ylim = self._ax.get_xlim(), self._ax.get_ylim()
        dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
        ratio = dy/dx
        self._fig.set_figheight(ratio * self._fig.get_figwidth())
        #Save figure with correct dpi to get desired size in pixels.
        dpi = max(minwidth/self._fig.get_figwidth(), minheight/self._fig.get_figheight())
        self._fig.savefig(filepath, dpi=dpi)
     
    #Adding elements to the figure.
    
    def draw_background(self, source:Union[GeoDataFrame, str], *, 
                       alpha:float=0.3, color:str='grey', linewidth:float=0.15, **kwargs) -> None:
        """Draw 'source' (geodataframe or shape file) to the figure. All kwargs 
        are passed to the plot function."""
        # Get data.
        if isinstance(source, GeoDataFrame): 
            gdf = source
        else:    
            #assue this is a file
            gdf = gpd.read_file(source)[['geometry']] #Workaround: only interested in geometry, but clip function only plays well with geodataframes
        # Clip.
        gdf = self._clip_and_reproject(gdf)
        # Plot.
        fig, ax = self.fig_and_ax()
        gdf.plot(ax=ax, **{'alpha':alpha, 'color':color, 'linewidth':linewidth, **kwargs})        

    def _draw_locations(self, locas:Iterable[Location], **kwargs) -> None:
        """Draw locations to figure."""
        # Turn locations into points.
        gdf = self.gdf_point_fromlocations(locas)
        # Create/get image and add elements.
        fig, ax = self.fig_and_ax()
        gdf.plot(ax=ax, **kwargs)

    def _draw_routes(self, df:pd.DataFrame, var_width:str, minimum_width:float, 
                  **kwargs) -> None:
        """Draw routes, to get from start to end locations, as lines to figure.
        'df' must contain columns 'path' (list of tuples), 'count' (int), 'color'."""
        # Turn usage count into width.
        df['linewidth'] = minimum_width
        if var_width[:3] == 'lin':
            df['linewidth'] *= df['count']
        elif var_width[:3] == 'log':
            df['linewidth'] *= np.log(df['count']) + 1
        elif var_width[:3] != 'not':
            raise ValueError("Value for parameter 'var_width' must be in {'lin', 'log', 'not'}.")
        gdf_path = self.gdf_linestring_fromroutes(df['path'])
        # Create/get image and add elements.
        fig, ax = self.fig_and_ax()  
        gdf_path.plot(ax=ax, **{'color':df['color'], 'linewidth':df['linewidth'], 
                                **kwargs})
        
    def _draw_voronoi(self, locas, start, vals, cax, **kwargs) -> None:
        """Draw colored voronoi cells to figure."""
        # Create (and clip) Voronoi cells.
        #   Center points.
        points = self.gdf_point_fromlocations(np.append(locas, [start])) #Add start point to keep region around it empty.
        #   Regions.
        vor = Voronoi([(point.x, point.y) for point in points.geometry])
        regions, vertices = voronoi_finite_polygons_2d(vor)
        #   Remove region around start again, so it won't be drawn (because we don't have a value for it).
        regions, points = regions[:-1], points[:-1]
        polys = [sg.Polygon(vertices[region]) for region in regions]
        gdf_voronoi = GeoDataFrame({'location': locas, 
                                    'point': points.geometry, 'geometry': polys},
                                   crs=points.crs)
        #   Clip to 'land' and visualization's clipping area.
        land = self._clip_and_reproject(ml._geodf()) 
        gdf_voronoi = gpd.clip(gdf_voronoi, land) #both in self._crs
        
        # Create/get image and add elements.
        fig, ax = self.fig_and_ax()
    
        
        # ax.plot(gdf_voronoi.geometry, **kwargs)           
        gdf_voronoi.plot(ax=ax, column=vals, cax=cax, **kwargs)
        # gdf_voronoi.plot(ax=ax, column=vals, **kwargs)
    

class MapViz(_Base):
    """
    Class to visualize the data (locations, directions) in a Map object.
    
    mapp: prepared Map object.
    crs: which coordinate reference system the data must be displayed in.
    clipping_margin: 'breathing room' between the locations and the figure borders, expressed
        as fraction of the height/width the locations take up. Used to discard
        elements outside the view port. Make this value as large as you think 
        you'll conceivably use, but no larger. When actually displayng the figure, 
        a smaller margin can be used.
    """    
    
    def __init__(self, mapp:Map, crs:str='epsg:3395', clipping_margin:float=1):
        self._map = mapp
        self._crs = pyproj.CRS(crs)     
        #Use all locations on the map to calculate some initial things.
        gdf = GeoDataFrame(geometry=[e.point for e in self._map.ends], 
                           crs=CRS_LONLAT)
        super().__init__(gdf, clipping_margin)
        
    def draw_startpoint(self, *, color:str='black', alpha:float=0.9, marker:str='o', 
                       markersize:float=200, **kwargs) -> None:
        """Draw start location to figure. All kwargs are passed to the plot 
        (GeoSeries.plot) function."""
        # Draw. 
        self._draw_locations([self._map._start], **{'alpha':alpha, 
            'color':color, 'marker':marker, 'markersize':markersize, **kwargs})
        
    def draw_endpoints(self, inter:bool=False,
                      *, color:str='black', alpha:float=0.9, marker:str='o', markersize:float=3, **kwargs) -> None:
        """Draw end locations to figure. Locations that no route has been found
        to are excluded. If inter==True, also plot intermediate points returned
        by google api (default False). All kwargs are passed to the plot 
        (GeoSeries.plot) function."""
        # Get data.
        if inter:
            locas = self._map.steps(0).apply(lambda m: m.end)
        else:
            locas = self._map.ends
        # Draw.
        self._draw_locations(locas, **{'alpha':alpha, 
            'color':color, 'marker':marker, 'markersize':markersize, **kwargs})
        
    def draw_routes(self, var_width:str='log', minimum_width:float=0.35, 
                  *, alpha:float=0.7, **kwargs) -> None:
        """Draw routes, to get from start to end locations, as lines to figure.
        var_width ('not', 'lin', or 'log') sets how linewidth varies (not, linearly,
        or logarithmically) with number of routes that include it. minimum_width is 
        in points. By default, the color changes with transportation carrier. All 
        kwargs (e.g., color) are passed to the plot (GeoSeries.plot) function 
        and can be used to override these values.
        """
        # Get data.
        df_path = pd.DataFrame([{'path': p, 'count': cnt, 'carrier': carrier} 
                                for carrier, subpaths in self._map.carriers_subpaths.items() 
                                for cnt, paths in subpaths.items() for p in paths])
        df_path['color'] = df_path['carrier'].apply(lambda x: Styles.color(x))
        self._map.save() #Save, as previous action might have caused many api-calls.
        # Draw.
        self._draw_routes(df_path, var_width, minimum_width, **{'alpha':alpha, **kwargs})
    
    def draw_quiver(self, inter:bool=True, 
                   *, cmap:str='RdYlGn_r', width:float=0.003, **kwargs) -> None:
        """Draw arrows, from 'actual' to 'corrected' locations, where corrected
        location is where the point would be, if all points could be reached at
        same speed. If inter==True (default), also draw arrow for intermediate
        points returned by google. All kwargs are passed to the plot (ax.quiver)
        function."""
        # Get data.
        if inter:
            s_mov = self._map.steps(0)
        else:
            s_mov = self._map.directions
        df = pd.DataFrame({'speed': s_mov.apply(lambda m: m.crow_speed),
                           'duration': s_mov.apply(lambda m: m.duration)})
        av_speed = df['speed'].mean()        
        gs1 = self.gdf_point_fromlocations(s_mov.apply(lambda m: m.end)).geometry
        gs2 = self.gdf_point_fromlocations(s_mov.apply(lambda m: m.end_durationcorrected(av_speed))).geometry
        self._map.save()#Save, as previous action might have caused many api-calls.
           
        # Create arrows (quiver) of displacement.
        self.df_quiv = df_quiv = pd.DataFrame()
        df_quiv['x'] = gs1.x
        df_quiv['y'] = gs1.y
        df_quiv['u'] = gs2.x - gs1.x
        df_quiv['v'] = gs2.y - gs1.y
        df_quiv['c'] = df['speed']

        # Create/get image and add elements.
        fig, ax = self.fig_and_ax()
        ax.quiver(df_quiv['x'], df_quiv['y'], df_quiv['u'], df_quiv['v'], df_quiv['c'], angles='xy', 
                  scale_units='xy', scale=1, **{'cmap':cmap, 'width': width, **kwargs})
    
    def draw_voronoi(self, show:str='duration', inter:bool=True,
                    *, cmap:str=None, alpha:float=0.5, **kwargs) -> None:
        """Draw colored (voronoi) cells to figure, coloring the area around a 
        location. If inter==True (default), also color the area around 
        intermediate points returned by google. Pick color based on time it 
        takes to get there (if show == 'duration', default) or speed with which
        one gets there (if show == 'speed'). All kwargs are passed to plot 
        (geopandas.plot) function."""
        if show == 'duration':
            show_func = lambda m: m.duration / 3600
            label = 'Time needed to get to point [h]'
            if cmap is None:
                cmap = 'RdYlGn_r'
        elif show == 'speed':
            show_func = lambda m: m.crow_speed * 3.6
            label = 'Velocity to get to point, measured by air-distance [km/h]'
            if cmap is None:
                cmap = 'RdYlGn'
        else:
            raise ValueError("Parameter 'show' must be 'duration' or 'speed'.")
                
        # Get data.
        if inter:
            s_mov = self._map.steps(0)
        else:
            s_mov = self._map.directions
        locas = s_mov.apply(lambda m: m.end).values
        vals = s_mov.apply(show_func)
        vmin, vmax = vals.quantile([0.01, 0.99]) #domain of color bar.
        self._map.save() #save, as previous action might have caused many api-calls.
        # Draw.
        legend = {'label': label, 'orientation': "horizontal", 'shrink':1, 'aspect':30, 'pad':0.02}
        self._draw_voronoi(locas, self._map.start, vals, **{'vmin':vmin,
                           'vmax':vmax, 'legend':True, 'legend_kwds':legend,
                           'cmap':cmap, 'alpha':alpha, **kwargs})


class MultimapViz(_Base):
    
    def __init__(self, mumap:Multimap, crs:str='epsg:3395', clipping_margin:float=1):
        self._mumap = mumap
        self._crs = pyproj.CRS(crs)     
        #Use all locations on the map to calculate some initial things.
        gdf = GeoDataFrame(geometry=[e.point for e in self._mumap.ends], 
                           crs=CRS_LONLAT)
        super().__init__(gdf, clipping_margin)
    
    def draw_startpoint(self, *, color:str='black', alpha:float=0.9, marker:str='o', 
                       markersize:float=200, **kwargs) -> None:
        """Draw start location to figure. All kwargs are passed to the plot 
        (GeoSeries.plot) function."""
        # Draw. 
        self._draw_locations([self._mumap._start], **{'alpha':alpha, 
            'color':color, 'marker':marker, 'markersize':markersize, **kwargs})
        
    def draw_endpoints(self, *, color:str='black', alpha:float=0.9,
                       marker:str='o', markersize:float=3, **kwargs) -> None:
        """Draw end locations to figure. Locations that no route has been found
        to are excluded. All kwargs are passed to the plot (GeoSeries.plot)
        function."""
        # Draw.
        self._draw_locations(self._mumap.ends, **{'alpha':alpha, 
            'color':color, 'marker':marker, 'markersize':markersize, **kwargs})
        
    def draw_routes(self, var_width:str='log', minimum_width:float=0.35, 
                  *, alpha:float=0.7, **kwargs) -> None:
        """Draw routes, to get from start to end locations, as lines to figure.
        var_width ('not', 'lin', or 'log') sets how linewidth varies (not, linearly,
        or logarithmically) with number of routes that include it. minimum_width is 
        in points. By default, the color changes with transportation mode. All 
        kwargs (e.g., color) are passed to the plot (GeoSeries.plot) function 
        and can be used to override these values.
        """
        # Get data.
        df_paths = pd.DataFrame()
        for mo, ma in zip(self._mumap.modes, self._mumap.maps):
            df_path = pd.DataFrame([{'path': p, 'count': cnt} 
                                    for subpaths in ma.carriers_subpaths.values() 
                                    for cnt, paths in subpaths.items() for p in paths])
            df_path['color'] = Styles.color(mo)
            df_paths = df_paths.append(df_path)
        self._mumap.save() #Save, as previous action might have caused many api-calls.
        # Draw.
        self._draw_routes(df_paths, var_width, minimum_width, 
                          **{'alpha':alpha, **kwargs})

    def draw_voronoi(self, show:str='ratio', colors:Iterable=None,
                    *, alpha:float=0.5, **kwargs) -> None:
        """Draw colored (voronoi) cells to figure, coloring the area around a 
        location, based on mode that the location is reached by in the shortest
        time. Pick color based on absolute (if show=='abs') or relative (if how
        == 'ratio', default) time difference between the modes.
        Preset colors are used if 'colors' is not specified.
        
        All kwargs are passed to plot (geopandas.plot) function."""
        # Get data.
        df_mov = self._mumap.directions
        #   Locations.
        locas = df_mov.iloc[:, 0].apply(lambda m: m.end).values
        #   Duration for each mode and location.
        df_val = pd.DataFrame({c: df_mov[c].apply(lambda d: d.duration)
                               for c in df_mov.columns})
        if show.lower().startswith('abs'): #Show absolute difference in duration, relative to slowest mode.
            #   Normalize to worst value for each location.
            df_relative_duration = df_val.subtract(df_val.max(axis=1), axis=0)
            #df_relative_duration: per location: value 0 for slowest mode, value<0 for other modes.
            #   Find largest difference (from 0) we want to display.
            lim = -np.nanquantile(df_relative_duration.min(axis=1), 0.1)
        else: #Show relative difference in duration, relative to slowest mode.
            df_relative_duration = df_val.div(df_val.max(axis=1), axis=0)
            #df_relative_duration: per location: value 1 for slowest mode, 0<value<1 for other modes.
            #   Find largest difference (from 1) we want to display.
            lim = 1 - np.nanquantile(df_relative_duration.min(axis=1), 0.1)
            
        #   Handle the number of modes.
        if len(self._mumap.modes) == 2:
            mo0, mo1 = self._mumap.modes
            vals = (df_val.iloc[:,0] - df_val.iloc[:,1]).values #negative if mode0 faster0
            label = f'Time comparison: {mo0} vs {mo1}\n'
            if show.lower().startswith('abs'):
                label += f'e.g.: "-1.5" (+1.5) means, that {mo0} needs 1.5 minutes less (more) time.'
            else:
                label += f'e.g.: "-12%" (+12%) means, that {mo0} needs 12% less (more) time.'
            try:
                clr0, clr1 = colors
            except:
                clr0, clr1 = Styles.color(mo0), Styles.color(mo1)                
            cmap = LinearSegmentedColormap.from_list('test', (
                adjust_lightness(clr0, -0.4), to_rgb(clr0), to_rgb('#eee'),
                to_rgb(clr1), adjust_lightness(clr1, -0.4)))
        elif len(self._mumap.modes) == 3:
            mo0, mo1, mo2 = self._mumap.modes

            
            try:
                clr0, clr1, clr2 = colors
            except:
                clr0, clr1, clr2 = Styles.color(mo0), Styles.color(mo1), Styles.color(mo2)                
            
            raise ValueError("Not yet implemented for 3 modes.")
        else:
            raise ValueError("Not yet implemented for >3 modes.")
            
            
        # Draw.
        legend = {'label': label, 'orientation': "horizontal", 'shrink':1, 
                  'aspect':30, 'pad':0.02, 'ticks': [-50, 0, 50]}
        
        ax = self.fig_and_ax()[1]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        self._draw_voronoi(locas, self._mumap.start, vals, cax, **{'vmin':-lim,
                           'vmax':lim, 'legend':True,
                           'cmap':cmap, 'alpha':alpha, **kwargs})
        cax.set_yticks([-50, 0, 50])
        cax.set_yticklabels(['10:00\nbike is faster', '-same', '10:00\ncar is faster'])
        # cax.set_yticklabels(['10:00\nbike is faster', '-same', '10:00\ncar is faster'])

        self.ucax=cax
        # self._draw_voronoi(locas, self._mumap.start, vals, cax, **{'vmin':-lim,
        #                    'vmax':lim, 'legend':True, 'legend_kwds':legend,
        #                    'cmap':cmap, 'alpha':alpha, **kwargs})
        # self.uvals = vals
        # ax = self.fig_and_ax()[1]
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # fig, ax = self.fig_and_ax()
        # cbar = fig.colorbar(fig)
        
#%%
