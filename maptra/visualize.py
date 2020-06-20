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
from maptra.maps import Map, MultiMap
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
import shapely.geometry as sg
import pyproj
import warnings
from scipy.spatial import Voronoi
from maptra.voronoi_finite_polygons_2d import voronoi_finite_polygons_2d
from typing import Dict, List, Union, Iterable, Tuple, Optional, Callable

class Styles:
    def style(carrier):
        default = {'color': '#646613', 'linestyle': '-'}
        if carrier == 'WALKING':
            return {**default, 'color': '#66131d', 'linestyle': '--'}
        elif carrier == 'SUBWAY':
            return {**default, 'color': '#136622'}
        elif carrier == 'COMMUTER_TRAIN':
            return {**default, 'color': '#135766'}
        elif carrier == 'HEAVY_RAIL' or carrier == 'LONG_DISTANCE_TRAIN':
            return {**default, 'color': '#5c6613'}
        elif carrier == 'BUS':
            return {**default, 'color': '#131d66'}
        elif carrier == 'DRIVING':
            return {**default, 'color': '#661346'}
        elif carrier == 'BICYCLING':
            return {**default, 'color': '#664413'}
        elif carrier == 'FERRY':
            return {**default, 'color': '#331366'}
        else:
            return default
    def color(carrier):
        return Styles.style(carrier)['color']
    def linestyle(carrier):
        return Styles.style(carrier)['linestyle']

CRS_LONLAT = 'epsg:4326'

class Visualization:
    """Class to visualize the data (locations, directions) in a Map object.
    
    mapp: prepared Map object.
    crs: which coordinate reference system the data must be displayed in.
    clipping_margin: 'breathing room' between the locations and the figure borders, expressed
        as fraction of the height/width the locations take up. Used to discard
        elements outside the view port. Make this value as large as you think 
        you'll conceivably use, but no larger. When actually displayng the figure, 
        a smaller margin can be used."""    
    
    def __init__(self, mapp:Map, crs:str='epsg:3395', clipping_margin:float=1):
        self._map = mapp
        self._crs = pyproj.CRS(crs)
        self._fig = self._ax = None
        self._data = {} #To save data of calculations. Unclear if needed; TODO: remove if not needed.
        
        #Use all locations on the map to calculate some initial things.
        gdf = GeoDataFrame(geometry=[d.end.point for d in self._map.directions], 
                           crs=CRS_LONLAT)
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
    
    def gdf_linestring_fromroutes(self, routes:Iterable[Iterable[Iterable[float]]]) -> GeoDataFrame:
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
    
    def add_background(self, gdf:GeoDataFrame, *, 
                       alpha:float=0.3, color:str='grey', linewidth:float=0.15, **kwargs) -> None:
        """Add geodataframe to the figure. All kwargs are passed to the plot function."""
        # Clip.
        gdf = self._clip_and_reproject(gdf)
        # Plot.
        fig, ax = self.fig_and_ax()
        gdf.plot(ax=ax, **{'alpha':alpha, 'color':color, 'linewidth':linewidth, **kwargs})        
    
    def add_background_fromfile(self, filepath:str, **kwargs) -> None:
        """Add background shape file to the figure. filepath: path to shape file.
        All kwargs are passed to the plot function."""
        # Get dataframe and add.
        gdf = gpd.read_file(filepath)[['geometry']] #Workaround: only interested in geometry, but clip function only plays well with geodataframes
        self.add_background(gdf, **kwargs)
        
    def add_lines(self, var_width:str='log', minimum_width:float=0.35, 
                  *, alpha:float=0.7, **kwargs) -> None:
        """Add routes, to get from start to end locations, as lines to figure.
        var_width ('not', 'lin', or 'log') sets how linewidth varies (not, linearly,
        or logarithmically) with number of routes that include it. minimum_width is 
        in points. By default, the color changes with transportation carrier. All 
        kwargs (e.g., color) are passed to the plot (GeoSeries.plot) function 
        and can be used to override these values.
        """
        #Get data.
        df_path = pd.DataFrame([{'path': p, 'count': cnt, 'carrier': carrier} 
                                for carrier, subpaths in self._map.carriers_subpaths.items() 
                                for cnt, paths in subpaths.items() for p in paths])
        df_path['linewidth'] = minimum_width 
        if var_width[:3].lower() == 'lin':
            df_path['linewidth'] *= df_path['count']
        elif var_width[:3].lower() == 'log':
            df_path['linewidth'] *= np.log(df_path['count']) + 1
        df_path['color'] = df_path['carrier'].apply(lambda x: Styles.color(x))
        gdf_path = self.gdf_linestring_fromroutes(df_path['path'])
        #self._map.save() #Save, as previous action might have caused many api-calls.
        #TODO uncomment
        #Save.
        self._data['add_lines'] = [gdf_path] #TODO: save bounding box instead of entire data?  

        #Create/get image and add elements.
        fig, ax = self.fig_and_ax()  
        gdf_path.plot(ax=ax, **{'alpha':alpha, 'color':df_path['color'], 
                                'linewidth':df_path['linewidth'], **kwargs})
    
    def add_quiver(self, *, cmap:str='RdYlGn_r', width:float=0.003, **kwargs) -> None:
        """Add arrows, from 'actual' to 'corrected' locations, where corrected
        location is where the point would be, if all points could be reached at
        same speed. All kwargs are passed to the plot (ax.quiver) function."""
        #Get data.
        s_dirs = self._map.directions
        df = pd.DataFrame({'speed': s_dirs.apply(lambda x: x.crow_speed),
                           'duration': s_dirs.apply(lambda x: x.duration)})
        av_speed = df['speed'].mean()        
        gs1 = self.gdf_point_fromlocations(s_dirs.apply(lambda x: x.end)).geometry
        gs2 = self.gdf_point_fromlocations(s_dirs.apply(lambda x: x.end_durationcorrected(av_speed))).geometry
        self._map.save()#Save, as previous action might have caused many api-calls.
           
        #Create arrows (quiver) of displacement, and save.
        self.df_quiv = df_quiv = pd.DataFrame()
        df_quiv['x'] = gs1.x
        df_quiv['y'] = gs1.y
        df_quiv['u'] = gs2.x - gs1.x
        df_quiv['v'] = gs2.y - gs1.y
        df_quiv['c'] = df['speed']

        #Save.
        self._data['add_quiver'] = [gs1, gs2] #df_quiv] geoseries/geodf only
        #Create/get image and add elements.
        fig, ax = self.fig_and_ax()
        ax.quiver(df_quiv['x'], df_quiv['y'], df_quiv['u'], df_quiv['v'], df_quiv['c'], angles='xy', 
                  scale_units='xy', scale=1, **{'cmap':cmap, 'width': width, **kwargs})
    
    def add_voronoi(self, show:str='duration', inter:bool=False, detach=0,
                    *, cmap:str=None, alpha:float=0.5, **kwargs) -> None:
        """Add colored (voronoi) cells to figure, coloring the area around a 
        location. If inter==True (default), also color the area around inter-
        mediate points returned by google. Pick color based on time it takes 
        to get there (if show == 'duration', default) or speed with which one 
        gets there (if show == 'speed'). If parameter 'detach' > 0 (i.e. 0.05), 
        a gutter is drawn between cells of 5% of their typical 1D-size. All 
        kwargs are passed to plot (geopandas.plot) function."""
        if show.lower() == 'duration':
            show_func = lambda d: d.duration / 3600
            label = 'Time needed to get to point [h]'
            if cmap is None:
                cmap = 'RdYlGn_r'
        elif show.lower() == 'speed':
            show_func = lambda d: d.crow_speed * 3.6
            label = 'Velocity to get to point, measured by air-distance [km/h]'
            if cmap is None:
                cmap = 'RdYlGn'
        else:
            raise ValueError("Parameter 'show' must be 'duration' or 'speed'.")
                
        #Get data.
        if inter:
            s_mov = self._map.steps(0)
        else:
            s_mov = self._map.directions
            
        mask = s_mov.apply(lambda x: x.route[-1]).duplicated()
        s_mov = s_mov[~mask]  #keep only one route per end point.
        locas = s_mov.apply(lambda x: x.end).values
        points = self.gdf_point_fromlocations(np.append(locas, [self._map.start])) #Add start point to keep region around it empty.
        self._map.save() #save, as previous action might have caused many api-calls.
        
        #Create (and clip) Voronoi cells.
        vor = Voronoi([(point.x, point.y) for point in points.geometry])
        regions, vertices = voronoi_finite_polygons_2d(vor)
        #Remove region around start again, so it won't be drawn (because we don't have a value for it).
        regions, points = regions[:-1], points[:-1]
        # points.drop(points.index[-1], inplace=True)     
        polys = [sg.Polygon(vertices[region]) for region in regions]
        gdf_voronoi = GeoDataFrame({'mov': s_mov.values, 'location': locas, 
                                    'point': points.geometry, 'geometry': polys},
                                   crs=points.crs)
        
        #Detach polygons from each other so they don't touch.
        if detach:
            dist = -(detach / 2) * np.sqrt(gdf_voronoi.geometry.area.median()) # typical size of cell.
            gdf_voronoi.geometry = gdf_voronoi.geometry.buffer(dist)
            mask = gdf_voronoi.geometry.apply(lambda p: p.area) > 0
            gdf_voronoi = gdf_voronoi[mask]
        #Clip to 'land' and visualization's clipping area.
        land = self._clip_and_reproject(ml._geodf()) 
        gdf_voronoi = gpd.clip(gdf_voronoi, land) #both in self._crs
        # remove = self._clip_and_reproject(gpd.read_file(remove))
        # gdf_voronoi = gpd.overlay(gdf_voronoi, remove, how='difference') #both in self._crs
        values = gdf_voronoi['mov'].apply(show_func)
        vmin, vmax = values.quantile([0.01, 0.99]) #domain of color bar.
        
        #Save.
        self._data['add_voronoi'] = [gdf_voronoi]
        #Create/get image and add elements.
        fig, ax = self.fig_and_ax()                
        legend = {'label': label, 'orientation': "horizontal", 'shrink':1, 'aspect':30, 'pad':0.02}
        gdf_voronoi.plot(ax=ax, column=values, vmin=vmin, vmax=vmax, legend=True, 
                         legend_kwds=legend, **{'cmap':cmap, 'alpha':alpha, **kwargs})
        
    def add_startpoint(self, *, color:str='black', alpha:float=0.9, marker:str='o', 
                       markersize:float=200, **kwargs) -> None:
        """Add start location to figure. All kwargs are passed to the plot 
        (GeoSeries.plot) function."""
        #Get data. 
        gdf = self.gdf_point_fromlocations([self._map.start])
        #Create/get image and add elements.
        fig, ax = self.fig_and_ax()   
        kwargs = {'alpha':alpha, 'color':color, 'marker':marker, 'markersize':markersize, **kwargs}
        gdf.plot(ax=ax, **kwargs)
    
    def add_endpoints(self, asfound:bool=False, inter:bool=False,
                      *, color:str='black', alpha:float=0.9, marker:str='o', markersize:float=3, **kwargs) -> None:
        """Add end locations to figure. Locations that no route has been found to
        are excluded. If asfound==False, put marker at specified location. If 
        asfound==True, put marker at location that a route was found to. (always 
        True if inter==True). If inter==True, also plot intermediate points returned 
        by google api (default False). All kwargs are passed to the plot 
        (GeoSeries.plot) function."""
        #Get data.
        if inter:
            s_mov = self._map.steps(0)
        else:
            s_mov = self._map.directions
                
        if asfound:
            s = [Location(coords) for coords in s_mov.apply(lambda m: m.route[-1]).unique()]
        else:
            s = s_mov.apply(lambda x: x.end)
        gdf = self.gdf_point_fromlocations(s)
        
        #Save.
        self._data['add_endpoints'] = [gdf]
        #Create image and add elements.
        fig, ax = self.fig_and_ax()
        gdf.plot(ax=ax, **{'alpha':alpha, 'color':color, 'marker':marker, 'markersize':markersize, **kwargs})