#TODO: Add type hints
#TODO: Find alternatives for 'apply' (https://stackoverflow.com/questions/54432583/when-should-i-ever-want-to-use-pandas-apply-in-my-code)


import numpy as np
import pandas as pd
import googlemaps
import pickle
from rw_gmaps import Location, Directions

gmaps = googlemaps.Client(key='')

start = Location(gmaps, address="Stresemannstrasse 326, Hamburg")
print(f"Starting location: {start.coords}, which is {start.address}.")

#%% Create the grid with locations.

# Create dataframe to hold the grid.
grid_stepnum = 3 #4 means, we will go 4 steps in each direction (north, south, east, west). So: total number of points: (n*2+1)^2
grid_distance = 1000 #4000 means, we will go 4000 m in each direction
grid_stepsize = grid_distance / grid_stepnum
print(f'The grid:\n. There will be {2*grid_stepnum+1} points on each side, i.e., {(grid_stepnum*2+1)**2} points on a square grid.')
print(f'. The distance between points on the central axes (north/south, and east/west) is {grid_stepsize} m.') # On the other lines it may be slightly different due to curvature.
grid = pd.DataFrame(data=None, columns=np.arange(-grid_stepnum, grid_stepnum+1), index=-np.arange(-grid_stepnum, grid_stepnum+1))
print(f'. Empty grid:\n{grid}')

#Find points on main axes.
for s in np.arange(-grid_stepnum, grid_stepnum+1):
    if s==0: 
        grid.loc[0, 0] = start
    grid.loc[s, 0] = Location(gmaps, ll=start.ll.destination(s*grid_stepsize, 0))
    grid.loc[0, s] = Location(gmaps, ll=start.ll.destination(s*grid_stepsize, 90))

#Find points away from main axes. Which are the crossing of a point A on the east/west, and a point B on the north/south axis.
for s_east in np.arange(-grid_stepnum, grid_stepnum+1):
    if s_east == 0:
        continue
    pointA = grid.loc[0, s_east].ll
    bearAtoStart = pointA.initialBearingTo(start.ll) #looking towards start point: 'eastish' (westish) for points west (east) of start point (on main east/west axis).
    
    for s_north in np.arange(-grid_stepnum, grid_stepnum+1):
        if s_north == 0:
            continue    
        pointB = grid.loc[s_north, 0].ll
        bearBtoStart = 0 if s_north > 0 else 180 #looking toward start point: south (north) for points north (south) of start point (on main north/south axis).
        
        bearA = bearAtoStart + 90 * np.sign(s_north) * np.sign(s_east) #turn 90 degrees to point into correct quadrant.
        bearB = bearBtoStart - 90 * np.sign(s_north) * np.sign(s_east) #turn 90 degrees to point into correct quadrant.
        # Find crossing between point A and B.
        grid.loc[s_north, s_east] = Location(gmaps, ll=pointA.intersection(bearA, pointB, bearB))

#Show the coordinates.
u = np.vectorize(lambda x: f"({x.coords[0]:.3f}, {x.coords[1]:.3f})" if hasattr(x, 'coords') else 0)
print(f'. (Coordinate attributes of) locations of filled grid:\n{grid.apply(u)}')

#%% Find the direction to each location of the grid.

grid_dirs = grid.applymap(lambda end: Directions(gmaps, grid.loc[0,0], end))

#Show the directions for a single location.
d = grid_dirs.loc[1, 1]
print('Directions to the location [1, 1] in the grid:')
print(f'. End location: {d.end.coords}, which is {d.end.address}.')
print(f'. Distance-by-air: {d.distance_by_air} m, over the road: {d.distance} m.')
print(f'. Route by coordinates: {d.route}.')

grid_dirs.to_pickle('pickle/grid_dirs_storedondisk.pkl')


#%% Get coordinates of 'shifted' locations, in wanted coordinate reference system.
grid_dirs = pd.read_pickle('pickle/grid_dirs_storedondisk.pkl')

import geopandas as gpd
import shapely.geometry as sg

list_dirs = pd.DataFrame(data={'object': grid_dirs.values.flatten()})
list_dirs['coords1'] = list_dirs['object'].apply(lambda x: x.end.coords)
list_dirs['point1'] = list_dirs['coords1'].apply(lambda x: sg.Point((x[1], x[0])))
av_distancing_speed = list_dirs['object'].apply(lambda x: x.distancing_speed).mean()
list_dirs['coords2'] = list_dirs['object'].apply(lambda x: x.corrected_end(av_distancing_speed).coords)
list_dirs['point2'] = list_dirs['coords2'].apply(lambda x: sg.Point((x[1], x[0])))
list_dirs['route'] = list_dirs['object'].apply(lambda x: [sg.Point(s[1], s[0]) for s in x.route])
print(list_dirs)

#Get geodataframes.
path_base = 'data/bbbike/planet_9.8655,53.5285_10.0298,53.5986-shp/shape/'
gdf = {
    'roads': gpd.read_file(path_base + 'roads.shp'),
    'landuse': gpd.read_file(path_base + 'landuse.shp'),
    'rail': gpd.read_file(path_base + 'railways.shp'),
    'locs1': gpd.GeoDataFrame(data=list_dirs[['coords1', 'point1']], geometry='point1', crs={'init': 'epsg:4326'}),
    'locs2': gpd.GeoDataFrame(data=list_dirs[['coords2', 'point2']], geometry='point2', crs={'init': 'epsg:4326'})
    } 

#Reproject.
to_crs = {'init': 'epsg:3395'}
for key in gdf.keys():
    gdf[key] = gdf[key].to_crs(to_crs)


#%% Make images; convert coordinates to pixels.

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw

#Save images without distortion.
fig, ax = plt.subplots(figsize=(30, 20))
gdf['roads'].plot(ax=ax, alpha=0.3, color='grey')
gdf['rail'].plot(ax=ax, alpha=0.3, color='grey')
plt.tight_layout()
fig.savefig('output/map_nowarp_empty.png')
gdf['locs1'].plot(ax=ax, markersize=100, color='orange', marker='o')
gdf['locs2'].plot(ax=ax, markersize=100, color='red', marker='o')
fig.savefig('output/map_nowarp_locations.png')


#Find pixel positions of end locations and of shifted end locations.
def geometry_to_pixel_function(fig, ax):
    """Returns function to turn geometry into pixel x, y position."""
    cnvs = FigureCanvasAgg(fig)
    fig.set_canvas(cnvs)
    _, height = cnvs.get_width_height() #height needed because mpl has origin in bottom left.    
    def function(geometry):
        x, y = ax.transData.transform(geometry)
        return x, height - y
    return function

geo2pix = geometry_to_pixel_function(fig, ax)

pixelpos1 = gdf['locs1'].geometry.apply(geo2pix)
pixelpos2 = gdf['locs2'].geometry.apply(geo2pix)


command = "convert output/map_nowarp_grid.png -virtual-pixel Black -distort Shepards '"
command += ''.join([f'{x0:.0f},{y0:.0f}, {x1:.0f},{y1:.0f}   ' for (x0, y0), (x1, y1) in zip(pixelpos1, pixelpos2)])
command += "' +repage map_warped_grid.png"

#Draw intermediate image to check warp vectors.
image = Image.open('output/map_nowarp_locations.png')
draw = ImageDraw.Draw(image)
for (x0, y0), (x1, y1) in zip(pixelpos1, pixelpos2):
    draw.ellipse([x0-5, y0-5, x0+5, y0+5], fill='orange')
    draw.line([x0, y0, x1, y1], fill='red', width=2)
image.save('output/map_nowarp_vectors.png')


#%%





#Do warp.
def warp(image, points):
    """Warp image according to list of (x0, y0, x1, y1)-tuples. Returns warped
    image."""
    def distance(p1, p2) -> float:
        return np.sqrt((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) **2)
    
    warpinfo = []
    for x0, y0, x1, y1 in points:
        end_point = (x1, y1)
        shift_vector = (x1-x0, y1-y0)
        len_shift_vector = distance((x0, y0), (x1, y1))
        warpinfo.append([end_point, shift_vector, len_shift_vector])
    
    result = Image.new("RGB", image.size)
    
    image_pixels = image.load()
    result_pixels = result.load()
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            offset = [0, 0]
            for end_point, shift_vector, len_shift_vector in warpinfo:
                if len_shift_vector > 0:
                    helper = 1 / (3 * (distance((x, y), end_point) / len_shift_vector) ** 4 + 1)
                else:
                    helper = 0
                    
                offset[0] -= helper * shift_vector[0]
                offset[1] -= helper * shift_vector[1]

            coords = (int(np.clip(x + offset[0], 0, image.size[0])), 
                      int(np.clip(y + offset[1], 0, image.size[1])))
            
            result_pixels[x, y] = image_pixels[coords]

    return result

image = Image.open('map_nolocations.png')
res = warp(image, (points_original + points_shifted).tolist())
res.save('map_warped.png')