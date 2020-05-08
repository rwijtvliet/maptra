#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:59:51 2020

@author: ruud
"""
from rwmap.visualize import Visualization

viz = Visualization()



#%%Start here with a viz object that has been drawn to your liking.



import numpy as np
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw

def fn(stage:int=0, is_test:bool=False, add:str=''):
    base = f'stage {stage}'
    if add:
        base += f' {add}'
    if is_test:
        base += ' - test'
    return base + '.png'
    
    
#%% Stage 0: Save image and testimage without distortion. Find distortion vectors.
fig, ax = viz.fig_and_ax(figsize=(5,5))
fig.savefig(fn(0))

image = Image.open(fn(0))
draw = ImageDraw.Draw(image)
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
for i, x in enumerate(range(0, image.width, 30)):
    draw.line([x, 0, x, image.height], fill=colors[i%len(colors)], width=2)
for i, y in enumerate(range(0, image.height, 30)):
    draw.line([0, y, image.width, y], fill=colors[i%len(colors)], width=2)
image.save(fn(0, True))

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

pixelpos1 = viz._data['add_quiver'][0].apply(geo2pix)
pixelpos1 = np.array([*pixelpos1])
pixelpos2 = viz._data['add_quiver'][1].apply(geo2pix)
pixelpos2 = np.array([*pixelpos2])

#%% Stage 1: Save image and testimage with Shepard's distortion

def shepards(power, src, dest):
    command = f"convert '{src}' -virtual-pixel Black -define shepards:power={power} -distort Shepards '"
    command += ''.join([f'{x0:.0f},{y0:.0f}, {x1:.0f},{y1:.0f} ' for (x0, y0), (x1, y1) in zip(pixelpos1, pixelpos2)])
    command += f"' +repage '{dest}'"
    return command

for powr in [0.5, *range(1,8)]:    
    c = shepards(powr, fn(0, False), fn(1, False, f'shep{powr}'))
    print(f'executing {c}')
    os.system(c)
    os.system(shepards(powr, fn(0, True), fn(1, True, f'shep{powr}')))

# #Draw intermediate image to check warp vectors.
# image = Image.open('output/map_nowarp_empty.png')
# draw = ImageDraw.Draw(image)
# for (x0, y0), (x1, y1) in zip(pixelpos1, pixelpos2):
#     draw.ellipse([x0-5, y0-5, x0+5, y0+5], fill='orange')
#     draw.line([x0, y0, x1, y1], fill='red', width=2)
# image.save('output/map_nowarp_vectors.png')


#%% Stage 2: thin-plate spline distortion (?)
#Source: https://github.com/scikit-image/scikit-image/issues/2429


from scipy.interpolate import Rbf
import skimage.transform

class PointsRBF:
    def __init__(self, src, dst):
         xsrc = src[:,0]
         ysrc = src[:,1]
         xdst = dst[:,0]
         ydst = dst[:,1]
         self.rbf_x = Rbf( xsrc, ysrc, xdst)
         self.rbf_y = Rbf( xsrc, ysrc, ydst)

    def __call__(self, xy):
        x = xy[:,0]
        y = xy[:,1]
        xdst = self.rbf_x(x,y)
        ydst = self.rbf_y(x,y)
        return np.transpose( [xdst,ydst] )

def warpRBF(image, src, dst):
    prbf = PointsRBF( dst, src)
    warped = skimage.transform.warp(image, prbf)
    warped = 255*warped                         # 0..1 => 0..255
    warped = warped.astype(np.uint8)            # convert from float64 to uint8
    return warped

w = warpRBF(image, np.array([*pixelpos1]), np.array([*pixelpos2]))
i = Image.fromarray(w)
i.save(fn(5))

#TODO: check https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.interpolate.Rbf.html

#%% Stage 3: Thin-plate

# import morphops as mops

# res = mops.gpa([pixelpos1, pixelpos2, image])

from rwmap.image_warp import warp_images

inp = np.asarray(image)
warped = warp_images(pixelpos1, pixelpos2, [inp[:,:,0].T, inp[:,:,1].T, inp[:,:,2].T], [0, 0, image.width-1, image.height-1])
i = Image.fromarray(warped[0].T)
i.save(fn(6))
i

#%%Do warp.
import numpy as np
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
            weightsum = 0
            for end_point, shift_vector, len_shift_vector in warpinfo:
                if len_shift_vector > 0:
                    weight = 1 / (distance((x, y), end_point) ** 2 + 0.5)                
                    weightsum += weight
                    offset[0] -= weight * shift_vector[0]
                    offset[1] -= weight * shift_vector[1]
            if weightsum > 0:
                offset[0] = offset[0] /weightsum
                offset[1] = offset[1] /weightsum

            coords = (int(np.clip(x + offset[0], 0, image.size[0]-1)), 
                      int(np.clip(y + offset[1], 0, image.size[1]-1)))
            
            result_pixels[x, y] = image_pixels[coords]

    return result

image = Image.open(filename)
# res = warp(image, (points_original + points_shifted).tolist())
res = warp(image, (pixelpos1 + pixelpos2).tolist())
res.save('map_warped_2b.png')

