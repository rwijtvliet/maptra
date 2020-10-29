#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:06:39 2020

@author: ruud
"""


import matplotlib.pyplot as plt
import maptra.external.point2color as p2c
from numpy import random as rnd

cm = ColorMap3(['#D87A11', '#F39935'], 
               ['#6C2FB1', '#8249C2'], 
               ['#22751E', '#3C9537'],
               maxdiff=2)

fig, ax = plt.subplots(figsize=(10, 9))
cax = fig.add_axes([0.6, 0, 0.4, 0.4])
cax2 = fig.add_axes([0, 0, 0.4, 0.25])

for i in range(100):
    a, b, c = rnd.rand(3)
    circle = plt.Circle((a, b), c/10, color=cm.color(a, b, c))
    ax.add_artist(circle)

cm.colortriangle(cax)
cax.set_ticks([(2,0,0), (0,2,0), (0,0,2)], '')
cax.set_ticklabels(['a', 'b', 'c'], bbox={'alpha':0})


cm.colorbars(cax2)
# cax.subaxes[1].set_title('b')
# cax.subaxes[2].set_title('c')
