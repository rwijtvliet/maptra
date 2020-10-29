# -*- coding: utf-8 -*-
"""
Module to turn a set of points into a set of nonoverlapping polygons.
Done by identifying the removable edges, and then removing one of them.
Rinse-repeat until no further walls can be removed.

Not optimized or anything. For example, all candidiates are recalculated
after removing a wall, and the Delaunay grid is calculated several times.

2020-10
rwijtvliet@gmail.com
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import Iterable

    
def is_convex_polygon(polygon):
    """Return True if the polynomial defined by the sequence of 2D
    points is convex, which is the case if 'driving around the polygon'
    means 'always steer to the left' or 'always steer to the right'. No checks
    are done for complex cases such as self-intersecting polygons etc.
    """
    polygon = np.array(polygon)
    if len(polygon) < 3: # Check for too few points
        return False
    orientation = 0
    for p1, p2, p3 in zip(*[np.roll(polygon, i, axis=0) for i in range(3)]):
        dxa, dya = p2 - p1
        dxb, dyb = p3 - p2
        cross = dxa*dyb - dya*dxb
        if not np.isclose(cross, 0.0):
            if orientation == 0:
                orientation = np.sign(cross)
            elif orientation != np.sign(cross):
                return False
    return True    
    
class Polygonate:
    """
    Turn a set of points into a set of polygons.
    
    Arguments:
        points: Iterable of (x, y)-points
        pickwall: 
            'long'to remove the longest walls first;
            'acute' to remove the most acute angles first (default);
            'round' to remove wall to create polygon with roundest corners.
        convex: True if resulting polygons must be convex (default).
    """
    
    def __init__(self, points:Iterable, pickwall:str='', convex:bool=True):
        self._points = np.array(points)
        self.__convex = convex
        self._shapes, self._neighbors_of_shapes, self._descendent_of_simplex = \
            self.__polygonation(pickwall)
    
    @property
    def points(self):
        """The (x, y) coordinates of the points."""
        return self._points
    
    @property
    def vertices(self):
        """The point-indices of the vertices of each shape."""
        return self._shapes
    
    @property
    def shapes(self):        
        """The point-indices of the vertices of each shape."""
        return self._shapes
    
    @property
    def neighbors_of_shapes(self):
        return self._neighbors_of_shapes
    
    def __polygonation(self, pickwall):
        self._delaunay = Delaunay(self._points)
        shapes = self._delaunay.simplices.tolist()
        descendent_of_simplex = np.arange(len(shapes))
        neighbors_of_shapes = [[si for si in neighbors if si != -1] 
                               for neighbors in self._delaunay.neighbors]
        
        if pickwall.startswith('long'):
            pickwallfunc = lambda cands: np.argmax([cand['len'] 
                                                    for cand in cands])
        elif pickwall.startswith('round'):
            pickwallfunc = lambda cands: np.argmin([cand['error'] 
                                                    for cand in cands])
        else:
            pickwallfunc = lambda cands: np.argmin([cand['angles_before'].min()
                                                    for cand in cands])
    
        def melt(si1, si2, shape3): #remove shapes with indices si1 and si2. Add shape with vertices shape3.
            nonlocal shapes, neighbors_of_shapes, descendent_of_simplex
            if si1 > si2: si1, si2 = si2, si1 #so that always si1 < si2
            shapes.pop(si2)
            shapes.pop(si1)  
            si3 = len(shapes)
            shapes.append(shape3)
            nei3 = [*neighbors_of_shapes.pop(si2), *neighbors_of_shapes.pop(si1)]
            nei3 = [si for si in nei3 if si != si1 and si != si2]
            neighbors_of_shapes.append(nei3)
            def new_si(si):
                if si==si1 or si==si2: return si3
                if si<si1: return si
                if si<si2: return si-1
                return si-2
            neighbors_of_shapes = [[new_si(si) for si in neighbors] 
                                   for neighbors in neighbors_of_shapes]
            descendent_of_simplex = [new_si(si) for si in descendent_of_simplex]
        
        while True:
            cands = self._candidates(shapes, neighbors_of_shapes)
            if len(cands) == 0: break
            # Find which one to remove.
            picked = cands[pickwallfunc(cands)]
            melt(*picked['si'], picked['shape3'])
        
        return shapes, neighbors_of_shapes, descendent_of_simplex
        
    def find_shape(self, point:Iterable, method=0):
        """Returns index of shape that contain the point."""  
        sim = self._delaunay.find_simplex(point)
        if sim > -1:
            return self._descendent_of_simplex[sim]
        return -1
    
    def _candidates(self, shapes, neighbors_of_shapes):
        """
        Find the edges that could be removed. Also store additional information, 
        such as wall length and existing angles.
        """
        def prepshape(shape, wall): #rotate/flip shape so, that wall[0] is at start and wall[1] is at end.
            while len(np.intersect1d(shape[0:2], wall)) != 2:
                shape = np.roll(shape, 1)
            shape = np.roll(shape, -1) #one vwall vertice at start, the other at the end.
            if shape[0] == wall[1]:
                shape = np.flip(shape) #vwall[0] is at beginning, vwall[1] is at end
            return shape
        def vec(*vi):
            return self._points[vi[1]] - self._points[vi[0]]
        def angle(vecA, vecB):
            cosangle = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
            return np.arccos(np.clip(cosangle, -1, 1))
        def PolyArea(vi):
            x, y = self._points[vi, 0], self._points[vi, 1]
            return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        
        candidates = []
        for si1, neighbors in enumerate(neighbors_of_shapes):
            shape1 = shapes[si1]
            for si2 in neighbors:
                if si1 > si2: continue #only add each wall once
                shape2 = shapes[si2] 
                # Find vertices of shared wall.
                wall = np.intersect1d(shape1, shape2)
                if len(wall) != 2: continue
                # Prepare by putting wall vertice 0 (1) at position 0 (-1) in each shape
                shape1, shape2 = prepshape(shape1, wall), prepshape(shape2, wall)
                # Get candidate-polygon
                shape3 = [*shape1[:-1], *shape2[::-1][:-1]]
                if self.__convex and not is_convex_polygon(self._points[shape3]): 
                    continue
                # Add characteristics.
                wallvec = vec(*wall) # pointing 0->1
                # Vectors pointing along the edges, starting at where wall is.
                vecs = [[vec(*vi) for vi in [shape1[:2], shape2[:2]]],
                        [vec(*vi) for vi in [shape1[-2:], shape2[-2:]]]]
                angles = np.array([[angle(wallvec, v) for v in vec] for vec in vecs]) #angles at corner 0, angles at corner 1
                candidates.append({
                    'wall': [*wall],
                    'si': [si1, si2], 
                    'shape3': shape3,
                    'len': np.linalg.norm(wallvec),
                    'angles_before': angles,
                    'angles_after': angles.sum(axis=1),
                    'error': np.abs(angles.sum(axis=1) - (np.pi * (1 - 2/len(shape3)))).mean()
                    })
        return candidates

    def plotpoints(self, ax, *args, **kwargs):
        ax.plot(*self._points.T, *args, **kwargs)
    def plotdelaunay(self, ax, *args, **kwargs):
        indptr, indices = self._delaunay.vertex_neighbor_vertices
        for vi1 in np.arange(len(self._points)):
            for vi2 in indices[indptr[vi1]:indptr[vi1+1]]:
                if vi1 < vi2:
                    ax.plot(*self._points[[vi1, vi2],:].T, *args, **{'alpha':1, **kwargs})
    def plotremovablewalls(self, ax, *args, **kwargs):
        cands = self._candidates(self._delaunay.simplices, self._delaunay.neighbors)
        for w in [cand['wall'] for cand in cands]:
            ax.plot(*self._points[w, :].T, *args, **{'color':'k', **kwargs})
    def plotpolygons(self, ax, *args, **kwargs):
        for shape in self._shapes:
            for vi in zip(shape, np.roll(shape, 1)):
                ax.plot(*self._points[vi,:].T, *args, **{'color':'b', **kwargs})
