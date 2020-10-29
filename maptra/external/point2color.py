"""
Interpolation functions, mainly for color.

2020-09 rwijtvliet@gmail.com
"""    

from matplotlib.colors import to_rgba, LinearSegmentedColormap
import numpy as np
from typing import Iterable, Callable, Tuple
from scipy.spatial import Delaunay, ConvexHull
from maptra.external.polygonate import Polygonate


SR3 = np.sqrt(3)


#%% Interpolation function in single polygon.

def polygon(anchorpoints:Iterable, anchorvalues:Iterable) -> Callable:
    """
    Generalised barycentric interpolation function for point in a single 
    polygon.
    
    Polygon has n vertices ("anchorpoints" of shape (n, 2)) in order
    and n values ("anchorvalues", e.g. iterable of n floats, or np-array of 
    shape (n, 3) with colors).
    
    The returned function takes a point (2-Iterable) and returns the inter-
    polated value (or extrapolated value if the point lies outside the shape).
    
    Source: https://www.inf.usi.ch/hormann/papers/Hormann.2006.MVC.pdf
    """
    anchorpoints = np.array(anchorpoints)
    anchorvalues = np.array(anchorvalues)
    if len(anchorpoints) != len(anchorvalues):
        raise ValueError("Parameters 'anchorpoints' and 'anchorvalues' must be of equal length.")
    if len(anchorpoints) < 3:
        raise ValueError("At least 3 anchorpoints must be specified.")
    
    F = anchorvalues
    F_next = np.roll(anchorvalues, -1, axis=0)
    eps = 1e-7
    def interp(point):  
        S = anchorpoints - point #distance vector to each anchorpoint
        R = np.linalg.norm(S, axis=1) #distances to each anchorpoint
        for r, f in zip(R, F): 
            if -eps < r < eps: 
                return f #point is on anchor point
            
        S_next = np.roll(S, -1, axis=0) # neighbors
        R_next = np.roll(R, -1)
        A = np.array([np.linalg.det([s, s_next]) 
                      for s, s_next in zip(S, S_next)]) #determinant of each displ. vector with neighbor
        D = np.array([np.dot(s, s_next) 
                      for s, s_next in zip(S, S_next)])
        for a, d, r, r_next, f, f_next in zip(A, D, R, R_next, F, F_next):
            if -eps < a < eps and d < 0:
                return (r_next*f + r*f_next) / (r + r_next) #point is on edge between anchor points
        
        T = np.array([a / (r*r_next + d) 
                      for a, d, r, r_next in zip(A, D, R, R_next)])
        T_prev = np.roll(T, 1)
        W = np.array([(t_prev + t) / r for t_prev, t, r in zip(T_prev, T, R)])
        return W.dot(F) / W.sum()
    
    return interp


#%% Interpolation functions in plane with points.

def triangles(anchorpoints:Iterable, anchorvalues:Iterable, valoutside=None) -> Callable:
    """
    Standard barycentric interpolation function for point inside a single 
    triangle, or, more generally, for point inside the convex hull around a 
    set of anchor points that is tesselated with triangles.
    
    Triangle has 3 vertices ("anchorpoints" of shape (3, 2)) and 3 values
    ("anchorvalues", e.g. Iterable of 3 floats, or of 3 colors). When passed 
    >3 anchorpoints, the area is tesselated with triangles.
    
    The returned function takes a point (2-Iterable) and returns the inter-
    polated value. If the point lies outside the shape, "valoutside" is 
    returned (if specified) or an attempt at extrapolation is done (default).
    
    Source: https://stackoverflow.com/questions/57863618/how-to-vectorize-calculation-of-barycentric-coordinates-in-python
    """
    anchorpoints = np.array(anchorpoints)
    anchorvalues = np.array(anchorvalues)
    if len(anchorpoints) != len(anchorvalues):
        raise ValueError("Parameters 'anchorpoints' and 'anchorvalues' must be of equal length.")
    if len(anchorpoints) < 3:
        raise ValueError("At least 3 anchorpoints must be specified.")
    
    # Tesselate into simplexes (individual triangles).
    delaunay = Delaunay(anchorpoints) # each row has indices of the 3 anchorpoints that are the simplex corners.
    
    def interp(point):
        # Find simplex point is in.
        s = delaunay.find_simplex(point) #simplex-index that contains point. (-1 means point is in none)
        if s > -1: #normal point, inside the hull
            # Get barycentric coordinates of the triangle.
            b0 = delaunay.transform[s, :2].dot((point - delaunay.transform[s, 2]))
            weights = np.array([*b0, 1 - b0.sum()]) # add final coordinate / weight.
            indices = delaunay.simplices[s]
        else: #point outside the hull
            if valoutside:
                return valoutside
            # Find the 2 anchorpoints on the hull line nearest to the point
            hull = ConvexHull([*anchorpoints, point], qhull_options='QG' + str(len(anchorpoints))) 
            visible = hull.simplices[hull.good] #lines (anchorpoints) visible from the point
            for indices in visible: #anchor-indices of visible line
                p01 = point - anchorpoints[indices] #from line anchors to point
                lin = anchorpoints[indices[0]] - anchorpoints[indices[1]]
                dot12 = p01.dot(lin)
                if np.sign(dot12).sum() == 0: #inside line 'shadow' if one dot product <0, >0
                    lens = np.linalg.norm(p01, axis=1)
                    lens = np.abs(dot12)
                    weights = np.flip(lens) / lens.sum()
                    break
            else: #not in shadow of line - use value of nearest anchor.
                #Find nearest anchor (="anchor 0"). Must be included in 2 lines.
                indices = list(set(visible.flatten()))
                sd = ((anchorpoints[indices] - point)**2).sum(axis=1) #squared distance to each anchorpoint
                indices = [indices[np.argmin(sd)]] #keep only nearest one
                weights = [1]               

        # Get interpolated value.
        value = np.dot(anchorvalues[indices].T, weights)
        return value
    
    interp.delaunay = delaunay #attach to function for debugging/visualisation
    return interp


def polygons(anchorpoints:Iterable, anchorvalues:Iterable) -> Callable:
    """
    Generalised barycentric interpolation function for point in-/outside a 
    single polygon, or, more generally, for point in-/outside the convex hull 
    around a set of anchor points that is tesselated with polygons.
    
    The function tesselates when necessary.
    
    The returned function takes a point (2-Iterable) and returns the inter-
    polated value (or extrapolated value if the point lies outside the shape).
    """
    anchorpoints = np.array(anchorpoints)
    anchorvalues = np.array(anchorvalues)
    if len(anchorpoints) != len(anchorvalues):
        raise ValueError("Parameters 'anchorpoints' and 'anchorvalues' must be of equal length.")
    if len(anchorpoints) < 3:
        raise ValueError("At least 3 anchorpoints must be specified.")
    
    # Tesselate into polygons.
    pg = Polygonate(anchorpoints, convex=False)
    # Interpolation function for each polygon...
    interpf = [polygon(anchorpoints[shape], anchorvalues[shape])
               for shape in pg.shapes]
    # ...and inter(extra)polation function for the hull.
    hull = ConvexHull(anchorpoints).vertices
    interpf_hull = polygon(anchorpoints[hull], anchorvalues[hull])
    
    def interp(point):
        # Find simplex point is in.
        s = pg.find_shape(point) #simplex that contains point. (-1 means point is in none)
        if s > -1: #normal point, inside the hull
            return interpf[s](point)
        else: #point outside the hull
            return interpf_hull(point)
    
    interp.polygonate = pg #attach to function for debugging/visualitation
    return interp


# %% ColorMap objects

class ColorMap2:
    """
    Create colormap to find color from the relative sizes of 2 float values.
    
    Arguments:
        colorsA (colorsB) is a list of colors, which are used if the value on 
        axis A (B) is largest. colorCenter, if specified, is the color used if
        values on both axes are equal.
        maxdiff: if the difference between the values is larger than this 
        maximum, the colors are clipped.
    """
    def __init__(self, 
                 colorsA:Iterable=['red'], 
                 colorsB:Iterable=['green'], 
                 colorCenter=None, 
                 maxdiff=1):
        self.maxdiff = maxdiff
        self._interp = self._interpf(colorsA, colorsB, colorCenter)
        
    def _interpf(self, colorsA:Iterable, colorsB:Iterable, colorCenter=None):
        """Returns interpolation function to find color for normalized values u and v."""
        lst = []
        for c, v in zip(colorsA, np.linspace(0, 0.5, len(colorsA), False)):
            lst.append((v, to_rgba(c)))
        if colorCenter:
            lst.append((0.5, to_rgba(colorCenter)))
        for c, v in zip(np.flip(colorsB, axis=0),
                        np.flip(np.linspace(1, 0.5, len(colorsB), False))):
            lst.append((v, to_rgba(c)))
        cmap = LinearSegmentedColormap.from_list('2axes', lst)
        f = lambda u, v: cmap(self._uv_to_x((u, v), True))
        return f
    
    def _uv_to_x(self, uv:Iterable[float], clip:bool=False) -> float:
        """
        Map normed values uv along axes A and B onto x-coordinate [0..1] along
        1D-colormap.
        . If clip==True, clips the x-coordinate so that it stays inside [0, 1].
        """
        x = 0.5 + 0.5*(uv[1] - uv[0])
        if clip:
            x = np.clip(x, 0, 1)
        return x

    def color(self, *ab:float) -> Tuple[float]:
        """
        Function that accepts values a and b, and returns the color that 
        best fits this datapoint. Smallest value is used as reference for the
        other one.
        """
        uv = np.array(ab) / self.maxdiff #normalize
        return to_rgba(np.clip(self._interp(*uv), 0, 1))
    
    def colorbar(self, cax, pixels=200, orientation='horizontal', aspect=20):
        """Draws a legend into the cax axes object."""
        img = np.array([[self.color(0, b) for b in self.maxdiff * np.linspace(-1, 1, pixels)]])
        extent = np.array([0, 1, -0.5/aspect, 0.5/aspect]) #* self.maxdiff
        if orientation=='vertical':
            img = img.transpose(1, 0, 2)
            extent = np.roll(extent, 2)
        cax.imshow(img, extent=extent, origin='lower', aspect='equal')
        if orientation=='vertical':
            cax.set_xticks([])
        else:
            cax.set_yticks([])
        
        def setticks(ab_list:Iterable[Iterable[float]],  *args, **kwargs):
            ticks = []
            for ab in ab_list:
                uv = np.array(ab) / self.maxdiff
                x = self._uv_to_x(uv, clip=True)
                ticks.append(x)
            if orientation == 'vertical':
                cax.set_yticks(ticks)
            else:
                cax.set_xticks(ticks)
            cax.get_ticks = lambda: ab_list
        cax.set_ticks = setticks
        
        def setticklabels(label_list:Iterable[str], *args, **kwargs):   
            if orientation == 'vertical':
                cax.set_yticklabels(label_list, *args, **kwargs)
            else:
                cax.set_xticklabels(label_list, *args, **kwargs)
        cax.set_ticklabels = setticklabels


class ColorMap3:
    """
    Create colormap to find color from the relative sizes of 3 float values.
    
    Arguments:
        colorsA (colorsB, colorsC) is a list of colors, which are used if value 
        a (b, c) is largest, and the other two are equal. 
        colorCenter, if specified, is the color used if values a, b, and c are 
        equal. 
        (In other cases, there is an interpolation between the specified colors.)
        maxdiff: if the difference between values is larger than this maximum, 
        the colors are clipped.
    """

    _cornersABC = np.array([(0, 1), (-SR3/2, -0.5), (SR3/2, -0.5)])

    def __init__(self, 
                 colorsA:Iterable=['red'], 
                 colorsB:Iterable=['green'], 
                 colorsC:Iterable=['blue'], 
                 colorCenter=None, 
                 maxdiff=1):
        self.maxdiff = maxdiff
        self._interp = self._interpf(colorsA, colorsB, colorsC, colorCenter)
    
    def _interpf(self, colorsA:Iterable, colorsB:Iterable, colorsC:Iterable,
                        colorCenter=None) -> Callable:
        """
        Returns interpolation function to find color for normalized u, v and w.
                
        Implementation details:
            Interpolation in 3 triangular sections, namely (A, B, center), (B,
            C, center), (C, A, center).
            colorCenter not specified: Center color is calculated from innermost
            elements in colorsA, colorsB, colorsC.
            If a shape is triangular and has its color specified only at the 
            corners, 'normal' barycentric coordinates are used. Otherwise, 
            'generalised' barycentric coordinates are used.
        """
        anchorpoints, anchorcolors = [], []
        colorsABC = [[to_rgba(col) for col in cols] 
                     for cols in [colorsA, colorsB, colorsC]]
        for corner, cols in zip(self._cornersABC, colorsABC):
            for i, col in enumerate(cols):
                anchorpoints.append(corner * (1 - i/len(cols)))
                anchorcolors.append(col)
        if colorCenter: 
            anchorpoints.append([0, 0])
            anchorcolors.append(to_rgba(colorCenter))
        anchorpoints, anchorcolors = np.array(anchorpoints), np.array(anchorcolors)
        
        pg = Polygonate(anchorpoints, convex=True) #divide into up to 4 convex shapes
        interpf = [polygon(anchorpoints[ai], anchorcolors[ai]) for ai in pg.shapes]

        def f(*uvw:float):
            xy = self._uvw_to_xy(uvw, clip=True)
            s = pg.find_shape(xy)
            if s > -1:
                return interpf[s](xy)
            return to_rgba('k') #should never happen
        
        return f

    def _uvw_to_xy(self, uvw:Iterable[float], clip:bool=False) -> Tuple[float]:
        """
        Map normed values uvw along axes ABC onto xy-coordinates inside triangle.
        . If clip==True, clips the xy-coordinates so that they stay inside the
          triangle.
        """
        uvw = np.array(uvw)
        uvw = uvw - uvw.min() #does not change location of point
        if clip and uvw.sum() > 1:
            uvw /= uvw.sum() #make sure it stays inside triangle
        xy = uvw.dot(self._cornersABC)    
        return xy
    def _xy_to_uvw(self, xy:Iterable[float], positive:bool=False) -> Tuple[float]:
        """
        Turn xy-coordinates into a tuple of normed values uvw that maps onto it.
        . If positive==True, the tuple contains nonnegative values with at least 
          being 0.
        """
        xy = np.array(xy)
        uvw = self._cornersABC.dot(xy)
        if positive:
            uvw -= uvw.min()
        return uvw

    def color(self, *abc:float) -> Tuple[float]:
        """
        Function that accepts values a, b, and c, and returns the color that 
        best fits this datapoint. Smallest value is used as reference for the
        other ones.
        """
        uvw = np.array(abc) / self.maxdiff #normalize
        return to_rgba(np.clip(self._interp(*uvw), 0, 1))
    
    def colortriangle(self, cax, pixels=100):
        """Draws a triangular legend into the cax axes object."""
        shape = (int(pixels*0.8), pixels)
        img = np.zeros((*shape, 4)) # transparent image
        es = np.array([(0, 1), (-SR3/2, -0.5), (SR3/2, -0.5)]) #unit vectors in axes' directions
        for j, x in enumerate(np.linspace(-1, 1, shape[1])):
            for i, y in enumerate(np.linspace(-0.65, 1.05, shape[0])):
                xy = np.array((x, y))
                uvw = self._xy_to_uvw(xy)
                k = np.argmin(uvw)
                if uvw[k] > -0.52:
                    abc = uvw*self.maxdiff
                    img[i, j, :] = self.color(*abc)
    
        cax.imshow(img, extent=[-1, 1, -0.65, 1.05], origin='lower', aspect='equal',
                   interpolation='bicubic')
        cax.set_yticks([])
        cax.set_xticks([])
        cax.patch.set_alpha(0)
        for s in cax.spines.values():
            s.set_visible(False)
        
        line = np.array([es[k] for k in [0,1,2,0]])
        mask = np.append(line, (line*1.2)[::-1], axis=0)
        cax.fill(mask[:,0], mask[:,1], 'w')
        cax.plot(line[:,0], line[:,1], 'k', linewidth=1)
        
        def setticks(abc_list:Iterable[Iterable[float]],  *args, **kwargs):
            for abc in abc_list:
                uvw = np.array(abc) / self.maxdiff
                xy = self._uvw_to_xy(uvw, clip=True)
                cax.plot(*xy, *args, **kwargs)
            cax.get_ticks = lambda: abc_list
        cax.set_ticks = setticks
        
        def setticklabels(label_list:Iterable[str], *args, **kwargs):
            for abc, label in zip(cax.get_ticks(), label_list):
                uvw = np.array(abc) / self.maxdiff
                xy = self._uvw_to_xy(uvw, clip=True)
                cax.text(*xy + [0,-0.1], label, *args, **{'ha':'center', 'va':'top', 'bbox':{
                    'color':'w', 'alpha':0.3}, **kwargs})
        cax.set_ticklabels = setticklabels
        
    def colorbars(self, cax, pixels=200, orientation='horizontal', vstacked=False):
        """Draws a 3 colorbar legend into the cax axes object."""
        from mpl_toolkits.axes_grid.inset_locator import inset_axes
        cax.set_yticks([])
        cax.set_xticks([])
        cax.patch.set_alpha(0)
        for s in cax.spines.values():
            s.set_visible(False)

        if (orientation=='vertical') ^ (not vstacked):
            width, height, locs = '100%', '23%', [9, 10, 8]
        else:
            width, height, locs = '25%', '100%', [6, 10, 7]
        axs = [inset_axes(cax, width, height, loc) for loc in locs]
        for i, ax in enumerate(axs):
            if i == 0:   fun = lambda a: self.color(a, 0, 0)
            elif i == 1: fun = lambda b: self.color(0, b, 0)
            else:        fun = lambda c: self.color(0, 0, c)
            img = np.array([[fun(d) for d in np.linspace(0, self.maxdiff, pixels)]])
            if orientation=='vertical':
                img = img.transpose(1, 0, 2)
            ax.imshow(img, extent=[0,self.maxdiff,0,self.maxdiff], origin='lower', 
                      aspect='auto', interpolation='bicubic')
            if orientation=='vertical':
                ax.set_xticks([])
                ax.set_ticks = ax.set_yticks
                ax.set_ticklabels = ax.set_yticklabels
            else:
                ax.set_yticks([])
                ax.set_ticks = ax.set_xticks
                ax.set_ticklabels = ax.set_xticklabels
            ax.set_ticks([0, self.maxdiff])
            ax.set_ticklabels(['same', 'max'])
        cax.subaxes = axs