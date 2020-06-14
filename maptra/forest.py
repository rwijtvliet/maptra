#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 00:46:17 2020

@author: ruud
https://en.wikipedia.org/wiki/Tree_(data_structure)#Terminology_used_in_trees
"""

from typing import List, Tuple, Dict, Iterable, Callable
import colorama #to draw in color to console

class ForestStruct:
    """Class to create and analyse a forest data structure; i.e., a collection
    of tree data structures. It is created by consecutively adding complete 
    (root-to-leaf) paths. About the internal data types:
    * A root-to-leaf path is represented by a list of nodes, starting with the 
    root. Other, partial (i.e., sub)paths also start with the node closest to 
    the root. 
    * A node may be a number, a string, or a complete object; JUST NOT A LIST!
    * A tree / a branch (='subtree') is also lists of nodes, but the final 
    element may be a 'branching point', which represents a fork.
    * A forest / a fork (='subforest') is a list of trees / branches. 
    So: there is a recursive pattern. The top-level is a forest. This is a list,
    which as elements contains trees. Each tree is a list, which as elements con-
    tains nodes before (possibly) ending with a fork (='subforest'). That fork
    is a list of branches (=subtrees), with each branch being a list of nodes 
    possibly ending in a fork, etc. 
    As the top-level is a forest, the class variable is called _forest."""
    def __init__(self):
        self._forest = []
        
    def add_paths(self, *paths:Iterable) -> None:
        """Integrate one or more complete root-to-leaf paths into forest."""
        for path in paths:
            if path is not None and len(path) > 0:
                self.__integrate(self._forest, path)
    
    def clear(self) -> None:
        """Clears forest."""
        self._forest = []
    
    @property
    def forest(self) -> List:
        """Return current forest as a list of trees, starting at their roots.
        Whereever a fork occurs, a list is inserted."""
        return self._forest
    @property
    def subpaths(self) -> Dict[int, List]:
        """Return non-forking subpaths of current forest, in a {count: paths}-
        dictionary. Key = number of root-to-leaf paths that contain the path.
        The paths also contain the last node before the fork, to ensure there
        are no gaps when drawing them."""
        return self.__split(self._forest)  
    
    @staticmethod
    def __integrate(forest:List, path:List) -> None:
        """Integrate a new root-to-leaf path into the forest by modifying/fork-
        ing/extending it where necessary, in place."""
        def integrate_into_fork(fork:List, path:List) -> bool:
            for branch in fork:                 #Check each branch:
                if integrate_into_branch(branch, path): #found correct branch and added path to it.
                    break
            else:                                       #did not find correct branch; make new branch.
                fork.append(path.copy())
            return True
        def integrate_into_branch(branch:List, path:List) -> bool:
            if len(branch) == 0:                #This branch is a leaf:
                return False                        #leave branch alone - don't add anything here.
            elif len(path) == 0:                #We're adding a leaf:
                return False                        #don't add anywhere - add as new sibbling branch.
            elif branch[0] != path[0]:          #Not even first node is equal - we're on wrong branch:
                return False                        #don't add here; maybe add to one of the sibbling branches.
            else:                               #We're on the correct branch - find where to integrate.
                for i, node in enumerate(branch):
                    if isinstance(node, List):              #Reached fork:
                        integrate_into_fork(node, path[i:])     #follow each branch and see where to merge.
                        break
                    if i >= len(path) or node != path[i]:   #Reached leaf of path or mismatching node:
                        branch[i:] = [[branch[i:], path[i:]]]   #create forking point here.        
                        break                                   
                else:                                       #Reached leaf (of tree, and possible also of path):
                    branch[i+1:] = [[branch[i+1:], path[i+1:]]] #create forking point here.        
                return True
        integrate_into_fork(forest, path)
    
    @staticmethod
    def __split(forest:List) -> Dict[int, List]:
        """Cut forest into non-forking (sub)paths, and groups by 'thickness' (i.e., 
        how many root-to-leaf paths contain them). Returns {count: paths} dictionary."""
        def split_fork(fork:List, node_to_prepend=None) -> Tuple[int, Dict[int, List]]:
            dg_f, dictio = 0, {} #degeneracy, dictionary
            for branch in fork:
                dg_b, dict_b = split_branch(branch)
                if node_to_prepend is not None:     #Prepend last node before fork to all branches, to make gapless.
                    dict_b[dg_b] = [[node_to_prepend] + path for path in dict_b[dg_b]]
                dict_b[dg_b] = [path for path in dict_b[dg_b] if len(path)>1] #Only keep paths with 2 or more nodes.
                dict_b = {dg: paths for dg, paths in dict_b.items() if len(paths)>0} #Only keep keys with 1 or more paths.
                for dg, paths in dict_b.items():    #Merge dictionaries.
                    dictio.setdefault(dg, []).extend(paths)
                dg_f += dg_b                        #Degeneracy of fork is sum of degeneracies of branches.
            return dg_f, dictio
        def split_branch(branch:List) -> Tuple[int, Dict[int, List]]:
            if len(branch) == 0 or not isinstance(branch[-1], List):
                return 1, {1: [branch]}             #Straight branch (i.e., ends in leaf).
            else:                                   #Forked branch.
                dg_b, dictio = split_fork(branch[-1], branch[-2]) #Degeneracy of branch is degeneracy of fork at its tip.
                dictio[dg_b] = [branch[:-1]]        #Add straight part (i.e., before fork) to dictionary.
                return dg_b, dictio
        _, dictio = split_fork(forest)
        return dictio

    def print_forest(self, charset:int=0) -> None:
        """Print forest to the console (mainly for debugging purposes).
        If charset==0, prints full string representation of each node. This may 
        be much text for large trees. For other values of charset (1..4), hash 
        each node and print a single colored character to represent it."""
        node2insetcol, print_path = self.__print_functions(charset)
        
        def print_fork(fork, inset=None) -> None:
            for idx, branch in enumerate(fork):
                print_branch(branch, inset, (idx == len(fork)-1))
        def print_branch(branch, inset=None, last_sibling:bool=False) -> None:
             #Prepare inset to accept last character. So, characters at pos. -3 and -2.
            if inset is None:                       #we are at the root
                inset = node2insetcol(0) + '─'
            else:                                   #we are not at the root
                inset = inset[:-12] + ('│' if inset[-12] in '│├┬' else ' ') + inset[-11:]   #pos -3
                inset = inset[:-2] + ('└' if last_sibling else '├')                          #pos -2
            #Add character at last position (i.e., pos. -1), and add nodes.
            if len(branch) == 0 or not isinstance(branch[-1], List):
                inset += '─'                        #Straight branch (i.e., ends in leaf).
                print_path(inset, branch)
            else:                                   #Forked branch.
                inset += '┬'
                print_path(inset, branch[:-1])      #Print straight part (i.e., before fork).
                inset = inset[:-1] + node2insetcol(branch[-2]) + inset[-1] + ' ' #insert style
                print_fork(branch[-1], inset)
            if last_sibling:
                inset = inset[:-11] + inset[-2:]

        print_fork(self._forest)
        
    def print_subpaths(self, charset:int=0) -> None:
        """Print non-forking subpaths to console, and how many root-to-leaf 
        paths contain them (mainly for debugging purposes). If charset==0, prints 
        full string representation of each node. This may be much text for large 
        trees. For other values of charset (1..4), hash each node and print a 
        single colored character to represent it."""
        _, print_path = self.__print_functions(charset)
        subpaths = self.subpaths
        for cnt in sorted(list(subpaths.keys())):
            for path in subpaths[cnt]:
                print_path(str(cnt) + ':', path)
                
    @staticmethod
    def __print_functions(charset:int=0) -> Tuple[Callable, Callable]:
        """Return two functions: (1) that takes an object and returns a string 
        to set the style of a treeline, and (2) that takes an string and a path,
        and prints it to the console, repectively."""
        forenames = backnames = ['BLACK', 'RED', 'GREEN', 'YELLOW', 'BLUE', 'MAGENTA', 'CYAN', 'WHITE']
        stylnames = ['DIM', 'BRIGHT']
        
        #Unique color for the inset (line), based on node.
        insetcolors = [colorama.Style.__dict__[s] + colorama.Fore.__dict__[f] 
                        for s in stylnames for f in forenames] #important: all elements have len()==9. Therefore, can't use NORMAL, as that has 5 instead of 4 characters
        def node2insetcol(node):
            return insetcolors[hash(node) % 39323 % len(insetcolors)] 
        #Unique colored character for each node.
        if charset == 0:
            def node2char(node): 
                return colorama.Style.RESET_ALL + str(node) + ' '
        else:
            if charset == 2:        # letters a-zA-Z, with extra vowels
                chars = list('aeiou')
                chars += [u"\u0304" + c for c in chars] + [u"\u0331" + c for c in chars]
                chars += list('bcdfghjklmnpqrstvwxyz')
                chars += [c.upper() for c in chars]
                backnames = ['RESET']
            elif charset == 3:      # full-width blocks of varying height
                chars = list('▁▂▃▄▅▆▇█')
            elif charset == 4:      # full-height blocks of varying width
                chars = list('▉▊▋▌▍▎')
            else: #charset == 1     # single line
                chars = list("━")
            colorchars = [colorama.Fore.__dict__[f] + colorama.Back.__dict__[b] +
                          colorama.Style.__dict__[s] + c for s in stylnames
                          for f in forenames for b in backnames for c in chars
                          if f != b]
            def node2char(node):
                return colorchars[hash(node) % 39119 % len(colorchars)] #some large prime
        #Print inset and path.
        def print_path(inset:str, path:List=[]) -> None:
            linelen = 90
            maxx = linelen - 1 - int(len(inset)/10) #due to special chararcters, inset has 10 characters for every printing character.
            end = ('(empty)', '', '…')[ (len(path)>0) + (len(path)>maxx) ]
            print(colorama.Style.RESET_ALL + inset + ' ' 
                  + ''.join([node2char(node) for node in path[:maxx]]) 
                  + colorama.Style.RESET_ALL + end)

        return node2insetcol, print_path

