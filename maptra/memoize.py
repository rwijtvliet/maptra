#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-03-26

@author: Ruud Wijtvliet rwijtvliet@gmail.com
"""


import pickle

def memoize_immutable(f):
    """Memoization for functions that have immutable objects as arguments (faster
    than for functions that have mutable objects as arguments)."""
    memo = {}
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items())) #Must use frozenset because kwargs (= a dictionary) cannot be used as part of dictionary key 
        if not key in memo:
            memo[key] = f(*args, **kwargs)
            #print(f'Calculated "{f.__name__}" for args: {str(args)[:100]} and kwargs: {str(kwargs)[:100]}')
        else:
            pass
            #print(f'Looked-up "{f.__name__}" for args: {str(args)[:100]} and kwargs: {str(kwargs)[:100]}')
        return memo[key]
    return wrapper

def memoize_mutable(f):
    """Memoization for functions that have mutable objects as arguments."""
    memo = {}
    def wrapper(*args, **kwargs):
        key = pickle.dumps(args) + pickle.dumps(kwargs) #To use as hash for mutable objects.
        if not key in memo:
            memo[key] = f(*args, **kwargs)
            #print(f'Calculated "{f.__name__}" for args: {str(args)[:100]} and kwargs: {str(kwargs)[:100]}')
        else:
            pass
            #print(f'Looked-up "{f.__name__}" for args: {str(args)[:100]} and kwargs: {str(kwargs)[:100]}')
        return memo[key]
    return wrapper