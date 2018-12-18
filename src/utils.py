
'''
This module provides helper routines for the rest of the modules in this library.
'''

from types import FunctionType
from inspect import isclass


def iterable(x):
    '''
    Checks if an object is iterable or not.
    An object is iterable if it defines the method __iter__
    :param x: Is the object to check if its iterable.
    :return: Returns True if x is iterable, False otherwise.
    '''
    try:
        iter(x)
        return True
    except:
        pass
    return False

def hashable(x):
    '''
    Checks if the given object implements the method __hash__
    :param x:
    :return:
    '''
    try:
        hash(x)
        return True
    except:
        pass
    return False

def islambda(x):
    '''
    Check if an object is a lambda function.
    :param x:
    :return:
    '''
    return isinstance(x, FunctionType) and x.__name__ == '<lambda>'


def anyinstanceof(x, t):
    '''
    This method returns True if any item of the iterable x is an instance of any of the classes indicated in t (or any of their
    subclasses)
    False otherwise
    :param x: Is an iterable of any type
    :param t: Must be a type or a tuple of types
    :return:
    '''
    assert iterable(x)
    return any(map(instanceofchecker(t), x))

def allinstanceof(x, t):
    '''
    This method returns True if all items in the iterable x are instances of any of the classes indicated in t (or any of their
    subclasses)
    :param x: Is an iterable of any type
    :param y: Must be a type or a tuple of types
    :return:
    '''
    assert iterable(x)
    return all(map(instanceofchecker(t), x))


def anyoftype(x, t):
    '''
    This method return true if any item in the iterable x is an instance of one of the classes indicated in t
    :param x: Is an iterable of any type
    :param t: Is a type or a tuple of types
    :return:
    '''
    assert iterable(x)
    return any(map(typechecker(t), x))


def alloftype(x, t):
    '''
    This method returns true if all items in the iterable x are instance of one of the classes indicated in t
    :param x: Is an iterable of any type
    :param t: Is a type or a tuple of types
    :return:
    '''
    assert iterable(x)
    return all(map(typechecker(t), x))


def instanceofchecker(t):
    '''
    Returns a callable object that is equivalent to: lambda x: isinstance(x, t)
    :param t: Must be a type or a tuple of types
    :return:
    '''
    assert isinstance(t, type) or (iterable(t) and len(tuple(t)) > 0 and all(map(lambda item: isinstance(item, type), t)))
    if not isinstance(t, type):
        t = tuple(t)
    return lambda x: isinstance(x, t)

def typechecker(t):
    '''
    Returns a callable object that is equivalent to: lambda x: type(x) == t   if the given argument t is a type.
    if its instead of a tuple of types, then its equivalent to lambda x: type(x) in t
    :param t: Is a type or a tuple of types
    :return:
    '''
    assert isinstance(t, type) or (iterable(t) and len(tuple(t)) > 0 and all(map(lambda item: isinstance(item, type), t)))
    if isinstance(t, type):
        return lambda x: type(x) == t
    t = tuple(t)
    return lambda x: type(x) in t

def findinstanceof(x, t):
    '''
    Find and return the first item that satisfies the next predicate: isinstance(item, t)
    Raises TypeError() if there isnt an item that satisfies the condition
    :param t: Is a type or a tuple of types.
    :return:
    '''
    assert isinstance(t, type) or (iterable(t) and len(tuple(t)) > 0 and all(map(lambda item: isinstance(item, type), t)))

    for item in x:
        if isinstance(item, t):
            return item
    raise TypeError()

def findsubclassof(x, cls):
    '''
    Find and return the first item in the container x that satisfies the next predicate: isclass(item) and issubclass(item, cls)
    Raises TypeError() if there isnt an item that satisfies the condition
    '''
    assert iterable(x) and isclass(cls)

    for item in x:
        if isclass(item) and issubclass(item, cls):
            return item
    raise TypeError()

def findsuperclassof(x, cls):
    '''
    Find and return the first item in the container x that satisfies the next predicate: isclass(item) and issubclass(cls, item)
    Raises TypeError() if there isnt an item that satisfies the condition
    '''
    assert iterable(x) and isclass(cls)

    for item in x:
        if isclass(item) and issubclass(cls, item):
            return item
    raise TypeError()



def enclose(s, chars='()'):
    '''
    Enclose the given string with the characters indicated
    :param x: Must be a string object
    :param chars: Must be an iterable with two items that will be used to enclose the given string
    :return: The same string but enclosed with the given chars.
    '''
    assert isinstance(s, str)
    assert iterable(chars) and allinstanceof(chars, str)

    chars = tuple(chars)
    assert 0 < len(chars) <= 2

    if len(chars) == 1:
        chars = 2 * chars
    return chars[0] + s + chars[1]


def slice_to_str(x):
    '''
    Stringifies a slice object.
    :param x: Must be an instance of the class slice
    :return:
    '''
    assert isinstance(x, slice)

    start, stop, step = tuple(map(lambda value: str(value) if value is not None else None, (x.start, x.stop, x.step)))
    sep = ':'

    if start is None and stop is None and step is None:
        items = sep,

    elif step is None:
        if stop is None:
            items = start, sep
        else:
            if start is None:
                items = sep, stop
            else:
                items = start, sep, stop
    else:
        if start is None and stop is None:
            items = sep, sep, step
        elif start is None:
            items = sep, stop, sep, step
        elif stop is None:
            items = start, sep, sep, step
        else:
            items = start, sep, stop, sep, step

    return ''.join(items)