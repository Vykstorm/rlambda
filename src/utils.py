
'''
This module provides helper routines for the rest of the modules in this library.
'''

from types import FunctionType


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
    :param x:
    :param t:
    :return:
    '''
    assert iterable(x)
    return any(map(instanceofchecker(t), x))

def allinstanceof(x, t):
    '''
    This method returns True if all items in the iterable x are instances of any of the classes indicated in t (or any of their
    subclasses)
    :param x:
    :param y:
    :return:
    '''
    assert iterable(x)
    return all(map(instanceofchecker(t), x))


def anyoftype(x, t):
    '''
    This method return true if any item in the iterable x is an instance of one of the classes indicated in t
    :param x:
    :param t:
    :return:
    '''
    assert iterable(x)
    return any(map(typechecker(t), x))


def alloftype(x, t):
    '''
    This method returns true if all items in the iterable x are instance of one of the classes indicated in t
    :param x:
    :param t:
    :return:
    '''
    assert iterable(x)
    return all(map(typechecker(t), x))


def instanceofchecker(t):
    '''
    Returns a callable object that is equivalent to: lambda x: isinstance(x, t)
    :param x:
    :param t:
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
    :param t:
    :return:
    '''
    assert isinstance(t, type) or (iterable(t) and len(tuple(t)) > 0 and all(map(lambda item: isinstance(item, type), t)))
    if isinstance(t, type):
        return lambda x: type(x) == t
    t = tuple(t)
    return lambda x: type(x) in t


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