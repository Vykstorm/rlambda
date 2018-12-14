
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


def enclose(s, chars='()'):
    '''
    Enclose the given string with the characters indicated
    :param x: Must be a string object
    :param chars: Must be an iterable with two items that will be used to enclose the given string
    :return: The same string but enclosed with the given chars.
    '''
    if not isinstance(s, str):
        raise TypeError()
    if not iterable(chars) or any(map(lambda c: not isinstance(c, str), chars)):
        raise TypeError()
    chars = tuple(chars)
    if not (0 < len(chars) <= 2):
        raise ValueError()
    if len(chars) == 1:
        chars = 2 * chars
    return chars[0] + s + chars[1]