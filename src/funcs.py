


'''
Additional functions to operate with rlambdas
'''

import builtins
import math
import operator
from types import SimpleNamespace
from functools import wraps
from itertools import chain
from src.rlambda import RLambda

_funcs = dict(builtins.__dict__)
_funcs.update(math.__dict__)
_builtins = SimpleNamespace(**_funcs)



def _build_wrappers(*args):

    def _build_wrapper(funcname):
        func = _builtins.__getattribute__(funcname)
        isinstance = _builtins.isinstance
        any, map = _builtins.any, _builtins.map

        @wraps(func)
        def _wrapper(*args, **kwargs):
            if any(map(lambda x: isinstance(x, RLambda), chain(args, kwargs.values()))):
                return RLambda._call_op(func, *args, **kwargs)
            return func(*args, **kwargs)

        globals()[funcname] = _wrapper

    for arg in args:
        _build_wrapper(arg)



# Built-in wrappers
_build_wrappers(
    'len', 'int', 'float', 'bool', 'str', 'complex', 'list', 'tuple', 'set', 'frozenset',
    'max', 'min'
)

# Math function wrappers
_build_wrappers(
    'ceil', 'copysign', 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'fsum', 'gcd',
    'isclose', 'isfinite', 'isinf', 'isnan', 'ldexp', 'modf', 'trunc',

    'exp', 'expm1', 'log', 'log1p', 'log2', 'pow', 'sqrt',

    'acos', 'asin', 'atan', 'atan2', 'cos', 'hypot', 'sin', 'tan',
    'acosh', 'asinh', 'atanh', 'cosh', 'sinh', 'tanh',

    'degrees', 'radians',

    'erf', 'erfc', 'gamma', 'lgamma'
)