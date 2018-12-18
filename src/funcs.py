


'''
Additional functions to operate with rlambdas
'''

import builtins
import math
from types import SimpleNamespace
from functools import wraps
from itertools import chain
from .rlambda import RLambda
from .utils import anyinstanceof

_funcs = dict(builtins.__dict__)
_funcs.update(math.__dict__)
_builtins = SimpleNamespace(**_funcs)



# Built-in overrides


def _build_wrapper(func):
    assert callable(func)

    @wraps(func)
    def _wrapper(*args, **kwargs):
        if anyinstanceof(chain(args, kwargs.values()), RLambda):
            return RLambda._call_op(func, *args, **kwargs)
        return func(*args, **kwargs)

    return _wrapper



len = length = _build_wrapper(_builtins.len)
min = _build_wrapper(_builtins.min)
max = _build_wrapper(_builtins.max)

ceil = _build_wrapper(_builtins.ceil)
copysign = _build_wrapper(_builtins.copysign)
fabs = _build_wrapper(_builtins.fabs)
factorial = _build_wrapper(_builtins.factorial)
floor = _build_wrapper(_builtins.floor)
fmod = _build_wrapper(_builtins.fmod)
frexp = _build_wrapper(_builtins.frexp)
gcd = _build_wrapper(_builtins.gcd)
isclose = _build_wrapper(_builtins.isclose)
isfinite = _build_wrapper(_builtins.isfinite)
isinf = _build_wrapper(_builtins.isinf)
isnan = _build_wrapper(_builtins.isnan)
ldexp = _build_wrapper(_builtins.ldexp)
modf = _build_wrapper(_builtins.modf)
trunc = _build_wrapper(_builtins.trunc)
exp = _build_wrapper(_builtins.exp)
expm1 = _build_wrapper(_builtins.expm1)
log = _build_wrapper(_builtins.log)
log1p = _build_wrapper(_builtins.log1p)
log2 = _build_wrapper(_builtins.log2)
pow = _build_wrapper(_builtins.pow)
sqrt = _build_wrapper(_builtins.sqrt)
acos = _build_wrapper(_builtins.acos)
asin = _build_wrapper(_builtins.asin)
atan = _build_wrapper(_builtins.atan)
atan2 = _build_wrapper(_builtins.atan2)
cos = _build_wrapper(_builtins.cos)
hypot = _build_wrapper(_builtins.hypot)
sin = _build_wrapper(_builtins.sin)
tan = _build_wrapper(_builtins.tan)
acosh = _build_wrapper(_builtins.acosh)
asinh = _build_wrapper(_builtins.asinh)
atanh = _build_wrapper(_builtins.atanh)
cosh = _build_wrapper(_builtins.cosh)
sinh = _build_wrapper(_builtins.sinh)
tanh = _build_wrapper(_builtins.tanh)
degrees = _build_wrapper(_builtins.degrees)
radians = _build_wrapper(_builtins.radians)
erf = _build_wrapper(_builtins.erf)
erfc = _build_wrapper(_builtins.erfc)
gamma = _build_wrapper(_builtins.gamma)
lgamma = _build_wrapper(_builtins.lgamma)




