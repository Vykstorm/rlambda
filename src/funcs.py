


'''
Additional functions to operate with rlambdas
'''

import builtins
import math
import operator
from types import SimpleNamespace
from itertools import chain
from .rlambda import RLambda
from .utils import anyinstanceof

_funcs = dict(operator.__dict__)
_funcs.update(builtins.__dict__)
_funcs.update(math.__dict__)
_builtins = SimpleNamespace(**_funcs)



# Built-in overrides

class Wrapper:
    def __init__(self, func):
        assert callable(func)
        self.func = func

    def __call__(self, *args, **kwargs):
        if anyinstanceof(chain(args, kwargs.values()), RLambda):
            return RLambda._call_op(self.func, *args, **kwargs)
        return self.func(*args, **kwargs)

    def __eq__(self, other):
        if isinstance(other, Wrapper):
            return self.func is other.func
        return self.func is other

    def __wrapped__(self):
        return self.func

    def __str__(self):
        return str(self.func)

    def __repr__(self):
        return repr(self.func)


len = length = Wrapper(_builtins.len)
min = Wrapper(_builtins.min)
max = Wrapper(_builtins.max)
contains = Wrapper(_builtins.contains)

ceil = Wrapper(_builtins.ceil)
copysign = Wrapper(_builtins.copysign)
fabs = Wrapper(_builtins.fabs)
factorial = Wrapper(_builtins.factorial)
floor = Wrapper(_builtins.floor)
fmod = Wrapper(_builtins.fmod)
frexp = Wrapper(_builtins.frexp)
gcd = Wrapper(_builtins.gcd)
isclose = Wrapper(_builtins.isclose)
isfinite = Wrapper(_builtins.isfinite)
isinf = Wrapper(_builtins.isinf)
isnan = Wrapper(_builtins.isnan)
ldexp = Wrapper(_builtins.ldexp)
modf = Wrapper(_builtins.modf)
trunc = Wrapper(_builtins.trunc)
exp = Wrapper(_builtins.exp)
expm1 = Wrapper(_builtins.expm1)
log = Wrapper(_builtins.log)
log1p = Wrapper(_builtins.log1p)
log2 = Wrapper(_builtins.log2)
log10 = Wrapper(_builtins.log10)
pow = Wrapper(_builtins.pow)
sqrt = Wrapper(_builtins.sqrt)
acos = Wrapper(_builtins.acos)
asin = Wrapper(_builtins.asin)
atan = Wrapper(_builtins.atan)
atan2 = Wrapper(_builtins.atan2)
cos = Wrapper(_builtins.cos)
hypot = Wrapper(_builtins.hypot)
sin = Wrapper(_builtins.sin)
tan = Wrapper(_builtins.tan)
acosh = Wrapper(_builtins.acosh)
asinh = Wrapper(_builtins.asinh)
atanh = Wrapper(_builtins.atanh)
cosh = Wrapper(_builtins.cosh)
sinh = Wrapper(_builtins.sinh)
tanh = Wrapper(_builtins.tanh)
degrees = Wrapper(_builtins.degrees)
radians = Wrapper(_builtins.radians)
erf = Wrapper(_builtins.erf)
erfc = Wrapper(_builtins.erfc)
gamma = Wrapper(_builtins.gamma)
lgamma = Wrapper(_builtins.lgamma)


