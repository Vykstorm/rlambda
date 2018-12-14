

from src.utils import iterable, allinstanceof, anyinstanceof, instanceofchecker
import ast
import operator
from functools import reduce
from itertools import chain
import builtins

from src.astwrappers import Expression, Lambda, Variable, Placeholder, UnaryOperator, BinaryOperator, CompareOperator
from src.astwrappers import Index,Slice, ExtendedSlice
from src.astwrappers import encode_value
from src.astwrappers import BinaryOperation, UnaryOperation, CompareOperation, AttributeOperation, CallOperation, SubscriptOperation
from src.astwrappers import UnaryAdd, UnarySub, Invert
from src.astwrappers import Add, Sub, Mult, Div, FloorDiv, Mod, Pow, LShift, RShift, BitOr, BitAnd, BitXor, MatMul
from src.astwrappers import EqualThan, NotEqualThan, GreaterThan, GreaterEqualThan, LowerThan, LowerEqualThan


class RLambda:
    def __init__(self, inputs, body):
        assert iterable(inputs) and allinstanceof(inputs, str)
        assert isinstance(body, ast.AST)

        inputs = tuple(inputs)
        assert len(inputs) > 0

        self._inputs = tuple(sorted(inputs, key=str.lower))
        self._body = body
        self._expr = None
        self._func = None


    def _build_expr(self):
        if self._expr is None:
            self._expr = Expression(
                Lambda(
                    args=self._inputs,
                    body=self._body
                ))

    def _build_func(self):
        if self._func is None:
            self._build_expr()
            self._func = self._expr.eval()


    def __call__(self, *args, **kwargs):
        self._build_func()
        return self._func(*args, **kwargs)

    def __str__(self):
        self._build_expr()
        return str(self._expr)

    def __repr__(self):
        return str(self)

    def _binary_op(self, op, other):
        assert isinstance(op, BinaryOperator)

        if isinstance(other, RLambda):
            return RLambda(
                inputs=self._inputs+other._inputs,
                body=BinaryOperation(self._body, op, other._body)
            )

        return RLambda(
            inputs=self._inputs,
            body=BinaryOperation(self._body, op, encode_value(other))
        )

    def _unary_op(self, op):
        assert isinstance(op, UnaryOperator)

        return RLambda(
            inputs=self._inputs,
            body=UnaryOperation(op, self._body)
        )

    def _compare_op(self, op, other):
        assert isinstance(op, CompareOperator)

        if isinstance(other, RLambda):
            return RLambda(
                inputs=self._inputs+other._inputs,
                body=CompareOperation(self._body, op, other._body)
            )

        return RLambda(
            inputs=self._inputs,
            body=CompareOperation(self._body, op, encode_value(other))
        )

    def _attr_op(self, key):
        assert isinstance(key, str)

        return RLambda(
            inputs=self._inputs,
            body=AttributeOperation(self._body, key)
        )

    def _subscript_op(self, item):
        def encode_slice(x):
            return Slice(
                encode_value(x.start) if x.start is not None else None,
                encode_value(x.stop) if x.stop is not None else None,
                encode_value(x.step) if x.step is not None else None)

        def encode_index(x):
            return Index(encode_value(x))

        def encode_extslice(x):
            return ExtendedSlice(*[encode_slice(item) if isinstance(item, slice) else encode_index(item) for item in x])

        if isinstance(item, slice):
            index = encode_slice(item)
        elif isinstance(item, tuple):
            index = encode_extslice(item)
        else:
            index = encode_index(item)

        return RLambda(
            body=SubscriptOperation(self._body, index),
            inputs=self._inputs
        )


    # Calling operations

    @staticmethod
    def _call_op(func, *args, **kwargs):
        assert callable(func)
        assert anyinstanceof(chain(args, kwargs.values()), RLambda) and allinstanceof(kwargs.values(), str)

        encode_arg = lambda arg: encode_value(arg) if not isinstance(arg, RLambda) else arg._body

        inputs = reduce(
            operator.add,
            map(operator.attrgetter('_inputs'), filter(instanceofchecker(RLambda), chain(args, kwargs.values())))
        )
        args = tuple(map(encode_arg, args))
        kwargs = dict(zip(kwargs.keys(), map(encode_arg, kwargs.values())))

        return RLambda(
            body=CallOperation(Placeholder(func), *args, **kwargs),
            inputs=inputs
        )

    def _abs_op(self):
        return self._call_op(builtins.abs, self)


    # Arithmetic operations

    def __neg__(self):
        return self._unary_op(UnarySub)

    def __pos__(self):
        return self._unary_op(UnaryAdd)

    def __invert__(self):
        return self._unary_op(Invert)

    def __add__(self, other):
        return self._binary_op(Add, other)

    def __sub__(self, other):
        return self._binary_op(Sub, other)

    def __mul__(self, other):
        return self._binary_op(Mult, other)

    def __truediv__(self, other):
        return self._binary_op(Div, other)

    def __floordiv__(self, other):
        return self._binary_op(FloorDiv, other)

    def __mod__(self, other):
        return self._binary_op(Mod, other)

    def __pow__(self, power):
        return self._binary_op(Pow, power)

    def __matmul__(self, other):
        return self._binary_op(MatMul, other)

    def __abs__(self):
        return self._abs_op()

    # Bitwise operators

    def __lshift__(self, other):
        return self._binary_op(LShift, other)

    def __rshift__(self, other):
        return self._binary_op(RShift, other)

    def __or__(self, other):
        return self._binary_op(BitOr, other)

    def __and__(self, other):
        return self._binary_op(BitAnd, other)

    def __xor__(self, other):
        return self._binary_op(BitXor, other)


    # Comparision operators

    def __eq__(self, other):
        return self._compare_op(EqualThan, other)

    def __ne__(self, other):
        return self._compare_op(NotEqualThan, other)

    def __lt__(self, other):
        return self._compare_op(LowerThan, other)

    def __le__(self, other):
        return self._compare_op(LowerEqualThan, other)

    def __gt__(self, other):
        return self._compare_op(GreaterThan, other)

    def __ge__(self, other):
        return self._compare_op(GreaterEqualThan, other)


    # Attribute access

    def __getattribute__(self, key):
        if key.startswith('_'):
            return super().__getattribute__(key)
        return self._attr_op(key)


    def __setattr__(self, key, value):
        if not key.startswith('_'):
            raise KeyError()
        super().__setattr__(key, value)


    # Container operations

    def __getitem__(self, item):
        return self._subscript_op(item)



class RLambdaIdentity(RLambda):
    def __init__(self, input):
        assert isinstance(input, str)
        input = str.lower(input)
        super().__init__(
            inputs=(input,),
            body=Variable(input)
        )