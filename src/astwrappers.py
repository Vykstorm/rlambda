
'''
This module provides functions and class helpers to build ast expression nodes
'''

import ast
from functools import reduce
from itertools import count, chain
from types import FunctionType, LambdaType, BuiltinFunctionType
from inspect import isclass
import operator
from src.utils import enclose, iterable


class Variable(ast.Name):
    '''
    This kind of node represents a named variable
    '''
    def __init__(self, name):
        if not isinstance(name, str):
            raise TypeError()
        super().__init__(name, ast.Load())

    def __str__(self):
        return self.id


class Placeholder(ast.Name):
    '''
    This kind of node represents a placeholder variable
    '''
    def __init__(self, value):
        '''
        Initializes this instance.
        :param value: Arbtitrary value this variable holds
        '''

        super().__init__('_0', ast.Load())
        self.value = value
        self._index = 0

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, x):
        if not isinstance(x, int):
            raise TypeError()
        if x < 0:
            raise ValueError()
        self._index = x
        self.id = '_' + repr(self._index)

    def __str__(self):
        value = self.value
        if isinstance(value, (FunctionType, LambdaType, BuiltinFunctionType)):
            return value.__qualname__
        if isclass(value):
            return value.__name__
        return str(value)


class Literal:
    '''
    Represents any kind of literal
    '''
    pass

class LiteralNumber(ast.Num, Literal):
    '''
    Represents a kind of node which holds a numeric literal (int or float)
    '''
    def __init__(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError()
        ast.Num.__init__(self, value)
        Literal.__init__(self)

    def __str__(self):
        return repr(self.n)

class LiteralStr(ast.Str, Literal):
    '''
    Represents a kind of node which holds a string literal
    '''
    def __init__(self, value):
        if not isinstance(value, str):
            raise TypeError()
        ast.Str.__init__(self, value)
        Literal.__init__(self)

    def __str__(self):
        return repr(self.s)


class LiteralBytes(ast.Bytes, Literal):
    '''
    Represents a kind of node which holds a bytes literal
    '''
    def __init__(self, value):
        if not isinstance(value, bytes):
            raise TypeError()
        ast.Bytes.__init__(self, value)
        Literal.__init__(self)

    def __str__(self):
        return repr(self.s)


class LiteralEllipsis(ast.Ellipsis, Literal):
    '''
    Represents a node which holds the "ellipsis" literal ("..." dots)
    '''
    def __init__(self):
        ast.Ellipsis.__init__(self)
        Literal.__init__(self)

    def __str__(self):
        return '...'


class LiteralBool(ast.NameConstant, Literal):
    '''
    Represents a node which holds either True/False literal
    '''
    def __init__(self, value):
        if value is not True and value is not False:
            raise TypeError()
        ast.NameConstant.__init__(self, value)
        Literal.__init__(self)

    def __str__(self):
        return repr(self.value)

class LiteralNone(ast.NameConstant, Literal):
    '''
    Represents a node which holds the literal "None"
    '''
    def __init__(self):
        ast.NameConstant.__init__(self, None)
        Literal.__init__(self)

    def __str__(self):
        return repr(None)



def literal(x):
    '''
    Returns a kind of node which holds a literal (True, False, None, Ellipsis, str, bytes, int or float) that when
    compiling it on 'eval' mode and evaluating, the result would be the value given as argument.
    Raises a ValueError exception if the given argument is not True, False, None, Ellipsis neither is of type
    str, bytes, int or float
    '''

    if x is True or x is False:
        return LiteralBool(x)

    if x is None:
        return LiteralNone()

    if isinstance(x, (int, float)):
        return LiteralNumber(x)

    if isinstance(x, str):
        return LiteralStr(x)

    if isinstance(x, bytes):
        return LiteralBytes(x)

    if x is Ellipsis:
        return LiteralEllipsis()

    raise ValueError()


def encode_value(x):
    '''
    Returns literal(x). If literal(x) raises an exception, it just returns Placeholder(x)
    :param x: An arbitrary value
    :return:
    '''
    try:
        if not isinstance(x, (str, bytes)) or len(x) < 32:
            return literal(x)
    except:
        pass
    return Placeholder(value=x)


class Operator:
    '''
    Represents an operator of any kind
    '''
    def __init__(self, symbol):
        self.symbol = symbol

    def __str__(self):
        return str(self.symbol)

class UnaryOperator(Operator):
    '''
    Represents any unary operator
    '''
    pass

class BinaryOperator(Operator):
    '''
    Represents any binary operator
    '''
    pass

class CompareOperator(Operator):
    '''
    Represents any comparision operator
    '''
    pass


# Unary operations

class UnaryAdd(UnaryOperator, ast.UAdd):
    def __init__(self):
        UnaryOperator.__init__(self, '+')
        ast.UAdd.__init__(self)


class UnarySub(UnaryOperator, ast.USub):
    def __init__(self):
        UnaryOperator.__init__(self, '-')
        ast.USub.__init__(self)

class Invert(UnaryOperator, ast.Invert):
    def __init__(self):
        UnaryOperator.__init__(self, '~')
        ast.Invert.__init__(self)

# Binary operations

class Add(BinaryOperator, ast.Add):
    def __init__(self):
        BinaryOperator.__init__(self, '+')
        ast.Add.__init__(self)

class Sub(BinaryOperator, ast.Sub):
    def __init__(self):
        BinaryOperator.__init__(self, '-')
        ast.Sub.__init__(self)

class Mult(BinaryOperator, ast.Mult):
    def __init__(self):
        BinaryOperator.__init__(self, '*')
        ast.Mult.__init__(self)

class Div(BinaryOperator, ast.Div):
    def __init__(self):
        BinaryOperator.__init__(self, '/')
        ast.Div.__init__(self)

class FloorDiv(BinaryOperator, ast.FloorDiv):
    def __init__(self):
        BinaryOperator.__init__(self, '//')
        ast.FloorDiv.__init__(self)

class Mod(BinaryOperator, ast.Mod):
    def __init__(self):
        BinaryOperator.__init__(self, '%')
        ast.Mod.__init__(self)

class Pow(BinaryOperator, ast.Pow):
    def __init__(self):
        BinaryOperator.__init__(self, '**')
        ast.Pow.__init__(self)

class LShift(BinaryOperator, ast.LShift):
    def __init__(self):
        BinaryOperator.__init__(self, '<<')
        ast.LShift.__init__(self)

class RShift(BinaryOperator, ast.RShift):
    def __init__(self):
        BinaryOperator.__init__(self, '>>')
        ast.RShift.__init__(self)

class BitOr(BinaryOperator, ast.BitOr):
    def __init__(self):
        BinaryOperator.__init__(self, '|')
        ast.BitOr.__init__(self)
        
class BitAnd(BinaryOperator, ast.BitAnd):
    def __init__(self):
        BinaryOperator.__init__(self, '&')
        ast.BitAnd.__init__(self)

class BitXor(BinaryOperator, ast.BitXor):
    def __init__(self):
        BinaryOperator.__init__(self, '^')
        ast.BitXor.__init__(self)

class MatMul(BinaryOperator, ast.MatMult):
    def __init__(self):
        BinaryOperator.__init__(self, '@')
        ast.MatMult.__init__(self)



# Comparision operators

class EqualThan(CompareOperator, ast.Eq):
    def __init__(self):
        CompareOperator.__init__(self, '==')
        ast.Eq.__init__(self)

class NotEqualThan(CompareOperator, ast.NotEq):
    def __init__(self):
        CompareOperator.__init__(self, '!=')
        ast.NotEq.__init__(self)

class LowerThan(CompareOperator, ast.Lt):
    def __init__(self):
        CompareOperator.__init__(self, '<')
        ast.Lt.__init__(self)

class LowerEqualThan(CompareOperator, ast.LtE):
    def __init__(self):
        CompareOperator.__init__(self, '<=')
        ast.LtE.__init__(self)
        
class GreaterThan(CompareOperator, ast.Gt):
    def __init__(self):
        CompareOperator.__init__(self, '>')
        ast.Gt.__init__(self)
        
class GreaterEqualThan(CompareOperator, ast.GtE):
    def __init__(self):
        CompareOperator.__init__(self, '>=')
        ast.GtE.__init__(self)


UnaryAdd = UnaryAdd()
UnarySub = UnarySub()
Invert = Invert()

Add = Add()
Sub = Sub()
Mult = Mult()
Div = Div()
FloorDiv = FloorDiv()
Mod = Mod()
Pow = Pow()
LShift = LShift()
RShift = RShift()
BitOr = BitOr()
BitAnd = BitAnd()
BitXor = BitXor()
MatMul = MatMul()

EqualThan = EqualThan()
NotEqualThan = NotEqualThan()
LowerThan = LowerThan()
LowerEqualThan = LowerEqualThan()
GreaterThan = GreaterThan()
GreaterEqualThan = GreaterEqualThan()






class UnaryOperation(ast.UnaryOp):
    '''
    This kind of node represents a unary operation which involves one operand and one unary operator.
    '''
    def __init__(self, op, operand):
        '''
        Initializes this instance
        :param op: Must be a unary operator
        :param operand: Must be the operand (another AST node)
        '''
        if not isinstance(op, ast.unaryop):
            raise TypeError()

        if not isinstance(operand, ast.AST):
            raise TypeError()

        super().__init__(op, operand)

    def __str__(self):
        return str(self.op) +\
               (str(self.operand) if isinstance(self.operand, (Literal, Variable, CallOperation)) else enclose(str(self.operand), '()'))

class BinaryOperation(ast.BinOp):
    '''
    This kind of node represents a binary operation which involves two operands and one binary operator
    '''
    def __init__(self, left, op, right):
        '''
        Initializes this instance
        :param left: Left operand
        :param op: A binary operator
        :param right: Right operand
        '''
        if not isinstance(left, ast.AST) or not isinstance(right, ast.AST):
            raise TypeError()
        if not isinstance(op, ast.operator):
            raise TypeError()
        super().__init__(left, op, right)

    def __str__(self):
        format_operand = lambda x: str(x) if isinstance(x, (Literal, Variable, Placeholder, CallOperation)) else enclose(str(x), '()')
        return ' '.join((format_operand(self.left), str(self.op), format_operand(self.right)))


class CompareOperation(ast.Compare):
    '''
    This kind of node represents a comparision operation. Involves 2 ore more operands and 2 or more comparision operators
    e.g:
    1 < x > 10
    '''
    def __init__(self, left, op, right, *args):
        '''
        Initializes this instance
        :param left: Left most operand
        :param op: The left most comparision operator
        :param right: The second left most operand
        :param args: An optional list of aditional pairs of operand/operator
        '''
        if len(args) % 2 != 0:
            raise ValueError()


        operands = (left,) + args[1::2] + (right,)
        operators = (op,) + args[0::2]

        for operand in operands:
            if not isinstance(operand, ast.AST):
                raise TypeError()

        for operator in operators:
            if not isinstance(operator, ast.cmpop):
                raise TypeError()

        super().__init__(operands[0], list(operators), list(operands[1:]))


    def __str__(self):
        format_operand = lambda x: str(x) if isinstance(x, (Literal, Variable, Placeholder, CallOperation)) else enclose(str(x), '()')

        return ' '.join(
            (format_operand(self.left),) +
            reduce(operator.add, zip(map(str, self.ops), map(format_operand, self.comparators))))



class Index(ast.Index):
    '''
    This node represents an index (for simple subscripting with a single value)
    '''
    def __init__(self, value):
        if not isinstance(value, ast.AST):
            raise TypeError()
        super().__init__(value)

    def __str__(self):
        return str(self.value)


class Slice(ast.Slice):
    '''
    This node represents a regular slice (for subscripting)
    '''
    def __init__(self, lower, upper, step):
        if not isinstance(lower, ast.AST) and lower is not None:
            raise TypeError()
        if not isinstance(upper, ast.AST) and upper is not None:
            raise TypeError()
        if not isinstance(step, ast.AST) and step is not None:
            raise TypeError()
        super().__init__(lower, upper, step)

    def __str__(self):
        items = (self.lower, self.upper, self.step)
        if self.lower is not None and self.upper is not None and self.step is None:
            items = items[:-1]
        return ':'.join(map(lambda x: str(x) if x is not None else '', items))


class ExtendedSlice(ast.ExtSlice):
    '''
    This node represents an extended slice (for subscripting)
    '''
    def __init__(self, *args):
        '''
        Initializes this instance
        :param args: Must be sequence of Slice and Index nodes
        '''
        if any(map(lambda arg: not isinstance(arg, (Slice, Index)), args)):
            raise TypeError()

        args = list(args)
        super().__init__(args)

    def __str__(self):
        dims = self.dims
        return ','.join(map(str, dims))


class SubscriptOperation(ast.Subscript):
    '''
    Its a node that represents a subscript operation.
    '''
    def __init__(self, value, index):
        '''
        Initializes this instance
        :param value: The node to be subscripted
        :param slice: An instance of ast.Index, ast.Slice or ast.ExtSlice
        '''
        if not isinstance(value, ast.AST):
            raise TypeError()
        if not isinstance(index, (ast.Index, ast.Slice, ast.ExtSlice)):
            raise TypeError()

        super().__init__(value, index, ast.Load())

    def __str__(self):
        return str(self.value) + enclose(str(self.slice), '[]')



class AttributeOperation(ast.Attribute):
    '''
    This node represents an attribute access node operation
    '''
    def __init__(self, value, attr):
        '''
        Initializes this instance
        :param value: Must be another node (representing the object being accessed)
        :param attr: Must be a string with the name of the attribute
        '''
        if not isinstance(value, ast.AST):
            raise TypeError()
        if not isinstance(attr, str):
            raise TypeError()

        super().__init__(value, attr, ast.Load())

    def __str__(self):
        return '.'.join(map(str, (str(self.value), self.attr)))


class CallOperation(ast.Call):
    '''
    Kind of node which defines a function call.
    '''
    def __init__(self, func, *args, **kwargs):
        '''
        Initializes this instance.
        :param func: Is the function that is called (another node, normally a Variable node)
        :param args: Is a list of arguments (each of them is a node)
        :param kwargs: Is a dictionary that indicates keyword arguments (keys are strings and values are nodes)
        '''
        if not isinstance(func, ast.AST):
            raise TypeError()

        if any(map(lambda arg: not isinstance(arg, ast.AST), args)):
            raise TypeError()

        if any(map(lambda key: not isinstance(key, str), kwargs.keys())):
            raise TypeError()
        if any(map(lambda value: not isinstance(value, ast.AST), kwargs.values())):
            raise TypeError()

        super().__init__(func, list(args), [ast.keyword(key, value) for key, value in kwargs.items()])


    def __str__(self):
        func = self.func
        args = self.args
        kwargs = map(operator.attrgetter('arg', 'value'), self.keywords)

        return str(func) + enclose(', '.join(chain(map(str, args), [key+'='+str(value) for key, value in kwargs])), '()')


class Lambda(ast.Lambda):
    '''
    This node represents a lambda object definition with only positional arguments
    '''
    def __init__(self, args, body):
        if not iterable(args):
            raise TypeError()
        args = list(args)
        if len(args) == 0:
            raise ValueError()

        if not isinstance(body, ast.AST):
            raise TypeError()

        args = list(map(lambda arg: ast.arg(arg, None), args))

        super().__init__(
            body=body,
            args=ast.arguments(
                args=args,
                kwonlyargs=[],
                vararg=None,
                kwarg=None,
                defaults=[],
                kw_defaults=[]
            )
        )


    def __str__(self):
        args = map(operator.attrgetter('arg'), self.args.args)
        return ', '.join(args) + ' : ' + str(self.body)


class Expression(ast.Expression):
    '''
    This node represents an expression (a wrapper of ast.Expression)
    '''
    def __init__(self, body):
        '''
        Initializes this instance
        :param body: Must be another node (it will be the body of this expression node)
        :param vars: An optional argument
        '''
        if not isinstance(body, ast.AST):
            raise TypeError()
        super().__init__(body)

    def __str__(self):
        return str(self.body)


    @property
    def placeholders(self):
        class Placeholders(ast.NodeVisitor, list):
            def __init__(self, node):
                ast.NodeVisitor.__init__(self)
                list.__init__(self)
                self.generic_visit(node)

            def generic_visit(self, node):
                if isinstance(node, Placeholder):
                    self.append(node)
                super().generic_visit(node)
        return tuple(Placeholders(self))


    def eval(self):
        '''
        Evaluates this expression and returns the result of such evaluation.
        :return:
        '''
        placeholders = self.placeholders
        for index, placeholder in zip(count(0), placeholders):
            placeholder.index = index

        def _compile():
            ast.fix_missing_locations(self)
            return compile(self, '<string>', 'eval')

        globals=dict(zip(map(operator.attrgetter('id'), placeholders), map(operator.attrgetter('value'), placeholders)))
        code = _compile()
        return eval(code, globals)