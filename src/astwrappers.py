
'''
This module provides classes that are wrappers of different kind of AST nodes that are used later by this library
to build rlambdas objects.
'''

import ast
from itertools import count, chain
import operator
from operator import attrgetter
from copy import copy, deepcopy
from inspect import getmro
from .utils import enclose, iterable, allinstanceof, findsubclassof


class Node:
    '''
    Represents any kind of node
    '''
    def __str__(self):
        return get_default_formatter()._format_node(self)

    def __repr__(self):
        return str(self)

    def __copy__(self):
        '''
        :return: Returns a shallow copy of this instance.
        '''
        cls = self.__class__
        obj = findsubclassof(getmro(cls), ast.AST).__new__(cls)
        for key, value in self.__dict__.items():
            obj.__dict__[key] = value if isinstance(value, (Node, ast.AST)) else copy(value)
        return obj

    def __deepcopy__(self, memodict={}):
        '''
        :return: Returns a deep copy of this instance
        '''
        cls = self.__class__
        obj = findsubclassof(getmro(cls), ast.AST).__new__(cls)
        for key, value in self.__dict__.items():
            if isinstance(value, Node) or not isinstance(value, ast.AST):
                obj.__dict__[key] = deepcopy(value, memodict)
            else:
                obj.__dict__[key] = value
        return obj

    def __eq__(self, other):
        '''
        Compare two nodes. They will be equal if they have the same node type and all its fields
        are equal.
        '''
        if type(other) != type(self):
            return False

        for key in self.__dict__:
            a, b = self.__dict__[key], other.__dict__[key]
            if type(a) != type(b):
                return False
            if isinstance(a, Node) or not isinstance(a, ast.AST):
                if a != b:
                    return False
            else:
                if a is not b:
                    return False
        return True


class Variable(ast.Name, Node):
    '''
    This kind of node represents a named variable
    '''

    def __init__(self, name):
        assert isinstance(name, str)
        ast.Name.__init__(self, name, ast.Load())
        Node.__init__(self)


class Placeholder(ast.Name, Node):
    '''
    This kind of node represents a placeholder variable
    '''

    def __init__(self, value):
        '''
        Initializes this instance.
        :param value: Arbtitrary value this variable holds
        '''

        ast.Name.__init__(self, '_0', ast.Load())
        Node.__init__(self)
        self.value = value
        self._index = 0

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, x):
        assert isinstance(x, int) and x >= 0
        self._index = x
        self.id = '_' + repr(self._index)

    def __copy__(self):
        obj = ast.Name.__new__(self.__class__)
        obj._index = self._index
        obj.ctx = self.ctx
        obj.id = self.id
        obj.value = self.value
        return obj

    def __deepcopy__(self, memodict={}):
        return copy(self)


    def __eq__(self, other):
        if not isinstance(other, Placeholder):
            return False
        return self.index == other.index and self.value is other.value

class Literal(Node):
    '''
    Represents any kind of literal
    '''
    pass



class LiteralNumber(ast.Num, Literal):
    '''
    Represents a kind of node which holds a numeric literal (int or float)
    '''
    def __init__(self, value):
        assert isinstance(value, (int, float))
        ast.Num.__init__(self, value)
        Literal.__init__(self)


class LiteralStr(ast.Str, Literal):
    '''
    Represents a kind of node which holds a string literal
    '''
    def __init__(self, value):
        assert isinstance(value, str)

        ast.Str.__init__(self, value)
        Literal.__init__(self)


class LiteralBytes(ast.Bytes, Literal):
    '''
    Represents a kind of node which holds a bytes literal
    '''
    def __init__(self, value):
        assert isinstance(value, bytes)

        ast.Bytes.__init__(self, value)
        Literal.__init__(self)



class LiteralEllipsis(ast.Ellipsis, Literal):
    '''
    Represents a node which holds the "ellipsis" literal ("..." dots)
    '''
    def __init__(self):
        ast.Ellipsis.__init__(self)
        Literal.__init__(self)



class LiteralBool(ast.NameConstant, Literal):
    '''
    Represents a node which holds either True/False literal
    '''
    def __init__(self, value):
        assert value is True or value is False

        ast.NameConstant.__init__(self, value)
        Literal.__init__(self)



class LiteralNone(ast.NameConstant, Literal):
    '''
    Represents a node which holds the literal "None"
    '''
    def __init__(self):
        ast.NameConstant.__init__(self, None)
        Literal.__init__(self)



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


class Operator(Node):
    '''
    Represents an operator of any kind
    '''

    def __init__(self, symbol, precedence):
        '''
        Initializes this instance
        :param symbol: Is the symbol of the operator
        :param precedence: Is the precedence level of this operator. Must be an integer greater or equal than 0. 0 is the
        lowest operator precedence lower.
        '''
        assert isinstance(symbol, str) and isinstance(precedence, int) and precedence >= 0
        super().__init__()
        self.symbol = symbol
        self.precedence = precedence

    def __lt__(self, other):
        '''
        Compares operator level precedences.
        :param other: Another instance of class Operator
        :return: Returns True if this operator precedence is lower than the other operator precedence. Otherwise returns False
        '''
        if not isinstance(other, Operator):
            raise NotImplemented()
        return self.precedence < other.precedence

    def __le__(self, other):
        '''
        Compares operator level precedences
        :param other: Another instance of class Operator
        :return: Returns True if this operator precedence is lower or equal than the other operator precedence.
        Otherwise returns False
        '''
        if not isinstance(other, Operator):
            raise NotImplemented()
        return self.precedence <= other.precedence



class UnaryOperator(Operator):
    '''
    Represents any unary operator
    '''
    pass

class BinaryOperator(Operator):
    '''
    Represents any binary operator
    '''

    def __init__(self, symbol, precedence, associative):
        '''
        Initializes this instance.
        :param associative: This is set to True to indicate that this kind of binary operator follows the associativity rule.
        a ⊕ (b ⊕ c) == (a ⊕ b) ⊕ c  for any a, b, c where ⊕ is the binary operator
        '''
        assert isinstance(associative, bool)
        super().__init__(symbol, precedence)
        self.associative = associative

class CompareOperator(Operator):
    '''
    Represents any comparision operator
    '''
    def __init__(self, symbol):
        # All comparision operators has the same precedence level
        super().__init__(symbol, 3)


# Unary operators

class UnaryAdd(UnaryOperator, ast.UAdd):
    def __init__(self):
        UnaryOperator.__init__(self, '+', 10)
        ast.UAdd.__init__(self)


class UnarySub(UnaryOperator, ast.USub):
    def __init__(self):
        UnaryOperator.__init__(self, '-', 10)
        ast.USub.__init__(self)

class Invert(UnaryOperator, ast.Invert):
    def __init__(self):
        UnaryOperator.__init__(self, '~', 10)
        ast.Invert.__init__(self)


# Binary operator

class Add(BinaryOperator, ast.Add):
    def __init__(self):
        BinaryOperator.__init__(self, '+', 8, True)
        ast.Add.__init__(self)

class Sub(BinaryOperator, ast.Sub):
    def __init__(self):
        BinaryOperator.__init__(self, '-', 8, False)
        ast.Sub.__init__(self)

class Mult(BinaryOperator, ast.Mult):
    def __init__(self):
        BinaryOperator.__init__(self, '*', 9, True)
        ast.Mult.__init__(self)

class Div(BinaryOperator, ast.Div):
    def __init__(self):
        BinaryOperator.__init__(self, '/', 9, False)
        ast.Div.__init__(self)

class FloorDiv(BinaryOperator, ast.FloorDiv):
    def __init__(self):
        BinaryOperator.__init__(self, '//', 9, False)
        ast.FloorDiv.__init__(self)

class Mod(BinaryOperator, ast.Mod):
    def __init__(self):
        BinaryOperator.__init__(self, '%', 9, False)
        ast.Mod.__init__(self)

class Pow(BinaryOperator, ast.Pow):
    def __init__(self):
        BinaryOperator.__init__(self, '**', 11, False)
        ast.Pow.__init__(self)

class LShift(BinaryOperator, ast.LShift):
    def __init__(self):
        BinaryOperator.__init__(self, '<<', 7, False)
        ast.LShift.__init__(self)

class RShift(BinaryOperator, ast.RShift):
    def __init__(self):
        BinaryOperator.__init__(self, '>>', 7, False)
        ast.RShift.__init__(self)

class BitOr(BinaryOperator, ast.BitOr):
    def __init__(self):
        BinaryOperator.__init__(self, '|', 4, True)
        ast.BitOr.__init__(self)

class BitAnd(BinaryOperator, ast.BitAnd):
    def __init__(self):
        BinaryOperator.__init__(self, '&', 6, True)
        ast.BitAnd.__init__(self)

class BitXor(BinaryOperator, ast.BitXor):
    def __init__(self):
        BinaryOperator.__init__(self, '^', 5, True)
        ast.BitXor.__init__(self)

class MatMul(BinaryOperator, ast.MatMult):
    def __init__(self):
        BinaryOperator.__init__(self, '@', 9, True)
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





# All unary operators
unary_operators = (UnaryAdd, UnarySub, Invert)

# All binary operators
binary_operators = (
    Add, Sub, Mult, Div, FloorDiv, Mod, Pow,
    LShift, RShift, BitOr, BitAnd, BitXor,
    MatMul
)

# All comparision operators
compare_operators = (
    EqualThan, NotEqualThan, LowerThan,
    LowerEqualThan, GreaterThan, GreaterEqualThan
)

# This list contains all operators
operators = unary_operators + binary_operators + compare_operators



class Operation(Node):
    '''
    Represents any kind of operation
    '''

    @property
    def precedence(self):
        '''
        This must be implemented by subclasses. It may return the precedence level of this operation
        :return:
        '''
        raise NotImplementedError()

    def __lt__(self, other):
        '''
        Compares operation level precedences.
        :param other: Another instance of class Operation
        :return: Returns True if this operation precedence is lower than the other operation precedence. Otherwise returns False
        '''
        if not isinstance(other, Operation):
            raise NotImplemented()
        return self.precedence < other.precedence

    def __le__(self, other):
        '''
        Compares operation level precedences
        :param other: Another instance of class Operation
        :return: Returns True if this operation precedence is lower or equal than the other operation precedence.
        Otherwise returns False
        '''
        if not isinstance(other, Operation):
            raise NotImplemented()
        return self.precedence <= other.precedence



class UnaryOperation(ast.UnaryOp, Operation):
    '''
    This kind of node represents a unary operation which involves one operand and one unary operator.
    '''
    def __init__(self, op, operand):
        '''
        Initializes this instance
        :param op: Must be a unary operator
        :param operand: Must be the operand (another AST node)
        '''
        assert isinstance(op, ast.unaryop) and isinstance(operand, ast.AST)

        ast.UnaryOp.__init__(self, op, operand)
        Operation.__init__(self)

    @property
    def precedence(self):
        return self.op.precedence


class BinaryOperation(ast.BinOp, Operation):
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
        assert isinstance(left, ast.AST) and isinstance(right, ast.AST)
        assert isinstance(op, ast.operator)

        ast.BinOp.__init__(self, left, op, right)
        Operation.__init__(self)

    @property
    def precedence(self):
        return self.op.precedence

    @property
    def associative(self):
        return self.op.associative

class CompareOperation(ast.Compare, Operation):
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
        assert len(args) % 2 == 0

        operands = (left,) + args[1::2] + (right,)
        operators = (op,) + args[0::2]

        assert allinstanceof(operands, ast.AST) and allinstanceof(operators, ast.cmpop)

        ast.Compare.__init__(self, operands[0], list(operators), list(operands[1:]))
        Operation.__init__(self)


    @property
    def precedence(self):
        # All compare operations has the same precedence level
        return self.ops[0].precedence


class Index(ast.Index, Node):
    '''
    This node represents an index (for simple subscripting with a single value)
    '''
    def __init__(self, value):
        assert isinstance(value, ast.AST)

        ast.Index.__init__(self, value)
        Node.__init__(self)



class Slice(ast.Slice, Node):
    '''
    This node represents a regular slice (for subscripting)
    '''
    def __init__(self, lower, upper, step):
        assert isinstance(lower, ast.AST) or lower is None
        assert isinstance(upper, ast.AST) or upper is None
        assert isinstance(step, ast.AST) or step is None

        ast.Slice.__init__(self, lower, upper, step)
        Node.__init__(self)



class ExtendedSlice(ast.ExtSlice, Node):
    '''
    This node represents an extended slice (for subscripting)
    '''
    def __init__(self, *args):
        '''
        Initializes this instance
        :param args: Must be sequence of Slice and Index nodes
        '''
        assert allinstanceof(args, (Slice, Index))

        args = list(args)
        ast.ExtSlice.__init__(self, args)
        Node.__init__(self)


class SubscriptOperation(ast.Subscript, Operation):
    '''
    Its a node that represents a subscript operation.
    '''
    def __init__(self, value, index):
        '''
        Initializes this instance
        :param value: The node to be subscripted
        :param slice: An instance of ast.Index, ast.Slice or ast.ExtSlice
        '''
        assert isinstance(value, ast.AST)
        assert isinstance(index, (ast.Index, ast.Slice, ast.ExtSlice))

        ast.Subscript.__init__(self, value, index, ast.Load())
        Operation.__init__(self)

    @property
    def precedence(self):
        return 13


class AttributeOperation(ast.Attribute, Operation):
    '''
    This node represents an attribute access node operation
    '''
    def __init__(self, value, attr):
        '''
        Initializes this instance
        :param value: Must be another node (representing the object being accessed)
        :param attr: Must be a string with the name of the attribute
        '''
        assert isinstance(value, ast.AST) and isinstance(attr, str)

        ast.Attribute.__init__(self, value, attr, ast.Load())
        Operation.__init__(self)

    @property
    def precedence(self):
        return 13


class CallOperation(ast.Call, Operation):
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
        assert isinstance(func, ast.AST)
        assert allinstanceof(args, ast.AST)
        assert allinstanceof(kwargs.keys(), str) and allinstanceof(kwargs.values(), ast.AST)

        ast.Call.__init__(self, func, list(args), [ast.keyword(key, value) for key, value in kwargs.items()])
        Operation.__init__(self)

    @property
    def precedence(self):
        return 13


class Lambda(ast.Lambda, Node):
    '''
    This node represents a lambda object definition with only positional arguments
    '''
    def __init__(self, args, body):
        assert iterable(args) and isinstance(body, ast.AST)

        args = list(map(lambda arg: ast.arg(arg, None), args))

        Node.__init__(self)
        ast.Lambda.__init__(self,
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



class Expression(ast.Expression, Node):
    '''
    This node represents an expression (a wrapper of ast.Expression)
    '''
    def __init__(self, body):
        '''
        Initializes this instance
        :param body: Must be another node (it will be the body of this expression node)
        :param vars: An optional argument
        '''
        assert isinstance(body, ast.AST)

        ast.Expression.__init__(self, body)
        Node.__init__(self)


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
        Evaluates this expression and returns the output.
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


class VariableBinder(ast.NodeTransformer):
    '''
    This class can be used to replace Variable nodes with Placeholder or literal nodes
    '''
    def __init__(self, **kwargs):
        assert allinstanceof(kwargs.keys(), str)

        super().__init__()
        self.kwargs = kwargs

    def bind(self, node):
        return self.generic_visit(deepcopy(node))


    def generic_visit(self, node):
        if isinstance(node, Variable):
            if node.id in self.kwargs:
                return encode_value(self.kwargs[node.id])
        return super().generic_visit(node)


from .formatter import get_default_formatter
