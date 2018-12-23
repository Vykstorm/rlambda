

import operator
import builtins
import math
from operator import attrgetter
from itertools import chain
from functools import reduce
from types import BuiltinFunctionType, FunctionType, LambdaType
from inspect import isclass, signature

from .utils import enclose, slice_to_str, findsuperclassof, unicode_subscript, unicode_superscript
from .astwrappers import Node, Lambda, Expression, Variable, Operator, Placeholder, Index, Slice, ExtendedSlice
from .astwrappers import Literal, LiteralNumber, LiteralEllipsis, LiteralBytes, LiteralBool, LiteralNone, LiteralStr
from .astwrappers import Operation, UnaryOperation, BinaryOperation, CompareOperation,\
    SubscriptOperation, AttributeOperation, CallOperation
from .astwrappers import Pow, Sub, Add
from .singleton import singleton


class RLambdaFormatter:
    '''
    This class is used to stringify rlambda objects. You can subclass it to change its default behaviour.
    '''

    def __init__(self):
        pass


    def format(self, f):
        '''
        Stringifies the rlambda object given as argument. e.g:
        RLambdaFormatter().format( (x+1) // 2 ) ->  "x : (x+1) // 2"
        This method should not be overrided. Look the next methods with the name format_X  to customize rlambda formatting.
        '''
        from .rlambda import RLambda
        if not isinstance(f, RLambda):
            raise TypeError('Expected rlambda object at argument 1 but got {}'.format(type(f).__name__))

        f._build_expr()
        expr = f._expr
        return self._format_node(expr)

    def format_body(self, f):
        '''
        Its like format() but only stringifies the body expression of the rlambda object (the signature is ommited)
        :return:
        '''
        from .rlambda import RLambda
        if not isinstance(f, RLambda):
            raise TypeError('Expected rlambda object at argument 1 but got {}'.format(type(f).__name__))
        f._build_expr()
        expr = f._expr
        return self._format_node(expr.body.body)


    def _format_node(self, node):
        assert isinstance(node, Node)

        # Call the right formatter to stringify the node depending on its kind.
        formatters = {
            Variable: self._format_variable,
            Literal: self._format_literal,
            Placeholder: self._format_placeholder,
            Operator: self._format_operator,
            Index: self._format_index,
            Slice: self._format_slice,
            ExtendedSlice: self._format_extslice,
            Operation: self._format_operation,
            Lambda: self._format_lambda,
            Expression: self._format_expression
        }
        formatter = formatters[findsuperclassof(formatters.keys(), type(node))]
        return formatter(node)

    def _format_subnode(self, node, parent=None):

        s = self._format_node(node)
        if not isinstance(node, Operation):
            return s
        if isinstance(parent, Operation):
            # Compare operations precedence. If the inner operation has more precedence than outer operation, we dont
            # need to enclose inner operation with brackets
            if node > parent:
                return s

            if isinstance(node, BinaryOperation) and isinstance(parent, BinaryOperation) and\
                    node.op == parent.op and node.associative:
                # Parent and child are binary operations of the same kind and follows the associativity rule.
                # We dont need to enclose the inner operation with brackets as the expression is evaluated left to right.
                return s

        return self.format_precedence(s)




    def _format_variable(self, node):
        assert isinstance(node, Variable)
        return self.format_arg(node.id)


    def _format_literal(self, node):
        assert isinstance(node, Literal)

        if isinstance(node, LiteralNumber):
            return self.format_value(node.n)
        if isinstance(node, (LiteralStr, LiteralBytes)):
            return self.format_value(node.s)
        if isinstance(node, LiteralEllipsis):
            return self.format_value(Ellipsis)
        if isinstance(node, LiteralNone):
            return self.format_value(None)
        if isinstance(node, LiteralBool):
            return self.format_value(node.value)

        raise NotImplementedError()


    def _format_placeholder(self, node):
        assert isinstance(node, Placeholder)
        return self.format_value(node.value)


    def _format_operator(self, node):
        assert isinstance(node, Operator)
        return self.format_operator(node.symbol)


    def _format_index(self, index):
        return self._format_node(index.value)


    def _format_slice(self, index):
        return self.format_slice(
            slice(
            None if index.lower is None else self._format_subnode(index.lower),
            None if index.upper is None else self._format_subnode(index.upper),
            None if index.step is None else self._format_subnode(index.step))
        )

    def _format_extslice(self, indexes):
        return self.format_extslice(
            tuple(
                map(lambda index: self._format_index(index) if isinstance(index, Index) else self._format_slice(index),
                    indexes.dims)
            )
        )

    def _format_operation(self, node):
        assert isinstance(node, Operation)

        formatters = {
            UnaryOperation: self._format_unary_operation,
            BinaryOperation: self._format_binary_operation,
            CompareOperation: self._format_compare_operation,
            SubscriptOperation: self._format_subscript_operation,
            AttributeOperation: self._format_getattr_operation,
            CallOperation: self._format_call_operation
        }
        formatter = formatters[findsuperclassof(formatters.keys(), type(node))]
        return formatter(node)


    def _format_unary_operation(self, node):
        assert isinstance(node, UnaryOperation)
        return self.format_unary_operation(
            self._format_subnode(node.op, node),
            self._format_subnode(node.operand, node)
        )

    def _format_binary_operation(self, node):
        assert isinstance(node, BinaryOperation)
        return self.format_binary_operation(
            self._format_subnode(node.left, node),
            self._format_subnode(node.op, node),
            self._format_subnode(node.right, node)
        )

    def _format_compare_operation(self, node):
        assert isinstance(node, CompareOperation)
        items = tuple(
            map(lambda subnode: self._format_subnode(subnode, node),
                chain((node.left,), reduce(operator.add, zip(node.ops, node.comparators)))))
        return self.format_compare_operation(*items)

    def _format_subscript_operation(self, node):
        assert isinstance(node, SubscriptOperation)
        return self.format_subscript_operation(
            self._format_subnode(node.value, node),
            self._format_subnode(node.slice, node)
        )

    def _format_getattr_operation(self, node):
        assert isinstance(node, AttributeOperation)
        return self.format_getattr_operation(
            self._format_subnode(node.value, node),
            node.attr
        )

    def _format_call_operation(self, node):
        assert isinstance(node, CallOperation)

        return self.format_call_operation(
            self._format_subnode(node.func, node),
            *tuple(map(self._format_node, node.args)),
            **dict(zip(
                map(attrgetter('arg'), node.keywords),
                map(lambda kwarg: self._format_subnode(kwarg.value, node), node.keywords)))
        )


    def _format_lambda(self, node):
        assert isinstance(node, Lambda)

        params = node.args.args

        # If signature parameters list is empty, just print the body.
        if  len(params) == 0:
            return self._format_node(node.body)

        return self.format_lambda(
            self.format_signature(
                tuple(map(lambda param: self.format_param(param.arg), params))
            ),
            self._format_node(node.body))


    def _format_expression(self, node):
        return self._format_node(node.body)




    def format_lambda(self, signature, body):
        '''
        Formats the lambda node given the signature and the body formatted. e.g:
        format_lambda('x, y, z', 'x + y + z') -> "x, y, z : x + y + z"
        :param signature: Its the signature node already formatted
        :param body: Its the body expression node already formatted
        '''
        return signature + ' : ' + body


    def format_signature(self, params):
        '''
        Formats the signature node given the input parameters already formatted and sorted by name. e.g:
        format_signature(['x', 'y', 'z']) -> 'x, y, z'
        :param params: A tuple of parameters (strings)
        '''
        return ', '.join(params)


    def format_param(self, name):
        '''
        Formats an input parameter when it is shown in the function signature.
        If format_param would be defined like
        format_param('x') -> '$x'
        then
        format((x + y) // 2) -> "$x, $y : (x + y) // 2"

        :param name: The parameter name to be formatted
        :return:
        '''
        return name

    def format_arg(self, name):
        '''
        Format an input parameter when it is shown in the body expression. e.g:
        If format_arg would be define like
        format_arg('x') -> '$x'
        then
        format((x + y) // 2) -> "x, y: ($x+ $y) // 2"

        :param name:
        :return:
        '''
        return name


    def format_operator(self, op):
        '''
        Format an operator node.
        :param op: Is the symbol of the operator. Could be one in the next list:
        '+', '-', '~', '*', '/', '//', '%', '**', '<<', '>>', '|' '&', '^', '@',
        '<', '<=', '>', '>=', '==', '!=',
        :return:
        '''
        return op


    def format_value(self, value):
        '''
        Format value nodes (literals used to build the body expression)
        :param value: The value to be formatted (could be any kind of object)
        e.g:
        When calling RLambdaFormatter().format( (x+1)*2 )

        this function is called two times to format the literals 1 and 2:
        format_value(1) -> "1"
        format_value(2) -> "2"

        '''
        if isinstance(value, (FunctionType, LambdaType, BuiltinFunctionType)):
            return value.__qualname__
        if isclass(value):
            return value.__name__

        if value is Ellipsis:
            return '...'

        if isinstance(value, (str, bytes)):
            return str(value)

        return repr(value)


    def format_unary_operation(self, op, operand):
        '''
        Format a unary operation node given the operator and the single operand nodes as arguments already formatted
        :param op: Is the unary operator already formatted (a string object)
        :param operand: Is the unique operand already formatted (a string object)
        e.g:
        format_unary_operation('-', 'x') -> "-x"
        '''
        return op + operand


    def format_binary_operation(self, left, op, right):
        '''
        Format a binary operation node given the operator and the left and right operand nodes as arguments already formatted
        :param left: Is the left operand already formatted
        :param op: Is the binary operator already formatted
        :param right: Is the right operand already formatted

        e.g:
        format_binary_operation('x', '+', 'y') -> "x + y"
        '''
        return ' '.join((left, op, right))


    def format_compare_operation(self, first_operand, first_op, second_operand, *args):
        '''
        Formats a comparision operation node given all the operators and operands noodes as arguments already formatted
        :param first_operand: Is the first operand already formatted
        :param first_op: Is the first operator already formatted
        :param second_operand: Is the second operator already formatted
        :param args: Is an additional sequence of operators and operands (this is specified when more
        than 2 operands appears in the comparision operation)

        If len(args)>0, it will contain an even number of elements (at least 2). At positions with even indexes, an operand
        will appear whereas in positions with odd indexes, and operator will appear instead.

        For example, if the operation would be 1 <= x <= y <= z <= 2
        This function will be called like:
        format_compare_operation('1', '<=', 'x', '<=', 'y', '<=', 'z', '<=', '2')

        Operators could take one of the next values:
        '<', '<=', '>', '>=', '==', '!='
        '''
        return ' '.join(chain((first_operand, first_op, second_operand), args))


    def format_index(self, index):
        '''
        Format an index node.
        e.g:
        Doing RLambdaFormatter().format( x[1] ) will invoke format_index('1') and trivially return '1'
        Then format_subscript_operation('x', '1') is called to build the final string. It will return 'x[1]'
        '''
        return index


    def format_slice(self, index):
        '''
        Format a slice node.
        e.g:
        Doing RLambdaFormatter().format( x[1:10:2] ) will call format_slice(slice('1', '10', '2')). The string
        '1:10:2' will be returned. Then format_subscript_operation('x', '1:10:2') its called to build the final string, 'x[1:10:2]'
        '''
        return slice_to_str(index)


    def format_extslice(self, indexes):
        '''
        Format a extended slice node
        Doing RLambdaFormatter().format( x[1, 1:10] ) will invoke format_extslice('1', slice('1', '10'))
        Such call will return the string '1, 1:10' which is later passed to format_subscript_operation.
        It will be called like format_subscript_operation('x', '1, 1:10') and will return 'x[1, 1:10]'
        '''
        return ', '.join(map(lambda x: x if not isinstance(x, slice) else slice_to_str(x), indexes))


    def format_subscript_operation(self, container, index):
        '''
        Format a subscript operation node.
        Doing RLambdaFormatter().format( x[1, 1:10, 1:10:2] ) will call format_subscript_operation('x', '1, 1:10, 1:10:2')
        and returns the string 'x[1, 1:10, 1:10:2]'
        '''
        return container + enclose(index, '[]')


    def format_getattr_operation(self, obj, key):
        '''
        Format an attribute access operation node.
        :param obj: Is the object being accessed already formatted (string object)
        :param key: Is the name of the attribute of the object being requested in the operation.
        e.g:
        RLambdaFormatter().format( x.name ) will call
        format_getattr_operation( 'x', 'name' ) and will return 'x.name'
        '''
        return obj + '.' + key

    def format_call_operation(self, func, *args, **kwargs):
        '''
        Format a function call operation node.
        :param func: Is the calling function name
        :param args: All the positional arguments that will be passed to the function already formatted (string objects)
        :param kwargs: All keyword arguments that will be passed to the function already formatted (keys & values are strings)
        :return:
        '''
        return func + enclose(', '.join(chain(args, [key+'='+value for key, value in kwargs.items()])), '()')

    def format_precedence(self, expr):
        '''
        Enclose the given expression node to show that its body should have higher precedence
        e.g:
        format_precedence('x+1') -> "(x+1)"
        :param expr:
        :return:
        '''
        return enclose(expr, '()')



@singleton
class DefaultRLambdaFormatter(RLambdaFormatter):
    '''
    This class will be the default rlambda formatter. It can be used as a singleton.
    '''
    pass


class MathRLambdaFormatter(RLambdaFormatter):
    '''
    An alternative formatter for printing rlambda objects:

    - Variables a, b, c and d are printed as lowercased greek letters: α, β, γ, δ
    Also A, B, C, D are formatted as uppercased greek letters: Α, Β, Γ, Δ

    - Any other single character different than a,b,c,d and A,B,C,D are formatted as normal latin characters.

    - The next variable names are formatted to its corresponding lowercase greek letter:
    alpha, beta, gamma, delta, zeta, eta, theta, iota, kappa, lambda, mu, nu, xi, omicron, rho, varsigma,
    sigma, tau, upsilon, phi, chi, psi, omega
    If you upper case the first character, the variable will be formatted to its corresponding uppercase greek letter:
    Alpha, Beta, Gamma, Delta, ...

    pi and epsilon are reserved (they are used for math constants pi and e)

    - Absolute value operations like abs(x + y) are formatted like |x + y|
    - Ceiling and floor operations are formatted like  ⌈x + y⌉,  ⌊x + y⌋
    - Sqrt operations are formatted like √x, √(x + y)
    - Factorials are printed like x!, (x + y)!

    - Logarithms with integer base are printed like log₂(x), log₁₀(x + y)
    - Power operator is replaced with '^' symbol.
    - Integer exponents are printed as superindices: x², (x+y)², (x+y)³

    - Regular Multiplication symbol will be '×'
    - Bitwise not, and, or, xor operators symbols will be ¬, ∧ ∨, ⊕ respectively
    - Comparision operators ==, !=, <=, >= will be =, ≠, ≤, ≥ respectively


    - float('inf') prints the infinite symbol ∞
    - complex numbers are printed with the format xi + yj, e.g: 2i + 1, 3i
    If only the real part is present, they are printed like regular numbers.
    '''



    def format_operator(self, op):
        math_op = {
            # Bitwise operators
            '~': '\u00ac',
            '|': '\u2228',
            '&': '\u2227',
            '^': '\u2295',

            # Comparision operators
            '==': '=',
            '!=': '\u2260',
            '>=': '\u2265',
            '<=': '\u2264',

            # Arithmetic operators
            '*': '\u00d7',
            '**': '^'
        }
        if op in math_op:
            return math_op[op]
        return op


    def format_value(self, value):
        # Natural numbers with float type
        if isinstance(value, float) and int(value) == value:
            return self.format_value(int(value))

        # Infinite or minus infinite values
        if isinstance(value, float) and abs(value) == float('inf'):
            return '\u221e' if value > 0 else '-\u221e'

        # Complex numbers
        if isinstance(value, complex):
            if value.imag == 0:
                return self.format_value(value.real)
            elif value.real == 0:
                return self.format_value(value.imag) + 'i'
            elif value.imag == 0 and value.real == 0:
                return self.format_value(0)
            return '{}i {} {}'.format(
                self.format_value(value.imag),
                '+' if value.real > 0 else '-',
                self.format_value(abs(value.real)))

        return super().format_value(value)


    def format_param(self, name):
        return self.format_arg(name)

    def format_arg(self, name):

        class GreekAlphabet:
            def __init__(self):
                self.names = [
                    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',
                    'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi',
                    'omicron', 'pi', 'rho', 'varsigma',
                    'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']


            def get_symbol(self, name):
                index = self.names.index(name)
                return chr(0x03b1 + index)

            def __contains__(self, item):
                return item in self.names

        greek_alphabet = GreekAlphabet()


        # A greek letter name is translated to its corresponding character
        if name.lower() in greek_alphabet:
            greek_letter = greek_alphabet.get_symbol(name.lower())
            return greek_letter.lower() if not name.istitle() else greek_letter.upper()

        # a, b, c and d latin chars will be formatted to its corresponding greek letters.
        if name.lower() in ('a', 'b', 'c', 'd'):
            greek_name = {'a': 'alpha', 'b': 'beta', 'c': 'gamma', 'd': 'delta'}[name.lower()]
            if name.isupper():
                greek_name = greek_name.title()
            return self.format_arg(greek_name)

        return name


    def _format_call_operation(self, node):
        if isinstance(node.func, Placeholder) and callable(node.func.value):
            func = node.func.value

            args = tuple(node.args)
            kwargs = dict(zip(
                map(attrgetter('arg'), node.keywords),
                node.keywords))

            if len(kwargs) == 0 and len(args) == 1:
                arg = args[0]
                formatted_arg = self._format_node(arg)


                # Absolute function calls
                if func == builtins.abs:
                    return '|' + formatted_arg + '|'

                # Ceil & floor functions calls
                if func == math.ceil:
                    return '\u2308' + formatted_arg + '\u2309'

                if func == math.floor:
                    return '\u230a' + formatted_arg + '\u230b'

                # Factorial
                if func == math.factorial:
                    if isinstance(arg, Variable):
                        return formatted_arg + '!'
                    return self.format_precedence(formatted_arg) + '!'

                # Square roots
                if func == math.sqrt:
                    if isinstance(arg, Variable):
                        return '\u221a' + formatted_arg
                    return '\u221a' + self.format_precedence(formatted_arg)

                # e raised to a power of x minus 1
                if func == math.expm1:
                    return self._format_node(BinaryOperation(Variable('e'), Pow, BinaryOperation(arg, Sub, LiteralNumber(1))))

                # e raised to the power x
                if func == math.exp:
                    return self._format_node(BinaryOperation(Variable('e'), Pow, arg))

                # Logarithms
                if func in (math.log, math.log2, math.log10, math.log1p):
                    if func in (math.log, math.log1p):
                        base = 'log'
                        if isinstance(arg, Variable):
                            base += ' '
                    else:
                        if func == math.log10:
                            n = 10
                        else:
                            n = 2
                        base = 'log' + unicode_subscript(n)

                    if func == math.log1p:
                        body = self._format_subnode(BinaryOperation(LiteralNumber(1), Add, arg))
                    else:
                        if isinstance(arg, Variable):
                            body = formatted_arg
                        else:
                            body = self.format_precedence(formatted_arg)

                    return base + body




            elif len(kwargs) == 0 and len(args) == 2:
                # pow
                if func == math.pow:
                    return self._format_node(BinaryOperation(args[0], Pow,args[1]))

        return super()._format_call_operation(node)



    def _format_binary_operation(self, node):

        # Power binary operations
        if node.op == Pow and isinstance(node.right, LiteralNumber) and isinstance(node.right.n, int):
            base = self._format_node(node.left)
            if not isinstance(node.left, Variable):
                base = self.format_precedence(base)
            exp = unicode_superscript(node.right.n)
            return base + exp

        return super()._format_binary_operation(node)


    def format_binary_operation(self, left, op, right):
        if op == '^':
            return left + op + right
        return super().format_binary_operation(left, op, right)


    def _format_subnode(self, node, parent=None):

        # Complex numbers enclosed with parenthesis if both real and imaginary parts are non-zeto
        if isinstance(node, Placeholder) and isinstance(node.value, complex):
            value = node.value
            if value.imag != 0 and value.real != 0:
                return self.format_precedence(self._format_node(node))
            return self._format_node(node)

        return super()._format_subnode(node, parent)