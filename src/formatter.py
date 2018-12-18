

import operator
from operator import attrgetter
from itertools import chain
from functools import reduce
from types import BuiltinFunctionType, FunctionType, LambdaType
from inspect import isclass

from .utils import enclose, slice_to_str
from .astwrappers import Node, Lambda, Expression, Variable, Operator, Placeholder, Index, Slice, ExtendedSlice
from .astwrappers import Literal, LiteralNumber, LiteralEllipsis, LiteralBytes, LiteralBool, LiteralNone, LiteralStr
from .astwrappers import Operation, UnaryOperation, BinaryOperation, CompareOperation,\
    SubscriptOperation, AttributeOperation, CallOperation
from .singleton import singleton

@singleton
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
        This method should not be overrided. Look the next methods below to customize rlambda formatting.
        '''
        from .rlambda import RLambda
        if not isinstance(f, RLambda):
            raise TypeError('Expected rlambda object at argument 1 but got {}'.format(type(f).__name__))

        f._build_expr()
        expr = f._expr
        return self.format_node(expr)

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
        return self.format_node(expr.body.body)


    def format_node(self, node):
        assert isinstance(node, Node)

        # Variables
        if isinstance(node, Variable):
            return self.format_arg(node.id)

        # Literals
        if isinstance(node, Literal):
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

        # Placeholders
        if isinstance(node, Placeholder):
            return self.format_value(node.value)

        # Operators
        if isinstance(node, Operator):
            return self.format_operator(node.symbol)

        # Index
        if isinstance(node, Index):
            return self._format_index(node)

        # Slice
        if isinstance(node, Slice):
            return repr(self._format_slice(node))

        # Extended slice
        if isinstance(node, ExtendedSlice):
            return repr(self._format_extslice(node))

        # Operations
        if isinstance(node, Operation):
            # Unary operations
            if isinstance(node, UnaryOperation):
                return self.format_unary_operation(
                    self.format_node(node.op),
                    self.format_node(node.operand)
                )

            # Binary operations
            if isinstance(node, BinaryOperation):
                return self.format_binary_operation(
                    self.format_node(node.left),
                    self.format_node(node.op),
                    self.format_node(node.right)
                )

            # Compare operations
            if isinstance(node, CompareOperation):
                items = tuple(map(self.format_node, chain((node.left,), reduce(operator.add, zip(node.ops, node.comparators)))))
                return self.format_compare_operation(*items)

            # Subscript operations
            if isinstance(node, SubscriptOperation):
                if isinstance(node.slice, Index):
                    return self.format_indexing_operation(
                        self.format_node(node.value),
                        self._format_index(node.slice)
                    )
                if isinstance(node.slice, Slice):
                    return self.format_slicing_operation(
                        self.format_node(node.value),
                        self._format_slice(node.slice)
                    )

                if isinstance(node.slice, ExtendedSlice):
                    return self.format_extslicing_operation(
                        self.format_node(node.value),
                        self._format_extslice(node.slice)
                    )


            # Attribute access operations
            if isinstance(node, AttributeOperation):
                return self.format_getattr_operation(
                    self._format_sunode(node.value),
                    node.attr
                )

            # Call function operation
            if isinstance(node, CallOperation):
                return self.format_call_operation(
                    self._format_sunode(node.func),
                    *tuple(map(self.format_node, node.args)),
                    **dict(map(
                        lambda key, value: (key, self._format_sunode(value)),
                        map(attrgetter('arg', 'value'), node.keywords)
                    ))

                )

        # Lambdas
        if isinstance(node, Lambda):
            return self.format_lambda(
                self.format_signature(
                    tuple(map(lambda arg: self.format_param(arg.arg), node.args.args))
                ),
                self.format_node(node.body))

        # Expressions
        if isinstance(node, Expression):
            return self.format_node(node.body)

        raise NotImplementedError()



    def _format_index(self, index):
        return self.format_node(index.value)


    def _format_slice(self, index):
        return slice(
            None if index.lower is None else self.format_node(index.lower),
            None if index.upper is None else self.format_node(index.upper),
            None if index.step is None else self.format_node(index.step)
        )

    def _format_extslice(self, indexes):
        return tuple(
            map(lambda index: self._format_index(index) if isinstance(index, Index) else self._format_slice(index),
                indexes.dims)
        )



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


    def format_indexing_operation(self, container, index):
        '''
        Formats a basic indexing operation node.
        :param container: Is the object being indexed already formatted (string object)
        :param index: Is the index already formatted
        e.g:
        RLambdaFormatter().format( x[1] ) will call format_indexing_operation('x', '1') and it will return "x[1]"
        '''
        return container + enclose(index, '[]')


    def format_slicing_operation(self, container, index):
        '''
        Format a basic slice indexing operation node.
        :param container: Is the object being indexed already formatted (string object)
        :param index: Is an object of the class slice such that "start", "stop" and "step" attributes are the values use
        for indexing the container already formatted (either strings or value None to indicate they are not set)
        e.g:
        RLambdaFormatter().format( x[1:2:1] ) will call format_slicing_operation('x', slice('1', '2', '1')) and will
        return 'x[1:2:1]'
        '''
        return container + enclose(slice_to_str(index), '[]')


    def format_extslicing_operation(self, container, indexes):
        '''
        Format a extended slice indexing operation node.
        :param container: Is the object being indexed already formatted (string object)
        :param indexes: Is a tuple of values used for indexing already formatted (either strings or instances of class slice)
        e.g:
        RLambdaFormatter().format( x[1:2, 3:4:1] ) will call
        format_extslicing_operation('x', ( slice('1', '2', None), slice('3', 4', '1') ) )
        '''

        s = ', '.join(map(lambda x: x if not isinstance(x, slice) else slice_to_str(x), indexes))
        return container + enclose(s if len(indexes) > 1 else s + ',', '[]')


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

    def format_expression_enclosed(self, expr):
        '''
        Enclose the given expression node to show that its body should have higher precedence
        e.g:
        format_expression_enclosed('x+1') -> "(x+1)"
        :param expr:
        :return:
        '''
        return enclose(expr, '()')

