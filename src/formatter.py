

import operator
from operator import attrgetter
from itertools import chain
from functools import reduce
from types import BuiltinFunctionType, FunctionType, LambdaType
from inspect import isclass

from .utils import enclose, slice_to_str, findsuperclassof
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
            None if index.lower is None else self._format_node(index.lower),
            None if index.upper is None else self._format_node(index.upper),
            None if index.step is None else self._format_node(index.step))
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
            self._format_node(node.op),
            self._format_node(node.operand)
        )

    def _format_binary_operation(self, node):
        assert isinstance(node, BinaryOperation)
        return self.format_binary_operation(
            self._format_node(node.left),
            self._format_node(node.op),
            self._format_node(node.right)
        )

    def _format_compare_operation(self, node):
        assert isinstance(node, CompareOperation)
        items = tuple(
            map(self._format_node, chain((node.left,), reduce(operator.add, zip(node.ops, node.comparators)))))
        return self.format_compare_operation(*items)

    def _format_subscript_operation(self, node):
        assert isinstance(node, SubscriptOperation)
        return self.format_subscript_operation(
            self._format_node(node.value),
            self._format_node(node.slice)
        )

    def _format_getattr_operation(self, node):
        assert isinstance(node, AttributeOperation)
        return self.format_getattr_operation(
            self._format_node(node.value),
            node.attr
        )

    def _format_call_operation(self, node):
        assert isinstance(node, CallOperation)
        return self.format_call_operation(
            self._format_node(node.func),
            *tuple(map(self._format_node, node.args)),
            **dict(map(
                lambda key, value: (key, self._format_node(value)),
                map(attrgetter('arg', 'value'), node.keywords)
            ))

        )

    def _format_lambda(self, node):
        assert isinstance(node, Lambda)

        return self.format_lambda(
            self.format_signature(
                tuple(map(lambda arg: self.format_param(arg.arg), node.args.args))
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
        return index


    def format_slice(self, index):
        return slice_to_str(index)


    def format_extslice(self, indexes):
        return ', '.join(map(lambda x: x if not isinstance(x, slice) else slice_to_str(x), indexes))


    def format_subscript_operation(self, container, index):
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

    def format_expression_enclosed(self, expr):
        '''
        Enclose the given expression node to show that its body should have higher precedence
        e.g:
        format_expression_enclosed('x+1') -> "(x+1)"
        :param expr:
        :return:
        '''
        return enclose(expr, '()')

