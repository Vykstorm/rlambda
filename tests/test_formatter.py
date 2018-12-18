
'''
Unitary test for RLambdaFormatter class
'''

import unittest
from unittest import TestCase
from itertools import product
from operator import attrgetter


from src.utils import enclose, slice_to_str
from src.abc import x, y, z
from src.formatter import RLambdaFormatter as Formatter
from src.astwrappers import Variable
from src.astwrappers import LiteralStr, LiteralBytes, LiteralNumber, LiteralEllipsis, LiteralBool, LiteralNone
from src.astwrappers import operators, compare_operators, unary_operators, binary_operators
from src.astwrappers import UnaryOperation, BinaryOperation, CompareOperation, AttributeOperation, SubscriptOperation, CallOperation
from src.astwrappers import Index, Slice, ExtendedSlice


class RLambdaFormatterTest(TestCase):


    def test_variables(self):
        '''
        Test variables are formatted correctly
        :return:
        '''
        format = Formatter()._format_node

        self.assertEqual(format(Variable('x')), 'x')
        self.assertEqual(format(Variable('y')), 'y')
        self.assertEqual(format(Variable('z')), 'z')

    def test_literals(self):
        '''
        Test literals are formatted correctly
        :return:
        '''
        format = Formatter()._format_node
        self.assertEqual(format(LiteralNumber(10)), repr(10))
        self.assertEqual(format(LiteralEllipsis()), '...')
        self.assertEqual(format(LiteralStr('Hello World')), 'Hello World')
        self.assertEqual(format(LiteralBytes(b'Hello World')), str(b'Hello World'))
        self.assertEqual(format(LiteralBool(True)), str(True))
        self.assertEqual(format(LiteralBool(False)), str(False))
        self.assertEqual(format(LiteralNone()), str(None))


    def test_operators(self):
        '''
        Test operators are formatted correctly
        '''
        format = Formatter()._format_node

        for operator in operators:
            self.assertEqual(format(operator), operator.symbol)


    def test_unary_operations(self):
        '''
        Test unary operations are formatted correctly
        '''
        format = Formatter()._format_node
        operands = tuple(map(Variable, ('x', 'y', 'z')))
        for operator, operand in product(unary_operators, operands):
            self.assertEqual(format(UnaryOperation(operator, operand)), operator.symbol + operand.id)


    def test_binary_operations(self):
        '''
        Test binary operations are formatted correctly
        '''
        format = Formatter()._format_node
        operands = tuple(map(Variable, ('x', 'y', 'z')))
        for left, operator, right in product(operands, binary_operators, operands):
            self.assertEqual(format(BinaryOperation(left, operator, right)), ' '.join([left.id, operator.symbol, right.id]))

    def test_compare_operations(self):
        '''
        Test compare operations are formatted correctly
        '''
        format = Formatter()._format_node
        operands = tuple(map(Variable, ('x', 'y', 'z')))
        for left, operator, right in product(operands, compare_operators, operands):
            self.assertEqual(format(CompareOperation(left, operator, right)), ' '.join([left.id, operator.symbol, right.id]))

    def test_getattr_operations(self):
        '''
        Test get attribute operations are formatted properly
        '''
        format = Formatter()._format_node

        attrs = tuple(map(chr, range(ord('a'), ord('z')+1)))
        containers = tuple(map(Variable, ('x', 'y', 'z')))

        for container, attr in product(containers, attrs):
            self.assertEqual(format(AttributeOperation(container, attr)), container.id+'.'+attr)

    def test_subscript_operations(self):
        '''
        Test subscripting operations are formatted properly
        :return:
        '''
        format = Formatter()._format_node

        containers = tuple(map(Variable, ('x', 'y', 'z')))

        indexes = map(lambda x: Index(LiteralNumber(x)), range(0, 20))
        for container, index in product(containers, indexes):
            self.assertEqual(format(SubscriptOperation(container, index)), container.id+enclose(repr(index.value.n),'[]'))

        container = Variable('x')
        for start, stop, step in product(map(LiteralNumber, range(0, 3)), repeat=3):
            self.assertEqual(
                format(SubscriptOperation(container, Slice(start, stop, step))),
                container.id+enclose(slice_to_str(slice(start.n, stop.n, step.n)), '[]'))

        container = Variable('x')
        for a, b in product(map(lambda x: Index(LiteralNumber(x)), range(0, 3)), repeat=2):
            indexes = (a, b)
            self.assertEqual(
                format(SubscriptOperation(container, ExtendedSlice(*indexes))),
                container.id+enclose(', '.join(map(lambda index: repr(index.value.n), indexes)), '[]'))

    def test_call_operations(self):
        '''
        Test calling operations are formatted properly
        '''
        format = Formatter()._format_node
        containers = tuple(map(Variable, ('x', 'y', 'z')))
        for container, args in product(containers, product(map(LiteralNumber, range(0, 3)),repeat=3)):
            self.assertEqual(
                format(CallOperation(container, *args)),
                container.id + enclose(', '.join(map(lambda arg: repr(arg.n), args)),'()'))


if __name__ == '__main__':
    unittest.main()