

'''
Tests for aswrappers module classes
'''

import unittest
from unittest import TestCase
from copy import copy, deepcopy
from src.astwrappers import *
import ast



class TestAstWrappers(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The next list is a set of nodes (instances of Node subclasses) for test purposes
        nodes = [
            Variable('x'),
            Placeholder([1, 2, 3]),

            LiteralNumber(1),
            LiteralStr('Hello World!'),
            LiteralBytes(b'Hello World!'),
            LiteralBool(True), LiteralBool(False),
            LiteralNone(),
            LiteralEllipsis(),

            UnaryOperation(UnarySub, LiteralNumber(1)),
            BinaryOperation(LiteralNumber(1), Add, LiteralNumber(2)),
            CompareOperation(LiteralNumber(1), LowerThan, LiteralNumber(2), LowerThan, LiteralNumber(3)),

            Index(LiteralNumber(1)),
            Slice(LiteralNumber(1), LiteralNumber(10), LiteralNumber(1)),
            ExtendedSlice(
                Index(LiteralNumber(1)),
                Slice(LiteralNumber(1), LiteralNumber(10), LiteralNumber(1))
            ),

            SubscriptOperation(Variable('x'), Index(LiteralNumber(1))),
            AttributeOperation(Variable('x'), 'imag'),
            CallOperation(Variable('x'), LiteralNumber(1), v=LiteralNumber(2)),

            Lambda(('x', 'y'), BinaryOperation(Variable('x'), Sub, Variable('y'))),
            Expression(UnaryOperation(UnaryAdd, LiteralNumber(-10)))
        ]
        nodes.extend(operators)
        self.nodes = nodes


    def test_copy(self):
        '''
        All instances of any subclass of Node can be copied
        :return:
        '''
        for node in self.nodes:
            clone = copy(node)
            self.assertIsNot(clone, node)
            self.assertEqual(type(clone), type(node))
            self.assertEqual(clone, node)
            self.assertEqual(clone.__dict__.keys(), node.__dict__.keys())

            # Attributes that are instances of Node subclasses or ast.AST are copied by reference (on shallow copies)
            for key in node.__dict__:
                a, b = node.__dict__[key], clone.__dict__[key]
                if isinstance(a, (Node, ast.AST)):
                    self.assertIs(a, b)
                else:
                    self.assertEqual(a, b)

    def test_deepcopy(self):
        '''
        All instance of any subclass of Node can be deep-copied
        '''
        for node in self.nodes:
            clone = deepcopy(node)
            self.assertIsNot(clone, node)
            self.assertEqual(type(clone), type(node))
            self.assertEqual(clone.__dict__.keys(), node.__dict__.keys())

            # Attributes that are instances of Node subclasses are copied by value (deep copy)
            # Instances of class ast.AST that are not instances of class Node also are copied by reference (shallow copy)
            for key in node.__dict__:
                a, b = node.__dict__[key], clone.__dict__[key]

                if isinstance(a, Node):
                    self.assertIsNot(a, b)
                elif isinstance(a, ast.AST):
                    self.assertIs(a, b)
                else:
                    if not iterable(a):
                        self.assertEqual(a, b)

    


if __name__ == '__main__':
    unittest.main()