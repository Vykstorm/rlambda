
'''
Unitary test for rlambda class
'''



import unittest
from unittest import TestCase
from inspect import signature
from itertools import combinations_with_replacement, combinations, chain

from src.rlambda import RLambda
from src.abc import x, y, z, w


class RLambdaTest(TestCase):
    def test_introspection(self):
        '''
        rlambda objects are introspectable. Signature information can be fetched using the standard module "inspect"
        :return:
        '''
        f = (x + 1) * 2
        params = signature(f).parameters
        self.assertTrue(len(params) == 1 and next(iter(params.keys())) == 'x')
        self.assertTrue(hasattr(f, '__name__'))

    def test_identities(self):
        '''
        identities x, y, z, w are rlambda objects equivalent to lambda functions defined like lambda x: x, lambda y: y, ...
        :return:
        '''
        for identity in (x, y, z, w):
            params = signature(identity).parameters
            self.assertTrue(len(params) == 1)
            self.assertEqual(identity(1), 1)


    def test_arithmetic_operations(self):
        '''
        test rlambda arithmetic operations
        :return:
        '''

        # Unary operations

        for a in range(-10, 10, 2):
            self.assertEqual((+x)(-a), +(-a))
            self.assertEqual((-x)(+a), -(+a))

        # Binary operations

        for a, b in combinations(range(4), 2):
            self.assertEqual((x+y)(a, b), a+b)
            self.assertEqual((x-y)(a, b), a-b)
            self.assertEqual((x*y)(a, b), a*b)
            self.assertEqual((x/y)(a, b), a/b)
            self.assertEqual((x//y)(a, b), a//b)
            self.assertEqual((x%y)(a, b), a%b)
            self.assertEqual((x**y)(a, b), a**b)

            self.assertEqual((x+b)(a), a+b)
            self.assertEqual((x-b)(a), a-b)
            self.assertEqual((x*b)(a), a*b)
            self.assertEqual((x/b)(a), a/b)
            self.assertEqual((x//b)(a), a//b)
            self.assertEqual((x%b)(a), a%b)
            self.assertEqual((x**b)(a), a**b)

        # Test matmult operator
        try:
            import numpy as np

            a, b = np.identity(2), np.ones(2)
            self.assertEqual((x @ y)(a, b), a @ b)
        except:
            pass


    def test_bitwise_operations(self):
        '''
        Test rlambda bitwise operations.
        :return:
        '''
        # Unary operations

        for a in range(0, 10, 2):
            self.assertEqual((~x)(a), ~a)

        # Binary operations

        for a, b in combinations(range(4), 2):
            self.assertEqual((x<<y)(a, b), a<<b)
            self.assertEqual((x>>y)(a, b), a>>b)
            self.assertEqual((x&y)(a, b), a&b)
            self.assertEqual((x|y)(a, b), a|b)
            self.assertEqual((x^y)(a, b), a^b)

            self.assertEqual((x<<b)(a), a<<b)
            self.assertEqual((x>>b)(a), a>>b)
            self.assertEqual((x&b)(a), a&b)
            self.assertEqual((x|b)(a), a|b)
            self.assertEqual((x^b)(a), a^b)


    def test_comparision_operators(self):
        '''
        Test rlambda comparision operators
        :return:
        '''
        for a, b in chain(combinations_with_replacement([1, 2], 2), combinations_with_replacement([2, 1], 2)):
            self.assertEqual((x>y)(a,b), a>b)
            self.assertEqual((x>=y)(a,b), a>=b)
            self.assertEqual((x<y)(a,b), a<b)
            self.assertEqual((x<=y)(a,b), a<=b)
            self.assertEqual((x==y)(a,b), a==b)
            self.assertEqual((x!=y)(a,b), a!=b)

            self.assertEqual((x > b)(a), a > b)
            self.assertEqual((x >= b)(a), a >= b)
            self.assertEqual((x < b)(a), a < b)
            self.assertEqual((x <= b)(a), a <= b)
            self.assertEqual((x == b)(a), a == b)
            self.assertEqual((x != b)(a), a != b)


    def test_subscripting_operation(self):
        '''
        Test rlambda subscripting operations.
        :return:
        '''
        items = (1, 2, 3, 4)

        # Test indexing
        for i in range(len(items)):
            self.assertEqual((x[i])(items), items[i])

        # Test basic slicing
        for a, b in combinations(range(len(items)+1), 2):
            self.assertEqual((x[a:b])(items), items[a:b])

        for a in range(len(items)+1):
            self.assertEqual((x[:a])(items), items[:a])
            self.assertEqual((x[:-a])(items), items[:-a])
            self.assertEqual((x[a:])(items), items[a:])
            self.assertEqual((x[-a:])(items), items[-a:])

        # Test extended slicing
        try:
            import numpy as np

            m = np.identity(4)

            self.assertEqual((x[2:4, 2:4])(m), m[2:4, 2:4])
            self.assertEqual((x[2:4, 1])(m), m[2:4, 1])
            self.assertEqual((x[2, 2])(m), m[2, 2])
        except:
            pass


    def test_getattribute_operation(self):
        '''
        Test rlambda getattriute operations
        :return:
        '''
        a = complex(1, 2)

        self.assertEqual((x.real ** 2)(a), a.real ** 2)
        self.assertEqual((x.imag ** 2)(a), a.imag ** 2)

    def test_variable_binding(self):
        '''
        Test rlambda _build method
        :return:
        '''
        f = (((x + y + z + 1) * 2) // 3) * z
        g = f._bind(x=1, y=2)
        self.assertEqual(g(z=3), f(x=1, y=2, z=3))

        g = f._bind(x=1, y=2, z=3)
        self.assertEqual(g(), f(x=1, y=2, z=3))


    def test_math(self):
        '''
        Test custom built-in math functions for rlambdas
        :return:
        '''
        # TODO
        pass

if __name__ == '__main__':
    unittest.main()