

from src.rlambda import RLambda
from src.astwrappers import Variable


'''
Variables x, y, z, w and its uppercase versions X,Y,Z,W are
the basic construction blocks to build rlambda objects
'''

w = RLambda(('w',), Variable('w'))
x = RLambda(('x',), Variable('x'))
y = RLambda(('y',), Variable('y'))
z = RLambda(('z',), Variable('z'))


# This variables are not case sensitive
W = w
X = x
Y = y
Z = z