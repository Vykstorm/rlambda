

from src.rlambda import RLambda
from src.astwrappers import Variable

x = RLambda(('x',), Variable('x'))
y = RLambda(('y',), Variable('y'))
w = RLambda(('w',), Variable('w'))
z = RLambda(('z',), Variable('z'))

X = x
Y = y
Z = z
W = w
