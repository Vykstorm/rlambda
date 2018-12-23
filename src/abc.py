
import math
from .rlambda import RLambdaIdentity as Identity


'''
Common rlambda object identities.
'''

w = Identity('w')
x = Identity('x')
y = Identity('y')
z = Identity('z')

W = Identity('W')
X = Identity('X')
Y = Identity('Y')
Z = Identity('Z')


a = Identity('a')
b = Identity('b')
c = Identity('c')
d = Identity('d')

A = Identity('A')
B = Identity('B')
C = Identity('C')
D = Identity('D')



'''
Special identities that will be formatted with greek characters using RLambdaFormatter()
'''

# Lowercase
alpha = a
beta = b
gamma = c
delta = d
epsilon = Identity('epsilon')
zeta = Identity('zeta')
eta = Identity('eta')
theta = Identity('theta')
iota = Identity('iota')
kappa = Identity('kappa')
lambda_ = Identity('lambda')
mu = Identity('mu')
nu = Identity('nu')
xi = Identity('xi')
omicron = Identity('omicron')
rho = Identity('rho')
varsigma = Identity('varsigma')
sigma = Identity('sigma')
upsilon = Identity('upsilon')
phi = Identity('phi')
chi = Identity('chi')
psi = Identity('psi')
omega = Identity('omega')

# Uppercase
Alpha = A
Beta = B
Gamma = C
Delta = D
Epsilon = Identity('Epsilon')
Zeta = Identity('Zeta')
Eta = Identity('Eta')
Theta = Identity('Theta')
Iota = Identity('Iota')
Kappa = Identity('Kappa')
Lambda = Identity('Lambda')
Mu = Identity('Mu')
Nu = Identity('Nu')
Xi = Identity('Xi')
Omicron = Identity('Omicron')
Rho = Identity('Rho')
Varsigma = Identity('Varsigma')
Sigma = Identity('Sigma')
Upsilon = Identity('Upsilon')
Phi = Identity('Phi')
Chi = Identity('Chi')
Psi = Identity('Psi')
Omega = Identity('Omega')



'''
Math constants that will be formatted nicely
'''

class MathConstant(float):
    def __new__(cls, value):
        return float.__new__(cls, value)


pi = Pi = PI = MathConstant(math.pi)
e = E = MathConstant(math.e)
tau = Tau = TAU = MathConstant(math.tau)
inf = Inf = INF = MathConstant(math.inf)
nan = Nan = NAN = MathConstant(math.nan)
