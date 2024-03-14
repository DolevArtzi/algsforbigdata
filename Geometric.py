from RandomVariable import RandomVariable
import math

from Uniform import Uniform

"""
Geometric RV

Number of independent trials until success with success probability p per round
"""
class Geometric(RandomVariable):
    def __init__(self,p):
        super().__init__()
        self.min = 1
        self.max = float('inf')
        self.p = p
        self.params.append(p)
        self.name = 'geometric'

    def pdf(self,i):
        if i > 0:
            return self.p * math.pow((1-self.p),i-1)
        return 0

    def cdf(self,x):
        if x >= 1:
            return 1 - math.pow((1 - self.p),x)
        return 0

    def expectedValue(self):
        return 1/self.p

    def _expectedValue(self,*params):
        p = params[0]
        return 1/p

    @staticmethod
    def _valid(*params):
        p = params[0]
        return 0 <= p <= 1


    def variance(self):
        return (1 - self.p) / (self.p ** 2)

    """
    Inverse transform method for geometric RV: find j such that 
    F_x(j-1) <= U(0,1) < F_x(j). Note F_x(j) = 1 - q^j
    1 - q^(j-1) <= U < 1 - q^j
    q^j - 1 < - U <= q^(j-1) - 1
    q^j < 1 - U <= q^(j-1)
    
    X = min{j : q^j < 1 - U} <==> min{j : j log q < log(1 - U)} <==> min{j : j > log(U)/log(q)} (since U ~ 1 - U)
    
    So, X can be expressed as Int(log[U] / log[q]) + 1
    """
    def genVar(self):
        U = Uniform(0,1).genVar()
        r = int(math.log(U)/math.log(1 - self.p)) + 1
        return r