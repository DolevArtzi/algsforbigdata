import math

from RandomVariable import RandomVariable
import random

"""
Uniform RV

uniformly random value in [a,b]
"""
class Uniform(RandomVariable):
    def __init__(self,a,b):
        super().__init__()
        self.a = a
        self.b = b
        self.min = a
        self.max = b
        self.discrete = False
        self.params.append(a)
        self.params.append(b)
        self.name = 'uniform'

    def pdf(self,x):
        return 1/(self.b-self.a) if self.a <= x <= self.b else 0

    def cdf(self,x):
        if self.a <= x <= self.b:
            return (x-self.a)/(self.b-self.a)
        return 0

    def expectedValue(self):
        return (self.b + self.a) / 2

    def _expectedValue(self,*params):
        a = params[0]
        b = params[1]
        return (a+b)/2

    def _valid(*params):
        a = params[0]
        b = params[1]
        return a < b

    def genVar(self):
        return random.uniform(self.a,self.b)

    def variance(self):
        return ((self.b - self.a) ** 2)/12

    def mgf(self):
        return lambda t: (math.e ** (t * self.b) - (math.e * (t * self.a))) / (t * (self.b - self.a)) if t else 1
    def moment(self,k):
        c = math.pow(self.b,k)
        s = c
        for i in range(k-1):
            c *= self.a/self.b
            s+=c
        return s/(k+1)

    def _getMomentRelatedFunction(self):
        return self.mgf()


if __name__ == '__main__':
    U = Uniform(0,5)
