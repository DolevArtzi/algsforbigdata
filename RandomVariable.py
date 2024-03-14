import math
import random
from abc import ABC, abstractmethod
from sympy import Derivative, symbols

class RandomVariable(ABC):
    def __init__(self):
        self.strictLower = True
        self.symmetric = False
        self.params = []
        self.min = None
        self.max = None
        self.discrete = True
        self.name = None
        pass

    """
    t: boolean, true if the distribution's support is inclusive of self.min, false otherwise.
    """
    def setStrictLower(self,t):
        self.strictLower = t

    def setSymmetric(self,s):
        self.symmetric = s

    def getMin(self):
        return self.min

    def getMax(self):
        return self.max

    def isDiscrete(self):
        return self.discrete

    def _expectedValue(self,*params):
        pass

    def numParams(self):
        return len(self.params)

    @staticmethod
    @abstractmethod
    def _valid(*params):
        pass

    def valid(self,*params):
        return len(params) == self.numParams() and self._valid(*params)

    def laplace(self):
        pass

    def confirm2ndMoment(self):
        v = self.variance()
        v2 = self.moment(2) - self.moment(1) ** 2
        print(f'{self}: Variance: {v}; Variance via Moments: {v2:.5f}')

    def mgf(self):
        pass

    #@abstractmethod
    def _getMomentRelatedFunction(self):
        pass

    def moment(self,k,verbose=False):
        f = self._getMomentRelatedFunction()
        x = symbols('x')
        expr = f(x)
        curr = expr
        if verbose:
            print(curr)
        for i in range(k):
            if i == k-1:
                curr = Derivative(curr, x)
                return abs(curr.doit_numerically(0))
            else:
                curr = Derivative(curr,x).doit()
                if verbose:
                    print(curr)

    """
    Probability Density Function 
    
    returns P[X == a]
    """
    @abstractmethod
    def pdf(self,a):
        pass

    @abstractmethod
    def expectedValue(self):
        pass

    """
    Cumulative Distribution Function

    P[X <= k]
    """
    @abstractmethod
    def cdf(self, k):
        pass


    """
    Tail Probability

    P[X > k]
    """
    def tail(self, k):
        return 1 - self.cdf(k)

    @abstractmethod
    def variance(self):
        pass

    """
    Generates an instance of a random variable, given the distribution.
    
    Similar to Mathematica's RandomVariate[..] function.
    """
    @abstractmethod
    def genVar(self):
        pass

    """
    For distributions with the following property:
    
    P[X = i+1] = C f(i) * P[X = i]
    
    uses the inverse transform method to generate a random variate of the distribution
    
    C: see above
    f: see above
    pr: P[X = X.min]
    
    Time Complexity: O(n * cost(f)) (note cost(f) is constant for both binomial and hypergeometric)
    """
    def inverseTransform(self,C,pr,f):
        U = random.random()
        F = pr
        i = self.min
        while U >= F:
            pr *= (C * f(i))
            F += pr
            if i == self.max:
                break
            i += 1
        return i

    """
    Generates an instance of a random variable using the inverse transform method. 
    
    Time Complexity: O(n * cost(pdf)) (note cost(pdf) can be very large, e.g. binomial/hypergeometric)
    """
    def _slowInverseTransform(self):
        U = random.random()
        i = self.min
        pr = self.pdf(i)
        F = pr
        while U >= F:
            i += 1
            F += self.pdf(i)
            pr = pr+self.pdf(i)
            if i == self.max:
                break
        return i

    """
    Calculates the cdf naively
    """
    def _cdfSlow(self, k):
        if k < self.min:
            return 0
        if k > self.max:
            return 1
        return sum([self.pdf(i) for i in range(k + 1)])

    """
    Simulates k independent generations of the random variable
    
    - output: if true, will print each generated value  
    - aggregate: if true, will print the average value
    """
    def simulate(self,k,output=False,aggregate=True):
        r = []
        μ = self.expectedValue()
        for _ in range(k):
            r.append(self.genVar())
            if output:
                print(r[-1])
        if aggregate:
            s = f'Mean = {(sum(r)/k):.4f}' + '\n' + (f'Sample Variance =  {(sum([(xi - μ) ** 2 for xi in r]) / (k - 1)):.4f}' if k > 1 else '')
            print(f'{self}: Average = {(sum(r)/k):.5f}' + (f'  Sample Variance =  {(sum([(xi - μ) ** 2 for xi in r]) / (k - 1)):.5f}' if k > 1 else ''))
            return r#, s
            if output:
                print(f'Outcomes: {r}')
        return r

    """
    Simulates k independent generations of the random variable, "rounds" times. 
    
    - short: if false (and aggregate is true), will print the average value for each round
    - output: if true, will print each generated value  
    - aggregate: if true, will print the average value across all rounds
    """
    def simRounds(self,rounds,k,short=True,output=False,aggregate=True):
        r = []
        for _ in range(rounds):
            r.append(self.simulate(k,output,False))
        if aggregate:
            avgs = [sum(ri)/k for ri in r]
            if not short:
                for i,a in enumerate(avgs):
                    print(f'Average round {i+1} = {a}')
            print(f'Total Average = {sum(avgs)/rounds:.5f}')

    def _paramString(self):
        s = ''
        for p in self.params[:-1]:
            s += str(p) + ','
        return s + str(self.params[-1])

    """
    Accept-Reject method for generating CRVs
    
    [PnC pg. 234]
    """
    def acceptRejectSim(self,subject,target,c):
        Y = subject.genVar()
        while not (target.pdf(Y) / (c * subject.pdf(Y)) - random.random() > 0):
            Y = subject.genVar()
        return Y

    def __str__(self):
        return f'{"".join([x.capitalize() for x in self.name.split(" ")])}({self._paramString()})'

    def __hash__(self):
        return self.name