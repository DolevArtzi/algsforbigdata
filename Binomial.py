from RandomVariable import RandomVariable
from Uniform import Uniform
import math

"""
Binomial RV

Number of successes in n independent trails with success probability p in each trial
"""
class Binomial(RandomVariable):
    def __init__(self, n, p,symmetric = True):
        super().__init__()
        self.min = 0
        self.max = n
        self.n = n
        self.p = p
        self.qn = math.pow(1 - p, n)
        self.params.append(n)
        self.params.append(p)
        self.name = 'binomial'
        if p == .5:
            self.setSymmetric(True)

    def mgf(self):
        return lambda t: (1-self.p + self.p * math.e ** t) ** self.n

    def pdf(self, k):
        if k < 0 or k > self.n or k % 1:
            return 0
        return math.comb(self.n, k) * math.pow(self.p, k) * math.pow(1 - self.p, self.n - k)

    def expectedValue(self):
        return self.n * self.p
    def _expectedValue(self,*params):
        print(params)
        print(self)
        n = params[0]
        p = params[1]
        return n*p
    @staticmethod
    def _valid(*params):
        n = params[0]
        p = params[1]
        return 0 <= p <= 1 and 0 <= n == int(n)

    def variance(self):
        return self.n * self.p * (1 - self.p)

    """
    Generates a Bin(n,p) random variable using the Inverse Transform Method 
    (PnC book pg. 231, "Simulation" [Ross] pg. 57).

    Exploits the following recursive property for pdfs of binomials:

        p_x(0) = (1-p)^n

        p_x(i+1) = p/[1-p] * [n-i]/[i+1] * p_x(i)
        
        * Currently left as an artifact, not meant for use. Use genVar instead.
    """
    def _binomialInverseTransform(self):
        C = self.p / (1 - self.p)
        U = Uniform(0, 1).genVar()
        pr = self.qn
        F = pr
        i = 0
        while U >= F:
            pr *= (C * (self.n - i) / (i + 1))
            F += pr
            if i == self.n:
                break
            i += 1
        return i

    def _f(self,i):
        return (self.n - i) / (i+1)
    def genVar(self):
        return self.inverseTransform(self.p / (1 - self.p), self.qn , self._f)

    """Uses same recursive trick as above"""
    def cdf(self, k):
        if k < 0:
            return 0
        if k > self.n:
            return 1
        C = self.p / (1 - self.p)
        pr = self.qn
        F = pr
        i = 0
        while i < k:
            pr *= (C * (self.n - i) / (i + 1))
            F += pr
            i += 1
        return min(F, 1)

    def _getMomentRelatedFunction(self):
        return self.mgf()

if __name__ == '__main__':
    X = Binomial(20,.33)
    Y = Binomial(21,.33)
    print(X.pdf(13)*20,Y.pdf(13) * 21)
    # print(x.cdf(15))
