from RandomVariable import RandomVariable
import math
from Uniform import Uniform
class Exponential(RandomVariable):

    def __init__(self,λ):
        super().__init__()
        self.min = 0
        self.max = float('inf')
        self.λ = λ
        self.params.append(λ)
        self.name = 'exponential'
        self.eNegλ = math.exp(-λ)
        self.setStrictLower(False)
        self.discrete = False

    def pdf(self,a):
        if a <= 0:
            return 0
        return self.λ * math.pow(self.eNegλ,a)

    def cdf(self, k):
        if k > 0:
            return 1 - math.pow(self.eNegλ,k)
        return 0

    def expectedValue(self):
        return 1/self.λ

    def _expectedValue(self,*params):
        λ = params[0]
        return 1/λ
    @staticmethod
    def _valid(*params):
        λ = params[0]
        return λ > 0

    def variance(self):
        return 1/(self.λ ** 2)

    def laplace(self):
        return lambda s: self.λ/(s+self.λ)

    """
    u ~ U(0,1), x = Fx^-1(u)
    Fx(x) = 1 - exp(-λx)
    u = 1-exp(-λx)
    exp(-λx) = 1 - u
    -λx = ln(1-u), but 1 - u ~ U(0,1) too
    x = -1/λ ln(U(0,1)) [Ross pg. 68]
    """
    def genVar(self):
        return -(1/self.λ) * math.log(Uniform(0,1).genVar())
    def _getMomentRelatedFunction(self):
        return self.laplace()

if __name__ == '__main__':
    X = Exponential(0.1)
    samples = [X.genVar() for _ in range(1000)]
    print(samples)
    # print(X.moment(10),math.factorial(10)/(3**10))


