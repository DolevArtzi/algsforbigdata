from RandomVariable import RandomVariable
import math

class Poisson(RandomVariable):
    def __init__(self,λ):
        super().__init__()
        self.min = 0
        self.max = float('inf')
        self.λ = λ
        self.params.append(λ)
        self.name = 'poisson'
        self.eNegλ = math.exp(-λ)

    def pdf(self,i):
        if i >= 0:
            return math.pow(math.e,-self.λ) * math.pow(self.λ,i) / (math.factorial(i))

    def cdf(self,k):
        pdf0 = self.pdf(0)
        if k == 0:
            return pdf0
        tot = pdf0
        for i in range(1,k+1):
            pdf0 *= self.λ / i
            tot += pdf0
        return min(1,tot)

    def expectedValue(self):
        return self.λ

    def _expectedValue(self,*params):
        λ = params[0]
        return λ
    @staticmethod
    def _valid(*params):
        λ = params[0]
        return λ > 0

    def variance(self):
        return self.λ

    def f(self,i):
        return 1 / (i + 1)
    def genVar(self):
        return self.inverseTransform(self.λ,self.eNegλ,self.f)

if __name__ == '__main__':
    X = Poisson(50)
    X.simRounds(30,500)
