from RandomVariable import RandomVariable
import math

class HyperGeometric(RandomVariable):
    def __init__(self,N,M,n):
        if n > N + M:
            raise ValueError
        super().__init__()
        self.N = N
        self.M = M
        self.n = n
        self.min = 0
        self.max = n
        self.params.append(N)
        self.params.append(M)
        self.params.append(n)
        self.name = 'hyper geometric'
        self.Mcn = math.comb(M,n)
        self.denom = math.comb(N + M, n)

    def pdf(self,a):
        return math.comb(self.N,a) * math.comb(self.M,self.n - a) / self.denom

    def cdf(self, k):
        if k < 0:
            return 0
        if k > self.n:
            return 1
        pr = self.Mcn / self.denom
        F = pr
        i = 0
        while i < k:
            pr *= ((self.N - i) * (self.n - i) / (i + 1)) / (self.M - self.n + i + 1)
            F += pr
            i += 1
        return min(F, 1)

    def expectedValue(self):
        return self.N * self.n / (self.N + self.M)

    def _expectedValue(self,*params):
        N = params[0]
        M = params[1]
        n = params[2]
        return N * n / (N + M)
    @staticmethod
    def _valid(*params):
        N = params[0]
        M = params[1]
        n = params[2]
        return N >= 0 and M >= 0 and 0 <= n <= N
    def variance(self):
        return self.M * self.expectedValue() * (1 - (self.n - 1) / (self.N + self.M -1)) / (self.M + self.N)

    """
    _f(i) = P[X = i + 1] / P[X = i]
    """
    def _f(self,i):
        return ((self.N - i) * (self.n - i) / (i + 1)) / (self.M - self.n + i + 1)

    def genVar(self):
        #selects the method of inverse transform depending on the relative sizes of M and n to avoid div by 0
        if self.M < self.n:
            return self._slowInverseTransform()
        return self.inverseTransform(1, self.Mcn / self.denom, self._f)

if __name__ == '__main__':
    x = HyperGeometric(5,5,2)
    print(x.pdf(0))
