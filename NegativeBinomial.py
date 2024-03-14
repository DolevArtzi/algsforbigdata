import math

class NegativeBinomial:
    def __init__(self, k, p):
        super().__init__()
        self.min = k
        self.max = float('inf')
        self.k = k
        self.p = p
        # self.params.append(k)
        # self.params.append(p)
        self.name = 'negative binomial'

    def pdf(self,a):
        if a < self.k:
            return 0
        return math.comb(a-1,self.k-1) * math.pow(self.p,self.k) * math.pow(1 - self.p,a - self.k)

    # def cdf(self,a):
    def expectedValue(self):
        return self.k/self.p

    def variance(self):
        return self.k * (1-self.p) / (self.p ** 2)

if __name__ == '__main__':
    X = NegativeBinomial(10,.5)
    print(X.pdf(15))