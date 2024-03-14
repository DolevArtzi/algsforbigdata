from RandomVariable import RandomVariable
import random

""""
Bernoulli random variable, coin flip with probability p of heads
"""
class Bernoulli(RandomVariable):
    def __init__(self,p):
        super().__init__()
        self.min = 0
        self.max = 1
        self.p = p
        self.params.append(p)
        self.name = 'bernoulli'

    def pdf(self,a):
        if a == 1:
            return self.p
        elif a == 0:
            return 1-self.p
        return 0

    def genVar(self):
        if self.p - random.random() > 0:
            return 1
        return 0

    def expectedValue(self):
        return self.p

    def cdf(self,k):
        if k < 0:
            return 0
        if k == 1:
            return 1
        return 1 - self.p

    def variance(self):
        return self.p * (1 - self.p)

    def _expectedValue(self,*params):
        p = params[0]
        return p

    @staticmethod
    def _valid(*params):
        print(params)
        p = params[0]
        return 0 <= p <= 1
    def moment(self,k):
        return self.p
    def _getMomentRelatedFunction(self):
        pass #not relevant for this distribution

if __name__ == '__main__':
    X = Bernoulli(.3)
    px = [1]
    print(Bernoulli._valid(1))