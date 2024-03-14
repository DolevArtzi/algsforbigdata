from RandomVariable import RandomVariable
import math
from scipy.stats import norm
from Exponential import Exponential
from Bernoulli import Bernoulli


class Normal(RandomVariable):
    def __init__(self,μ,var):
        super().__init__()
        self.min = -float('inf')
        self.max = float('inf')
        self.μ = μ
        self.var = var
        self.σ = math.sqrt(var)
        self.params.append(μ)
        self.params.append(var)
        self.name = 'normal'
        self.discrete = False
        self.setStrictLower(False)

    def pdf(self,a):
        return (1 / (math.sqrt(2 * math.pi) * self.σ)) * math.exp(-.5 * (((a - self.μ) / self.σ) ** 2))

    """ CDF of standard normal """
    def Phi(self, k):
        return norm.cdf(k)

    """
    X ~ N(μ,σ^2)
    Y := (X - μ) / σ ~ N(0,1) 
    Fx(x) = Phi(x - μ / σ)
    """
    def cdf(self,k):
        print(self.μ,self.σ)
        return norm.cdf((k-self.μ)/self.σ)

    def expectedValue(self):
        return self.μ

    def _expectedValue(self,*params):
        μ = params[0]
        return μ
    @staticmethod
    def _valid(*params):
        σ = params[1]
        return σ > 0

    def variance(self):
        return self.var

    """
    Proof of method used for generating standard normal RVs [Ross 75-76] or [PnC 235-236]
    X ~ N(0,1), 
    |fx(x)| = 2/sqrt(2π) * exp(-x^2/2) (symmetric about 0)
    
    We'll use the accept-reject method with our subject being Exp(1)
    Y ~ Exp(1)
    fy(y) = g(y) = exp(-x), 0<x<inf
    
    max_x(f(x)/g(x)) = c = sqrt(2e/π) 
    """
    def genVar(self):
        # note the .5 in c is really to multiply the pdf of N(0,1) by 2 to make it |N(0,1)|
        x = self.acceptRejectSim(Exponential(1),Normal(0,1),.5 * math.sqrt(2 * math.e / math.pi))
        if Bernoulli(.5).genVar():
            y = x
        else:
            y = -x
        return self.μ + y * self.σ

if __name__ == '__main__':
    N = Normal(0,1)
    samples = [N.genVar() for _ in range(50000)]
    print(samples)
