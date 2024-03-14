import math
from allRVs import *
from mathutil import avgVar
class Util:
    def __init__(self):
        self.rvs = {'binomial':(Binomial,(20,.5)),
                    'uniform':(Uniform,(0,1)),
                    'bernoulli':(Bernoulli,(.5,)),
                    'geometric':(Geometric,(0.1,)),
                    'hyper geometric':(HyperGeometric, (20,20,20)),
                    'poisson':(Poisson,(12,)),
                    'exponential':(Exponential, (1/10,)),
                    'normal':(Normal, (0,1)),
                    'erlang': (Erlang, (10, 1))
                    }
    """
    Markov's Inequality

    P[X >= a] <= E[X] / a

    if X is a non-negative RV and a > 0 (finite mean)
    """
    def markovs(self,X,a):
        if a > 0:
            return X.expectedValue() / a

    """
    Chebyshev's Inequality
    
    P[|X - E[X]| >= a] <= Var(x)/a^2 (finite mean, variance, a > 0)
    """
    def chebyshevs(self,X,a):
        if a > 0:
            return X.variance() / (a * a)

    def hoeffdingsBin(self,X,λ,upperTail=True):
        μ = X.expectedValue()
        if upperTail:
            return math.exp(-(λ**2)/(2*μ + λ))
        return math.exp(-(λ**2)/(3*μ))

    """
    Sn = Σ_i X_i, X_i independent RVs in [0,1]
    E[Sn] = μ
    then for all λ >= 0: 
        
        P[Sn >= μ + λ] <= e^(-λ^2/(2μ + λ))
        
        P[Sn <= μ - λ] <= e^(-λ^2/3μ)
    
    """
    def hoeffdings(self,Sn,λ,upperTail=True):
        μ = sum([X.expectedValue() for X in Sn])
        if upperTail:
            return math.exp(-(λ**2)/(2*μ + λ))
        return math.exp(-(λ**2)/(3*μ))

    def flipFairCoin(self,n):
        return self.flipCoin(n,.5)

    def flipCoin(self,n,p):
        return Binomial(n,p).genVar()

    """
    Generates k random instances of each distribution in self.rvs, and displays the averages
    
    """
    def simAll(self,k):
        res = []
        avgs = {}
        print('=' * 75)
        print(f'---  Statistics for k = {k} iterations ---')
        for rv_name in self.rvs:
            rv, conditions = self.rvs[rv_name]
            X = rv(*conditions)
            r = X.simulate(k,aggregate=True)
            res.append(r)
            if r:
                # print(r,k,sep='--><--')
                avgs[rv_name] = sum(r) / k
            else:
                avgs.append(0)
        print('=' * 75)

    """
    Returns a random instance of an RV given the name of the distribution and its necessary parameters
    """
    def generateRV(self,rv_name,*conditions,display=True):
        rv = self.rvs[rv_name][0]
        X = rv(*conditions)
        y = X.genVar()
        if display:
            print(f'{X}: {y}')
        return y

    def guess(self,data=None,k=None,target=None,verbose=False):
        if data:
            avg,var = avgVar(data)
            m = {}
            for rv_name in self.rvs:
                rv, conditions = self.rvs[rv_name]
                X = rv(*conditions)
                m[str(X)] = X.expectedValue(),X.variance()
            best = float('inf')
            name = None
            r = []
            for x in m:
                diff = abs(m[x][0] - avg) + abs(m[x][1] - var)
                r.append((x,diff))
                if diff < best:
                    best = diff
                    name = x
            if verbose:
                print(r)
            print(f'Best fit: {name}')
            return name
        if k != None and target:
            data = [target.genVar() for _ in range(k)]
            return self.guess(data)
        return -1

    def iterate(self, f, vars=None, print=False):
        if not vars:
            vars = self.rvs
        if print:
            r = []
        for rv_name in vars:
            rv, conditions = vars[rv_name]
            X = rv(*conditions)
            if print:
                r.append(f(X))
            else:
                f(X)
        if print:
            print(f'Outcomes: {r}')

    def compareAll2ndMoments(self, rv_names):
        m = {}
        for s in rv_names:
            m[s] = self.rvs[s]
        self.iterate(RandomVariable.confirm2ndMoment, vars=m)

u = Util()
if __name__ == '__main__':
    # for rv_name in u.rvs:
    #     print(rv_name,u.guess(None,1000,u.rvs[rv_name][0](*(u.rvs[rv_name][1]))))
    # u.compareAll2ndMoments(['uniform'])
    #'erlang','binomial','bernoulli','exponential'
    u.simAll(k=10000)
    # X = Binomial(100,.5)
    # print(u.chebyshevs(X,25))