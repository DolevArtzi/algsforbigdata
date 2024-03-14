import math
from allRVs import *
from util import Util
from matplotlib import pyplot as plt
from collections import Counter
from mathutil import avgVar
import numpy as np
from mpl_toolkits import mplot3d


util = Util()
class Plot:
    def __init__(self):
        self.ax = None

    def _fillInPDFCDFTail(self,X,name,text=None,show=True):
        print(name)
        t = '='
        if name == 'cdf':
            t = '<='
        elif name == 'tail':
            t = '>'

        m = {
            'xlabel' : f'Value of {X}',
            'ylabel': f'P[X {t} x]',
            'title': f'{name} of {X}',
        }

        if text:
            m['text'] = text
        self.fillInChartInfo(m)
        if show:
            self._showPlt()

    def _showPlt(self):
        plt.show()
        plt.close()

    def plotPDF(self,X:RandomVariable,mx=None,mn=None,δ=0.05):
        self.plot({X.name:([X.params],mx if mx != None else 2 * X.expectedValue(),mn,δ)},'pdf',dontShow=True)
        self._fillInPDFCDFTail(X,'pmf' if X.isDiscrete() else 'pdf')

    def plotTail(self,X:RandomVariable,mx=None,mn=None,δ=0.05):
        self.plot({X.name:([X.params],mx if mx != None else 2 * X.expectedValue(),mn,δ)},'tail',dontShow=True)
        self._fillInPDFCDFTail(X,'tail')

    def plotCDF(self,X:RandomVariable,mx=None,mn=None,δ=0.05):
        self.plot({X.name:([X.params],mx if mx != None else 2 * X.expectedValue(),mn,δ)},'cdf',dontShow=True)
        self._fillInPDFCDFTail(X,'cdf')

    """
    m: a map of the form
        k: RV class name (eg 'binomial')
        v: (conditions, mx, δ)
        
        conditions: a list of parameters to be instantiated with k eg. [(20,.5), (20,.7)]
        mx = max value to plot for distribution k
        δ = step value for calculation

    f: the name of a function of type Real --> Real that belongs to all RVs (eg pdf or cdf)
    
    together: 
        true: all plotted on same graph
        false: different graphs for each instance
    """
    def plot(self,m,f,together=True,dontShow=False):
        legend = []
        m = {util.rvs[k][0]:m[k] for k in m}
        for rv in m:
            conditions, mx, mn, δ = m[rv]
            for c in conditions:
                X = rv(*c)
                self._plot(X,mx,mn,δ,f)
                if not together:
                    plt.legend([f'{X}'])
                    if not dontShow:
                        self._showPlt()
                else:
                    legend.append(f'{X}')
        if together:
            plt.legend(legend)
            if not dontShow:
                self._showPlt()

    def _plot(self,X,mx,mn,δ,f):
        x, y = [],[]
        m1 = X.getMin()
        MAX = 20000
        i = 0
        if mn != None:
            m1 = mn
        if not X.strictLower:
            if m1 == -float('inf'):
                μ = X.expectedValue()
                if μ > 0:
                    m1 = .4 * μ
                elif μ < 0:
                    m1 = 2.5 * μ
                else:
                    m1 = -math.sqrt(X.variance())
            m1 += δ
        if X.isDiscrete():
            δ = max(δ,1)
        while m1 <= mx and i <= MAX:
            x.append(m1)
            F = getattr(X,f)
            y.append(F(m1))
            m1+=δ
            i+=1
        plt.plot(x, y)

    """
    Fills in the information necessary to properly label a chart
    Assumptions: plt.plot has been called before calling this (in other words, there's data to be plotted)
    Params:
    - m: map of the form {s:object} where s : string and plt has an attribute plt.s
    
    Example Usage: 
    
        P = Plot()
        X = Binomial(20,.5)
        P.plotPDF(X)
        chart = {
            'xlabel' : f'Value of {X}',
            'ylabel': f'P[X = x]',
            'title': f'PMF of Binomial(20,.5)',
        }
        P.fillInChartInfo(chart,output=True)
    
    """
    def fillInChartInfo(self,m,show=False,threeD=None,wx=1,wy=1):
        x0,x1 = plt.xlim()
        y0,y1 = plt.ylim()
        x,y = .6*(x1-x0)*wx,.6*(y1-y0)*wy
        # print(x0,x1,x)
        # print(y0,y1,y)
        if threeD:
            # ax = plt.axes(projection='3d')
            z_label = m.pop('zlabel')
            self.ax.set_zlabel(z_label)
            # print(m,'hi')
        if 'lobf' in m:
            del m['lobf']
        for k in m:
            f = getattr(plt,k)
            if k == 'text':
                f(x,y,m[k])
            else:
                f(m[k])
        if show:
            self._showPlt()

    """
    Compares k samples of X to the scaled pdf of X graphically
    """
    def plotSamples(self,X:RandomVariable=None,k=10000,mx=None,mn=None,buckets=50,data=None):
        if not data:
            r = X.simulate(k)
            # r = [math.ceil(x) for x in r]
        elif data:
            r = data
        counts = Counter(r)
        x = []
        y = []
        for val in counts:
            x.append(val)
            if X.isDiscrete():
                y.append(counts[val]/k)
        if X.isDiscrete():
            plt.scatter(x,y)
        else:
            plt.hist(r, bins=buckets, density=True, alpha=0.6)
        if X.isDiscrete():
            self.plot({X.name:([X.params],mx if mx != None else 2 * X.expectedValue(),mn,1 if X.isDiscrete() else 0.05)},'pdf',dontShow=True)
        else:
            self.plot({X.name:([X.params],mx if mx != None else 2 * X.expectedValue(),mn,1 if X.isDiscrete() else 0.05)},'pdf',dontShow=True)
        m = {
            'xlabel' : f'Value of {X}',
            'ylabel': f'P[X = x]',
            'title': f'Sample of {X} for {k} iters vs. {"pmf" if X.isDiscrete() else "pdf"} of X',
            # 'text': 'hello'
        }
        self.fillInChartInfo(m)
        self._showPlt()

    """ plotGeneric
    
    Generic multimodal plotting function. 
    Either plot given data (set data param) or give a function and a range to iterate it over (set fInfo param)
    
    Params:
    - [data]: (xs:list[num], ys:list[num): numerical data to plot
    - [output]: boolean: if true, prints x,y data to the console
    - [chart]: map object containing information for labelling the plot. See fillInChartInfo
    - [wait]: boolean: if true, will not show the plot after creating it
    - [addStats]: boolean: if true, will add the sample mean and variance to the plot
    - [onlyMean]: boolean: if [chart] and [addStats], if [onlyMean], only returns avg, not (avg,variance)
    - [fInfo]: map containing information about what to plot if data == None. 
        Required Parameters:
            - f: a function with 0 or more arguments, the function of interest
                - normally, we plot f(x) for x in (mn,mx,δ)
                - if 'prefixArgs' is set to the list L = [arg1,arg2,...,argN], we plot f(arg1,arg2,...,argN,[x])
                - the x above is optional because f can also take no arguments or be dependent only on prefixArgs
            - Either 'mn' and 'mx' or 'iters'
                - mn and mx are set to the minimum and maximum of the range you want to consider
                - iters is set to the number of iterations to run f, if f is not dependent on range
        Optional Parameters:
            - [prefixArgs]: list of arguments to be passed to f before the other potential range argument
            - [δ]: step size when iterating over a range. Defaults to 1
            -[checkNone]: boolean: if true, treats f as type ?args --> Option[num]
                (e.g. we exclude x,f(x) if f(x) == None)
            - [mn/mx] or [iters], but not both. See above.
            - [lobf]: if true, will plot the line of best fit of the data
            - [plotIt]: currently unused, but if false will not plot the data 
                (was in now deleted combine.iterativeBinMaxPlot)
            
    ex:
        P = Plot()
        m = {
            'mn': 0,
            'mx': int(k*d),
            'f': self.walkRangeA,
            'prefixArgs': [k],
            'checkNone': True
        }

        chart = {
            'xlabel': f'x: Final Value of Random Walk with {k} Steps (S)',
            'ylabel': f'P[-x <= S <= x]',
            'title': f'Probability of Simple Bounded {k}-step Random Walk Ending in [-x,x]',
        }

        if d != 1:
            s = chart['title']
            t = s + f' up to S = {int(k*d)}'
            chart['title'] = t

        P.plotGeneric(fInfo=m, output=True, chart=chart)
    """
    def plotGeneric(self,data=None,fInfo=None,output=False,chart=None,wait=False,addStats=False,onlyMean=False):
        lobf = 0
        if not data and not fInfo:
            return
        plotIt = True #for preventing intermediate plotting
        if data:
            xs,ys = data
        else:
            f = fInfo['f']
            iters = self._getOptionalParam(fInfo,'iters')
            prefixArgs = self._getOptionalParam(fInfo,'prefixArgs')
            checkNone = self._getOptionalParam(fInfo,'checkNone')
            plotIt = self._getOptionalParam(fInfo,'plotIt',default=True)
            δ = self._getOptionalParam(fInfo,'δ',default=1)
            lobf = self._getOptionalParam(fInfo,'lobf')

            if iters:
                xs,ys = [],[]
                for i in range(1,iters+1):
                    res = self._compute(prefixArgs,f)
                    if checkNone:
                        if res != None:
                            xs.append(i)
                            ys.append(res)
                    else:
                        xs.append(i)
                        ys.append(res)
            else:
                mn,mx = self._getOptionalParam(fInfo,'mn'),self._getOptionalParam(fInfo,'mx')
                xs,ys = [],[]
                if δ == int(δ):
                    r = range(mn,mx+1,δ)
                else:
                    from numpy import linspace
                    r = linspace(mn,mx+1,int((mx+1-mn)/δ)+1)
                for i in r:
                    res = self._compute(prefixArgs,f,i)
                    if checkNone:
                        if res != None:
                            xs.append(i)
                            ys.append(res)
                    else:
                        xs.append(i)
                        ys.append(res)
        if output:
            print(xs)
            print(ys)

        if plotIt:
            plt.plot(xs, ys) #actually plot the data
        stats = None
        if chart:
            if addStats:
                avg,var = avgVar(ys)
                s = f'Mean = {avg}' + '\n' + (f'Sample Variance =  {var:.5f}')
                chart['text'] = s
                print(s)
                stats = avg,var
            elif lobf:
                print('here')
                args = np.polyfit(xs, ys, 2)
                # chart['text'] = f'y = {m}x + {b}'
                chart['text'] = str(args)
                print(args)
            self.fillInChartInfo(chart) #TODO: fix weights/centering of text on charts
        if not wait:
            self._showPlt()
        return stats if not onlyMean else stats[0]

    def _getOptionalParam(self,m,k,default=None):
        return m[k] if k in m else default

    def _compute(self,prefixArgs,f,x=None):
        if x != None:
            if prefixArgs:
                return f(*prefixArgs,x)
            return f(x)
        return f(*prefixArgs)

    def plot3dData(self,x,y,z,chart=None,show=True):
        if not self.ax:
            self.ax = plt.axes(projection='3d')
        self.ax.plot3D(x, y, z)
        if chart:
            self.fillInChartInfo(chart,show=show,threeD=True,wx=.9,wy=3.7)
            return
        if show:
            self._showPlt()


if __name__ == '__main__':
    # P = Plot()
    # # P.plot({'binomial':([(20,.3),(20,.5),(20,.7)],20,0,1)},'pdf')
    # # P.plotTail(Exponential(.10),mx=50,mn=0)
    # P.plotPDF(Normal(0,10),20,-20)
    # # P.plotPDF(Erlang(3,1/3),100,0)
    # # P.plotSamples(Poisson(10),10000,25,0)
    # # X = Normal(0,.01)
    # # P.plotPDF(Binomial(20,.5))
    # # print(X.pdf(0))
    # # P.plotPDF(Normal(0,.0001),mx=1,mn=-1)
    # chart = {
    #     'xlabel': f'iteration i',
    #     'ylabel': f'Max Across Foods',
    #     'zlabel': 'hello',
    #     'title': f'Max Value fors for Iterations',
    #
    # }
    #
    U = Exponential(0.1)
    print(U.moment(2))
    # P.plot3dData([1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],show=False)
    # P.plot3dData([10,20,30,40,50,60],[10,20,30,40,50,60],[1,2,3,4,5,6],show=True)