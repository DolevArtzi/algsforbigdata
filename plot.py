import math
from allRVs import *
from util import Util
from matplotlib import pyplot as plt
from collections import Counter
from mathutil import avgVar
import numpy as np
from mpl_toolkits import mplot3d
import time
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

util = Util()
class Plot:
    def __init__(self):
        self.ax = None
        self.rng = np.random.default_rng()
        self.df = None
        self.process_csv('data.csv')
        self.classifier = None
        self.param = 5
        self.avgs = None
        self.set_classifier('knn',self.param)

    """ set_classifier
    Sets the classifier for plot according to the name and param provided

    Parameters:
        - name : 'knn' | 'r_nn': k nearest neighbors, or radius neighbors
        - param : parameters to pass to the classifier ctor: defaults to 3 neighbors, but is interpreted as the radius if name == 'r_nn'
    """
    def set_classifier(self,name='knn',param=3):
        if name == 'knn':
            self.classifier = KNeighborsClassifier(n_neighbors=param)
        elif name == 'r_nn':
            self.classifier = RadiusNeighborsClassifier(radius=param)


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

    def _legend(self):
        plt.legend()

    def _showPlt(self,legend=False):
        if legend:
            plt.legend(fontsize='medium')
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
    - [scatter]: boolean: if true, scatter plots instead of linear plots the data
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
    def plotGeneric(self,data=None,fInfo=None,scatter=False,label=None,output=False,chart=None,wait=False,addStats=False,onlyMean=False,scatterArgs=None):
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

        def label_maybe_and_plot():
            if label:
                if scatterArgs:
                    plt.scatter(xs,ys,s=scatterArgs['s'],c=scatterArgs['color'],label=label)
                elif scatter:
                    plt.scatter(xs,ys,label=label)
                elif plotIt:
                    plt.plot(xs, ys,label=label) #actually plot the data
            else:
                if scatterArgs:
                    plt.scatter(xs,ys,s=scatterArgs['s'],c=scatterArgs['color'])
                elif scatter:
                    plt.scatter(xs,ys)
                elif plotIt:
                    plt.plot(xs, ys) #actually plot the data
        label_maybe_and_plot()
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

    """ benchmark_np
        benchmark a numpy function on vectors or matrices

        args:
            - f: array -> 'a or mtx/tensor -> 'a, the numpy function to benchmark
            - data: list['a] | None: data to pass directly to f if not None
            - type: 'vec', 'sqmtx', 'mnmtx', 'tensor'
            - gen_info: map | None: used to describe the inputs to generate for f if no data provided directly
                      
            - default: depends on type, see plot_docs.md (https://github.com/DolevArtzi/algsforbigdata/blob/main/plot_docs.md) for full documentation
    """
    def benchmark_np(self,f,data=None,type_='sqmtx',gen_info=None,theory_info=None):
        if data:
            times = self._benchmark(f,data)
        # elif not gen_info:
        gen_info = self._get_default_np_gen_info(type_,**gen_info)
        data = self._generate_data(type_,gen_info)
        sizes = [max(d.shape) for d in data]
        times = self._benchmark(f,data)
        dim_str = 'Largest Dimension' if type_ != 'vec' else 'Dimension'
        chart = {
                    'xlabel': dim_str + ' ' + f"of {type_ if type_ in ['vec','tensor'] else 'matrix'}",
                    'ylabel': f'Time',
                    'lobf':1
                }

        chart['title'] = f'Size vs. Time for {f.__name__} for ' + (type_ if type_ in ['vec','tensor'] else 'matrix')
        chart['title'] += ' ' + ('[np.random]' if 'rand' not in gen_info else f'[{gen_info["rand"]}]')
        if not theory_info:
            self.plotGeneric(data=(sizes,times),chart=chart,wait=0,label='time')
            print('[Sizes, dimension of data]','first two:', f' {sizes[0]},{sizes[1]}, ...', 'last one:',sizes[-1])
            print('[Times, sec.             ]','first two:', f' {times[0]},{times[1]}, ...', 'last one:',times[-1])
        else:
            f_theory = theory_info['f']
            f_theory_label = theory_info['label']
            self.plotGeneric(data=(sizes,[t for t in times]),chart=chart,wait=1,label='time')
            self._legend()
            theory_times = [f_theory(s) for s in sizes]
            diffs = [theory_times[i]/times[i] for i in range(len(times))]
            diff = diffs[-1]
            normed_theory_times = [t/diff for t in theory_times]
            self.plotGeneric(data=(sizes,normed_theory_times),label=f_theory_label,wait=1)
            self._showPlt(legend=1)
            a = np.array(normed_theory_times)
            b = np.array(times)
            cos_sim = (a @ b.T) / (np.linalg.norm(a)* np.linalg.norm (b))
            print('Results:')
            print(f"Normalized Error: [Measured vs. Theoretical]  {100 * (1 - cos_sim)/2}%")
            polyfit = np.polyfit([t for t in times],sizes,theory_info['prediction'])
            dims = len(polyfit)-1
            fit_str = ' + '.join([f'{polyfit[i]:.2f} x^{dims-i}' for i in range(dims)])
            print('Best Fit Curve: y = ',fit_str + ' + ' + str(polyfit[-1])) 


    """ _benchmark
    Times f(data)

    Parameters:
        - f : function
        - data : 'a

    Returns:
        - the time for f(data)'s execution, in secs
    """
    def _benchmark(self,f,data):
        times = []
        for d in tqdm(data):
            t = time.time()
            f(d)
            total_t = time.time() - t
            times.append(total_t)
        return times
        
    """ _update_curr
    Updates the current size/shape for generation.
    
    Parameters:
        - curr : integer | tuple(integer): the current shape, which is an integer if type_ == 'vec', otherwise its a tuple
        - op : 'mult' | 'add': the type of operation we do to increase our size
        - delta : number: how much to add/multiply our current size by, depending on op
        - type_ : 'vec' | 'mnmtx' | 'sqmtx' | 'tensor': the type of data we're generating

    Returns:
        - the new shape
    """
    def _update_curr(self,curr,op,delta,type_='vec'):
        if type_ == 'vec':
            if op == 'add':
                return curr + delta
            return curr * delta
        new_curr = []
        for x in curr:
            new_curr.append(self._update_curr(x,op,delta))
        return tuple(new_curr)

    """ _gen_rand_array
    Generates a random numpy array of length N, according to the randomness specified by rand
    
    Parameters:
        - rand : str: either 'base' or the name of a probability distribution, e.g. 'poisson'
        - N : integer: a number specifying the length of the array to generate
        - params: list['a] | None: an optional list of parameters to augment the random variable chosen
    
    Returns: a np array of length N, with N independent random variables from the given distribution
    """
    def _gen_rand_array(self,rand,N,params=None):
        if rand == 'base':
            return self.rng.random((N,))
        if params:
            return np.array([util.generateRV(rand,params,display=0) for _ in range(N)])
        return np.array([util.generateRV(rand,display=0) for _ in range(N)])


    """ _get_rand_tensor
    Generates a random numpy tensor with dimensions = curr

    Parameters:
        - rand : str: either 'base' or the name of a probability distribution, e.g. 'poisson'
        - curr : tuple: the shape of the array to generate
        - params: list['a] | None: an optional list of parameters to augment the random variable chosen

    Returns: an np tensor of shape curr, with each element an independent random variable from the given distribution
    """
    def _gen_rand_tensor(self,rand,curr,params=None):
        if rand == 'base':
            return self.rng.random(curr)
        tensor = np.zeros(curr)
        for i,_ in np.ndenumerate(tensor):
            tensor[i] = util.generateRV(rand,params,display=0) if params else util.generateRV(rand,display=0)
        return tensor

    """ _generate_data
    Generates a test np array/matrix/tensor according to the parameters in gen_info

    Parameters:
        - type_ : the type of data to generate
        - gen_info : dict
            - the dictionary containing the information needed to generate our test data
            - Required Keys:
                - see plot_docs.md
    
    Returns: the generated numpy data
    """
    def _generate_data(self,type_,gen_info):
        gen_info = self._get_default_np_gen_info(type_,**gen_info) # I think
        range_ = gen_info['range']
        base = gen_info['base']
        op = gen_info['op']
        delta = gen_info['delta']
        rand = gen_info['rand']

        rand_params = []
        if rand != 'base':
            rand_params = gen_info['rand_params'] if 'rand_params' in gen_info else []
        data = []
        curr = base
        # for curr in range base, base *= delta
        # so the its 1 + log_delta(base * (delta ** range[-1]))-log_delta(range[0])
        if type_ == 'vec':
            while curr <= (range_[-1] if op == 'add' else delta ** range_[-1]):
                data.append(self._gen_rand_array(rand,curr,))
                curr = self._update_curr(curr,op,delta,type_)
        else:
            while curr[-1] <= (range_[-1] if op == 'add' else delta ** range_[-1]):
                if rand_params:
                    data.append(self._gen_rand_tensor(rand,curr,*rand_params))
                else:
                    data.append(self._gen_rand_tensor(rand,curr))
                curr = self._update_curr(curr,op,delta,type_)
        return data

    """ _get_default_np_gen_info

    Parameters:
        - type_ : 'vec' | 'mnmtx' | 'sqmtx' | 'tensor'
        - kw_override : dict: a dictionary of values to add/override to gen_info
    
    Returns: the default gen_info dict for the given type_, after possibly augmenting with k:v's in kw_override
    - see plot_docs.md for full documentation
    """
    def _get_default_np_gen_info(self,type_,**kw_override):
        gen_info = {'range':[2,11],'op':'mult','delta':2,'rand':'base'}            
    
        for k in kw_override: # to capture partial gen_infos being passed in, which affect other things
            gen_info[k] = kw_override[k]
        range_ = gen_info['range']
        b = range_[0]
        if type_ == 'vec':
            gen_info['type'] = 'vec'
            gen_info['dims'] = 'N'
            base = b
        elif type_ == 'tensor':
            gen_info['type'] = 'tensor'
            gen_info['dims'] = 3 if 'dims' not in gen_info else gen_info['dims']
            base = tuple([b] * gen_info['dims'])
        gen_info['type'] = 'mtx'
        if type_ == 'sqmtx':
            gen_info['dims'] = 'NN'
            base = (b,b)
        else:
            gen_info['dims'] = 'MN'
        for k in kw_override:
            gen_info[k] = kw_override[k]

        if type_ == 'mnmtx':
            base = gen_info['mn']
            del gen_info['mn']
        gen_info['base'] = base

        return gen_info
    
    def read_file(self,file='data.csv',preview=0):
        #https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who/data
        df = pd.read_csv(file)
        if preview:
            print(df.head(5))
        return df
    
    def get_relevant_df_col(self,df,ind_name='GDP',dep_name='Life expectancy '):
        df_rel = df[[dep_name, ind_name]]
        data = df_rel.to_dict()
        l = len(list(data[ind_name].values()))
        for k in data:
            data[k] = list(data[k].values())
        final_data = [(data[ind_name][i],data[dep_name][i]) for i in range(l)]
        data = final_data
        data = [(x,y) for x,y in data if not math.isnan(x) and not math.isnan(y)]
        indeps = [x[0] for x in data]
        deps = [x[1] for x in data]
        min_indeps = min(indeps)
        max_indeps = max(indeps)
        min_deps = min(deps)
        max_deps = max(deps)
        avg_indeps = sum(indeps)/len(indeps)
        avg_deps = sum(deps)/len(deps)
        self.avgs = [avg_indeps,avg_deps]
        print(f'Basic Statistics\n- [{ind_name}] min: {min_indeps}; max: {max_indeps}; avg: {avg_indeps}')
        print(f'- [{"LE" if dep_name == "Life expectancy " else dep_name}] min: {min_deps}; max: {max_deps}; avg: {avg_deps}')
        return data

    ''' _extract
    Given a list of indexes into a list, extracts the relevant elements according to their indices
    and the arrs[i]'s are disjoint. 
    
    Parameters:
        - idxs : list[integers]: a list of indices into the lists in arrs
        - arr : list['a]: a list representing one dimension of our data to pull from
    
    Returns: (arr[i] for i in idxs)
    '''
    def _extract(self,idxs,arr):
        out = []
        for i in idxs:
                out.append(arr[i])
        return tuple(out)

    ''' extract
    Given a list of indexes, and a 'list of lists' (*arrs) where each |arrs[i]| is equal, extracts 
    [[a[i] for a in *arrs] for i in idxs]
    and the arrs[i]'s are disjoint. 
    
    Parameters:
        - idxs : list[integers]: a list of indices into the lists in arrs
        - data : n x d 

    '''
    def extract(self,idxs,data):
        out = []
        for i in idxs:
            out.append(data[i])
        return out
        # for _,a in enumerate(*arrs):
        #         out.append(self._extract(idxs,a))
        # return out

    ''' _extract_category
    Given a map of our predictions and a category, extracts all datapoints that belong to that category 
    
    Parameters:
        - predictions : list[category]: predictions[i] is the numerical category that our i^th data point belongs to
        - cat_num : integer: an integer representing the category to extract
    
    Returns: the datapoints in the relevant category, by index
    '''
    def _extract_category(self,predictions,cat_num):
        return [i for i,x in enumerate(predictions) if x == cat_num]

    """ extract_categories
    Assuming you already called plot.process_csv, given two columns ind_name and dep_name, and a list of predictions, returns a 3D list defined below.

    Parameters:
        - predictions : list[category]: assuming data is of the form [(x,y) for (x,y) in zip(ind_col,dep_col)],
                                        predictions[i] is the predicted category for data[i]
        - ind_name : str: name of column that is the independent variable in our dataframe
        - dep_name : str: name of column that is the dependent variable

    Returns: a list, out, defined as follows:
        - out[i] is the list of datapoints belonging to the ith category
    """
    def extract_categories(self,predictions,ind_name='GDP',dep_name = 'Life expectancy '):
        data = self.get_relevant_df_col(self.df,ind_name=ind_name,dep_name=dep_name)
        out = []
        num_cats = len(list(set(predictions)))
        for cat_num in range(1,1+num_cats):
                out.append(self._extract_category(predictions,cat_num))
        for i in range(len(out)):
            # data is n x 2
            out[i] = self.extract(out[i],data)
        return out

    def process_csv(self,file='data.csv'):
        self.df = self.read_file(file)

    """ clustering
    Deterministically clusters data according to the class information in class_info
        - data: includes as many lists of length #(data points) as there are dimensions (usually 2)
            - independent variable data (x's) first, then the dependent variable (y's)

        - cat_info : dict[label : str |-> member : func] each key in class_info maps a label (str) to a function:
            - member : 'a -> bool: determines membership in the class
        - x_name/y_name : str: the names of the x and y columns in data
        - note: assumes the membership functions partition the data space
    """
    def clustering(self,x_name,y_name,cat_info,plot=1):
        data = self.get_relevant_df_col(self.df,x_name,y_name)
        cats = []
        labels = list(cat_info.keys())
        for label in labels:
            member = cat_info[label]
            cats.append([t for t in data if member(t)])

        n = self.classifier
        y = []
        for i,cat in enumerate(cats):
            for _ in cat:
                y.append(i+1)
        y = np.array(y)
        X = np.array(data)
        classifier = n.fit(X,y)
        predictions = classifier.predict(X)
        extracted_cats =  self.extract_categories(predictions,x_name,y_name)
        if not plot:
            return extracted_cats
        colors = ['g','y','b','r','teal','black'] #add more colors
        for i,cat in enumerate(extracted_cats):
            xs = [d[0] for d in cat]
            ys = [d[1] for d in cat]
            self.plotGeneric(data=[xs,ys],
                            #  label=labels[i],
                             scatterArgs={'s':.75,'color':colors[i % len(colors)]},
                             wait=1)
            # self._legend()
        chart={'title':f'{"life expectancy (country)" if y_name == "Life expectancy" else y_name} vs. {x_name}, 2000-2015 [KNN]', #fix knn hardcoding
                                    'xlabel':'GDP ($)',
                                    'ylabel':'Life Expectancy (years)'}
        plt.plot([1000,70000],[69.37066398390344,69.37066398390344],label='average LE',color='black')
        plt.plot([7494.210719388659,7494.210719388659],[55,84.5],label='average GDP',color='teal')
        self.set_chart(chart,show=1)

        # self.plotGeneric(data=cats[3],label='best',
        #                     scatterArgs={'s':1.5,'color':'g'},
        #                     wait=True)
        # p._legend()
        # p.plotGeneric(data=cats[1],label='bad gdp, good le',
        #                     scatterArgs={'s':1.5,'color':'y'},
        #                     wait=True)
        # p._legend()
        # p.plotGeneric(data=cats[2],label='good gdp, bad le',
        #                     scatterArgs={'s':1.5,'color':'b'},
        #                     wait=True)
        # p._legend()

        # p.plotGeneric(data=cats[0],label='worst',
        #                     scatterArgs={'s':0.2,'color':'r'},
        #                     wait=True,
        #                     )

    def set_chart(self,chart,show=0):
        plt.chart = chart
        if show:
            self._showPlt(legend=1)


if __name__ == '__main__':
    U = Exponential(0.1)
    