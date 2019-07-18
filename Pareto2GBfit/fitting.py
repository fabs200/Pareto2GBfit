import numpy as np
import scipy.optimize as opt
from scipy import linalg
from scipy.misc import derivative
from scipy.special import digamma, gammaln
from scipy.stats import norm
import matplotlib.pyplot as plt
import progressbar
from prettytable import PrettyTable
from .distributions import Pareto_pdf, IB1_pdf, GB1_pdf, GB_pdf, Pareto_icdf, IB1_icdf_ne, GB1_icdf_ne, GB_icdf_ne
from .testing import *
import warnings, sys, os

"""
-------------------------
Define functions
-------------------------
"""

def namestr(obj, namespace):
    """
    retrieve variable name of a list (can be used for example to retrieve variable and use it later)
    :param obj: e.g. Pareto_data which is an numpy.array
    :param namespace: globals()
    :return: ['Pareto_data']
    """
    return [name for name in namespace if namespace[name] is obj]

def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ source: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    NOTE: quantiles should be in [0, 1]
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

""" 
---------------------------------------------------
Goddness of fit measures
---------------------------------------------------
"""
class gof:
    """ Goodness of fit measures and descriptive statistics data vs fit """
    def __init__(self, x, x_hat, W, parms, b):
        """
        :param x: model with same shape as data
        :param x_hat: fitted data
        :param parms: np.array []
        :param b: location parameter, fixed
        """
        # if no weights or weighting=='expanded' (np.ones-array)
        if len(W) == np.sum(W):
            self.n = n = len(x)
            self.emp_mean = np.mean(x, dtype=np.float64)
            self.emp_var = np.var(x, dtype=np.float64)
        # if weighting=='multiply'
        else:
            self.n = n = np.sum(W)
            self.emp_mean = np.average(a=x, weights=W)
            self.emp_var = np.sqrt(np.cov(x, aweights=W))**2
        self.pred_mean = np.mean(x_hat, dtype=np.float64)
        self.pred_var = np.var(x_hat, dtype=np.float64)
        self.e = e = np.array(x) - np.array(x_hat)
        self.soe = soe = np.sum(e)
        # self.ssr = ssr = soe**2
        # self.sse = sse = np.sum(e**2)
        # self.sst = ssr + sse
        self.mse = 1/n * np.sum(e**2)
        self.rmse = np.sqrt(1/n * np.sum(e**2))
        self.mae = 1/n * np.sum(np.abs(e))
        self.mape = (100/n) * np.sum(np.abs(e/x))
        self.rrmse = np.sqrt(1/n * np.sum((e/x)**2))

        if len(parms) == 1:
            self.ll = ll = (-10000)*Pareto_ll(parms=parms, x=x, W=W, b=b) #normalization
            self.aic = -2*ll + 2
            self.bic = -2*ll + np.log(n)
        if len(parms) == 2:
            self.ll = ll = (-10000)*IB1_ll(parms=parms, x=x, b=b) #normalization
            self.aic = -2*ll + 2*2
            self.bic = -2*ll + np.log(n)*2
        if len(parms) == 3:
            self.ll = ll = (-100)*GB1_ll(parms=parms, x=x, b=b) #normalization
            self.aic = -2*ll + 2*3
            self.bic = -2*ll + np.log(n)*3
        if len(parms) == 4:
            self.ll = ll = (-100)*GB_ll(parms, x=x, b=b) #normalization
            self.aic = -2*ll + 2*4
            self.bic = -2*ll + np.log(n)*4

""" 
---------------------------------------------------
Neg. Log-Likelihoods
---------------------------------------------------
"""
def Pareto_ll(parms, x, W, b):
    """
    :param parms: np.array [p], optimized
    :param x: linspace, fixed
    :param W: weights, either np.ones() if no weights have been applied OR weighting='expand', or iweights Σw=1, fixed
    :param b: location parameter, fixed
    :return: neg. logliklihood of Pareto
    """
    p = parms[0]
    n = len(x)
    sum = np.sum(np.log(x)*W)
    ll = n*np.log(p) + p*n*np.log(b) - (p+1)*sum
    ll = -ll/10000
    return ll

def IB1_ll(parms, x, b):
    """
    :param parms: np.array [p, q] optimized
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :return: neg. log-likelihood of IB1
    """
    p = parms[0]
    q = parms[1]
    x = np.array(x)
    x = x[x>b]
    n = len(x)
    lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
    # ll = p*n*np.log(b) - n*np.log(beta(p,q)) + (q-1)*np.sum(np.log(1-b/x)) - (p+1)*np.sum(np.log(x)) # both work
    ll = p*n*np.log(b) - n*lnb + (q-1)*np.sum(np.log(1-b/x)) - (p+1)*np.sum(np.log(x))
    ll = -ll/10000
    return ll


# log-likelihood
# NOTE: optimize a,p,q = parms (first args here), fixed parameters: x, b
def GB1_ll(parms, x, b):
    """
    :param parms: np.array [a, p, q] optimized
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :return: neg. log-likelihood of GB1
    """
    a = parms[0]
    p = parms[1]
    q = parms[2]
    x = np.array(x)
    # x = x[x>b]
    n = len(x)
    lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
    ll = n*np.log(abs(a)) + (a*p-1)*np.sum(np.log(x)) + (q-1)*np.sum(np.log(1-(x/b)**a)) - n*a*p*np.log(b) - n*lnb
    ll = -ll/100
    return ll

def GB_ll(parms, x, b):
    """
    :param parms=np.array [a, c, p, q] optimized
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :return: neg. log-likelihood of GB1
    """
    a = parms[0]
    c = parms[1]
    p = parms[2]
    q = parms[3]
    n = len(x)
    x = np.array(x)
    x = x[x>b]
    sum1 = np.sum(np.log(x))
    sum2 = np.sum(np.log(1-(1-c)*(x/b)**a))
    sum3 = np.sum(np.log(1+c*((x/b)**a)))
    lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
    # lnb = np.log(beta(p,q))
    ll = n*(np.log(np.abs(a)) - a*p*np.log(b) - lnb) + (a*p-1)*sum1 + (q-1)*sum2 - (p+q)*sum3
    ll = -ll/100
    return ll

""" 
---------------------------------------------------
Fitting Functions
---------------------------------------------------
"""
def Paretofit(x, b, x0, weights=np.array([1]), weighting='expand', bootstraps=None, method='SLSQP', omit_missings=True,
              verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, suppress_warnings=True,
              return_parameters=False, return_gofs=False, #save_plot=False,
              plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'orange'},
              basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                                   'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
              slsqp_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                             'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit Pareto distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [p] or simply (p)
    :param weights: weight, default: numpy.ones array of same shape as x
    :param weighting: how to apply weights, default: 'expand' x expandes w times, else: 'multiply' x*w
    :param bootstraps: amount of bootstraps, default is size k (x>b)
    :param method: # default: SLSQP (local optimization, much faster), 'basinhopping' (global optimization technique)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhopping_options: dictionary with optimization options
    :param slsqp_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """

    #ignore by warning message (during the optimization process following messages may occur and can be suppressed)
    if suppress_warnings:
        warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        warnings.filterwarnings("ignore", message="invalid value encountered")
    else:
        warnings.filterwarnings("default", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("default", message="divide by zero encountered in divide")
        warnings.filterwarnings("default", message="divide by zero encountered")
        warnings.filterwarnings("default", message="invalid value encountered")

    # convert to numpy.array for easier data handling
    x = np.array(x)
    x0 = np.array(x0)

    # help flag
    weights_applied = False

    # check whether weights are applied
    if len(weights)>1:
        weights = np.array(weights)
        weights_applied = True
    else:
        weights = np.ones(len(x))

    # round weights
    if weighting == 'expand' and weights_applied is True:
        weights = np.around(weights, 0).astype(float)

    # handle nans (Note: length of x, w must be same)
    if omit_missings:
        if np.isnan(x).any():
            if verbose: print('data contains NaNs and will be omitted')
            x_nan_index = np.where(~np.isnan(x))[0]
            x = np.array(x)[x_nan_index]
            weights = np.array(weights)[x_nan_index]

        if np.isnan(weights).any():
            if verbose: print('weights contain NaNs and will be omitted')
            w_nan_index = np.where(~np.isnan(weights))[0]
            x = np.array(x)[w_nan_index]
            weights = np.array(weights)[w_nan_index]

        # check whether there are weights=0, if True, drop w, x
        if np.any(weights == 0):
            nonzero_index = np.where(weights != 0)[0]
            weights = np.array(weights)[nonzero_index]
            x = np.array(x)[nonzero_index]

    if len(weights) != len(x):
        raise Exception("error - the length of W: {} does not match the length of x: {}".format(len(weights), len(x)))

    # cut x at lower bound b, top tails condition; Note: due to MemoryError, we need to keep the x, weights small from beginning
    # expand:
    # multiply:
    # non-weights:
    k = len(x[x>b])
    if weights_applied is True:
        xlargerb_index = np.where(x > b)[0]
        weights = np.array(weights)[xlargerb_index]
        N = int(np.sum(weights))
    else:
        # As no weights are specified, are not needed anymore -> set vector W to 1
        weights = np.ones(k)
        N = int(np.sum(weights))
    x = x_backup = x[x>b]
    weights_backup = weights

    # create list with indexes of x (needed for bootstrapping)
    x_index = np.arange(0, k, 1)

    # bootstraps (default: size k)
    if bootstraps is None:
        bootstraps = k

    # prepare progress bar and printed tables
    widgets = ['Bootstrapping (Pareto)\t', progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=bootstraps).start()
    tbl, tbl_gof = PrettyTable(), PrettyTable()

    def Pareto_constraint(b):
        return np.min(boot_sample) - b

    bnd = (10**-14, np.inf)
    constr = {'type': 'ineq', 'fun': Pareto_constraint}
    bootstrapping, p_fit_bs = 1, []

    if method == 'SLSQP':

        # shorter variable name
        opts = slsqp_options

        # defaults
        if 'jac' not in opts.keys():
            opts['jac'] = None
        if 'tol' not in opts.keys():
            opts['tol'] = None
        if 'callback' not in opts.keys():
            opts['callback'] = None
        if 'func' not in opts.keys():
            opts['func'] = None
        if 'maxiter' not in opts.keys():
            opts['maxiter'] = 100
        if 'ftol' not in opts.keys():
            opts['ftol'] = 1e-14
        if 'iprint' not in opts.keys():
            opts['iprint'] = 1
        if 'disp' not in opts.keys():
            opts['disp'] = False
        if 'eps' not in opts.keys():
            opts['eps'] = 1.4901161193847656e-08

        while bootstrapping <= bootstraps:

            # first, bootstrap indexes of sample x: x_index
            boot_sample_idx = np.random.choice(x_index, size=k, replace=True)
            boot_sample_idx = boot_sample_idx.astype(int)

            # second, select x of sample based on bootstrapped idx (sometimes first, faster call not working)
            try:
                boot_sample = x[boot_sample_idx]
            except:
                boot_sample = [x[i] for i in boot_sample_idx]

            # third, prepare weights: if weights were applied, also select weights based on bootstrapped idx
            if weights_applied is True:
                try:
                    boot_sample_weights = weights[boot_sample_idx]
                except:
                    boot_sample_weights = [weights[i] for i in boot_sample_idx]

                # fourth, expand boot_sample by weight
                if weighting == 'expand':
                    try:
                        x_inflated = []
                        for idx, i in enumerate(boot_sample):
                            x_extendby = np.repeat(boot_sample[idx], boot_sample_weights[idx])
                            x_inflated.extend(x_extendby)
                        boot_sample = x_inflated
                        # As we inflated x now, weights are not needed anymore -> set to 1
                        boot_sample_weights = np.ones(len(x_inflated))
                    except MemoryError:
                        print("error - MemoryError, not enough memory! Try a higher value of lower bound b to keep the top tail sample small")
                    except:
                        print("error - something went wrong while inflating x by its weights!")
                # normalize in each bootstrap
                if weighting == 'multiply':
                    # normalize weights: Σw=1
                    boot_sample_weights = np.multiply(boot_sample_weights, 1/np.sum(boot_sample_weights))
            # if no weights applied, fill weights with ones
            else:
                boot_sample_weights = np.ones(len(boot_sample))

            # prepare vars for optimization
            x = boot_sample
            W = boot_sample_weights

            result = opt.minimize(Pareto_ll, x0,
                                  args=(x, W, b),
                                  method='SLSQP',
                                  jac=opts['jac'],
                                  bounds=(bnd,),
                                  constraints=constr,
                                  tol=opts['tol'],
                                  callback=opts['callback'],
                                  options={'maxiter': opts['maxiter'], 'ftol': opts['ftol'], #'func': opts['func'],
                                           'iprint': opts['iprint'], 'disp': opts['disp'], 'eps': opts['eps']})
            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))
            p_fit = result.x.item(0)
            if weighting == 'multiply':
                if p_fit==1.0:
                    p_fit_bs.append(p_fit_bs[-1])
                else:
                    p_fit = p_fit/1000000
                p_fit_bs.append(p_fit)
            else:
                p_fit_bs.append(p_fit)
            bar.update(bootstrapping)
            bootstrapping += 1
        bar.finish()

    # if method == 'basinhopping':
    #
    #     # shorter variable name
    #     opts = basinhopping_options
    #
    #     # defaults
    #     if 'niter' not in opts.keys():
    #         opts['niter'] = 20
    #     if 'T' not in opts.keys():
    #         opts['T'] = 1.0
    #     if 'stepsize' not in opts.keys():
    #         opts['stepsize'] = 0.5
    #     if 'take_step' not in opts.keys():
    #         opts['take_step'] = None
    #     if 'accept_test' not in opts.keys():
    #         opts['accept_test'] = None
    #     if 'callback' not in opts.keys():
    #         opts['callback'] = None
    #     if 'interval' not in opts.keys():
    #         opts['interval'] = 50
    #     if 'disp' not in opts.keys():
    #         opts['disp'] = False
    #     if 'niter_success' not in opts.keys():
    #         opts['niter_success'] = None
    #     if 'seed' not in opts.keys():
    #         opts['seed'] = 123
    #
    #     while bootstrapping <= bootstraps:
    #
    #         # first, bootstrap indexes of sample x: x_index
    #         boot_sample_idx = np.random.choice(x_index, size=k, replace=True)
    #
    #         # second, select x of sample based on bootstrapped idx
    #         boot_sample = [x[i] for i in boot_sample_idx]
    #
    #         # third, if weights were applied, also select W of weights based on bootstrapped idx
    #         if weights_applied is True:
    #             boot_sample_weights = [weights[i] for i in boot_sample_idx]
    #
    #         # fourth, expand/multiply x_inflated by weight
    #         if weighting == 'expand':
    #             try:
    #                 x_inflated = []
    #                 for idx, i in enumerate(boot_sample):
    #                     x_extendby = np.repeat(boot_sample[idx], boot_sample_weights[idx])
    #                     x_inflated.extend(x_extendby)
    #                 x_weighted = x_inflated
    #             except MemoryError:
    #                 print("error - MemoryError, not enough memory! Try a higher value of lower bound b to keep the top tail sample small")
    #             except:
    #                 print("error - something went wrong while inflating x by its weights!")
    #
    #         if weighting == 'multiply':
    #             try:
    #                 xw = np.multiply(boot_sample, boot_sample_weights)
    #                 x_weighted = xw
    #             except:
    #                 print("error - something went wrong when trying to apply weights to x!")
    #
    #         minimizer_kwargs = {"method": "SLSQP", "args": (x_weighted, b),
    #                             "bounds": (bnd,)} #, "constraints": constr} constraint not needed, because b not optimized
    #
    #         result = opt.basinhopping(Pareto_ll, x0,
    #                                   minimizer_kwargs=minimizer_kwargs,
    #                                   niter=opts['niter'],
    #                                   T=opts['T'],
    #                                   stepsize=opts['stepsize'],
    #                                   take_step=opts['take_step'],
    #                                   accept_test=opts['accept_test'],
    #                                   callback=opts['callback'],
    #                                   interval=opts['interval'],
    #                                   disp=opts['disp'],
    #                                   niter_success=opts['niter_success'],
    #                                   seed=opts['seed'])
    #
    #         if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))
    #         p_fit = result.x.item(0)
    #         p_fit_bs.append(p_fit)
    #         bar.update(bootstrapping)
    #         bootstrapping += 1
    #
    #     bar.finish()

    # set back x, weights
    x = x_backup
    weights = weights_backup

    if ci is False and verbose:
        tbl.field_names = ['parameter', 'value', 'se']
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), '{:.3f}'.format(np.std(p_fit_bs))])
        print(tbl)

    if ci and verbose:
        # calculation of zscores, p-values related to Stata's regression outputs
        p_zscore = np.mean(p_fit_bs)/np.std(p_fit_bs)
        p_pval = 2*norm.cdf(-np.abs((np.mean(p_fit_bs)/np.std(p_fit_bs))))
        p_cilo = np.percentile(p_fit_bs, 2.5) #CS
        p_cihi = np.percentile(p_fit_bs, 97.5)
        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'CI(2.5)', 'CI(97.5)', 'n', 'N']
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), '{:.3f}'.format(np.std(p_fit_bs)),
                     '{:.3f}'.format(p_zscore), '{:.3f}'.format(p_pval),
                     '{:.3f}'.format(p_cilo), '{:.3f}'.format(p_cihi), k, N])
        print(tbl)

    # if verbose:
    #     print(locals())

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit==True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 50
        else:
            num_bins = plot_cosmetics['bins']
        if 'col_data' not in plot_cosmetics.keys():
            col_fit = 'blue'
        else:
            col_fit = plot_cosmetics['col_data']
        if 'col_fit' not in plot_cosmetics.keys():
            col_model = 'orange'
        else:
            col_model = plot_cosmetics['col_fit']

        fig, ax = plt.subplots()
        # histogram of x (actual data)
        n, bins, patches = ax.hist(x, num_bins, density=1, label='Pareto fit', color=col_fit)
        # x2: model with fitted parameters
        x2 = Pareto_pdf(np.linspace(np.min(x), np.max(x), np.size(x)), b=b, p=np.mean(p_fit_bs))
        ax.plot(np.linspace(np.min(x), np.max(x), np.size(x)), x2, '--', label='model', color=col_model)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability density')
        ax.set_title('fit vs. data')
        ax.legend(['Pareto fit', 'data'])
        # fig.tight_layout()
        # if save_plot:
        #     figurename = 'figure_Pareto_fit_' + str(filename) + '.png'
        #     # figurename = 'figure_Pareto_fit_1.png'
        #     fig.savefig(figurename, dpi=300, format='png')
        plt.show()

    if return_gofs:
        fit = True
    if fit:
        u = np.array(np.random.uniform(.0, 1., len(x)))
        model = Pareto_icdf(u=u, b=b, p=np.mean(p_fit_bs))
        soe = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).soe
        # ssr = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).ssr
        # sse = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).sse
        # sst = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).sst
        emp_mean = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).emp_mean
        emp_var = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).emp_var
        pred_mean = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).pred_mean
        pred_var = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).pred_var
        mae = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).mae
        mse = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).mse
        rmse = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).rmse
        rrmse = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).rrmse
        aic = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).aic
        bic = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).bic
        n = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).n
        ll = gof(x=x, x_hat=model, W=W, b=b, parms=[np.mean(p_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'n', 'N']
            tbl_gof.add_row(['GOF', '{:.3f}'.format(aic), '{:.3f}'.format(bic), '{:.3f}'.format(mae), '{:.3f}'.format(mse),
                             '{:.3f}'.format(rmse), '{:.3f}'.format(rrmse), '{:.3f}'.format(ll), '{:.3f}'.format(soe),
                             '{:.3f}'.format(emp_mean), '{:.3f}'.format(emp_var), '{:.3f}'.format(pred_mean),
                             '{:.3f}'.format(pred_var), '{:d}'.format(k), '{:d}'.format(N)])
            print("\n{}\n".format(tbl_gof))

        if return_gofs:
            return_parameters = False
            return np.mean(p_fit_bs), np.std(p_fit_bs), aic, bic, mae, mse, rmse, rrmse, ll, soe, emp_mean, emp_var, pred_mean, pred_var, k, N

    if return_parameters:
        return np.mean(p_fit_bs), np.std(p_fit_bs)


def IB1fit(x, b, x0, weights=np.array([1]), weighting='multiply', bootstraps=None, method='SLSQP', omit_missings=True,
           verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, suppress_warnings=True,
           return_parameters=False, return_gofs=False, #save_plot=False,
           plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'orange'},
           basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                                'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
           slsqp_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                          'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit the IB1 distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [p,q] or simply (p,q)
    :param weights: weight, default: numpy.ones array of same shape as x
    :param weighting: how to apply weights, default: 'expand' x expandes w times, else: 'multiply' x*w
    :param bootstraps: amount of bootstraps, default is size k (x>b)
    :param method: # default: SLSQP (local optimization, much faster), 'basinhopping' (global optimization technique)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhopping_options: dictionary with optimization options
    :param slsqp_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """

    #ignore by warning message (during the optimization process following messages may occur and can be suppressed)
    if suppress_warnings:
        warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        warnings.filterwarnings("ignore", message="invalid value encountered")
    else:
        warnings.filterwarnings("default", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("default", message="divide by zero encountered in divide")
        warnings.filterwarnings("default", message="divide by zero encountered")
        warnings.filterwarnings("default", message="invalid value encountered")

    # convert to numpy.array for easier data handling
    x = np.array(x)
    x0 = np.array(x0)

    # help flag
    weights_applied = False

    # handle nans (Note: length of x, w must be same)
    if omit_missings:
        if np.isnan(x).any():
            if verbose: print('data contains NaNs and will be omitted')
            x_nan_index = np.where(~np.isnan(x))[0]
            # x = x[~np.isnan(x)]
            x = np.array(x)[x_nan_index]
            weights = np.array(weights)[x_nan_index]

        if np.isnan(weights).any():
            if verbose: print('weights contain NaNs and will be omitted')
            w_nan_index = np.where(~np.isnan(weights))[0]
            # weights = weights[~np.isnan(weights)]
            x = np.array(x)[w_nan_index]
            weights = np.array(weights)[w_nan_index]

    # check whether user specified both x and W with same shape, if True, calculate population (=N) above b
    if len(weights)>1:
        N = int(np.sum(weights))
        weights = np.array(weights)
        weights_applied = True

        # check whether there are weights=0, if True, drop w, x
        if np.any(weights == 0):
            nonzero_index = np.where(weights != 0)[0]
            weights = np.array(weights)[nonzero_index]
            x = np.array(x)[nonzero_index]

    if len(weights) != len(x):
        raise Exception("error - the length of W: {} does not match the length of x: {}".format(len(weights), len(x)))

    # round weights
    if weighting == 'expand':
        weights = np.around(weights, 0).astype(float)

    # normalize weights: Σw=1
    if weighting == 'multiply':
        weights = np.multiply(weights, 1/sum(weights))

    # cut x at lower bound b, top tails condition; Note: due to MemoryError, we need to keep the x, weights small from beginning
    x = x[x>b]
    if weights_applied is True:
        xlargerb_index = np.where(x > b)[0]
        weights = np.array(weights)[xlargerb_index]
    else:
        weights = np.ones(len(x))
    k = len(x[x>b])

    # create list with indexes of x
    x_index = np.arange(0, k, 1)

    # bootstraps (default: size k)
    if bootstraps is None:
        bootstraps = k

    # prepare progress bar and printed tables
    widgets = ['Bootstrapping (IB1)\t\t', progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=bootstraps).start()
    tbl, tbl_gof = PrettyTable(), PrettyTable()

    def IB1_constraint(parms):
        a = parms[0]
        return (np.min(boot_sample)/b)**a

    constr = {'type': 'ineq', 'fun': IB1_constraint}
    bnds = (10**-14, np.inf)
    bootstrapping, p_fit_bs, q_fit_bs = 1, [], []

    if method == 'SLSQP':

        # shorter variable name
        opts = slsqp_options

        # defaults
        if 'jac' not in opts.keys():
            opts['jac'] = None
        if 'tol' not in opts.keys():
            opts['tol'] = 1e-14
        if 'callback' not in opts.keys():
            opts['callback'] = None
        if 'func' not in opts.keys():
            opts['func'] = None
        if 'maxiter' not in opts.keys():
            opts['maxiter'] = 300
        if 'ftol' not in opts.keys():
            opts['ftol'] = 1e-14
        if 'iprint' not in opts.keys():
            opts['iprint'] = 1
        if 'disp' not in opts.keys():
            opts['disp'] = False
        if 'eps' not in opts.keys():
            opts['eps'] = 1.4901161193847656e-08

        while bootstrapping <= bootstraps:

            # first, bootstrap indexes of sample x: x_index
            boot_sample_idx = np.random.choice(x_index, size=k, replace=True)

            # second, select x of sample based on bootstrapped idx
            boot_sample = [x[i] for i in boot_sample_idx]

            # third, if weights were applied, also select W of weights based on bootstrapped idx
            if weights_applied is True:
                boot_sample_weights = [weights[i] for i in boot_sample_idx]

            # fourth, expand/multiply x_inflated by weight
            if weighting == 'expand':
                try:
                    x_inflated = []
                    for idx, i in enumerate(boot_sample):
                        x_extendby = np.repeat(boot_sample[idx], boot_sample_weights[idx])
                        x_inflated.extend(x_extendby)
                    x_weighted = x_inflated
                except MemoryError:
                    print("error - MemoryError, not enough memory! Try a higher value of lower bound b to keep the top tail sample small")
                except:
                    print("error - something went wrong while inflating x by its weights!")

            if weighting == 'multiply':
                try:
                    xw = np.multiply(boot_sample, boot_sample_weights)
                    x_weighted = xw
                except:
                    print("error - something went wrong while when trying to apply weights to x!")

            result = opt.minimize(IB1_ll, x0,
                                  args=(x_weighted, b),
                                  method='SLSQP',
                                  jac=opts['jac'],
                                  bounds=(bnds, bnds,),
                                  constraints=constr,
                                  tol=opts['tol'],
                                  callback=opts['callback'],
                                  options=({'maxiter': opts['maxiter'], 'ftol': opts['ftol'], #'func': opts['func'],
                                            'iprint': opts['iprint'], 'disp': opts['disp'], 'eps': opts['eps']}))

            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))

            p_fit, q_fit = result.x.item(0), result.x.item(1)
            p_fit_bs.append(p_fit)
            q_fit_bs.append(q_fit)
            bar.update(bootstrapping)
            bootstrapping += 1

        bar.finish()

    if method == 'basinhopping':

        # shorter variable name
        opts = basinhopping_options

        # defaults
        if 'niter' not in opts.keys():
            opts['niter'] = 20
        if 'T' not in opts.keys():
            opts['T'] = 1.0
        if 'stepsize' not in opts.keys():
            opts['stepsize'] = 0.5
        if 'take_step' not in opts.keys():
            opts['take_step'] = None
        if 'accept_test' not in opts.keys():
            opts['accept_test'] = None
        if 'callback' not in opts.keys():
            opts['callback'] = None
        if 'interval' not in opts.keys():
            opts['interval'] = 50
        if 'disp' not in opts.keys():
            opts['disp'] = False
        if 'niter_success' not in opts.keys():
            opts['niter_success'] = None
        if 'seed' not in opts.keys():
            opts['seed'] = 123

        while bootstrapping <= bootstraps:

            # first, bootstrap indexes of sample x: x_index
            boot_sample_idx = np.random.choice(x_index, size=k, replace=True)

            # second, select x of sample based on bootstrapped idx
            boot_sample = [x[i] for i in boot_sample_idx]

            # third, if weights were applied, also select W of weights based on bootstrapped idx
            if weights_applied is True:
                boot_sample_weights = [weights[i] for i in boot_sample_idx]

            # fourth, expand/multiply x_inflated by weight
            if weighting == 'expand':
                try:
                    x_inflated = []
                    for idx, i in enumerate(boot_sample):
                        x_extendby = np.repeat(boot_sample[idx], boot_sample_weights[idx])
                        x_inflated.extend(x_extendby)
                    x_weighted = x_inflated
                except MemoryError:
                    print("error - MemoryError, not enough memory! Try a higher value of lower bound b to keep the top tail sample small")
                except:
                    print("error - something went wrong while inflating x by its weights!")

            if weighting == 'multiply':
                try:
                    xw = np.multiply(boot_sample, boot_sample_weights)
                    x_weighted = xw
                except:
                    print("error - something went wrong when trying to apply weights to x!")

            minimizer_kwargs = {"method": "SLSQP", "args": (x_weighted, b),
                                "bounds": (bnds, bnds,)} # , "constraints": constr} # constr not needed here

            result = opt.basinhopping(IB1_ll, x0,
                                      minimizer_kwargs=minimizer_kwargs,
                                      niter=opts['niter'],
                                      T=opts['T'],
                                      stepsize=opts['stepsize'],
                                      take_step=opts['take_step'],
                                      accept_test=opts['accept_test'],
                                      callback=opts['callback'],
                                      interval=opts['interval'],
                                      disp=opts['disp'],
                                      niter_success=opts['niter_success'],
                                      seed=opts['seed'])

            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))

            p_fit, q_fit = result.x.item(0), result.x.item(1)
            p_fit_bs.append(p_fit)
            q_fit_bs.append(q_fit)
            bar.update(bootstrapping)
            bootstrapping += 1

        bar.finish()

    if ci is False and verbose is True:
        tbl.field_names = ['parameter', 'value', 'se']
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), '{:.3f}'.format(np.std(p_fit_bs))])
        tbl.add_row(['q', '{:.3f}'.format(np.mean(q_fit_bs)), '{:.3f}'.format(np.std(q_fit_bs))])
        print(tbl)

    if ci and verbose:
        p_zscore = np.mean(p_fit_bs)/np.std(p_fit_bs)
        q_zscore = np.mean(q_fit_bs)/np.std(q_fit_bs)
        p_pval = 2*norm.cdf(-np.abs((np.mean(p_fit_bs)/np.std(p_fit_bs))))
        q_pval = 2*norm.cdf(-np.abs((np.mean(q_fit_bs)/np.std(q_fit_bs))))
        p_cilo = np.percentile(p_fit_bs, 2.5) #CS
        p_cihi = np.percentile(p_fit_bs, 97.5)
        q_cilo = np.percentile(q_fit_bs, 2.5)
        q_cihi = np.percentile(q_fit_bs, 97.5)

        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'CI(2.5)', 'CI(97.5)', 'n']
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), '{:.3f}'.format(np.std(p_fit_bs)),
                     '{:.3f}'.format(p_zscore), "{:.3f}".format(p_pval),
                     '{:.3f}'.format(p_cilo), #'{:.3f}'.format(np.mean(p_fit_bs)-np.std(p_fit_bs)*1.96),
                     '{:.3f}'.format(p_cihi), #'{:.3f}'.format(np.mean(p_fit_bs)+np.std(p_fit_bs)*1.96),
                     k])
        tbl.add_row(['q', '{:.3f}'.format(np.mean(q_fit_bs)), '{:.3f}'.format(np.std(q_fit_bs)),
                     '{:.3f}'.format(q_zscore), "{:.3f}".format(q_pval),
                     '{:.3f}'.format(q_cilo),#'{:.3f}'.format(np.mean(q_fit_bs)-np.std(q_fit_bs)*1.96),
                     '{:.3f}'.format(q_cihi),#'{:.3f}'.format(np.mean(q_fit_bs)+np.std(q_fit_bs)*1.96),
                     k])
        print(tbl)

    if verbose:
        print(locals())

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit==True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 50
        else:
            num_bins = plot_cosmetics['bins']
        if 'col_data' not in plot_cosmetics.keys():
            col_fit = 'blue'
        else:
            col_fit = plot_cosmetics['col_data']
        if 'col_fit' not in plot_cosmetics.keys():
            col_model = 'orange'
        else:
            col_model = plot_cosmetics['col_fit']

        fig, ax = plt.subplots()
        # histogram of x (actual data)
        n, bins, patches = ax.hist(x, num_bins, density=1, label='IB1 fit', color=col_fit)
        # x2: model with fitted parameters
        x2 = IB1_pdf(np.linspace(np.min(x), np.max(x), np.size(x)), b=b, p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        ax.plot(np.linspace(np.min(x), np.max(x), np.size(x)), x2, '--', label='model', color=col_model)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability density')
        ax.set_title('fit vs. data')
        ax.legend(['IB1 fit', 'data'])
        # fig.tight_layout()
        # if save_plot:
        #     figurename = 'figure_IB1_fit_' + str(filename) + '.png'
        #     # figurename = 'figure_IB1_fit_2.png'
        #     fig.savefig(figurename, dpi=300, format='png')
        plt.show()

    if return_gofs:
        fit = True
    if fit:
        model, u = IB1_icdf_ne(x=x, b=b, p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        soe = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).soe
        # ssr = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs)]).ssr
        # sse = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs)]).sse
        # sst = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs)]).sst
        emp_mean = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_mean
        emp_var = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_var
        pred_mean = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_mean
        pred_var = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_var
        mae = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'n', 'N']
            tbl_gof.add_row(['GOF', '{:.3f}'.format(aic), '{:.3f}'.format(bic), '{:.3f}'.format(mae), '{:.3f}'.format(mse),
                             '{:.3f}'.format(rmse), '{:.3f}'.format(rrmse), '{:.3f}'.format(ll), '{:.3f}'.format(soe),
                             '{:.3f}'.format(emp_mean), '{:.3f}'.format(emp_var), '{:.3f}'.format(pred_mean),
                             '{:.3f}'.format(pred_var), '{:d}'.format(k), '{:d}'.format(N)])
            print("\n{}\n".format(tbl_gof))

        if return_gofs:
            return_parameters = False
            return np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs), aic, bic, mae, mse, rmse, rrmse, ll, soe, emp_mean, emp_var, pred_mean, pred_var, k, N

    if return_parameters:
        return np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs)


def GB1fit(x, b, x0, weights=np.array([1]), weighting='multiply', bootstraps=None, method='SLSQP', omit_missings=True,
           verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, suppress_warnings=True,
           return_parameters=False, return_gofs=False, #save_plot=False,
           plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'orange'},
           basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                                'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
           slsqp_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                          'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit the GB1 distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [a,p,q] or simply (a,p,q)
    :param weights: weight, default: numpy.ones array of same shape as x
    :param weighting: how to apply weights, default: 'expand' x expandes w times, else: 'multiply' x*w
    :param bootstraps: amount of bootstraps, default is size k (x>b)
    :param method: # default: SLSQP (local optimization, much faster), 'basinhopping' (global optimization technique)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhopping_options: dictionary with optimization options
    :param slsqp_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """

    #ignore by warning message (during the optimization process following messages may occur and can be suppressed)
    if suppress_warnings:
        warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        warnings.filterwarnings("ignore", message="invalid value encountered")
    else:
        warnings.filterwarnings("default", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("default", message="divide by zero encountered in divide")
        warnings.filterwarnings("default", message="divide by zero encountered")
        warnings.filterwarnings("default", message="invalid value encountered")

    # convert to numpy.array for easier data handling
    x = np.array(x)
    x0 = np.array(x0)

    # help flag
    weights_applied = False

    # handle nans (Note: length of x, w must be same)
    if omit_missings:
        if np.isnan(x).any():
            if verbose: print('data contains NaNs and will be omitted')
            x_nan_index = np.where(~np.isnan(x))[0]
            # x = x[~np.isnan(x)]
            x = np.array(x)[x_nan_index]
            weights = np.array(weights)[x_nan_index]

        if np.isnan(weights).any():
            if verbose: print('weights contain NaNs and will be omitted')
            w_nan_index = np.where(~np.isnan(weights))[0]
            # weights = weights[~np.isnan(weights)]
            x = np.array(x)[w_nan_index]
            weights = np.array(weights)[w_nan_index]

    # check whether user specified both x and W with same shape, if True, calculate population (=N) above b
    if len(weights)>1:
        N = int(np.sum(weights))
        weights = np.array(weights)
        weights_applied = True

        # check whether there are weights=0, if True, drop w, x
        if np.any(weights == 0):
            nonzero_index = np.where(weights != 0)[0]
            weights = np.array(weights)[nonzero_index]
            x = np.array(x)[nonzero_index]

    if len(weights) != len(x):
        raise Exception("error - the length of W: {} does not match the length of x: {}".format(len(weights), len(x)))

    # round weights
    weights = np.around(weights, 0).astype(float)

    # cut x at lower bound b, top tails condition; Note: due to MemoryError, we need to keep the x, weights small from beginning
    x = x[x>b]
    if weights_applied is True:
        xlargerb_index = np.where(x > b)[0]
        weights = np.array(weights)[xlargerb_index]
    else:
        weights = np.ones(len(x))
    k = len(x[x>b])

    # create list with indexes of x
    x_index = np.arange(0, k, 1)

    # bootstraps (default: size k)
    if bootstraps is None:
        bootstraps = k

    # prepare progress bar and printed tables
    widgets = ['Bootstrapping (GB1)\t\t', progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=bootstraps).start()
    tbl, tbl_gof = PrettyTable(), PrettyTable()

    def GB1_constraint(parms):
        a = parms[0]
        return (np.min(boot_sample)/b)**a

    constr = {'type': 'ineq', 'fun': GB1_constraint}
    a_bnd, bnds = (-10, -1e-10), (10**-14, np.inf)
    bootstrapping, a_fit_bs, p_fit_bs, q_fit_bs = 1, [], [], []

    if method == 'SLSQP':

        # shorter variable name
        opts = slsqp_options

        # defaults
        if 'jac' not in opts.keys():
            opts['jac'] = None
        if 'tol' not in opts.keys():
            opts['tol'] = None
        if 'callback' not in opts.keys():
            opts['callback'] = None
        if 'func' not in opts.keys():
            opts['func'] = None
        if 'maxiter' not in opts.keys():
            opts['maxiter'] = 300
        if 'ftol' not in opts.keys():
            opts['ftol'] = 1e-14
        if 'iprint' not in opts.keys():
            opts['iprint'] = 1
        if 'disp' not in opts.keys():
            opts['disp'] = False
        if 'eps' not in opts.keys():
            opts['eps'] = 1.4901161193847656e-08


        while bootstrapping <= bootstraps:

            # first, bootstrap indexes of sample x: x_index
            boot_sample_idx = np.random.choice(x_index, size=k, replace=True)

            # second, select x of sample based on bootstrapped idx
            boot_sample = [x[i] for i in boot_sample_idx]

            # third, if weights were applied, also select W of weights based on bootstrapped idx
            if weights_applied is True:
                boot_sample_weights = [weights[i] for i in boot_sample_idx]

            # fourth, expand/multiply x_inflated by weight
            if weighting == 'expand':
                try:
                    x_inflated = []
                    for idx, i in enumerate(boot_sample):
                        x_extendby = np.repeat(boot_sample[idx], boot_sample_weights[idx])
                        x_inflated.extend(x_extendby)
                    x_weighted = x_inflated
                except MemoryError:
                    print("error - MemoryError, not enough memory! Try a higher value of lower bound b to keep the top tail sample small")
                except:
                    print("error - something went wrong while inflating x by its weights!")

            if weighting == 'multiply':
                try:
                    xw = np.multiply(boot_sample, boot_sample_weights)
                    x_weighted = xw
                except:
                    print("error - something went wrong while when trying to apply weights to x!")

            result = opt.minimize(GB1_ll, x0,
                                  args=(x_weighted, b),
                                  method='SLSQP',
                                  jac=opts['jac'],
                                  bounds=(a_bnd, bnds, bnds,),
                                  constraints=constr,
                                  tol=opts['tol'],
                                  callback=opts['callback'],
                                  options=({'maxiter': opts['maxiter'], 'ftol': opts['ftol'], #'func': opts['func'],
                                            'iprint': opts['iprint'], 'disp': opts['disp'], 'eps': opts['eps']}))

            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))

            a_fit, p_fit, q_fit = result.x.item(0), result.x.item(1), result.x.item(2)
            a_fit_bs.append(a_fit)
            p_fit_bs.append(p_fit)
            q_fit_bs.append(q_fit)
            bar.update(bootstrapping)
            bootstrapping += 1

        bar.finish()

    if method == 'basinhopping':

        # shorter variable name
        opts = basinhopping_options

        # defaults
        if 'niter' not in opts.keys():
            opts['niter'] = 20
        if 'T' not in opts.keys():
            opts['T'] = 1.0
        if 'stepsize' not in opts.keys():
            opts['stepsize'] = 0.5
        if 'take_step' not in opts.keys():
            opts['take_step'] = None
        if 'accept_test' not in opts.keys():
            opts['accept_test'] = None
        if 'callback' not in opts.keys():
            opts['callback'] = None
        if 'interval' not in opts.keys():
            opts['interval'] = 50
        if 'disp' not in opts.keys():
            opts['disp'] = False
        if 'niter_success' not in opts.keys():
            opts['niter_success'] = None
        if 'seed' not in opts.keys():
            opts['seed'] = 123

        while bootstrapping <= bootstraps:

            # first, bootstrap indexes of sample x: x_index
            boot_sample_idx = np.random.choice(x_index, size=k, replace=True)

            # second, select x of sample based on bootstrapped idx
            boot_sample = [x[i] for i in boot_sample_idx]

            # third, if weights were applied, also select W of weights based on bootstrapped idx
            if weights_applied is True:
                boot_sample_weights = [weights[i] for i in boot_sample_idx]

            # fourth, expand/multiply x_inflated by weight
            if weighting == 'expand':
                try:
                    x_inflated = []
                    for idx, i in enumerate(boot_sample):
                        x_extendby = np.repeat(boot_sample[idx], boot_sample_weights[idx])
                        x_inflated.extend(x_extendby)
                    x_weighted = x_inflated
                except MemoryError:
                    print("error - MemoryError, not enough memory! Try a higher value of lower bound b to keep the top tail sample small")
                except:
                    print("error - something went wrong while inflating x by its weights!")

            if weighting == 'multiply':
                try:
                    xw = np.multiply(boot_sample, boot_sample_weights)
                    x_weighted = xw
                except:
                    print("error - something went wrong when trying to apply weights to x!")

            minimizer_kwargs = {"method": "SLSQP", "args": (x_weighted, b),
                                "bounds": (a_bnd, bnds, bnds,)} #, "constraints": constr} not needed

            result = opt.basinhopping(GB1_ll, x0,
                                      minimizer_kwargs=minimizer_kwargs,
                                      niter=opts['niter'],
                                      T=opts['T'],
                                      stepsize=opts['stepsize'],
                                      take_step=opts['take_step'],
                                      accept_test=opts['accept_test'],
                                      callback=opts['callback'],
                                      interval=opts['interval'],
                                      disp=opts['disp'],
                                      niter_success=opts['niter_success'],
                                      seed=opts['seed'])

            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))

            a_fit, p_fit, q_fit = result.x.item(0), result.x.item(1), result.x.item(2)
            a_fit_bs.append(a_fit)
            p_fit_bs.append(p_fit)
            q_fit_bs.append(q_fit)
            bar.update(bootstrapping)

            bootstrapping += 1

        bar.finish()

    if ci is False and verbose is True:
        tbl.field_names = ['parameter', 'value', 'se']
        tbl.add_row(['a', '{:.3f}'.format(np.mean(a_fit_bs)), '{:.3f}'.format(np.std(a_fit_bs))])
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), '{:.3f}'.format(np.std(p_fit_bs))])
        tbl.add_row(['q', '{:.3f}'.format(np.mean(q_fit_bs)), '{:.3f}'.format(np.std(q_fit_bs))])
        print(tbl)

    if ci and verbose:
        a_zscore = np.mean(a_fit_bs)/np.std(a_fit_bs)
        p_zscore = np.mean(p_fit_bs)/np.std(p_fit_bs)
        q_zscore = np.mean(q_fit_bs)/np.std(q_fit_bs)
        a_pval = 2*norm.cdf(-np.abs((np.mean(a_fit_bs)/np.std(a_fit_bs))))
        p_pval = 2*norm.cdf(-np.abs((np.mean(p_fit_bs)/np.std(p_fit_bs))))
        q_pval = 2*norm.cdf(-np.abs((np.mean(q_fit_bs)/np.std(q_fit_bs))))
        a_cilo = np.percentile(a_fit_bs, 2.5) #CS
        a_cihi = np.percentile(a_fit_bs, 97.5)
        p_cilo = np.percentile(p_fit_bs, 2.5)
        p_cihi = np.percentile(p_fit_bs, 97.5)
        q_cilo = np.percentile(q_fit_bs, 2.5)
        q_cihi = np.percentile(q_fit_bs, 97.5)

        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'CI(2.5)', 'CI(97.5)', 'n']
        tbl.add_row(['a', '{:.3f}'.format(np.mean(a_fit_bs)), ('{:.3f}'.format(np.std(a_fit_bs))),
                     '{:.3f}'.format(a_zscore), '{:.3f}'.format(a_pval),
                     '{:.3f}'.format(a_cilo),#'{:.3f}'.format(np.mean(a_fit_bs)-np.std(a_fit_bs)*1.96),
                     '{:.3f}'.format(a_cihi),#'{:.3f}'.format(np.mean(a_fit_bs)+np.std(a_fit_bs)*1.96),
                     k])
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), ('{:.3f}'.format(np.std(p_fit_bs))),
                     '{:.3f}'.format(p_zscore), '{:.3f}'.format(p_pval),
                     '{:.3f}'.format(p_cilo),#'{:.3f}'.format(np.mean(p_fit_bs)-np.std(p_fit_bs)*1.96),
                     '{:.3f}'.format(p_cihi),#'{:.3f}'.format(np.mean(p_fit_bs)+np.std(p_fit_bs)*1.96),
                     k])
        tbl.add_row(['q', '{:.3f}'.format(np.mean(q_fit_bs)), ('{:.3f}'.format(np.std(q_fit_bs))),
                     '{:.3f}'.format(q_zscore), '{:.3f}'.format(q_pval),
                     '{:.3f}'.format(q_cilo),#'{:.3f}'.format(np.mean(q_fit_bs)-np.std(q_fit_bs)*1.96),
                     '{:.3f}'.format(q_cihi),#'{:.3f}'.format(np.mean(q_fit_bs)+np.std(q_fit_bs)*1.96),
                     k])
        print(tbl)

    if verbose:
        print(locals())

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit==True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 50
        else:
            num_bins = plot_cosmetics['bins']
        if 'col_data' not in plot_cosmetics.keys():
            col_fit = 'blue'
        else:
            col_fit = plot_cosmetics['col_data']
        if 'col_fit' not in plot_cosmetics.keys():
            col_model = 'orange'
        else:
            col_model = plot_cosmetics['col_fit']

        fig, ax = plt.subplots()
        # histogram of x (actual data)
        n, bins, patches = ax.hist(x, num_bins, density=1, label='GB1 fit', color=col_fit)
        # x2: model with fitted parameters
        x2 = GB1_pdf(np.linspace(np.min(x), np.max(x), np.size(x)), b=b, a=np.mean(a_fit_bs), p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        ax.plot(np.linspace(np.min(x), np.max(x), np.size(x)), x2, '--', label='model', color=col_model)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability density')
        ax.set_title('fit vs. data')
        ax.legend(['GB1 fit', 'data'])
        # fig.tight_layout()
        # if save_plot:
        #     figurename = 'figure_GB1_fit_' + str(filename) + '.png'
        #     # figurename = 'figure_GB1_fit_3.png'
        #     fig.savefig(figurename, dpi=300, format='png')
        plt.show()

    if return_gofs:
        fit = True
    if fit:
        model, u = GB1_icdf_ne(x=x, b=b, a=np.mean(a_fit_bs), p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))

        soe = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).soe
        # ssr = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs)]).ssr
        # sse = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs)]).sse
        # sst = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs)]).sst
        emp_mean = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_mean
        emp_var = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_var
        pred_mean = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_mean
        pred_var = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_var
        mae = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'n', 'N']
            tbl_gof.add_row(['GOF', '{:.3f}'.format(aic), '{:.3f}'.format(bic), '{:.3f}'.format(mae), '{:.3f}'.format(mse),
                             '{:.3f}'.format(rmse), '{:.3f}'.format(rrmse), '{:.3f}'.format(ll), '{:.3f}'.format(soe),
                             '{:.3f}'.format(emp_mean), '{:.3f}'.format(emp_var), '{:.3f}'.format(pred_mean),
                             '{:.3f}'.format(pred_var), '{:d}'.format(k), '{:d}'.format(N)])
            print("\n{}\n".format(tbl_gof))

        if return_gofs:
            return_parameters = False
            return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs), aic, bic, mae, mse, rmse, rrmse, ll, soe, emp_mean, emp_var, pred_mean, pred_var, k, N

    if return_parameters:
        return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs)


def GBfit(x, b, x0, weights=np.array([1]), weighting='multiply', bootstraps=None, method='SLSQP', omit_missings=True,
          verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, suppress_warnings=True,
          return_parameters=False, return_gofs=False, #save_plot=False,
          plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'orange'},
    basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                         'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
          slsqp_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                         'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit the GB distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [a,c,p,q] or simply (q,c,p,q)
    :param weights: weight, default: numpy.ones array of same shape as x
    :param weighting: how to apply weights, default: 'expand' x expandes w times, else: 'multiply' x*w
    :param bootstraps: amount of bootstraps, default is size k (x>b)
    :param method: # default: SLSQP (local optimization, much faster), 'basinhopping' (global optimization technique)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhopping_options: dictionary with optimization options
    :param slsqp_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """

    #ignore by warning message (during the optimization process following messages may occur and can be suppressed)
    if suppress_warnings:
        warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        warnings.filterwarnings("ignore", message="invalid value encountered")
    else:
        warnings.filterwarnings("default", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("default", message="divide by zero encountered in divide")
        warnings.filterwarnings("default", message="divide by zero encountered")
        warnings.filterwarnings("default", message="invalid value encountered")

    # convert to numpy.array for easier data handling
    x = np.array(x)
    x0 = np.array(x0)

    # help flag
    weights_applied = False

    # handle nans (Note: length of x, w must be same)
    if omit_missings:
        if np.isnan(x).any():
            if verbose: print('data contains NaNs and will be omitted')
            x_nan_index = np.where(~np.isnan(x))[0]
            # x = x[~np.isnan(x)]
            x = np.array(x)[x_nan_index]
            weights = np.array(weights)[x_nan_index]

        if np.isnan(weights).any():
            if verbose: print('weights contain NaNs and will be omitted')
            w_nan_index = np.where(~np.isnan(weights))[0]
            # weights = weights[~np.isnan(weights)]
            x = np.array(x)[w_nan_index]
            weights = np.array(weights)[w_nan_index]

    # check whether user specified both x and W with same shape, if True, calculate population (=N) above b
    if len(weights)>1:
        N = int(np.sum(weights))
        weights = np.array(weights)
        weights_applied = True

        # check whether there are weights=0, if True, drop w, x
        if np.any(weights == 0):
            nonzero_index = np.where(weights != 0)[0]
            weights = np.array(weights)[nonzero_index]
            x = np.array(x)[nonzero_index]

    if len(weights) != len(x):
        raise Exception("error - the length of W: {} does not match the length of x: {}".format(len(weights), len(x)))

    # round weights
    weights = np.around(weights, 0).astype(float)

    # cut x at lower bound b, top tails condition; Note: due to MemoryError, we need to keep the x, weights small from beginning
    x = x[x>b]
    if weights_applied is True:
        xlargerb_index = np.where(x > b)[0]
        weights = np.array(weights)[xlargerb_index]
    else:
        weights = np.ones(len(x))
    k = len(x[x>b])

    # create list with indexes of x
    x_index = np.arange(0, k, 1)

    # bootstraps (default: size k)
    if bootstraps is None:
        bootstraps = k

    # prepare progress bar and printed tables
    widgets = ['Bootstrapping (GB)\t\t', progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=bootstraps).start()
    tbl, tbl_gof = PrettyTable(), PrettyTable()

    def GB_constraint1(parms):
        a = parms[0]
        c = parms[1]
        return (b**a)/(1-c) - np.min(boot_sample)**a

    def GB_constraint2(parms):
        a = parms[0]
        c = parms[1]
        return (b**a)/(1-c) - np.max(boot_sample)**a

    constr = ({'type': 'ineq', 'fun': GB_constraint1},
              {'type': 'ineq', 'fun': GB_constraint2})

    a_bnd, c_bnd, bnds = (-10, -1e-10), (0, 1), (10**-14, np.inf)
    bootstrapping, a_fit_bs, c_fit_bs, p_fit_bs, q_fit_bs = 1, [], [], [], []

    if method == 'SLSQP':

        # shorter variable name
        opts = slsqp_options

        # defaults
        if 'jac' not in opts.keys():
            opts['jac'] = None
        if 'tol' not in opts.keys():
            opts['tol'] = None
        if 'callback' not in opts.keys():
            opts['callback'] = None
        if 'func' not in opts.keys():
            opts['func'] = None
        if 'maxiter' not in opts.keys():
            opts['maxiter'] = 300
        if 'ftol' not in opts.keys():
            opts['ftol'] = 1e-14
        if 'iprint' not in opts.keys():
            opts['iprint'] = 1
        if 'disp' not in opts.keys():
            opts['disp'] = False
        if 'eps' not in opts.keys():
            opts['eps'] = 1.4901161193847656e-08

        while bootstrapping <= bootstraps:

            # first, bootstrap indexes of sample x: x_index
            boot_sample_idx = np.random.choice(x_index, size=k, replace=True)

            # second, select x of sample based on bootstrapped idx
            boot_sample = [x[i] for i in boot_sample_idx]

            # third, if weights were applied, also select W of weights based on bootstrapped idx
            if weights_applied is True:
                boot_sample_weights = [weights[i] for i in boot_sample_idx]

            # fourth, expand/multiply x_inflated by weight
            if weighting == 'expand':
                try:
                    x_inflated = []
                    for idx, i in enumerate(boot_sample):
                        x_extendby = np.repeat(boot_sample[idx], boot_sample_weights[idx])
                        x_inflated.extend(x_extendby)
                    x_weighted = x_inflated
                except MemoryError:
                    print("error - MemoryError, not enough memory! Try a higher value of lower bound b to keep the top tail sample small")
                except:
                    print("error - something went wrong while inflating x by its weights!")

            if weighting == 'multiply':
                try:
                    xw = np.multiply(boot_sample, boot_sample_weights)
                    x_weighted = xw
                except:
                    print("error - something went wrong while when trying to apply weights to x!")

            result = opt.minimize(GB_ll, x0,
                                  args=(x_weighted, b),
                                  method='SLSQP',
                                  jac=opts['jac'],
                                  bounds=(a_bnd, c_bnd, bnds, bnds,),
                                  constraints=constr,
                                  tol=opts['tol'],
                                  callback=opts['callback'],
                                  options=({'maxiter': opts['maxiter'], 'ftol': opts['ftol'], #'func': opts['func'],
                                            'iprint': opts['iprint'], 'disp': opts['disp'], 'eps': opts['eps']}))

            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))

            a_fit, c_fit, p_fit, q_fit = result.x.item(0), result.x.item(1), result.x.item(2), result.x.item(3)
            a_fit_bs.append(a_fit)
            c_fit_bs.append(c_fit)
            p_fit_bs.append(p_fit)
            q_fit_bs.append(q_fit)
            bar.update(bootstrapping)
            bootstrapping += 1

        bar.finish()

    if method == 'basinhopping':

        # shorter variable name
        opts = basinhopping_options

        # defaults
        if 'niter' not in opts.keys():
            opts['niter'] = 20
        if 'T' not in opts.keys():
            opts['T'] = 1.0
        if 'stepsize' not in opts.keys():
            opts['stepsize'] = 0.5
        if 'take_step' not in opts.keys():
            opts['take_step'] = None
        if 'accept_test' not in opts.keys():
            opts['accept_test'] = None
        if 'callback' not in opts.keys():
            opts['callback'] = None
        if 'interval' not in opts.keys():
            opts['interval'] = 50
        if 'disp' not in opts.keys():
            opts['disp'] = False
        if 'niter_success' not in opts.keys():
            opts['niter_success'] = None
        if 'seed' not in opts.keys():
            opts['seed'] = 123

        while bootstrapping <= bootstraps:

            # first, bootstrap indexes of sample x: x_index
            boot_sample_idx = np.random.choice(x_index, size=k, replace=True)

            # second, select x of sample based on bootstrapped idx
            boot_sample = [x[i] for i in boot_sample_idx]

            # third, if weights were applied, also select W of weights based on bootstrapped idx
            if weights_applied is True:
                boot_sample_weights = [weights[i] for i in boot_sample_idx]

            # fourth, expand/multiply x_inflated by weight
            if weighting == 'expand':
                try:
                    x_inflated = []
                    for idx, i in enumerate(boot_sample):
                        x_extendby = np.repeat(boot_sample[idx], boot_sample_weights[idx])
                        x_inflated.extend(x_extendby)
                    x_weighted = x_inflated
                except MemoryError:
                    print("error - MemoryError, not enough memory! Try a higher value of lower bound b to keep the top tail sample small")
                except:
                    print("error - something went wrong while inflating x by its weights!")

            if weighting == 'multiply':
                try:
                    xw = np.multiply(boot_sample, boot_sample_weights)
                    x_weighted = xw
                except:
                    print("error - something went wrong when trying to apply weights to x!")

            minimizer_kwargs = {"method": "SLSQP", "args": (x_weighted, b),
                                "bounds": (a_bnd, c_bnd, bnds, bnds,)} #, "constraints": constr} not needed

            result = opt.basinhopping(GB_ll, x0,
                                      minimizer_kwargs=minimizer_kwargs,
                                      niter=opts['niter'],
                                      T=opts['T'],
                                      stepsize=opts['stepsize'],
                                      take_step=opts['take_step'],
                                      accept_test=opts['accept_test'],
                                      callback=opts['callback'],
                                      interval=opts['interval'],
                                      disp=opts['disp'],
                                      niter_success=opts['niter_success'],
                                      seed=opts['seed'])

            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))

            a_fit, c_fit, p_fit, q_fit = result.x.item(0), result.x.item(1), result.x.item(2), result.x.item(3)
            a_fit_bs.append(a_fit)
            c_fit_bs.append(c_fit)
            p_fit_bs.append(p_fit)
            q_fit_bs.append(q_fit)
            bar.update(bootstrapping)

            bootstrapping += 1

        bar.finish()

    if ci is False and verbose is True:

        tbl.field_names = ['parameter', 'value', 'se']
        tbl.add_row(['a', '{:.3f}'.format(np.mean(a_fit_bs)), '{:.3f}'.format(np.std(a_fit_bs))])
        tbl.add_row(['c', '{:.3f}'.format(np.mean(c_fit_bs)), '{:.3f}'.format(np.std(c_fit_bs))])
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), '{:.3f}'.format(np.std(p_fit_bs))])
        tbl.add_row(['q', '{:.3f}'.format(np.mean(q_fit_bs)), '{:.3f}'.format(np.std(q_fit_bs))])
        print(tbl)

    if ci and verbose:
        a_zscore = np.mean(a_fit_bs)/np.std(a_fit_bs)
        c_zscore = np.mean(c_fit_bs)/np.std(c_fit_bs)
        p_zscore = np.mean(p_fit_bs)/np.std(p_fit_bs)
        q_zscore = np.mean(q_fit_bs)/np.std(q_fit_bs)
        a_pval = 2*norm.cdf(-np.abs((np.mean(a_fit_bs)/np.std(a_fit_bs))))
        c_pval = 2*norm.cdf(-np.abs((np.mean(c_fit_bs)/np.std(c_fit_bs))))
        p_pval = 2*norm.cdf(-np.abs((np.mean(p_fit_bs)/np.std(p_fit_bs))))
        q_pval = 2*norm.cdf(-np.abs((np.mean(q_fit_bs)/np.std(q_fit_bs))))
        a_cilo = np.percentile(a_fit_bs, 2.5) #CS
        a_cihi = np.percentile(a_fit_bs, 97.5)
        c_cilo = np.percentile(c_fit_bs, 2.5) #CS
        c_cihi = np.percentile(c_fit_bs, 97.5)
        p_cilo = np.percentile(p_fit_bs, 2.5)
        p_cihi = np.percentile(p_fit_bs, 97.5)
        q_cilo = np.percentile(q_fit_bs, 2.5)
        q_cihi = np.percentile(q_fit_bs, 97.5)

        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'CI(2.5)', 'CI(97.5)', 'n']
        tbl.add_row(['a', '{:.3f}'.format(np.mean(a_fit_bs)), '{:.3f}'.format(np.std(a_fit_bs)),
                     '{:.3f}'.format(a_zscore), '{:.3f}'.format(a_pval),
                     '{:.3f}'.format(a_cilo),#'{:.3f}'.format(np.mean(a_fit_bs)-np.std(a_fit_bs)*1.96),
                     '{:.3f}'.format(a_cihi),#'{:.3f}'.format(np.mean(a_fit_bs)+np.std(a_fit_bs)*1.96),
                     k])
        tbl.add_row(['c', '{:.3f}'.format(np.mean(c_fit_bs)), '{:.3f}'.format(np.std(c_fit_bs)),
                     '{:.3f}'.format(c_zscore), '{:.3f}'.format(c_pval),
                     '{:.3f}'.format(c_cilo),#'{:.3f}'.format(np.mean(c_fit_bs)-np.std(c_fit_bs)*1.96),
                     '{:.3f}'.format(c_cihi),#'{:.3f}'.format(np.mean(c_fit_bs)+np.std(c_fit_bs)*1.96),
                     k])
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), '{:.3f}'.format(np.std(p_fit_bs)),
                     '{:.3f}'.format(p_zscore), '{:.3f}'.format(p_pval),
                     '{:.3f}'.format(p_cilo),#'{:.3f}'.format(np.mean(p_fit_bs)-np.std(p_fit_bs)*1.96),
                     '{:.3f}'.format(p_cihi),#'{:.3f}'.format(np.mean(p_fit_bs)+np.std(p_fit_bs)*1.96),
                     k])
        tbl.add_row(['q', '{:.3f}'.format(np.mean(q_fit_bs)), '{:.3f}'.format(np.std(q_fit_bs)),
                     '{:.3f}'.format(q_zscore), '{:.3f}'.format(q_pval),
                     '{:.3f}'.format(q_cilo),#'{:.3f}'.format(np.mean(q_fit_bs)-np.std(q_fit_bs)*1.96),
                     '{:.3f}'.format(q_cihi),#'{:.3f}'.format(np.mean(q_fit_bs)+np.std(q_fit_bs)*1.96),
                     k])
        print(tbl)

    if verbose:
        print(locals())

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit=True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 50
        else:
            num_bins = plot_cosmetics['bins']
        if 'col_data' not in plot_cosmetics.keys():
            col_fit = 'blue'
        else:
            col_fit = plot_cosmetics['col_data']
        if 'col_fit' not in plot_cosmetics.keys():
            col_model = 'orange'
        else:
            col_model = plot_cosmetics['col_fit']

        fig, ax = plt.subplots()
        # histogram of x (actual data)
        n, bins, patches = ax.hist(x, num_bins, density=1, label='GB fit', color=col_fit)
        # x2: model with fitted parameters
        x2 = GB_pdf(np.linspace(np.min(x), np.max(x), np.size(x)), b=b,
                    a=np.mean(a_fit_bs), c=np.mean(c_fit_bs),
                    p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        ax.plot(np.linspace(np.min(x), np.max(x), np.size(x)), x2, '--', label='model', color=col_model)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability density')
        ax.set_title('fit vs. data')
        ax.legend(['GB fit', 'data'])
        # fig.tight_layout()
        # if save_plot:
        #     figurename = 'figure_GB_fit_' + str(filename) + '.png'
        #     # figurename = 'figure_GB_fit_4.png'
        #     fig.savefig(figurename, dpi=300, format='png')
        plt.show()

    if return_gofs:
        fit = True
    if fit:
        model, u = GB_icdf_ne(x=x, b=b, a=np.mean(a_fit_bs), c=np.mean(c_fit_bs), p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        soe = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).soe
        # ssr = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs)]).ssr
        # sse = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs)]).sse
        # sst = gof(x=x, x_hat=model, b=b, parms=[np.mean(p_fit_bs)]).sst
        emp_mean = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_mean
        emp_var = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_var
        pred_mean = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_mean
        pred_var = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_var
        mae = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=x, x_hat=model, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'n', 'N']
            tbl_gof.add_row(['GOF', '{:.3f}'.format(aic), '{:.3f}'.format(bic), '{:.3f}'.format(mae), '{:.3f}'.format(mse),
                             '{:.3f}'.format(rmse), '{:.3f}'.format(rrmse), '{:.3f}'.format(ll), '{:.3f}'.format(soe),
                             '{:.3f}'.format(emp_mean), '{:.3f}'.format(emp_var), '{:.3f}'.format(pred_mean),
                             '{:.3f}'.format(pred_var), '{:d}'.format(k), '{:d}'.format(N)])
            print("\n{}\n".format(tbl_gof))

        if return_gofs:
            return_parameters = False
            return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(c_fit_bs), np.std(c_fit_bs), np.mean(p_fit_bs), \
                   np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs), aic, bic, mae, mse, rmse, rrmse, ll, soe, emp_mean, emp_var, pred_mean, pred_var, k, N

    if return_parameters:
        return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(c_fit_bs), np.std(c_fit_bs), np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs)


""" 
---------------------------------------------------
Pareto branch fitting
---------------------------------------------------
"""

def Paretobranchfit(x, b, x0=np.array([-.1,.1,1,-.1]), weights=np.array([1]), weighting='multiply', bootstraps=None,
                    method='SLSQP', rejection_criterion='LRtest', alpha=.05,
                    verbose_bootstrap=False, verbose_single=False, verbose=True,
                    fit=False, plot=False, return_bestmodel=False, return_all=False, #save_all_plots=False,
                    suppress_warnings=True, omit_missings=True,
          plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'orange'},
    basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                         'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
          slsqp_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                         'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    This function fits the Pareto branch upwards, starting from the Pareto distribution. This function is a wrapper that
    calls all above fitting functions, runs all optimizations and compares the parameters according the Pareto branch
    restrictions. Comparing the Pareto distribution to the IB1, the LRtest (AIC) decides, whether there is a improvement.
    If the IB1 delivers a better fit, we go one level upwards and compare the IB1 vs the GB1 and so on.
    Following comparison structure: compare 1v2 -> 2 improvement -> compare 2v3 -> 3 improvement -> compare 3v4
    :param x: as above
    :param b: as above
    :param x0: either pass an 1x5 array (GB init guess structure) OR pass [[p_guess], [p_guess, q_guess], [a_guess, p_guess, q_guess], [a_guess, c_guess, p_guess, q_guess]]
    :param weights: as above
    :param weighting: as above
    :param bootstraps: either 1x1 OR 1x2 array (1st arg: Pareto+IB1, 2nd arg: GB1+GB) OR pass 1x4 array [Pareto_bs, IB1_bs, GB1_bs, GB_bs]
    :param method: as above
    :param verbose_bootstrap: as above
    :param rejection_criterion: LRtest or AIC (as recommended by McDonald)
    :param verbose: table with parameters and another with gofs, display only final result
    :param verbose_single: display each optimization results
    :param alpha: significance level of LRtest, default: 5%
    :param fit: as above
    :param plot: as above
    :param return_parameters: as above and parameters, se of all distributions are returned
    :param return_gofs: as above
    :param plot_cosmetics: as above
    :param basinhopping_options: as above
    :param slsqp_options: as above
    :return:
    """

    #ignore by warning message (during the optimization process following messages may occur and can be suppressed)
    if suppress_warnings:
        warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        warnings.filterwarnings("ignore", message="invalid value encountered")
    else:
        warnings.filterwarnings("default", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("default", message="divide by zero encountered in divide")
        warnings.filterwarnings("default", message="divide by zero encountered")
        warnings.filterwarnings("default", message="invalid value encountered")

    #ignore by warning message (during the optimization process following messages may occur and can be suppressed)
    if suppress_warnings:
        warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        warnings.filterwarnings("ignore", message="invalid value encountered")
    else:
        warnings.filterwarnings("default", message="divide by zero encountered in double_scalars")
        warnings.filterwarnings("default", message="divide by zero encountered in divide")
        warnings.filterwarnings("default", message="divide by zero encountered")
        warnings.filterwarnings("default", message="invalid value encountered")

    # convert to numpy.array for easier data handling
    x = np.array(x)
    x0 = np.array(x0)

    # help flag
    weights_applied = False

    # handle nans (Note: length of x, w must be same)
    if omit_missings:
        if np.isnan(x).any():
            if verbose: print('data contains NaNs and will be omitted')
            x_nan_index = np.where(~np.isnan(x))[0]
            # x = x[~np.isnan(x)]
            x = np.array(x)[x_nan_index]
            weights = np.array(weights)[x_nan_index]

        if np.isnan(weights).any():
            if verbose: print('weights contain NaNs and will be omitted')
            w_nan_index = np.where(~np.isnan(weights))[0]
            # weights = weights[~np.isnan(weights)]
            x = np.array(x)[w_nan_index]
            weights = np.array(weights)[w_nan_index]

    # check whether user specified both x and W with same shape, if True, calculate population (=N) above b
    if len(weights)>1:
        N = int(np.sum(weights))
        weights = np.array(weights)
        weights_applied = True

        # check whether there are weights=0, if True, drop w, x
        if np.any(weights == 0):
            nonzero_index = np.where(weights != 0)[0]
            weights = np.array(weights)[nonzero_index]
            x = np.array(x)[nonzero_index]

    if len(weights) != len(x):
        raise Exception("error - the length of W: {} does not match the length of x: {}".format(len(weights), len(x)))

    # round weights
    weights = np.around(weights, 0).astype(float)

    # cut x at lower bound b, top tails condition; Note: due to MemoryError, we need to keep the x, weights small from beginning
    x = x[x>b]
    if weights_applied is True:
        xlargerb_index = np.where(x > b)[0]
        weights = np.array(weights)[xlargerb_index]
    else:
        weights = np.ones(len(x))
    k = len(x[x>b])

    # create list with indexes of x
    x_index = np.arange(0, k, 1)

    # bootstraps (default: size k)
    if bootstraps is None:
        bootstraps = k
    try:
        if int(bootstraps):
            Pareto_bs, IB1_bs, GB1_bs, GB_bs = bootstraps, bootstraps, bootstraps, bootstraps
    except TypeError:
        if len(bootstraps) == 4:
            Pareto_bs, IB1_bs, GB1_bs, GB_bs = bootstraps[0], bootstraps[1], bootstraps[2], bootstraps[3]
        elif len(bootstraps) == 2:
            Pareto_bs, IB1_bs, GB1_bs, GB_bs = bootstraps[0], bootstraps[0], bootstraps[1], bootstraps[1]
        else:
            raise Exception("error - bootstrap not correctly specified")

    ### Prepare args for passing to below fitting functions
    # x0
    try:
        x0_temp = [item for sublist in x0 for item in sublist]
        if len(x0_temp) == 10:
            Pareto_x0, IB1_x0, GB1_x0, GB_x0 = x0_temp[0], x0_temp[1:3], x0_temp[3:5], x0_temp[6:]
    except TypeError:
        if len(x0) == 4:
            Pareto_x0 = x0[3]
            IB1_x0 = x0[2:4]
            GB1_x0 = (x0[0], x0[2], x0[3])
            GB_x0 = x0
        else:
            raise Exception("error - x0 not correctly specified")

    # plot_options
    plt_cosm = {'bins': plot_cosmetics['bins'], 'col_data': plot_cosmetics['col_data'], 'col_fit': plot_cosmetics['col_fit']}

    # basinhopping options
    bh_opts = {'niter': basinhopping_options['niter'], 'T': basinhopping_options['T'], 'stepsize': basinhopping_options['stepsize'],
             'take_step': basinhopping_options['take_step'], 'accept_test': basinhopping_options['accept_test'],
             'callback': basinhopping_options['callback'], 'interval': basinhopping_options['interval'],
             'disp': basinhopping_options['disp'], 'niter_success': basinhopping_options['niter_success'],
             'seed': basinhopping_options['seed']}

    # SLSQP options
    slsqp_opts = {'jac': slsqp_options['jac'], 'tol': slsqp_options['tol'], 'callback': slsqp_options['callback'],
                  'func': slsqp_options['func'], 'maxiter': slsqp_options['maxiter'], 'ftol': slsqp_options['ftol'],
                  'iprint': slsqp_options['iprint'], 'disp': slsqp_options['disp'], 'eps': slsqp_options['eps']}

    # fit distributions
    Pareto_fit = Paretofit(x=x, b=b, x0=Pareto_x0, weights=weights, weighting=weighting, bootstraps=Pareto_bs, method=method,
                           return_parameters=True, return_gofs=True, ci=True, verbose=verbose_single, omit_missings=omit_missings,
                           verbose_bootstrap=verbose_bootstrap, fit=fit, plot=plot, suppress_warnings=suppress_warnings,
                           plot_cosmetics=plt_cosm, basinhopping_options=bh_opts, slsqp_options=slsqp_opts)

    IB1_fit = IB1fit(x=x, b=b, x0=IB1_x0, weights=weights, weighting=weighting, bootstraps=IB1_bs, method=method,
                     return_parameters=True, return_gofs=True, ci=True, verbose=verbose_single, omit_missings=omit_missings,
                     verbose_bootstrap=verbose_bootstrap, fit=fit, plot=plot, suppress_warnings=suppress_warnings,
                     plot_cosmetics=plt_cosm, basinhopping_options=bh_opts, slsqp_options=slsqp_opts)

    GB1_fit = GB1fit(x=x, b=b, x0=GB1_x0, weights=weights, weighting=weighting, bootstraps=GB1_bs, method=method,
                     return_parameters=True, return_gofs=True, ci=True, verbose=verbose_single, omit_missings=omit_missings,
                     verbose_bootstrap=verbose_bootstrap, fit=fit, plot=plot, suppress_warnings=suppress_warnings,
                     plot_cosmetics=plt_cosm, basinhopping_options=bh_opts, slsqp_options=slsqp_opts)

    GB_fit = GBfit(x=x, b=b, x0=GB_x0, weights=weights, weighting=weighting, bootstraps=GB_bs, method=method,
                   return_parameters=True, return_gofs=True, ci=True, verbose=verbose_single, omit_missings=omit_missings,
                   verbose_bootstrap=verbose_bootstrap, fit=fit, plot=plot, suppress_warnings=suppress_warnings,
                   plot_cosmetics=plt_cosm, basinhopping_options=bh_opts, slsqp_options=slsqp_opts)

    # unpack parameters
    p_fit1, p_se1 = Pareto_fit[:2]
    p_fit2, p_se2, q_fit2, q_se2 = IB1_fit[:4]
    a_fit3, a_se3, p_fit3, p_se3, q_fit3, q_se3 = GB1_fit[:6]
    a_fit4, a_se4, c_fit4, c_se4, p_fit4, p_se4, q_fit4, q_se4 = GB_fit[:8]

    # run rejection based on LRtest
    if 'LRtest' in rejection_criterion or 'LRtest' in rejection_criterion:
        # alpha = .05
        # 1. LRtest IB1 restriction q=1
        LRtestIB1_restrict = LRtest(IB1(x=x, b=b, p=p_fit2, q=1).LL,
                           IB1(x=x, b=b, p=p_fit2, q=q_fit2).LL,
                           df=1, verbose=False) #df: # of tested parms
        # 2. LRtest GB1 restriction a=-1
        LRtestGB1_restrict = LRtest(GB1(x=x, b=b, a=-1, p=p_fit3, q=q_fit3).LL,
                           GB1(x=x, b=b, a=a_fit3, p=p_fit3, q=q_fit3).LL,
                           df=1, verbose=False) #df: # of tested parms
        # 3. LRtest GB restriction c=0
        LRtestGB_restrict = LRtest(GB(x=x, b=b, a=a_fit4, c=0, p=p_fit4, q=q_fit4).LL,
                           GB(x=x, b=b, a=a_fit4, c=c_fit4, p=p_fit4, q=q_fit4).LL,
                           df=1, verbose=False) #df: # of tested parms

        # LR testing procedure
        Pareto_bm = IB1_bm = GB1_bm = GB_bm = Pareto_marker = IB1_marker = GB1_marker = GB_marker = '--'
        GB_remaining = False
        if LRtestIB1_restrict.pval < alpha:
            # LRtestIB1_restrict.pval is smaller than alpha, reject H0,
            # q=1 not valid, go one parm. level up in pareto branch and test next GB1 restriction
            if LRtestGB1_restrict.pval < alpha:
                # LRtestGB1_restrict.pval is smaller than alpha, reject H0,
                # a=-1 not valid, go one parm. level up in pareto branch and test next GB restriction
                bestmodel, Pareto_marker = '--', 'IB1'
                if LRtestGB_restrict.pval < alpha:
                    # LRtestGB_restrict.pval is smaller than alpha, reject H0,
                    # c=0 not valid, go one parm. level up in pareto branch and test next GB1 restriction
                    GB_bm, bestmodel, GB_marker, GB_remaining = 'GB', 'GB', 'XX', True
                    Pareto_marker = IB1_marker = GB1_marker = GB1_marker = '--'
                else:
                    GB1_bm, bestmodel, GB1_marker = 'GB1', 'GB1', 'XX'
                    IB1_marker = Pareto_marker = '--'
            else:
                IB1_bm, bestmodel, Pareto_marker, IB1_marker = 'IB1', 'IB1', '--', 'XX'
        else:
            Pareto_bm, bestmodel, Pareto_marker = 'Pareto', 'Pareto', 'XX'

        # save LRtest results to tbl
        tbl = PrettyTable()
        tbl.field_names = ['test restriction', 'H0', 'LR test', '', 'stop', 'best model']
        tbl.add_row(['IB1 restriction', 'q=1', 'chi2({}) = '.format(1), '{:.3f}'.format(LRtestIB1_restrict.w), '{}'.format(Pareto_marker), '{}'.format(Pareto_bm)])
        tbl.add_row(['', '', 'Prob > chi2', '{:.3f}'.format(LRtestIB1_restrict.pval), '{}'.format(Pareto_marker), '{}'.format(Pareto_bm)])
        tbl.add_row(['GB1 restriction', 'a=-1', 'chi2({}) = '.format(1), '{:.3f}'.format(LRtestGB1_restrict.w), '{}'.format(IB1_marker), '{}'.format(IB1_bm)])
        tbl.add_row(['', '', 'Prob > chi2', '{:.3f}'.format(LRtestGB1_restrict.pval), '{}'.format(IB1_marker), '{}'.format(IB1_bm)])
        tbl.add_row(['GB restriction', 'c=0', 'chi2({}) = '.format(1), '{:.3f}'.format(LRtestGB_restrict.w), '{}'.format(GB1_marker), '{}'.format(GB1_bm)])
        tbl.add_row(['', '', 'Prob > chi2', '{:.3f}'.format(LRtestGB_restrict.pval), '{}'.format(GB1_marker), '{}'.format(GB1_bm)])
        if GB_remaining:
            tbl.add_row(['GB', '', '', '', '{}'.format(GB_marker), '{}'.format(GB_bm)])
        print("\n")
        print(tbl)

    # run rejection based on AIC
    if 'AIC' in rejection_criterion or 'AIC' in rejection_criterion:

        # unpack aic
        Pareto_aic = Pareto_fit[2]
        IB1_aic = IB1_fit[4]
        GB1_aic = GB1_fit[6]
        GB_aic = GB_fit[8]

        # 1v2, 2v3, 3v4
        Pareto_bm = IB1_bm = GB1_bm = GB_bm = Pareto_marker = IB1_marker = GB1_marker = GB_marker = '--'
        GB_remaining = False

        if Pareto_aic > IB1_aic:
            if IB1_aic > GB1_aic:
                bestmodel, Pareto_marker = '--', 'IB1'

                if GB1_aic > GB_aic:
                    GB_bm, bestmodel, GB_marker, GB_remaining = 'GB', 'GB', 'XX', True
                    Pareto_marker = IB1_marker = GB1_marker = GB1_marker = '--'
                else:
                    GB1_bm, bestmodel, IB1_marker, GB1_marker = 'GB1', 'GB1', '--', 'XX'
            else:
                IB1_bm, bestmodel, Pareto_marker, IB1_marker = 'IB1', 'IB1', '--', 'XX'
        else:
            Pareto_bm, bestmodel, Pareto_marker = 'Pareto', 'Pareto', 'XX'

        # save LRtest results to tbl
        tbl = PrettyTable()
        tbl.field_names = ['comparison', 'AIC1', 'AIC2', 'stop', 'best model']
        tbl.add_row(['Pareto vs IB1', '{:.3f}'.format(Pareto_aic), '{:.3f}'.format(IB1_aic), '{}'.format(Pareto_marker), '{}'.format(Pareto_bm)])
        tbl.add_row(['IB1 vs GB1', '{:.3f}'.format(IB1_aic), '{:.3f}'.format(GB1_aic), '{}'.format(IB1_marker), '{}'.format(IB1_bm)])
        tbl.add_row(['GB1 vs GB', '{:.3f}'.format(GB1_aic), '{:.3f}'.format(GB_aic), '{}'.format(GB1_marker), '{}'.format(GB1_bm)])

        if GB_remaining:
            tbl.add_row(['GB', '{:.3f}'.format(GB_aic), '', '{}'.format(GB_marker), '{}'.format(GB_bm)])

        print('\n{}'.format(tbl))

        # # recalculate AICs
        # Pareto_aic = -2*Pareto(x=x, b=b, p=Pareto_fit[0]).LL+2*1
        # IB1_aic = -2*IB1(x=x, b=b, p=IB1_fit[0], q=IB1_fit[2]).LL+2*2
        # GB1_aic = -2*GB1(x=x, b=b, a=GB1_fit[0], p=GB1_fit[2], q=GB1_fit[4]).LL+2*3
        # GB_aic  = -2*GB(x=x, b=b, a=GB_fit[0], c=GB_fit[2], p=GB_fit[4], q=GB_fit[6]).LL+2*4
        #
        # # calculate AICs of models with restriction of Pareto branch
        # IB1_aic_restrict = -2*IB1(x=x, b=b, p=IB1_fit[0], q=1).LL+2*2
        # GB1_aic_restrict = -2*GB1(x=x, b=b, a=-1, p=GB1_fit[2], q=GB1_fit[4]).LL+2*3
        # GB_aic_restrict  = -2*GB(x=x, b=b, a=GB_fit[0], c=0, p=GB_fit[4], q=GB_fit[6]).LL+2*4
        #
        # # AIC testing procedure
        # Pareto_bm = IB1_bm = GB1_bm = GB_bm = Pareto_marker = IB1_marker = GB1_marker = GB_marker = '--'
        # GB_remaining = False
        # if IB1_aic_restrict > IB1_aic:
        #     if GB1_aic_restrict > GB1_aic:
        #         bestmodel, Pareto_marker = '--', 'IB1'
        #         if GB_aic_restrict > GB_aic:
        #             GB_bm, bestmodel, GB_marker, GB_remaining = 'GB', 'GB', 'XX', True
        #             Pareto_marker = IB1_marker = GB1_marker = GB1_marker = '--'
        #         else:
        #             GB1_bm, bestmodel, GB1_marker = 'GB1', 'GB1', 'XX'
        #             IB1_marker = Pareto_marker = '--'
        #     else:
        #         IB1_bm, bestmodel, Pareto_marker, IB1_marker = 'IB1', 'IB1', '--', 'XX'
        # else:
        #     Pareto_bm, bestmodel, Pareto_marker = 'Pareto', 'Pareto', 'XX'
        #
        # # save LRtest results to tbl
        # tbl = PrettyTable()
        # tbl.field_names = ['AIC comparison', 'AIC (restricted)', 'AIC (full)', 'stop', 'best model']
        # tbl.add_row(['IB1(pfit, q=1) vs. IB1(pfit, qfit)', '{:.3f}'.format(IB1_aic_restrict), '{:.3f}'.format(IB1_aic), '{}'.format(Pareto_marker), '{}'.format(Pareto_bm)])
        # tbl.add_row(['GB1(a=-1, pfit, qfit) vs.', '', '', '', ''])
        # tbl.add_row(['GB1(afit, pfit, qfit)', '{:.3f}'.format(GB1_aic_restrict), '{:.3f}'.format(GB1_aic), '{}'.format(IB1_marker), '{}'.format(IB1_bm)])
        # tbl.add_row(['GB(afit, c=0, pfit, qfit) vs.', '', '', '', ''])
        # tbl.add_row(['GB(afit, cfit, pfit, qfit)', '{:.3f}'.format(GB_aic_restrict), '{:.3f}'.format(GB_aic), '{}'.format(GB1_marker), '{}'.format(GB1_bm)])
        #
        # if GB_remaining:
        #     tbl.add_row(['GB', '{:.3f}'.format(GB_aic), '', '{}'.format(GB_marker), '{}'.format(GB_bm)])
        #
        # print('\n{}'.format(tbl))

    if verbose:
        tbl_parms = PrettyTable()
        tbl_parms.field_names = ['parameter', 'Pareto', 'IB1', 'GB1', 'GB']
        tbl_parms.add_row(['a', '-', '-', '{:.3f}'.format(GB1_fit[0]), '{:.3f}'.format(GB_fit[0])])
        tbl_parms.add_row(['', '', '', '({:.3f})'.format(GB1_fit[1]), '({:.3f})'.format(GB_fit[1])])
        tbl_parms.add_row(['c', '-', '-', '-', '{:.3f}'.format(GB_fit[2])])
        tbl_parms.add_row(['', '', '', '', '({:.3f})'.format(GB_fit[3])])
        tbl_parms.add_row(['p', '{:.3f}'.format(Pareto_fit[0]), '{:.3f}'.format(IB1_fit[0]), '{:.3f}'.format(GB1_fit[2]), '{:.3f}'.format(GB_fit[4])])
        tbl_parms.add_row(['', '({:.3f})'.format(Pareto_fit[1]), '({:.3f})'.format(IB1_fit[1]), '({:.3f})'.format(GB1_fit[3]), '({:.3f})'.format(GB_fit[5])])
        tbl_parms.add_row(['q', '-', '{:.3f}'.format(IB1_fit[2]), '{:.3f}'.format(GB1_fit[4]), '{:.3f}'.format(GB_fit[6])])
        tbl_parms.add_row(['', '', '({:.3f})'.format(IB1_fit[3]), '({:.3f})'.format(GB1_fit[5]), '({:.3f})'.format(GB_fit[7])])

        tbl_gof = PrettyTable()
        tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'df', 'n', 'N']
        tbl_gof.add_row(['Pareto', '{:.3f}'.format(Pareto_fit[2]), '{:.3f}'.format(Pareto_fit[3]), '{:.3f}'.format(Pareto_fit[4]),
                         '{:.3f}'.format(Pareto_fit[5]), '{:.3f}'.format(Pareto_fit[6]), '{:.3f}'.format(Pareto_fit[7]),
                         '{:.3f}'.format(Pareto_fit[8]), '{:.3f}'.format(Pareto_fit[9]), '{:.3f}'.format(Pareto_fit[10]),
                         '{:.3f}'.format(Pareto_fit[11]), '{:.3f}'.format(Pareto_fit[12]), '{:.3f}'.format(Pareto_fit[13]), 1,
                         '{}'.format(Pareto_fit[14]), '{}'.format(Pareto_fit[15])])
        tbl_gof.add_row(['IB1', '{:.3f}'.format(IB1_fit[4]), '{:.3f}'.format(IB1_fit[5]), '{:.3f}'.format(IB1_fit[6]),
                         '{:.3f}'.format(IB1_fit[7]), '{:.3f}'.format(IB1_fit[8]), '{:.3f}'.format(IB1_fit[9]),
                         '{:.3f}'.format(IB1_fit[10]), '{:.3f}'.format(IB1_fit[11]), '{:.3f}'.format(IB1_fit[12]),
                         '{:.3f}'.format(IB1_fit[13]), '{:.3f}'.format(IB1_fit[14]), '{:.3f}'.format(IB1_fit[15]), 2,
                         '{}'.format(IB1_fit[16]), '{}'.format(IB1_fit[17])])
        tbl_gof.add_row(['GB1', '{:.3f}'.format(GB1_fit[6]), '{:.3f}'.format(GB1_fit[7]), '{:.3f}'.format(GB1_fit[8]),
                         '{:.3f}'.format(GB1_fit[9]), '{:.3f}'.format(GB1_fit[10]), '{:.3f}'.format(GB1_fit[11]),
                         '{:.3f}'.format(GB1_fit[12]), '{:.3f}'.format(GB1_fit[13]), '{:.3f}'.format(GB1_fit[14]),
                         '{:.3f}'.format(GB1_fit[15]), '{:.3f}'.format(GB1_fit[16]), '{:.3f}'.format(GB1_fit[17]), 3,
                         '{}'.format(GB1_fit[18]), '{}'.format(GB1_fit[19])])
        tbl_gof.add_row(['GB', '{:.3f}'.format(GB_fit[8]), '{:.3f}'.format(GB_fit[9]), '{:.3f}'.format(GB_fit[10]),
                         '{:.3f}'.format(GB_fit[11]), '{:.3f}'.format(GB_fit[12]), '{:.3f}'.format(GB_fit[13]),
                         '{:.3f}'.format(GB_fit[14]), '{:.3f}'.format(GB_fit[15]), '{:.3f}'.format(GB_fit[16]),
                         '{:.3f}'.format(GB_fit[17]), '{:.3f}'.format(GB_fit[18]), '{:.3f}'.format(GB_fit[19]), 4,
                         '{}'.format(GB_fit[20]), '{}'.format(GB_fit[21])])

        print('\n{}'.format(tbl_parms))
        print('\n{}'.format(tbl_gof))

    if return_bestmodel:
            if bestmodel == 'Pareto':
                return 'Pareto_best', Pareto_fit
            if bestmodel == 'IB1':
                return 'IB1_best', IB1_fit
            if bestmodel == 'GB1':
                return 'GB1_best', GB1_fit
            if bestmodel == 'GB':
                return 'GB_best', GB_fit

    if return_all:
        return Pareto_fit, IB1_fit, GB1_fit, GB_fit


""" 
---------------------------------------------------
Pareto and IB1: se extracting
---------------------------------------------------
"""

def Pareto_extract_se(x, b, p_fitted, method=1, verbose=True, hess=False):
    """
    NOTE: depreciated
    This function returns the standard errors of the fitted parameter p. Unfortunately, the analytical solution of
    the hessian only depends on n and p so varying b does not change the se. Alternatively, I tried to derive the
    Jacobian numerically with scipy's derivative fct but this resulted not in satisfactiory/plausible solutions.
    Finally, I bootstrap the se directly in the fitting fcts above.
    :param x: linspace or data
    :param b: location parameter, fixed
    :param p_fitted: fitted parameter, for which we want se
    :param method: 1=analytical solution, hessian, 2=scipy's derivative fct applied to Jacobian
    :param verbose: display
    :param hess: display hessian
    :return: returns se
    """
    p = p_fitted
    x = np.array(x)
    x = x[x>b]
    n = len(x)
    if method == 2: # derivative by hand
        δ2ll_δp2 = -n / (p**2)
        hess = [[δ2ll_δp2]]
    if (method == 1) or (method == None): # derivative numerically evaluated with 'central difference formula' (scipy.misc.derivative)
        def δll_δp(p,b,n):
            return (n/p) - np.sum(np.log(x))
        hess = [[derivative(δll_δp, p, args=[b,n], dx=1e-8)]]
    info_matrix = np.dot(-1/n, hess)
    # covvar = linalg.inv(info_matrix)
    p_se = np.sqrt(info_matrix[0][0])
    if verbose: print("p: {:.3f}, se: {:.3f}".format(p, p_se))
    # if hess: print("Hessian Matrix:", hess)
    return p_se


def IB1_extract_se(x, fitted_parms, method, dx, display, display_hessian):
    """
    NOTE: depreciated
    This function returns the standard errors of the fitted parameter p,q. Unfortunately, no plausible
    analytical solution of the hessian could be derived. Alternatively, I tried to derive the
    Jacobian numerically with scipy's derivative fct but this resulted not in satisfactiory/plausible solutions.
    Finally, I bootstrap the se directly in the fitting fcts above.
    :param x: linspace or data
    :param method: 1=analytical solution, hessian, 2=scipy's derivative fct applied to Jacobian
    :param fitted_parms:
    :param dx: tolerance of scipy.misc.derivative
    :param display: display results
    :param display_hessian: display hessian
    :return: returns se
    """
    # method == 2: # derivative by hand - NOTE: no analytical solution found
    b = fitted_parms[0]
    p = fitted_parms[1]
    q = fitted_parms[2]
    x = x[x>b]
    n = len(x)
    if dx == None: dx = 1e-6
    #dx = 1e-8 # temp
    if (method == 1) or (method == None):
        def δll_δb(b,p,q,n):
            return (n*p)/b + (q-1)*np.sum((-1/x)/(1-b/x))
        def δll_δp(p,b,q,n):
            return n*(np.log(b) - digamma(p) - digamma(p+q)) - np.sum(np.log(x))
        def δll_δq(p,q,n):
            return -n*(digamma(q) - digamma(q+p)) + np.sum(np.log(1-b/x))
        hess = [[derivative(δll_δb, b, args=[p,q,n], dx=dx), derivative(δll_δb, p, args=[b,q,n], dx=dx), derivative(δll_δb, q, args=[b,p,n], dx=dx)],
                [derivative(δll_δp, b, args=[p,q,n], dx=dx), derivative(δll_δp, p, args=[b,q,n], dx=dx), derivative(δll_δp, q, args=[b,p,n], dx=dx)],
                [0,                                            derivative(δll_δq, p, args=[q,n], dx=dx), derivative(δll_δq, q, args=[p,n], dx=dx)]]
    info_matrix = np.dot(-n, hess)
    covvar = linalg.inv(info_matrix)
    b_se = np.sqrt(covvar[0][0])
    p_se = np.sqrt(covvar[1][1])
    q_se = np.sqrt(covvar[2][2])
    if display == True:
        print("b: {:.3f}, se: {:.3f}\np: {:.3f}, se: {:.3f}\nq: {:.3f}, se: {:.3f}".format(b, b_se, p, p_se, q, q_se))
    if display_hessian == True:
        print("Hessian Matrix:", hess)
    return b_se, p_se, q_se



# ⣿⠄⡇⢸⣟⠄⠁⢸⡽⠖⠛⠈⡉⣉⠉⠋⣁⢘⠉⢉⠛⡿⢿⣿⣿⣿⣿⣿⣿⣿
# ⣷⣶⣷⣤⠄⣠⠖⠁⠄⠂⠁⠄⠄⠉⠄⠄⠎⠄⠠⠎⢐⠄⢑⣛⠻⣿⣿⣿⣿⣿
# ⣿⣿⣿⠓⠨⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠈⠐⠅⠄⠉⠄⠗⠆⣸⣿⣿⣿⣿⣿
# ⣿⣿⣿⡣⠁⠄⠄⠄⠄⠄⠄⠄⠄⠄⢰⣤⣦⠄⠄⠄⠄⠄⠄⠄⡀⡙⣿⣿⣿⣿
# ⣿⣿⡛⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠔⠿⡿⠿⠒⠄⠠⢤⡀⡀⠄⠁⠄⢻⣿⣿⣿
# ⣿⣿⠄⠄⠄⠄⠄⠄⣠⡖⠄⠁⠁⠄⠄⠄⠄⠄⠄⠄⣽⠟⡖⠄⠄⠄⣼⣿⣿⣿
# ⣿⣿⠄⠄⠄⠄⠄⠄⢠⣠⣀⠄⠄⠄⠄⢀⣾⣧⠄⠂⠸⣈⡏⠄⠄⠄⣿⣿⣿⣿
# ⣿⣿⡞⠄⠄⠄⠄⠄⢸⣿⣶⣶⣶⣶⣶⡿⢻⡿⣻⣶⣿⣿⡇⠄⠄⠄⣿⣿⣿⣿
# ⣿⣿⡷⡂⠄⠄⠁⠄⠸⣿⣿⣿⣿⣿⠟⠛⠉⠉⠙⠛⢿⣿⡇⠄⠄⢀⣿⣿⣿⣿
# ⣶⣶⠃⠄⠄⠄⠄⠄⠄⣾⣿⣿⡿⠁⣀⣀⣤⣤⣤⣄⢈⣿⡇⠄⠄⢸⣿⣿⣿⣿
# ⣿⣯⠄⠄⠄⠄⠄⠄⠄⢻⣿⣿⣷⣶⣿⣿⣥⣬⣿⣿⣟⣿⠃⠄⠨⠺⢿⣿⣿⣿
# ⠱⠂⠄⠄⠄⠄⠄⠄⠄⣬⣸⡝⠿⢿⣿⡿⣿⠻⠟⠻⢫⡁⠄⠄⠄⡐⣾⣿⣿⣿
# ⡜⠄⠄⠄⠄⠄⠆⡐⡇⢿⣽⣻⣷⣦⣧⡀⡀⠄⠄⣴⣺⡇⠄⠁⠄⢣⣿⣿⣿⣿
# ⠡⠱⠄⠄⠡⠄⢠⣷⠆⢸⣿⣿⣿⣿⣿⣿⣷⣿⣾⣿⣿⡇⠄⠄⠠⠁⠿⣿⣿⣿
# ⢀⣲⣧⣷⣿⢂⣄⡉⠄⠘⠿⣿⣿⣿⡟⣻⣯⠿⠟⠋⠉⢰⢦⠄⠊⢾⣷⣮⣽⣛
