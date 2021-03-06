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
    :param quantiles: array-like with many quantiles needed, between 0.0 and 1.0
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)

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
    """ Goodness of fit measures and descriptive statistics for comparison of data vs fit """
    def __init__(self, x, x_hat, W, parms, b):
        """
        :param x: model with same shape as data
        :param x_hat: fitted data
        :param parms: np.array []
        :param b: location parameter, fixed
        """
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

        # if weights = np.ones vector, pass, else normalize to ΣW=1
        if len(x) == np.sum(W):
            pass
        else:
            W = np.multiply(W, 1/np.sum(W))

        if len(parms) == 1:
            self.ll = ll = (-1)*Pareto_ll(parms=parms, x=x, W=W, b=b)
            self.aic = -2*ll + 2
            self.bic = -2*ll + np.log(n)
        if len(parms) == 2:
            self.ll = ll = (-1)*IB1_ll(parms=parms, x=x, W=W, b=b)
            self.aic = -2*ll + 2*2
            self.bic = -2*ll + np.log(n)*2
        if len(parms) == 3:
            self.ll = ll = (-1)*GB1_ll(parms=parms, x=x, W=W, b=b)
            self.aic = -2*ll + 2*3
            self.bic = -2*ll + np.log(n)*3
        if len(parms) == 4:
            self.ll = ll = (-1)*GB_ll(parms, x=x, W=W, b=b)
            self.aic = -2*ll + 2*4
            self.bic = -2*ll + np.log(n)*4

""" 
---------------------------------------------------
Neg. Log-Likelihoods
---------------------------------------------------
# NOTE: optimize a,c,p,q = parms (first args in below functions, must be array), fixed parameters: x, W, b
"""

def Pareto_ll(parms, x, W, b):
    """
    :param parms: np.array [p], optimized
    :param x: linspace, fixed
    :param W: weights, either np.ones() if no weights have been applied OR iweights Σw=1, fixed
    :param b: location parameter, fixed
    :return: neg. logliklihood of Pareto
    """
    p = parms[0]
    n = np.sum(W)
    x = x[x>b]
    sum = np.sum(np.log(x)*W)
    ll = n*np.log(p) + p*n*np.log(b) - (p+1)*sum
    ll = -ll
    return ll

def IB1_ll(parms, x, W, b):
    """
    :param parms: np.array [p, q] optimized
    :param x: linspace or data, fixed
    :param W: weights, either np.ones() if no weights have been applied OR iweights Σw=1, fixed
    :param b: location parameter, fixed
    :return: neg. log-likelihood of IB1
    """
    p = parms[0]
    q = parms[1]
    x = np.array(x)
    x = x[x>b]
    n = np.sum(W)
    lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
    sum1 = np.sum(np.log(1-b/x)*W)
    sum2 = np.sum(np.log(x)*W)
    ll = p*n*np.log(b) - n*lnb + (q-1)*sum1 - (p+1)*sum2
    ll = -ll
    return ll

def GB1_ll(parms, x, W, b):
    """
    :param parms: np.array [a, p, q] optimized
    :param x: linspace or data, fixed
    :param W: weights, either np.ones() if no weights have been applied OR iweights Σw=1, fixed
    :param b: location parameter, fixed
    :return: neg. log-likelihood of GB1
    """
    a = parms[0]
    p = parms[1]
    q = parms[2]
    x = np.array(x)
    x = x[x>b]
    n = np.sum(W)
    lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
    sum1 = np.sum(np.log(x)*W)
    sum2 = np.sum(np.log(1-(x/b)**a)*W)
    ll = n*np.log(abs(a)) + (a*p-1)*sum1 + (q-1)*sum2 - n*a*p*np.log(b) - n*lnb
    ll = -ll
    return ll

def GB_ll(parms, x, W, b):
    """
    :param parms=np.array [a, c, p, q] optimized
    :param x: linspace or data, fixed
    :param W: weights, either np.ones() if no weights have been applied OR iweights Σw=1, fixed
    :param b: location parameter, fixed
    :return: neg. log-likelihood of GB1
    """
    a = parms[0]
    c = parms[1]
    p = parms[2]
    q = parms[3]
    x = np.array(x)
    x = x[x>b]
    n = np.sum(W)
    sum1 = np.sum(np.log(x)*W)
    sum2 = np.sum(np.log(1-(1-c)*(x/b)**a)*W)
    sum3 = np.sum(np.log(1+c*((x/b)**a))*W)
    lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
    ll = n*(np.log(np.abs(a)) - a*p*np.log(b) - lnb) + (a*p-1)*sum1 + (q-1)*sum2 - (p+q)*sum3
    ll = -ll
    return ll

""" 
---------------------------------------------------
Fitting Functions
---------------------------------------------------
"""
def Paretofit(x, b, x0=1, weights=np.array([1]), bootstraps=None, method='SLSQP', omit_missings=True,
              verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, suppress_warnings=True,
              return_parameters=False, return_gofs=False, #save_plot=False,
              plot_cosmetics={'bins': 50, 'col_data': 'blue', 'col_fit': 'orange'},
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
    :param bootstraps: amount of bootstraps, default is size k (x>b)
    :param method: # default: SLSQP (local optimization, much faster), 'basinhopping' (global optimization technique)
    :param verbose_bootstrap: display each bootstrap round
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model, default false
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhopping_options: dictionary with optimization options
    :param slsqp_options: dictionary with optimization options
    :return: fitted parameters, gof, ci
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
    # non-weights: filled with ones
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
        return np.min(x) - b

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
                boot_sample = x_backup[boot_sample_idx]
            except:
                boot_sample = [x_backup[i] for i in boot_sample_idx]

            # third, prepare weights: if weights were applied, also select weights based on bootstrapped idx
            if weights_applied is True:
                try:
                    boot_sample_weights = weights[boot_sample_idx]
                except:
                    boot_sample_weights = [weights[i] for i in boot_sample_idx]

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

            # save results
            p_fit = result.x.item(0)

            # save bootstrapped parameters
            p_fit_bs.append(p_fit)

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
            boot_sample_idx = boot_sample_idx.astype(int)

            # second, select x of sample based on bootstrapped idx (sometimes first, faster call not working)
            try:
                boot_sample = x_backup[boot_sample_idx]
            except:
                boot_sample = [x_backup[i] for i in boot_sample_idx]

            # third, prepare weights: if weights were applied, also select weights based on bootstrapped idx
            if weights_applied is True:
                try:
                    boot_sample_weights = weights[boot_sample_idx]
                except:
                    boot_sample_weights = [weights[i] for i in boot_sample_idx]

                # normalize weights: Σw=1
                boot_sample_weights = np.multiply(boot_sample_weights, 1/np.sum(boot_sample_weights))

            # if no weights applied, fill weights with ones
            else:
                boot_sample_weights = np.ones(len(boot_sample))

            # prepare vars for optimization
            x = boot_sample
            W = boot_sample_weights

            minimizer_kwargs = {"method": "SLSQP", "args": (x, W, b),
                                "bounds": (bnd,)} #, "constraints": constr} constraint not needed, because b not optimized

            result = opt.basinhopping(Pareto_ll, x0,
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

            # save results
            p_fit = result.x.item(0)

            # save bootstrapped parameters
            p_fit_bs.append(p_fit)

            bar.update(bootstrapping)
            bootstrapping += 1

        bar.finish()

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
        p_cilo = np.percentile(p_fit_bs, 2.5)
        p_cihi = np.percentile(p_fit_bs, 97.5)
        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'CI(2.5)', 'CI(97.5)', 'n', 'N']
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), '{:.3f}'.format(np.std(p_fit_bs)),
                     '{:.3f}'.format(p_zscore), '{:.3f}'.format(p_pval),
                     '{:.3f}'.format(p_cilo), '{:.3f}'.format(p_cihi), k, N])
        print(tbl)

    # if verbose: print(locals())

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit==True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 100
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
        soe = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).soe
        emp_mean = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).emp_mean
        emp_var = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).emp_var
        pred_mean = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).pred_mean
        pred_var = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).pred_var
        mae = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).mae
        mse = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).mse
        rmse = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).rmse
        rrmse = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).rrmse
        aic = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).aic
        bic = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).bic
        n = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).n
        ll = gof(x=x, x_hat=model, W=weights, b=b, parms=[np.mean(p_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'n', 'N']
            tbl_gof.add_row(['GOF', '{:.3f}'.format(aic), '{:.3f}'.format(bic), '{:.3f}'.format(mae), '{:.3f}'.format(mse),
                             '{:.3f}'.format(rmse), '{:.3f}'.format(rrmse), '{:.3f}'.format(ll), '{:.3f}'.format(soe),
                             '{:.3f}'.format(emp_mean), '{:.3f}'.format(emp_var), '{:.3f}'.format(pred_mean),
                             '{:.3f}'.format(pred_var), '{:d}'.format(k), '{:d}'.format(N)])
            print("\n{}\n".format(tbl_gof))

        if return_gofs:
            return_parameters = False
            return np.mean(p_fit_bs), np.std(p_fit_bs), aic, bic, mae, mse, rmse, rrmse, ll, soe, emp_mean, emp_var, pred_mean, pred_var, k, N, np.percentile(p_fit_bs, 2.5), np.percentile(p_fit_bs, 97.5)

    if return_parameters:
        return np.mean(p_fit_bs), np.std(p_fit_bs), np.percentile(p_fit_bs, 2.5), np.percentile(p_fit_bs, 97.5)

def IB1fit(x, b, x0=(1,1), weights=np.array([1]), bootstraps=None, method='SLSQP', omit_missings=True,
           verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, suppress_warnings=True,
           return_parameters=False, return_gofs=False, #save_plot=False,
           plot_cosmetics={'bins': 50, 'col_data': 'blue', 'col_fit': 'orange'},
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
    :return: fitted parameters, gof, ci
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
    # non-weights: filled with ones
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
    widgets = ['Bootstrapping (IB1)\t', progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=bootstraps).start()
    tbl, tbl_gof = PrettyTable(), PrettyTable()

    def IB1_constraint(parms):
        return (np.min(x)/b)

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
            boot_sample_idx = boot_sample_idx.astype(int)

            # second, select x of sample based on bootstrapped idx (sometimes first, faster call not working)
            try:
                boot_sample = x_backup[boot_sample_idx]
            except:
                boot_sample = [x_backup[i] for i in boot_sample_idx]

            # third, prepare weights: if weights were applied, also select weights based on bootstrapped idx
            if weights_applied is True:
                try:
                    boot_sample_weights = weights[boot_sample_idx]
                except:
                    boot_sample_weights = [weights[i] for i in boot_sample_idx]

                # normalize weights: Σw=1
                boot_sample_weights = np.multiply(boot_sample_weights, 1/np.sum(boot_sample_weights))

            # if no weights applied, fill weights with ones
            else:
                boot_sample_weights = np.ones(len(boot_sample))

            # prepare vars for optimization
            x = boot_sample
            W = boot_sample_weights

            result = opt.minimize(IB1_ll, x0,
                                  args=(x, W, b),
                                  method='SLSQP',
                                  jac=opts['jac'],
                                  bounds=(bnds, bnds,),
                                  constraints=constr,
                                  tol=opts['tol'],
                                  callback=opts['callback'],
                                  options=({'maxiter': opts['maxiter'], 'ftol': opts['ftol'], #'func': opts['func'],
                                            'iprint': opts['iprint'], 'disp': opts['disp'], 'eps': opts['eps']}))

            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))

            # save results
            p_fit, q_fit = result.x.item(0), result.x.item(1)

            # save bootstrapped parameters
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
            boot_sample_idx = boot_sample_idx.astype(int)

            # second, select x of sample based on bootstrapped idx (sometimes first, faster call not working)
            try:
                boot_sample = x_backup[boot_sample_idx]
            except:
                boot_sample = [x_backup[i] for i in boot_sample_idx]

            # third, prepare weights: if weights were applied, also select weights based on bootstrapped idx
            if weights_applied is True:
                try:
                    boot_sample_weights = weights[boot_sample_idx]
                except:
                    boot_sample_weights = [weights[i] for i in boot_sample_idx]

                # normalize weights: Σw=1
                boot_sample_weights = np.multiply(boot_sample_weights, 1/np.sum(boot_sample_weights))

            # if no weights applied, fill weights with ones
            else:
                boot_sample_weights = np.ones(len(boot_sample))

            # prepare vars for optimization
            x = boot_sample
            W = boot_sample_weights

            minimizer_kwargs = {"method": "SLSQP", "args": (x, W, b),
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

            # save results
            p_fit, q_fit = result.x.item(0), result.x.item(1)

            # save bootstrapped parameters
            p_fit_bs.append(p_fit)
            q_fit_bs.append(q_fit)

            bar.update(bootstrapping)
            bootstrapping += 1

        bar.finish()

    # set back x, weights
    x = x_backup
    weights = weights_backup

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
        p_cilo = np.percentile(p_fit_bs, 2.5)
        p_cihi = np.percentile(p_fit_bs, 97.5)
        q_cilo = np.percentile(q_fit_bs, 2.5)
        q_cihi = np.percentile(q_fit_bs, 97.5)

        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'CI(2.5)', 'CI(97.5)', 'n', 'N']
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), '{:.3f}'.format(np.std(p_fit_bs)),
                     '{:.3f}'.format(p_zscore), "{:.3f}".format(p_pval),
                     '{:.3f}'.format(p_cilo), '{:.3f}'.format(p_cihi),
                     k, N])
        tbl.add_row(['q', '{:.3f}'.format(np.mean(q_fit_bs)), '{:.3f}'.format(np.std(q_fit_bs)),
                     '{:.3f}'.format(q_zscore), "{:.3f}".format(q_pval),
                     '{:.3f}'.format(q_cilo), '{:.3f}'.format(q_cihi),
                     k, N])
        print(tbl)

    # if verbose: print(locals())

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit==True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 100
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
        soe = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).soe
        emp_mean = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_mean
        emp_var = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_var
        pred_mean = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_mean
        pred_var = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_var
        mae = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'n', 'N']
            tbl_gof.add_row(['GOF', '{:.3f}'.format(aic), '{:.3f}'.format(bic), '{:.3f}'.format(mae), '{:.3f}'.format(mse),
                             '{:.3f}'.format(rmse), '{:.3f}'.format(rrmse), '{:.3f}'.format(ll), '{:.3f}'.format(soe),
                             '{:.3f}'.format(emp_mean), '{:.3f}'.format(emp_var), '{:.3f}'.format(pred_mean),
                             '{:.3f}'.format(pred_var), '{:d}'.format(k), '{:d}'.format(N)])
            print("\n{}\n".format(tbl_gof))

        if return_gofs:
            return_parameters = False
            return np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs), \
                   aic, bic, mae, mse, rmse, rrmse, ll, soe, emp_mean, emp_var, pred_mean, pred_var, k, N, \
                   np.percentile(p_fit_bs, 2.5), np.percentile(p_fit_bs, 97.5), np.percentile(q_fit_bs, 2.5), np.percentile(q_fit_bs, 97.5)

    if return_parameters:
        return np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs), \
               np.percentile(p_fit_bs, 2.5), np.percentile(p_fit_bs, 97.5), np.percentile(q_fit_bs, 2.5), np.percentile(q_fit_bs, 97.5)

def GB1fit(x, b, x0=(-.1, 1, 1), weights=np.array([1]), bootstraps=None, method='SLSQP', omit_missings=True,
           verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, suppress_warnings=True,
           return_parameters=False, return_gofs=False, #save_plot=False,
           plot_cosmetics={'bins': 50, 'col_data': 'blue', 'col_fit': 'orange'},
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
    :return: fitted parameters, gof, ci
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
    # non-weights: filled with ones
    k = len(x[x>b])
    if weights_applied is True:
        xlargerb_index = np.where(x > b)[0]
        weights = np.array(weights)[xlargerb_index]
        N = int(np.sum(weights))
    else:
        # As no weights are specified, are not needed anymore -> set vector W to 1
        weights = np.ones(k)
        N = int(np.sum(weights))

    # backup
    x = x_backup = x[x>b]
    weights_backup = weights

    # create list with indexes of x (needed for bootstrapping)
    x_index = np.arange(0, k, 1)

    # bootstraps (default: size k)
    if bootstraps is None:
        bootstraps = k

    # prepare progress bar and printed tables
    widgets = ['Bootstrapping (GB1)\t', progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=bootstraps).start()
    tbl, tbl_gof = PrettyTable(), PrettyTable()

    # Note: constraints not binding, SLSQP finds Minimum without
    # def GB1_constraint(parms):
    #     a = parms[0]
    #     return (np.min(x)/b)**a
    #
    # constr = {'type': 'ineq', 'fun': GB1_constraint}
    a_bnd, bnds = (-5, -1e-10), (10**-14, np.inf)
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
            opts['maxiter'] = 600
        if 'ftol' not in opts.keys():
            opts['ftol'] = 1e-16
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
                boot_sample = x_backup[boot_sample_idx]
            except:
                boot_sample = [x_backup[i] for i in boot_sample_idx]

            # third, prepare weights: if weights were applied, also select weights based on bootstrapped idx
            if weights_applied is True:
                try:
                    boot_sample_weights = weights[boot_sample_idx]
                except:
                    boot_sample_weights = [weights[i] for i in boot_sample_idx]

                # normalize weights: Σw=1
                boot_sample_weights = np.multiply(boot_sample_weights, 1/np.sum(boot_sample_weights))

            # if no weights applied, fill weights with ones
            else:
                boot_sample_weights = np.ones(len(boot_sample))

            # prepare vars for optimization
            x = boot_sample
            W = boot_sample_weights

            result = opt.minimize(GB1_ll, x0,
                                  args=(x, W, b),
                                  method='SLSQP',
                                  jac=opts['jac'],
                                  bounds=(a_bnd, bnds, bnds,),
                                  #constraints=constr,
                                  tol=opts['tol'],
                                  callback=opts['callback'],
                                  options=({'maxiter': opts['maxiter'], 'ftol': opts['ftol'], #'func': opts['func'],
                                            'iprint': opts['iprint'], 'disp': opts['disp'], 'eps': opts['eps']}))

            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))

            # save results
            a_fit, p_fit, q_fit = result.x.item(0), result.x.item(1), result.x.item(2)

            # save bootstrapped parameters
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
            boot_sample_idx = boot_sample_idx.astype(int)

            # second, select x of sample based on bootstrapped idx (sometimes first, faster call not working)
            try:
                boot_sample = x_backup[boot_sample_idx]
            except:
                boot_sample = [x_backup[i] for i in boot_sample_idx]

            # third, prepare weights: if weights were applied, also select weights based on bootstrapped idx
            if weights_applied is True:
                try:
                    boot_sample_weights = weights[boot_sample_idx]
                except:
                    boot_sample_weights = [weights[i] for i in boot_sample_idx]

                # normalize weights: Σw=1
                boot_sample_weights = np.multiply(boot_sample_weights, 1/np.sum(boot_sample_weights))

            # if no weights applied, fill weights with ones
            else:
                boot_sample_weights = np.ones(len(boot_sample))

            # prepare vars for optimization
            x = boot_sample
            W = boot_sample_weights

            minimizer_kwargs = {"method": "SLSQP", "args": (x, W, b),
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

            # save results
            a_fit, p_fit, q_fit = result.x.item(0), result.x.item(1), result.x.item(2)

            # save bootstrapped parameters
            a_fit_bs.append(a_fit)
            p_fit_bs.append(p_fit)
            q_fit_bs.append(q_fit)

            bar.update(bootstrapping)
            bootstrapping += 1

        bar.finish()

    # set back x, weights
    x = x_backup
    weights = weights_backup

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
        a_cilo = np.percentile(a_fit_bs, 2.5)
        a_cihi = np.percentile(a_fit_bs, 97.5)
        p_cilo = np.percentile(p_fit_bs, 2.5)
        p_cihi = np.percentile(p_fit_bs, 97.5)
        q_cilo = np.percentile(q_fit_bs, 2.5)
        q_cihi = np.percentile(q_fit_bs, 97.5)

        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'CI(2.5)', 'CI(97.5)', 'n', 'N']
        tbl.add_row(['a', '{:.3f}'.format(np.mean(a_fit_bs)), ('{:.3f}'.format(np.std(a_fit_bs))),
                     '{:.3f}'.format(a_zscore), '{:.3f}'.format(a_pval),
                     '{:.3f}'.format(a_cilo), '{:.3f}'.format(a_cihi), k, N])
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), ('{:.3f}'.format(np.std(p_fit_bs))),
                     '{:.3f}'.format(p_zscore), '{:.3f}'.format(p_pval),
                     '{:.3f}'.format(p_cilo), '{:.3f}'.format(p_cihi), k, N])
        tbl.add_row(['q', '{:.3f}'.format(np.mean(q_fit_bs)), ('{:.3f}'.format(np.std(q_fit_bs))),
                     '{:.3f}'.format(q_zscore), '{:.3f}'.format(q_pval),
                     '{:.3f}'.format(q_cilo), '{:.3f}'.format(q_cihi), k, N])
        print(tbl)

    # if verbose: print(locals())

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit==True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 100
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
        soe = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).soe
        emp_mean = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_mean
        emp_var = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_var
        pred_mean = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_mean
        pred_var = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_var
        mae = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'n', 'N']
            tbl_gof.add_row(['GOF', '{:.3f}'.format(aic), '{:.3f}'.format(bic), '{:.3f}'.format(mae), '{:.3f}'.format(mse),
                             '{:.3f}'.format(rmse), '{:.3f}'.format(rrmse), '{:.3f}'.format(ll), '{:.3f}'.format(soe),
                             '{:.3f}'.format(emp_mean), '{:.3f}'.format(emp_var), '{:.3f}'.format(pred_mean),
                             '{:.3f}'.format(pred_var), '{:d}'.format(k), '{:d}'.format(N)])
            print("\n{}\n".format(tbl_gof))

        if return_gofs:
            return_parameters = False
            return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs), \
                   aic, bic, mae, mse, rmse, rrmse, ll, soe, emp_mean, emp_var, pred_mean, pred_var, k, N,\
                   np.percentile(a_fit_bs, 2.5), np.percentile(a_fit_bs, 97.5), np.percentile(p_fit_bs, 2.5), np.percentile(p_fit_bs, 97.5),\
                   np.percentile(q_fit_bs, 2.5), np.percentile(q_fit_bs, 97.5)

    if return_parameters:
        return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs),
        np.percentile(a_fit_bs, 2.5), np.percentile(a_fit_bs, 97.5), np.percentile(p_fit_bs, 2.5), np.percentile(p_fit_bs, 97.5), \
        np.percentile(q_fit_bs, 2.5), np.percentile(q_fit_bs, 97.5)

def GBfit(x, b, x0=(-.1, .1, 1, 1), weights=np.array([1]), bootstraps=None, method='SLSQP', omit_missings=True,
          verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, suppress_warnings=True,
          return_parameters=False, return_gofs=False, #save_plot=False,
          plot_cosmetics={'bins': 50, 'col_data': 'blue', 'col_fit': 'orange'},
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
    :return: fitted parameters, gof, ci
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
    # non-weights: filled with ones
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
    widgets = ['Bootstrapping (GB)\t', progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=bootstraps).start()
    tbl, tbl_gof = PrettyTable(), PrettyTable()

    # Note: constraints not binding, SLSQP finds Minimum without
    # def GB_constraint1(parms):
    #     a = parms[0]
    #     c = parms[1]
    #     return (b**a)/(1-c) - np.min(x)**a
    #
    # def GB_constraint2(parms):
    #     a = parms[0]
    #     c = parms[1]
    #     return (b**a)/(1-c) - np.max(x)**a
    #
    # constr = ({'type': 'ineq', 'fun': GB_constraint1},
    #           {'type': 'ineq', 'fun': GB_constraint2})

    a_bnd, c_bnd, bnds = (-5, -1e-10), (0, 1), (10**-14, np.inf)
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
            boot_sample_idx = boot_sample_idx.astype(int)

            # second, select x of sample based on bootstrapped idx (sometimes first, faster call not working)
            try:
                boot_sample = x_backup[boot_sample_idx]
            except:
                boot_sample = [x_backup[i] for i in boot_sample_idx]

            # third, prepare weights: if weights were applied, also select weights based on bootstrapped idx
            if weights_applied is True:
                try:
                    boot_sample_weights = weights[boot_sample_idx]
                except:
                    boot_sample_weights = [weights[i] for i in boot_sample_idx]

                # normalize weights: Σw=1
                boot_sample_weights = np.multiply(boot_sample_weights, 1/np.sum(boot_sample_weights))

            # if no weights applied, fill weights with ones
            else:
                boot_sample_weights = np.ones(len(boot_sample))

            # prepare vars for optimization
            x = boot_sample
            W = boot_sample_weights

            result = opt.minimize(GB_ll, x0,
                                  args=(x, W, b),
                                  method='SLSQP',
                                  jac=opts['jac'],
                                  bounds=(a_bnd, c_bnd, bnds, bnds,),
                                  #constraints=constr,
                                  tol=opts['tol'],
                                  callback=opts['callback'],
                                  options=({'maxiter': opts['maxiter'], 'ftol': opts['ftol'], #'func': opts['func'],
                                            'iprint': opts['iprint'], 'disp': opts['disp'], 'eps': opts['eps']}))

            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))

            # save results
            a_fit, c_fit, p_fit, q_fit = result.x.item(0), result.x.item(1), result.x.item(2), result.x.item(3)

            # save bootstrapped parameters
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
            boot_sample_idx = boot_sample_idx.astype(int)

            # second, select x of sample based on bootstrapped idx (sometimes first, faster call not working)
            try:
                boot_sample = x_backup[boot_sample_idx]
            except:
                boot_sample = [x_backup[i] for i in boot_sample_idx]

            # third, prepare weights: if weights were applied, also select weights based on bootstrapped idx
            if weights_applied is True:
                try:
                    boot_sample_weights = weights[boot_sample_idx]
                except:
                    boot_sample_weights = [weights[i] for i in boot_sample_idx]

                # normalize weights: Σw=1
                boot_sample_weights = np.multiply(boot_sample_weights, 1/np.sum(boot_sample_weights))

            # if no weights applied, fill weights with ones
            else:
                boot_sample_weights = np.ones(len(boot_sample))

            # prepare vars for optimization
            x = boot_sample
            W = boot_sample_weights

            minimizer_kwargs = {"method": "SLSQP", "args": (x, W, b),
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

            # save results
            a_fit, c_fit, p_fit, q_fit = result.x.item(0), result.x.item(1), result.x.item(2), result.x.item(3)

            # save bootstrapped parameters
            a_fit_bs.append(a_fit)
            c_fit_bs.append(c_fit)
            p_fit_bs.append(p_fit)
            q_fit_bs.append(q_fit)

            bar.update(bootstrapping)
            bootstrapping += 1

        bar.finish()

    # set back x, weights
    x = x_backup
    weights = weights_backup

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
        a_cilo = np.percentile(a_fit_bs, 2.5)
        a_cihi = np.percentile(a_fit_bs, 97.5)
        c_cilo = np.percentile(c_fit_bs, 2.5)
        c_cihi = np.percentile(c_fit_bs, 97.5)
        p_cilo = np.percentile(p_fit_bs, 2.5)
        p_cihi = np.percentile(p_fit_bs, 97.5)
        q_cilo = np.percentile(q_fit_bs, 2.5)
        q_cihi = np.percentile(q_fit_bs, 97.5)

        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'CI(2.5)', 'CI(97.5)', 'n', 'N']
        tbl.add_row(['a', '{:.3f}'.format(np.mean(a_fit_bs)), '{:.3f}'.format(np.std(a_fit_bs)),
                     '{:.3f}'.format(a_zscore), '{:.3f}'.format(a_pval),
                     '{:.3f}'.format(a_cilo), '{:.3f}'.format(a_cihi), k, N])
        tbl.add_row(['c', '{:.3f}'.format(np.mean(c_fit_bs)), '{:.3f}'.format(np.std(c_fit_bs)),
                     '{:.3f}'.format(c_zscore), '{:.3f}'.format(c_pval),
                     '{:.3f}'.format(c_cilo), '{:.3f}'.format(c_cihi), k, N])
        tbl.add_row(['p', '{:.3f}'.format(np.mean(p_fit_bs)), '{:.3f}'.format(np.std(p_fit_bs)),
                     '{:.3f}'.format(p_zscore), '{:.3f}'.format(p_pval),
                     '{:.3f}'.format(p_cilo), '{:.3f}'.format(p_cihi), k, N])
        tbl.add_row(['q', '{:.3f}'.format(np.mean(q_fit_bs)), '{:.3f}'.format(np.std(q_fit_bs)),
                     '{:.3f}'.format(q_zscore), '{:.3f}'.format(q_pval),
                     '{:.3f}'.format(q_cilo), '{:.3f}'.format(q_cihi), k, N])
        print(tbl)

    # if verbose: print(locals())

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit=True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 100
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
        soe = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).soe
        emp_mean = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_mean
        emp_var = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_var
        pred_mean = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_mean
        pred_var = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_var
        mae = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=x, x_hat=model, b=b, W=weights, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
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
                   np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs), \
                   aic, bic, mae, mse, rmse, rrmse, ll, soe, emp_mean, emp_var, pred_mean, pred_var, k, N, \
                   np.percentile(a_fit_bs, 2.5), np.percentile(a_fit_bs, 97.5), np.percentile(c_fit_bs, 2.5), np.percentile(c_fit_bs, 97.5), \
                   np.percentile(p_fit_bs, 2.5), np.percentile(p_fit_bs, 97.5), np.percentile(q_fit_bs, 2.5), np.percentile(q_fit_bs, 97.5)

    if return_parameters:
        return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(c_fit_bs), np.std(c_fit_bs), np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs), \
               np.percentile(a_fit_bs, 2.5), np.percentile(a_fit_bs, 97.5), np.percentile(c_fit_bs, 2.5), np.percentile(c_fit_bs, 97.5), \
               np.percentile(p_fit_bs, 2.5), np.percentile(p_fit_bs, 97.5), np.percentile(q_fit_bs, 2.5), np.percentile(q_fit_bs, 97.5)

""" 
---------------------------------------------------
Pareto branch fitting
---------------------------------------------------
"""

def Paretobranchfit(x, b, x0=np.array([-.1,.1,1,1]), weights=np.array([1]), bootstraps=None,
                    method='SLSQP', rejection_criterion=['LRtest', 'AIC'], alpha=.05,
                    verbose_bootstrap=False, verbose_single=False, verbose=True, verbose_parms=False,
                    fit=False, plot=False, return_bestmodel=False, return_all=False, #save_all_plots=False,
                    suppress_warnings=True, omit_missings=True, ci=False,
          plot_cosmetics={'bins': 250, 'col_data': 'blue', 'col_fit': 'orange'},
    basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                         'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
          slsqp_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                         'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    This function fits the Pareto branch upwards, starting from the bottom with the Pareto distribution.
    This function is a wrapper that calls all above fitting functions, runs all optimizations and compares the
    parameters according the Pareto branch parameter restrictions.
    Comparing the Pareto distribution to the IB1, the LRtest (AIC, AIC_alternative) decides, whether there is a
    improvement in fit. If the IB1 delivers a better fit, we go one level upwards and compare the IB1 to GB1 and so on.
    :param x: as above
    :param b: as above
    :param x0: either pass an 1x5 array (GB init guess structure) OR pass
               [[p_guess], [p_guess, q_guess], [a_guess, p_guess, q_guess], [a_guess, c_guess, p_guess, q_guess]]
    :param weights: as above
    :param bootstraps: either 1x1 OR 1x2 array (1st arg: Pareto+IB1, 2nd arg: GB1+GB) OR pass 1x4 array [Pareto_bs, IB1_bs, GB1_bs, GB_bs]
    :param method: as above
    :param verbose_bootstrap: as above
    :param rejection_criterion: LRtest or AIC (as recommended by McDonald)
    :param verbose: table with parameters and another with gofs, display only final result
    :param verbose_single: display each optimization results
    :param ci: display cis of fitted bootstrapped parameters
    :param alpha: significance level of LRtest, default: 5%
    :param fit: as above
    :param plot: as above
    :param return_parameters: as above and parameters, se, ci of all distributions are returned (see Excel file in //Testing/)
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
    Pareto_fit = Paretofit(x=x, b=b, x0=Pareto_x0, weights=weights, bootstraps=Pareto_bs, method=method,
                           return_parameters=True, return_gofs=True, ci=True, verbose=verbose_single, omit_missings=omit_missings,
                           verbose_bootstrap=verbose_bootstrap, fit=fit, plot=plot, suppress_warnings=suppress_warnings,
                           plot_cosmetics=plt_cosm, basinhopping_options=bh_opts, slsqp_options=slsqp_opts)

    if verbose_parms: print(Pareto_fit)

    IB1_fit = IB1fit(x=x, b=b, x0=IB1_x0, weights=weights, bootstraps=IB1_bs, method=method,
                     return_parameters=True, return_gofs=True, ci=True, verbose=verbose_single, omit_missings=omit_missings,
                     verbose_bootstrap=verbose_bootstrap, fit=fit, plot=plot, suppress_warnings=suppress_warnings,
                     plot_cosmetics=plt_cosm, basinhopping_options=bh_opts, slsqp_options=slsqp_opts)

    if verbose_parms: print(IB1_fit)

    GB1_fit = GB1fit(x=x, b=b, x0=GB1_x0, weights=weights, bootstraps=GB1_bs, method=method,
                     return_parameters=True, return_gofs=True, ci=True, verbose=verbose_single, omit_missings=omit_missings,
                     verbose_bootstrap=verbose_bootstrap, fit=fit, plot=plot, suppress_warnings=suppress_warnings,
                     plot_cosmetics=plt_cosm, basinhopping_options=bh_opts, slsqp_options=slsqp_opts)

    if verbose_parms: print(GB1_fit)

    GB_fit = GBfit(x=x, b=b, x0=GB_x0, weights=weights, bootstraps=GB_bs, method=method,
                   return_parameters=True, return_gofs=True, ci=True, verbose=verbose_single, omit_missings=omit_missings,
                   verbose_bootstrap=verbose_bootstrap, fit=fit, plot=plot, suppress_warnings=suppress_warnings,
                   plot_cosmetics=plt_cosm, basinhopping_options=bh_opts, slsqp_options=slsqp_opts)

    if verbose_parms: print(GB_fit)

    # unpack parameters
    p_fit1, p_se1 = Pareto_fit[:2]
    p_fit2, p_se2, q_fit2, q_se2 = IB1_fit[:4]
    a_fit3, a_se3, p_fit3, p_se3, q_fit3, q_se3 = GB1_fit[:6]
    a_fit4, a_se4, c_fit4, c_se4, p_fit4, p_se4, q_fit4, q_se4 = GB_fit[:8]

    # unpack CIs
    p_cilo1, p_cihi1 = Pareto_fit[16:]
    p_cilo2, p_cihi2, q_cilo2, q_cihi2 = IB1_fit[18:]
    a_cilo3, a_cihi3, p_cilo3, p_cihi3, q_cilo3, q_cihi3 = GB1_fit[20:]
    a_cilo4, a_cihi4, c_cilo4, c_cihi4, p_cilo4, p_cihi4, q_cilo4, q_cihi4 = GB_fit[22:]

    # run rejection based on LRtest
    if rejection_criterion == 'LRtest' or 'LRtest' in rejection_criterion:
        # alpha = .05
        # 1. LRtest IB1 restriction q=1
        LRtestIB1_restrict = LRtest(LL1=IB1(x=x, W=weights, b=b, p=p_fit2, q=1).LL,
                                    LL2=IB1(x=x, W=weights, b=b, p=p_fit2, q=q_fit2).LL,
                                    df=1, verbose=False) #df: # of tested parms
        # 2. LRtest GB1 restriction a=-1
        LRtestGB1_restrict = LRtest(LL1=GB1(x=x, W=weights, b=b, a=-1, p=p_fit3, q=q_fit3).LL,
                                    LL2=GB1(x=x, W=weights, b=b, a=a_fit3, p=p_fit3, q=q_fit3).LL,
                                    df=1, verbose=False) #df: # of tested parms
        # 3. LRtest GB restriction c=0
        LRtestGB_restrict = LRtest(LL1=GB(x=x, W=weights, b=b, a=a_fit4, c=0, p=p_fit4, q=q_fit4).LL,
                                   LL2=GB(x=x, W=weights, b=b, a=a_fit4, c=c_fit4, p=p_fit4, q=q_fit4).LL,
                                   df=1, verbose=False) #df: # of tested parms

        # LR testing procedure (paper chp. 2.3)
        Pareto_bm = IB1_bm = GB1_bm = GB_bm = Pareto_marker = IB1_marker = GB1_marker = GB_marker = '--'
        GB_remaining = False
        if LRtestIB1_restrict.pval < alpha:
            # LRtestIB1_restrict.pval is smaller than alpha, reject H0,
            # q=1 not valid, go one parm. level up in pareto branch and test next GB1 restriction
            if LRtestGB1_restrict.pval < alpha:
                # LRtestGB1_restrict.pval is smaller than alpha, reject H0,
                # a=-1 not valid, go one parm. level up in pareto branch and test next GB restriction
                bestmodel_LR, Pareto_marker = '--', 'IB1'
                if LRtestGB_restrict.pval < alpha:
                    # LRtestGB_restrict.pval is smaller than alpha, reject H0,
                    # c=0 not valid, go one parm. level up in pareto branch and test next GB1 restriction
                    GB_bm, bestmodel_LR, GB_marker, GB_remaining = 'GB', 'GB', 'XX', True
                    Pareto_marker = IB1_marker = GB1_marker = GB1_marker = '--'
                else:
                    GB1_bm, bestmodel_LR, GB1_marker = 'GB1', 'GB1', 'XX'
                    IB1_marker = Pareto_marker = '--'
            else:
                IB1_bm, bestmodel_LR, Pareto_marker, IB1_marker = 'IB1', 'IB1', '--', 'XX'
        else:
            Pareto_bm, bestmodel_LR, Pareto_marker = 'Pareto', 'Pareto', 'XX'

        # save LRtest results to tbl
        tbl = PrettyTable()
        tbl.field_names = ['test restriction', 'H0', 'LR test', '', 'stop', 'best model']
        tbl.add_row(['IB1 restriction', 'q=1', 'chi2({}) = '.format(1), '{:.3f}'.format(LRtestIB1_restrict.LR), '{}'.format(Pareto_marker), '{}'.format(Pareto_bm)])
        tbl.add_row(['', '', 'Prob > chi2', '{:.3f}'.format(LRtestIB1_restrict.pval), '{}'.format(Pareto_marker), '{}'.format(Pareto_bm)])
        tbl.add_row(['GB1 restriction', 'a=-1', 'chi2({}) = '.format(1), '{:.3f}'.format(LRtestGB1_restrict.LR), '{}'.format(IB1_marker), '{}'.format(IB1_bm)])
        tbl.add_row(['', '', 'Prob > chi2', '{:.3f}'.format(LRtestGB1_restrict.pval), '{}'.format(IB1_marker), '{}'.format(IB1_bm)])
        tbl.add_row(['GB restriction', 'c=0', 'chi2({}) = '.format(1), '{:.3f}'.format(LRtestGB_restrict.LR), '{}'.format(GB1_marker), '{}'.format(GB1_bm)])
        tbl.add_row(['', '', 'Prob > chi2', '{:.3f}'.format(LRtestGB_restrict.pval), '{}'.format(GB1_marker), '{}'.format(GB1_bm)])
        if GB_remaining:
            tbl.add_row(['GB', '', '', '', '{}'.format(GB_marker), '{}'.format(GB_bm)])
        print('\n{}'.format(tbl))

        # save LR fit results
        Pareto_fit_LR, IB1_fit_LR, GB1_fit_LR, GB_fit_LR = Pareto_fit, IB1_fit, GB1_fit, GB_fit

    if 'AIC' == rejection_criterion or 'AIC' in rejection_criterion:
        # AIC alternative method #2 (paper chp. 2.4)
        # unpack aic
        Pareto_aic = Pareto_fit[2]
        IB1_aic = IB1_fit[4]
        GB1_aic = GB1_fit[6]
        GB_aic = GB_fit[8]

        # calculate AICs of models with restriction of Pareto branch (Note: x_hat not needed, x_hat=x placeholder)
        IB1_aic_restrict = gof(x=x, x_hat=x, b=b, W=weights, parms=[IB1_fit[0], 1]).aic
        GB1_aic_restrict = gof(x=x, x_hat=x, b=b, W=weights, parms=[-1, GB1_fit[2], GB1_fit[4]]).aic
        GB_aic_restrict  = gof(x=x, x_hat=x, b=b, W=weights, parms=[GB_fit[0], 0, GB_fit[4], GB_fit[6]]).aic

        # AIC testing procedure
        Pareto_bm = IB1_bm = GB1_bm = GB_bm = Pareto_marker = IB1_marker = GB1_marker = GB_marker = '--'
        GB_remaining = False
        if IB1_aic_restrict > IB1_aic:
            if GB1_aic_restrict > GB1_aic:
                if GB_aic_restrict > GB_aic:
                    GB_bm, bestmodel_AIC, GB_marker, GB_remaining = 'GB', 'GB', 'XX', True
                    Pareto_marker = IB1_marker = GB1_marker = GB1_marker = '--'
                else:
                    GB1_bm, bestmodel_AIC, GB1_marker = 'GB1', 'GB1', 'XX'
                    IB1_marker = Pareto_marker = '--'
            else:
                IB1_bm, bestmodel_AIC, Pareto_marker, IB1_marker = 'IB1', 'IB1', '--', 'XX'
        else:
            Pareto_bm, bestmodel_AIC, Pareto_marker = 'Pareto', 'Pareto', 'XX'

        # save LRtest results to tbl
        tbl = PrettyTable()
        tbl.field_names = ['AIC comparison', 'AIC (restricted)', 'AIC (full)', 'stop', 'best model']
        tbl.add_row(['IB1(pfit, q=1) vs. IB1(pfit, qfit)', '{:.3f}'.format(IB1_aic_restrict), '{:.3f}'.format(IB1_aic), '{}'.format(Pareto_marker), '{}'.format(Pareto_bm)])
        tbl.add_row(['GB1(a=-1, pfit, qfit) vs.', '', '', '', ''])
        tbl.add_row(['GB1(afit, pfit, qfit)', '{:.3f}'.format(GB1_aic_restrict), '{:.3f}'.format(GB1_aic), '{}'.format(IB1_marker), '{}'.format(IB1_bm)])
        tbl.add_row(['GB(afit, c=0, pfit, qfit) vs.', '', '', '', ''])
        tbl.add_row(['GB(afit, cfit, pfit, qfit)', '{:.3f}'.format(GB_aic_restrict), '{:.3f}'.format(GB_aic), '{}'.format(GB1_marker), '{}'.format(GB1_bm)])

        if GB_remaining:
            tbl.add_row(['GB', '{:.3f}'.format(GB_aic), '', '{}'.format(GB_marker), '{}'.format(GB_bm)])

        print('\n{}'.format(tbl))

        # save AIC fit results
        Pareto_fit_AIC, IB1_fit_AIC, GB1_fit_AIC, GB_fit_AIC = Pareto_fit, IB1_fit, GB1_fit, GB_fit

    # run rejection based on AIC alternative (AIC method #1: paper Appendix)
    if 'AIC_alternative' == rejection_criterion or 'AIC_alternative' in rejection_criterion:

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
                if GB1_aic > GB_aic:
                    GB_bm, bestmodel_AIC_altern, GB_marker, GB_remaining = 'GB', 'GB', 'XX', True
                    Pareto_marker = IB1_marker = GB1_marker = GB1_marker = '--'
                else:
                    GB1_bm, bestmodel_AIC_altern, IB1_marker, GB1_marker = 'GB1', 'GB1', '--', 'XX'
            else:
                IB1_bm, bestmodel_AIC_altern, Pareto_marker, IB1_marker = 'IB1', 'IB1', '--', 'XX'
        else:
            Pareto_bm, bestmodel_AIC_altern, Pareto_marker = 'Pareto', 'Pareto', 'XX'

        # save LRtest results to tbl
        tbl = PrettyTable()
        tbl.field_names = ['comparison', 'AIC1', 'AIC2', 'stop', 'best model']
        tbl.add_row(['Pareto vs IB1', '{:.3f}'.format(Pareto_aic), '{:.3f}'.format(IB1_aic), '{}'.format(Pareto_marker), '{}'.format(Pareto_bm)])
        tbl.add_row(['IB1 vs GB1', '{:.3f}'.format(IB1_aic), '{:.3f}'.format(GB1_aic), '{}'.format(IB1_marker), '{}'.format(IB1_bm)])
        tbl.add_row(['GB1 vs GB', '{:.3f}'.format(GB1_aic), '{:.3f}'.format(GB_aic), '{}'.format(GB1_marker), '{}'.format(GB1_bm)])

        if GB_remaining:
            tbl.add_row(['GB', '{:.3f}'.format(GB_aic), '', '{}'.format(GB_marker), '{}'.format(GB_bm)])

        print('\n{}'.format(tbl))

        # save AIC fit results
        Pareto_fit_AIC_altern, IB1_fit_AIC_altern, GB1_fit_AIC_altern, GB_fit_AIC_altern = Pareto_fit, IB1_fit, GB1_fit, GB_fit

    if verbose:
        tbl_parms = PrettyTable()
        tbl_parms.field_names = ['parameter', 'Pareto', 'IB1', 'GB1', 'GB']
        tbl_parms.add_row(['a', '-', '-', '{:.3f}'.format(GB1_fit[0]), '{:.3f}'.format(GB_fit[0])])
        if ci:
            tbl_parms.add_row(['', '', '', '[{:.3f};{:.3f}]'.format(a_cilo3, a_cihi3), '[{:.3f};{:.3f}]'.format(a_cilo4, a_cihi4)])
        tbl_parms.add_row(['', '', '', '({:.3f})'.format(GB1_fit[1]), '({:.3f})'.format(GB_fit[1])])
        tbl_parms.add_row(['c', '-', '-', '-', '{:.3f}'.format(GB_fit[2])])
        if ci:
            tbl_parms.add_row(['', '', '', '', '[{:.3f};{:.3f}]'.format(c_cilo4, c_cihi4)])
        tbl_parms.add_row(['', '', '', '', '({:.3f})'.format(GB_fit[3])])
        tbl_parms.add_row(['p', '{:.3f}'.format(Pareto_fit[0]), '{:.3f}'.format(IB1_fit[0]), '{:.3f}'.format(GB1_fit[2]), '{:.3f}'.format(GB_fit[4])])
        if ci:
            tbl_parms.add_row(['', '[{:.3f},{:.3f}]'.format(p_cilo1, p_cihi1), '[{:.3f},{:.3f}]'.format(p_cilo2, p_cihi2), '[{:.3f},{:.3f}]'.format(p_cilo3, p_cihi3), '[{:.3f},{:.3f}]'.format(p_cilo4, p_cihi4)])
        tbl_parms.add_row(['', '({:.3f})'.format(Pareto_fit[1]), '({:.3f})'.format(IB1_fit[1]), '({:.3f})'.format(GB1_fit[3]), '({:.3f})'.format(GB_fit[5])])
        tbl_parms.add_row(['q', '-', '{:.3f}'.format(IB1_fit[2]), '{:.3f}'.format(GB1_fit[4]), '{:.3f}'.format(GB_fit[6])])
        if ci:
            tbl_parms.add_row(['', '', '[{:.3f},{:.3f}]'.format(q_cilo2, q_cihi2), '[{:.3f},{:.3f}]'.format(q_cilo3, q_cihi3), '[{:.3f},{:.3f}]'.format(q_cilo4, q_cihi4)])
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

        printout_LR, printout_AIC, printout_AIC_alternative = list(), list(), list()

        if rejection_criterion == 'LRtest' or 'LRtest' in rejection_criterion:
            if bestmodel_LR == 'Pareto':
                printout_LR = 'Pareto_best', Pareto_fit_LR
            if bestmodel_LR == 'IB1':
                printout_LR = 'IB1_best', IB1_fit_LR
            if bestmodel_LR == 'GB1':
                printout_LR = 'GB1_best', GB1_fit_LR
            if bestmodel_LR == 'GB':
                printout_LR = 'GB_best', GB_fit_LR

        if rejection_criterion == 'AIC' or 'AIC' in rejection_criterion:
            if bestmodel_AIC == 'Pareto':
                printout_AIC = 'Pareto_best', Pareto_fit_AIC
            if bestmodel_AIC == 'IB1':
                printout_AIC = 'IB1_best', IB1_fit_AIC
            if bestmodel_AIC == 'GB1':
                printout_AIC = 'GB1_best', GB1_fit_AIC
            if bestmodel_AIC == 'GB':
                printout_AIC = 'GB_best', GB_fit_AIC

        if rejection_criterion == 'AIC_alternative' or 'AIC_alternative' in rejection_criterion:
            if bestmodel_AIC_altern == 'Pareto':
                printout_AIC_alternative = 'Pareto_best', Pareto_fit_AIC_altern
            if bestmodel_AIC_altern == 'IB1':
                printout_AIC_alternative = 'IB1_best', IB1_fit_AIC_altern
            if bestmodel_AIC_altern == 'GB1':
                printout_AIC_alternative = 'GB1_best', GB1_fit_AIC_altern
            if bestmodel_AIC_altern == 'GB':
                printout_AIC_alternative = 'GB_best', GB_fit_AIC_altern

        return printout_LR, printout_AIC, printout_AIC_alternative

    if return_all:
        return Pareto_fit, IB1_fit, GB1_fit, GB_fit

""" 
---------------------------------------------------
Pareto and IB1: se extracting
---------------------------------------------------
"""

def Pareto_extract_se(x, b, p_fitted, method=1, verbose=True, hess=False):
    """
    NOTE: depreciated but is still provided for future improvements
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
    NOTE: depreciated but is still provided for future improvements
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

⣿⠄⡇⢸⣟⠄⠁⢸⡽⠖⠛⠈⡉⣉⠉⠋⣁⢘⠉⢉⠛⡿⢿⣿⣿⣿⣿⣿⣿⣿
⣷⣶⣷⣤⠄⣠⠖⠁⠄⠂⠁⠄⠄⠉⠄⠄⠎⠄⠠⠎⢐⠄⢑⣛⠻⣿⣿⣿⣿⣿
⣿⣿⣿⠓⠨⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠈⠐⠅⠄⠉⠄⠗⠆⣸⣿⣿⣿⣿⣿
⣿⣿⣿⡣⠁⠄⠄⠄⠄⠄⠄⠄⠄⠄⢰⣤⣦⠄⠄⠄⠄⠄⠄⠄⡀⡙⣿⣿⣿⣿
⣿⣿⡛⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠔⠿⡿⠿⠒⠄⠠⢤⡀⡀⠄⠁⠄⢻⣿⣿⣿
⣿⣿⠄⠄⠄⠄⠄⠄⣠⡖⠄⠁⠁⠄⠄⠄⠄⠄⠄⠄⣽⠟⡖⠄⠄⠄⣼⣿⣿⣿
⣿⣿⠄⠄⠄⠄⠄⠄⢠⣠⣀⠄⠄⠄⠄⢀⣾⣧⠄⠂⠸⣈⡏⠄⠄⠄⣿⣿⣿⣿
⣿⣿⡞⠄⠄⠄⠄⠄⢸⣿⣶⣶⣶⣶⣶⡿⢻⡿⣻⣶⣿⣿⡇⠄⠄⠄⣿⣿⣿⣿
⣿⣿⡷⡂⠄⠄⠁⠄⠸⣿⣿⣿⣿⣿⠟⠛⠉⠉⠙⠛⢿⣿⡇⠄⠄⢀⣿⣿⣿⣿
⣶⣶⠃⠄⠄⠄⠄⠄⠄⣾⣿⣿⡿⠁⣀⣀⣤⣤⣤⣄⢈⣿⡇⠄⠄⢸⣿⣿⣿⣿
⣿⣯⠄⠄⠄⠄⠄⠄⠄⢻⣿⣿⣷⣶⣿⣿⣥⣬⣿⣿⣟⣿⠃⠄⠨⠺⢿⣿⣿⣿
⠱⠂⠄⠄⠄⠄⠄⠄⠄⣬⣸⡝⠿⢿⣿⡿⣿⠻⠟⠻⢫⡁⠄⠄⠄⡐⣾⣿⣿⣿
⡜⠄⠄⠄⠄⠄⠆⡐⡇⢿⣽⣻⣷⣦⣧⡀⡀⠄⠄⣴⣺⡇⠄⠁⠄⢣⣿⣿⣿⣿
⠡⠱⠄⠄⠡⠄⢠⣷⠆⢸⣿⣿⣿⣿⣿⣿⣷⣿⣾⣿⣿⡇⠄⠄⠠⠁⠿⣿⣿⣿
⢀⣲⣧⣷⣿⢂⣄⡉⠄⠘⠿⣿⣿⣿⡟⣻⣯⠿⠟⠋⠉⢰⢦⠄⠊⢾⣷⣮⣽⣛
