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

""" 
---------------------------------------------------
Goddness of fit measures
---------------------------------------------------
"""
class gof:
    """ Goodness of fit measures and descriptive statistics data vs fit """
    def __init__(self, x, x_hat, parms, b):
        """
        :param x: model with same shape as data
        :param x_hat: fitted data
        :param parms: np.array []
        :param b: location parameter, fixed
        """
        self.n = n = len(x)
        self.emp_mean = np.mean(x, dtype=np.float64)
        self.emp_var = np.var(x, dtype=np.float64)
        self.pred_mean = np.mean(x_hat, dtype=np.float64)
        self.pred_var = np.var(x_hat, dtype=np.float64)
        self.e = e = np.array(x) - np.array(x_hat)
        self.soe = soe = np.sum(e)
        self.ssr = ssr = soe**2
        self.sse = sse = np.sum(e**2)
        self.sst = ssr + sse
        self.mse = 1/n * np.sum(e**2)
        self.rmse = np.sqrt(1/n * np.sum(e**2))
        self.mae = 1/n * np.sum(np.abs(e))
        self.mape = (100/n) * np.sum(np.abs(e/x))
        self.rrmse = np.sqrt(1/n * np.sum((e/x)**2))
        if len(parms) == 1:
            self.ll = ll = (-10000)*Pareto_ll(parms=parms, x=x_hat, b=b)
            self.aic = -2*ll + 2
            self.bic = -2*ll + np.log(n)
        if len(parms) == 2:
            self.ll = ll = (-10000)*IB1_ll(parms=parms, x=x_hat, b=b)
            self.aic = -2*ll + 2*2
            self.bic = -2*ll + np.log(n)*2
        if len(parms) == 3:
            self.ll = ll = (-100)*GB1_ll(parms=parms, x=x_hat, b=b)
            self.aic = -2*ll + 2*3
            self.bic = -2*ll + np.log(n)*3
        if len(parms) == 4:
            self.ll = ll = (-100)*GB_ll(parms, x=x_hat, b=b)
            self.aic = -2*ll + 2*4
            self.bic = -2*ll + np.log(n)*4

""" 
---------------------------------------------------
Neg. Log-Likelihoods
---------------------------------------------------
"""
def Pareto_ll(parms, x, b):
    """
    :param parms: np.array [p], optimized
    :param x: linspace, fixed
    :param b: location parameter, fixed
    :return: neg. logliklihood of Pareto
    """
    p = parms[0]
    n = len(x)
    sum = np.sum(np.log(x))
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
def Paretofit(x, b, x0, weights=np.array([1]), bootstraps=500, method='SLSQP',
              verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False,
              return_parameters=False, return_gofs=False,
              plot_cosmetics={'bins': 50, 'col_fit': 'blue', 'col_model': 'orange'},
              basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                                   'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
              SLSQP_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                             'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit Pareto distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [p] or simply (p)
    :param weights: weight, default: numpy.ones array of same shape as x
    :param bootstraps: amount of bootstraps
    :param method: # default: SLSQP (local optimization, much faster), 'L-BFGS-B' (global optimization, but slower)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhopping_options: dictionary with optimization options
    :param SLSQP_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """
    x = np.array(x)
    n_sample = len(x[x>b])

    """ Weights """
    # if default, assign weight of ones if no W specified
    if len(weights) == 1:
        weights = np.ones(len(x))

    # if user specified both x and W with same shape, calculate x*W
    if len(weights) == len(x):
        pass
    else:
        raise Exception("error - the length of W does not match the length of x {}".format(len(weights), len(x)))

    # if weights not roundes (like Stata), raise error
    try:
        x_inflated = []
        for idx, i in enumerate(x):
            weight = np.int64(weights[idx])
            x_extended = [i] * weight
            x_inflated.append(x_extended)
    except:
        print("error - probably, your weights are no integers, round first!")

    # flatten list
    x = [item for sublist in x_inflated for item in sublist]
    x = np.array(x)

    x = x[x>b]
    k = len(x)
    x0 = np.array(x0)

    widgets = ['Bootstrapping ', progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=bootstraps).start()
    tbl, tbl_gof = PrettyTable(), PrettyTable()

    def Pareto_constraint(b):
        return np.min(boot_sample) - b

    bnd = (10**-14, np.inf)
    constr = {'type': 'ineq', 'fun': Pareto_constraint}
    bootstrapping, p_fit_bs = 1, []

    if method == 'L-BFGS-B':

        # shorter variable name
        opts = basinhopping_options

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

            boot_sample = np.random.choice(x, size=k, replace=True)

            minimizer_kwargs = {"method": "L-BFGS-B", "args": (boot_sample, b),
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
            p_fit = result.x.item(0)
            p_fit_bs.append(p_fit)
            bar.update(bootstrapping)
            bootstrapping += 1

        bar.finish()

    if method == 'SLSQP':

        # shorter variable name
        opts = SLSQP_options

        if 'jac' not in opts.keys():
            opts['jac'] = None
        if 'tol' not in opts.keys():
            opts['tol'] = None
        if 'callback' not in opts.keys():
            opts['callback'] = None
        # if 'func' not in opts.keys():
        #     opts['func'] = None
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

            boot_sample = np.random.choice(x, size=k, replace=True)

            result = opt.minimize(Pareto_ll, x0,
                                  args=(boot_sample, b),
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
            p_fit_bs.append(p_fit)
            bar.update(bootstrapping)
            bootstrapping += 1
        bar.finish()

    if ci is False and verbose:
        tbl.field_names = ['parameter', 'value', 'se']
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), "{:.4f}".format(np.around(np.std(p_fit_bs), 4))])
        print(tbl)

    if ci and verbose:
        # calculation of zscores, p-values related to Stata's regression outputs
        p_zscore = np.around(np.mean(p_fit_bs)/np.std(p_fit_bs), 4)
        p_pval = 2*norm.cdf(-np.abs((np.mean(p_fit_bs)/np.std(p_fit_bs))))
        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'cilo_95', 'cihi_95', 'n']
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), "{:.4f}".format(np.around(np.std(p_fit_bs), 4)),
                     p_zscore, "{:.4f}".format(p_pval),
                     np.around(np.mean(p_fit_bs)-np.std(p_fit_bs)*1.96, 4),
                     np.around(np.mean(p_fit_bs)+np.std(p_fit_bs)*1.96, 4), k])
        print(tbl)

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit==True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 50
        else:
            num_bins = plot_cosmetics['bins']
        if 'col_fit' not in plot_cosmetics.keys():
            col_fit = 'blue'
        else:
            col_fit = plot_cosmetics['col_fit']
        if 'col_model' not in plot_cosmetics.keys():
            col_model = 'orange'
        else:
            col_model = plot_cosmetics['col_model']

        fig, ax = plt.subplots()
        # histogram of x (actual data)
        n, bins, patches = ax.hist(x, num_bins, density=1, label='Pareto fit', color=col_fit)
        # x2: model with fitted parameters
        x2 = Pareto_pdf(np.linspace(np.min(x), np.max(x), np.size(x)), b=b, p=np.mean(p_fit_bs))
        ax.plot(np.linspace(np.min(x), np.max(x), np.size(x)), x2, '--', label='model', color=col_model)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability density')
        ax.set_title('fit vs. model')
        ax.legend(['Pareto model', 'fit'])
        # fig.tight_layout()
        plt.show()

    if return_gofs:
        fit = True
    if fit:
        u = np.array(np.random.uniform(.0, 1., len(x)))
        model = Pareto_icdf(u=u, b=b, p=np.mean(p_fit_bs))
        soe = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).soe
        # ssr = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).ssr
        # sse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).sse
        # sst = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).sst
        emp_mean = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).emp_mean
        emp_var = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).emp_var
        pred_mean = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).pred_mean
        pred_var = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).pred_var
        mae = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).mae
        mse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).mse
        rmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).rmse
        rrmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).rrmse
        aic = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).aic
        bic = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).bic
        n = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).n
        ll = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'n']
            tbl_gof.add_row(['GOF', np.around(aic, 3), np.around(bic, 3), np.around(mae, 3), np.around(mse, 3),
                             np.around(rmse, 3), np.around(rrmse, 3), np.around(ll, 3), np.around(soe, 3),
                             np.around(emp_mean, 3), np.around(emp_var, 3), np.around(pred_mean, 3),
                             np.around(pred_var, 3), np.around(n, 3)])
            print("\n{}\n".format(tbl_gof))

        if return_gofs:
            return_parameters = False
            return np.mean(p_fit_bs), np.std(p_fit_bs), aic, bic, mae, mse, rmse, rrmse, ll, n

    if return_parameters:
        return np.mean(p_fit_bs), np.std(p_fit_bs)


def IB1fit(x, b, x0, weights=np.array([1]), bootstraps=500, method='SLSQP',
           verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False,
           return_parameters=False, return_gofs=False,
           plot_cosmetics={'bins': 50, 'col_fit': 'blue', 'col_model': 'orange'},
           basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                                'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
           SLSQP_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                          'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit the IB1 distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [p,q] or simply (p,q)
    :param weights: weight, default: numpy.ones array of same shape as x
    :param bootstraps: amount of bootstraps
    :param method: # default: SLSQP (local optimization, much faster), 'L-BFGS-B' (global optimization, but slower)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhopping_options: dictionary with optimization options
    :param SLSQP_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """
    x = np.array(x)
    n_sample = len(x[x>b])

    """ Weights """
    # if default, assign weight of ones if no W specified
    if len(weights) == 1:
        weights = np.ones(len(x))

    # if user specified both x and W with same shape, calculate x*W
    if len(weights) == len(x):
        pass
    else:
        raise Exception("error - the length of W does not match the length of x {}".format(len(weights), len(x)))

    # if weights not roundes (like Stata), raise error
    try:
        x_inflated = []
        for idx, i in enumerate(x):
            weight = np.int64(weights[idx])
            x_extended = [i] * weight
            x_inflated.append(x_extended)
    except:
        print("error - probably, your weights are no integers, round first!")

    # flatten list
    x = [item for sublist in x_inflated for item in sublist]
    x = np.array(x)

    x = x[x>b]
    k = len(x)
    x0 = np.array(x0)

    widgets = ['Bootstrapping ', progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=bootstraps).start()
    tbl, tbl_gof = PrettyTable(), PrettyTable()

    def IB1_constraint(parms):
        a = parms[0]
        return (np.min(boot_sample)/b)**a

    constr = {'type': 'ineq', 'fun': IB1_constraint}
    bnds = (10**-14, np.inf)
    bootstrapping, p_fit_bs, q_fit_bs = 1, [], []

    if method == 'L-BFGS-B':

        # shorter variable name
        opts = basinhopping_options

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

            boot_sample = np.random.choice(x, size=k, replace=True)

            minimizer_kwargs = {"method": "L-BFGS-B", "args": (boot_sample, b),
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

    if method == 'SLSQP':

        # shorter variable name
        opts = SLSQP_options

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

            boot_sample = np.random.choice(x, size=k, replace=True)

            result = opt.minimize(IB1_ll, x0,
                                  args=(boot_sample, b),
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

    if ci is False and verbose is True:
        tbl.field_names = ['parameter', 'value', 'se']
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), "{:.4f}".format(np.around(np.std(p_fit_bs), 4))])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), "{:.4f}".format(np.around(np.std(q_fit_bs), 4))])
        print(tbl)

    if ci and verbose:
        p_zscore = np.around(np.mean(p_fit_bs)/np.std(p_fit_bs), 4)
        q_zscore = np.around(np.mean(q_fit_bs)/np.std(q_fit_bs), 4)
        p_pval = 2*norm.cdf(-np.abs((np.mean(p_fit_bs)/np.std(p_fit_bs))))
        q_pval = 2*norm.cdf(-np.abs((np.mean(q_fit_bs)/np.std(q_fit_bs))))

        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'cilo_95', 'cihi_95', 'n']
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), "{:.4f}".format(np.around(np.std(p_fit_bs), 4)),
                     p_zscore, "{:.4f}".format(p_pval),
                          np.around(np.mean(p_fit_bs)-np.std(p_fit_bs)*1.96, 4),
                          np.around(np.mean(p_fit_bs)+np.std(p_fit_bs)*1.96, 4), k])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), "{:.4f}".format(np.around(np.std(q_fit_bs), 4)), q_zscore, "{:.4f}".format(q_pval),
                          np.around(np.mean(q_fit_bs)-np.std(q_fit_bs)*1.96, 4),
                          np.around(np.mean(q_fit_bs)+np.std(q_fit_bs)*1.96, 4), k])
        print(tbl)

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit==True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 50
        else:
            num_bins = plot_cosmetics['bins']
        if 'col_fit' not in plot_cosmetics.keys():
            col_fit = 'blue'
        else:
            col_fit = plot_cosmetics['col_fit']
        if 'col_model' not in plot_cosmetics.keys():
            col_model = 'orange'
        else:
            col_model = plot_cosmetics['col_model']

        fig, ax = plt.subplots()
        # histogram of x (actual data)
        n, bins, patches = ax.hist(x, num_bins, density=1, label='IB1 fit', color=col_fit)
        # x2: model with fitted parameters
        x2 = IB1_pdf(np.linspace(np.min(x), np.max(x), np.size(x)), b=b, p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        ax.plot(np.linspace(np.min(x), np.max(x), np.size(x)), x2, '--', label='model', color=col_model)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability density')
        ax.set_title('fit vs. model')
        ax.legend(['IB1 model', 'fit'])
        # fig.tight_layout()
        plt.show()

    if return_gofs:
        fit = True
    if fit:
        # save calculation time and use Pareto_icdf which is equivalent if parms fall in Pareto branch restrictions
        if (.95<np.mean(q_fit_bs)<1.05) | (.95<q_fit<1.05):
            model = Pareto_icdf(u=np.array(np.random.uniform(.0, 1., len(x))), b=b, p=np.mean(p_fit_bs))
        else:
            model, u = IB1_icdf_ne(x=x, b=b, p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        soe = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).soe
        # ssr = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).ssr
        # sse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).sse
        # sst = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).sst
        emp_mean = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_mean
        emp_var = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_var
        pred_mean = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_mean
        pred_var = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_var
        mae = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'n']
            tbl_gof.add_row(['GOF', np.around(aic, 3), np.around(bic, 3), np.around(mae, 3), np.around(mse, 3),
                             np.around(rmse, 3), np.around(rrmse, 3), np.around(ll, 3), np.around(soe, 3),
                             np.around(emp_mean, 3), np.around(emp_var, 3), np.around(pred_mean, 3),
                             np.around(pred_var, 3), np.around(n, 3)])
            print("\n{}\n".format(tbl_gof))

        if return_gofs:
            return_parameters = False
            return np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs), aic, bic, mae, mse, rmse, rrmse, ll, n

    if return_parameters:
        return np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs)


def GB1fit(x, b, x0, weights=np.array([1]), bootstraps=250, method='SLSQP',
           verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False,
           return_parameters=False, return_gofs=False,
           plot_cosmetics={'bins': 50, 'col_fit': 'blue', 'col_model': 'orange'},
           basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                                'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
           SLSQP_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                          'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit the GB1 distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [a,p,q] or simply (a,p,q)
    :param weights: weight, default: numpy.ones array of same shape as x
    :param bootstraps: amount of bootstraps
    :param method: # default: SLSQP (local optimization, much faster), 'L-BFGS-B' (global optimization, but slower)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhopping_options: dictionary with optimization options
    :param SLSQP_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """
    x = np.array(x)
    n_sample = len(x[x>b])

    """ Weights """
    # if default, assign weight of ones if no W specified
    if len(weights) == 1:
        weights = np.ones(len(x))

    # if user specified both x and W with same shape, calculate x*W
    if len(weights) == len(x):
        pass
    else:
        raise Exception("error - the length of W does not match the length of x {}".format(len(weights), len(x)))

    # if weights not roundes (like Stata), raise error
    try:
        x_inflated = []
        for idx, i in enumerate(x):
            weight = np.int64(weights[idx])
            x_extended = [i] * weight
            x_inflated.append(x_extended)
    except:
        print("error - probably, your weights are no integers, round first!")

    # flatten list
    x = [item for sublist in x_inflated for item in sublist]
    x = np.array(x)

    x = x[x>b]
    k = len(x)
    x0 = np.array(x0)

    widgets = ['Bootstrapping ', progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=bootstraps).start()
    tbl, tbl_gof = PrettyTable(), PrettyTable()

    def GB1_constraint(parms):
        a = parms[0]
        return (np.min(boot_sample)/b)**a

    constr = {'type': 'ineq', 'fun': GB1_constraint}
    a_bnd, bnds = (-10, -1e-10), (10**-14, np.inf)
    bootstrapping, a_fit_bs, p_fit_bs, q_fit_bs = 1, [], [], []

    if method == 'L-BFGS-B':

        # shorter variable name
        opts = basinhopping_options

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

            boot_sample = np.random.choice(x, size=k, replace=True)

            minimizer_kwargs = {"method": "L-BFGS-B", "args": (boot_sample, b),
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

    if method == 'SLSQP':

        # shorter variable name
        opts = SLSQP_options

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
            boot_sample = np.random.choice(x, size=k, replace=True)

            result = opt.minimize(GB1_ll, x0,
                                  args=(boot_sample, b),
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

    if ci is False and verbose is True:
        tbl.field_names = ['parameter', 'value', 'se']
        tbl.add_row(['a', np.around(np.mean(a_fit_bs), 4), "{:.4f}".format(np.around(np.std(a_fit_bs), 4))])
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), "{:.4f}".format(np.around(np.std(p_fit_bs), 4))])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), "{:.4f}".format(np.around(np.std(q_fit_bs), 4))])
        print(tbl)

    if ci and verbose:
        a_zscore = np.around(np.mean(a_fit_bs)/np.std(a_fit_bs), 4)
        p_zscore = np.around(np.mean(p_fit_bs)/np.std(p_fit_bs), 4)
        q_zscore = np.around(np.mean(q_fit_bs)/np.std(q_fit_bs), 4)
        a_pval = 2*norm.cdf(-np.abs((np.mean(a_fit_bs)/np.std(a_fit_bs))))
        p_pval = 2*norm.cdf(-np.abs((np.mean(p_fit_bs)/np.std(p_fit_bs))))
        q_pval = 2*norm.cdf(-np.abs((np.mean(q_fit_bs)/np.std(q_fit_bs))))

        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'cilo_95', 'cihi_95', 'n']
        tbl.add_row(['a', np.around(np.mean(a_fit_bs), 4), "{:.4f}".format(np.around(np.std(a_fit_bs), 4)),
                     a_zscore, "{:.4f}".format(a_pval),
                     np.around(np.mean(a_fit_bs)-np.std(a_fit_bs)*1.96, 4),
                     np.around(np.mean(a_fit_bs)+np.std(a_fit_bs)*1.96, 4), k])
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), "{:.4f}".format(np.around(np.std(p_fit_bs), 4)),
                     p_zscore, "{:.4f}".format(p_pval),
                     np.around(np.mean(p_fit_bs)-np.std(p_fit_bs)*1.96, 4),
                     np.around(np.mean(p_fit_bs)+np.std(p_fit_bs)*1.96, 4), k])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), "{:.4f}".format(np.around(np.std(q_fit_bs), 4)),
                     q_zscore, "{:.4f}".format(q_pval),
                     np.around(np.mean(q_fit_bs)-np.std(q_fit_bs)*1.96, 4),
                     np.around(np.mean(q_fit_bs)+np.std(q_fit_bs)*1.96, 4), k])
        print(tbl)

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit==True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 50
        else:
            num_bins = plot_cosmetics['bins']
        if 'col_fit' not in plot_cosmetics.keys():
            col_fit = 'blue'
        else:
            col_fit = plot_cosmetics['col_fit']
        if 'col_model' not in plot_cosmetics.keys():
            col_model = 'orange'
        else:
            col_model = plot_cosmetics['col_model']

        fig, ax = plt.subplots()
        # histogram of x (actual data)
        n, bins, patches = ax.hist(x, num_bins, density=1, label='GB1 fit', color=col_fit)
        # x2: model with fitted parameters
        x2 = GB1_pdf(np.linspace(np.min(x), np.max(x), np.size(x)), b=b, a=np.mean(a_fit_bs), p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        ax.plot(np.linspace(np.min(x), np.max(x), np.size(x)), x2, '--', label='model', color=col_model)
        ax.set_xlabel('x')
        ax.set_ylabel('Probability density')
        ax.set_title('fit vs. model')
        ax.legend(['GB1 model', 'fit'])
        # fig.tight_layout()
        plt.show()

    if return_gofs:
        fit = True
    if fit:
        # save calculation time and use Pareto_icdf which is equivalent if parms fall in Pareto branch restrictions
        if (.95<np.mean(q_fit_bs)<1.05) | (.95<q_fit<1.05) and (-1.05<np.mean(a_fit_bs)<-.95) | (-1.05<a_fit<-.95):
            model = Pareto_icdf(u=np.array(np.random.uniform(.0, 1., len(x))), b=b, p=np.mean(p_fit_bs))
        else:
            model, u = GB1_icdf_ne(x=x, b=b, a=np.mean(a_fit_bs), p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        soe = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).soe
        # ssr = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).ssr
        # sse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).sse
        # sst = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).sst
        emp_mean = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_mean
        emp_var = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_var
        pred_mean = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_mean
        pred_var = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_var
        mae = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'n']
            tbl_gof.add_row(['GOF', np.around(aic, 3), np.around(bic, 3), np.around(mae, 3), np.around(mse, 3),
                             np.around(rmse, 3), np.around(rrmse, 3), np.around(ll, 3), np.around(soe, 3),
                             np.around(emp_mean, 3), np.around(emp_var, 3), np.around(pred_mean, 3),
                             np.around(pred_var, 3), np.around(n, 3)])
            print("\n{}\n".format(tbl_gof))

        if return_gofs:
            return_parameters = False
            return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs), aic, bic, mae, mse, rmse, rrmse, ll, n

    if return_parameters:
        return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs)


def GBfit(x, b, x0, weights=np.array([1]), bootstraps=250, method='SLSQP',
          verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False,
          return_parameters=False, return_gofs=False,
          plot_cosmetics={'bins': 50, 'col_fit': 'blue', 'col_model': 'orange'},
    basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                         'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
          SLSQP_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                         'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit the GB distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [a,c,p,q] or simply (q,c,p,q)
    :param weights: weight, default: numpy.ones array of same shape as x
    :param bootstraps: amount of bootstraps
    :param method: # default: SLSQP (local optimization, much faster), 'L-BFGS-B' (global optimization, but slower)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhopping_options: dictionary with optimization options
    :param SLSQP_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """
    x = np.array(x)
    n_sample = len(x[x>b])

    """ Weights """
    # if default, assign weight of ones if no W specified
    if len(weights) == 1:
        weights = np.ones(len(x))

    # if user specified both x and W with same shape, calculate x*W
    if len(weights) == len(x):
        pass
    else:
        raise Exception("error - the length of W does not match the length of x {}".format(len(weights), len(x)))

    # if weights not roundes (like Stata), raise error
    try:
        x_inflated = []
        for idx, i in enumerate(x):
            weight = np.int64(weights[idx])
            x_extended = [i] * weight
            x_inflated.append(x_extended)
    except:
        print("error - probably, your weights are no integers, round first!")

    # flatten list
    x = [item for sublist in x_inflated for item in sublist]
    x = np.array(x)

    x = x[x>b]
    k = len(x)
    x0 = np.array(x0)

    widgets = ['Bootstrapping ', progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
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

    if method == 'L-BFGS-B':

        # shorter variable name
        opts = basinhopping_options

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

            boot_sample = np.random.choice(x, size=k, replace=True)

            minimizer_kwargs = {"method": "L-BFGS-B", "args": (boot_sample, b),
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

    if method == 'SLSQP':

        # shorter variable name
        opts = SLSQP_options

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

            boot_sample = np.random.choice(x, size=k, replace=True)

            result = opt.minimize(GB_ll, x0,
                                  args=(boot_sample, b),
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

    if ci is False and verbose is True:

        tbl.field_names = ['parameter', 'value', 'se']
        tbl.add_row(['a', np.around(np.mean(a_fit_bs), 4), "{:.4f}".format(np.around(np.std(a_fit_bs), 4))])
        tbl.add_row(['c', np.around(np.mean(c_fit_bs), 4), "{:.4f}".format(np.around(np.std(c_fit_bs), 4))])
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), "{:.4f}".format(np.around(np.std(p_fit_bs), 4))])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), "{:.4f}".format(np.around(np.std(q_fit_bs), 4))])
        print(tbl)

    if ci and verbose:
        a_zscore = np.around(np.mean(a_fit_bs)/np.std(a_fit_bs), 4)
        c_zscore = np.around(np.mean(c_fit_bs)/np.std(c_fit_bs), 4)
        p_zscore = np.around(np.mean(p_fit_bs)/np.std(p_fit_bs), 4)
        q_zscore = np.around(np.mean(q_fit_bs)/np.std(q_fit_bs), 4)
        a_pval = 2*norm.cdf(-np.abs((np.mean(a_fit_bs)/np.std(a_fit_bs))))
        c_pval = 2*norm.cdf(-np.abs((np.mean(c_fit_bs)/np.std(c_fit_bs))))
        p_pval = 2*norm.cdf(-np.abs((np.mean(p_fit_bs)/np.std(p_fit_bs))))
        q_pval = 2*norm.cdf(-np.abs((np.mean(q_fit_bs)/np.std(q_fit_bs))))

        tbl.field_names = ['parameter', 'value', 'se', 'z', 'P>|z|', 'cilo_95', 'cihi_95', 'n']
        tbl.add_row(['a', np.around(np.mean(a_fit_bs), 4), "{:.4f}".format(np.around(np.std(a_fit_bs), 4)),
                     a_zscore, "{:.4f}".format(a_pval),
                     np.around(np.mean(a_fit_bs)-np.std(a_fit_bs)*1.96, 4),
                     np.around(np.mean(a_fit_bs)+np.std(a_fit_bs)*1.96, 4), k])
        tbl.add_row(['c', np.around(np.mean(c_fit_bs), 4), "{:.4f}".format(np.around(np.std(c_fit_bs), 4)),
                     c_zscore, "{:.4f}".format(c_pval),
                     np.around(np.mean(c_fit_bs)-np.std(c_fit_bs)*1.96, 4),
                     np.around(np.mean(c_fit_bs)+np.std(c_fit_bs)*1.96, 4), k])
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), "{:.4f}".format(np.around(np.std(p_fit_bs), 4)),
                     p_zscore, "{:.4f}".format(p_pval),
                     np.around(np.mean(p_fit_bs)-np.std(p_fit_bs)*1.96, 4),
                     np.around(np.mean(p_fit_bs)+np.std(p_fit_bs)*1.96, 4), k])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), "{:.4f}".format(np.around(np.std(q_fit_bs), 4)),
                     q_zscore, "{:.4f}".format(q_pval),
                     np.around(np.mean(q_fit_bs)-np.std(q_fit_bs)*1.96, 4),
                     np.around(np.mean(q_fit_bs)+np.std(q_fit_bs)*1.96, 4), k])
        print(tbl)

    if plot:
        fit = True # if plot is True, also display tbl_gof so set fit=True
        # Set defaults of plot_cosmetics in case plot_cosmetics-dictionary as arg has an empty key
        if 'bins' not in plot_cosmetics.keys():
            num_bins = 50
        else:
            num_bins = plot_cosmetics['bins']
        if 'col_fit' not in plot_cosmetics.keys():
            col_fit = 'blue'
        else:
            col_fit = plot_cosmetics['col_fit']
        if 'col_model' not in plot_cosmetics.keys():
            col_model = 'orange'
        else:
            col_model = plot_cosmetics['col_model']

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
        ax.set_title('fit vs. model')
        ax.legend(['GB model', 'fit'])
        # fig.tight_layout()
        plt.show()

    if return_gofs:
        fit = True
    if fit:
        # save calculation time and use Pareto_icdf which is equivalent if parms fall in Pareto branch restrictions
        if (.95<np.mean(q_fit_bs)<1.05) | (.95<q_fit<1.05) \
                and (-1.05<np.mean(a_fit_bs)<-.95) | (-1.05<a_fit<-.95) \
                and (-.05<np.mean(c_fit_bs)<.05) | (-.05<c_fit<-.05):
            model = Pareto_icdf(u=np.array(np.random.uniform(.0, 1., len(x))), b=b, p=np.mean(p_fit_bs))
        else:
            model, u = GB_icdf_ne(x=x, b=b, a=np.mean(a_fit_bs), c=np.mean(c_fit_bs), p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        soe = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).soe
        # ssr = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).ssr
        # sse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).sse
        # sst = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).sst
        emp_mean = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_mean
        emp_var = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).emp_var
        pred_mean = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_mean
        pred_var = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).pred_var
        mae = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'sum of errors', 'emp. mean', 'emp. var.', 'pred. mean', 'pred. var.', 'n']
            tbl_gof.add_row(['GOF', np.around(aic, 3), np.around(bic, 3), np.around(mae, 3), np.around(mse, 3),
                             np.around(rmse, 3), np.around(rrmse, 3), np.around(ll, 3), np.around(soe, 3),
                             np.around(emp_mean, 3), np.around(emp_var, 3), np.around(pred_mean, 3),
                             np.around(pred_var, 3), np.around(n, 3)])
            print("\n{}\n".format(tbl_gof))

        if return_gofs:
            return_parameters = False
            return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(c_fit_bs), np.std(c_fit_bs), np.mean(p_fit_bs), \
                   np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs), aic, bic, mae, mse, rmse, rrmse, ll, n

    if return_parameters:
        return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(c_fit_bs), np.std(c_fit_bs), np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs)


""" 
---------------------------------------------------
Pareto branch fitting
---------------------------------------------------
"""

def Paretobranchfit(x, b, x0, weights=np.array([1]), bootstraps=250, method='SLSQP', verbose_bootstrap=False,
                    ci=True, verbose=True, fit=False, plot=False, return_parameters=False, return_gofs=False,
                    rejecting_criteria="LRtest",
          plot_cosmetics={'bins': 50, 'col_fit': 'blue', 'col_model': 'orange'},
    basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                         'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
          SLSQP_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
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
    :param bootstraps: either 1x1 OR 1x2 array (1st arg: Pareto+IB1, 2nd arg: GB1+GB) OR pass 1x4 array [Pareto_bs, IB1_bs, GB1_bs, GB_bs]
    :param method: as above
    :param verbose_bootstrap: as above
    :param ci: as above
    :param verbose: as above
    :param fit: as above
    :param plot: as above
    :param return_parameters: as above and parameters, se of all distributions are returned
    :param return_gofs: as above
    :param plot_cosmetics: as above
    :param basinhopping_options: as above
    :param SLSQP_options: as above
    :return:
    """

    # Prepare args for passing to below fitting functions
    # TODO: preparation passing to
    # x0
    try:
        x0_temp = [item for sublist in x0 for item in sublist]
        if len(x0_temp) == 10:
            Pareto_x0, IB1_x0, GB1_x0, GB_x0 = x0_temp[0], x0_temp[1:3], x0_temp[3:5], x0_temp[6:]
    except TypeError:
        if len(x0) == 4:
            Pareto_x0, IB1_x0, GB1_x0, GB_x0 = x0[0], x0[:2], x0[:3], x0[:4]
        else:
            raise Exception("error - x0 not correctly specified")

    # bootstraps
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



    Pareto_fit = Paretofit(x=x, b=b, x0=Pareto_x0, weights=weights, bootstraps=Pareto_bs, method=method,
                           return_parameters=True, return_gofs=True,
                           verbose_bootstrap=verbose_bootstrap, ci=ci, verbose=verbose, fit=fit, plot=plot,
                           plot_cosmetics={'bins': plot_cosmetics['bins'], 'col_fit': plot_cosmetics['blue'],
                                           'col_model': plot_cosmetics['col_model']},
                           basinhopping_options={'niter': basinhopping_options['niter'], 'T': basinhopping_options['T'],
                                                 'stepsize': basinhopping_options['stepsize'], 'take_step': basinhopping_options['take_step'],
                                                 'accept_test': basinhopping_options['accept_test'], 'callback': basinhopping_options['callback'],
                                                 'interval': basinhopping_options['interval'], 'disp': basinhopping_options['disp'],
                                                 'niter_success': basinhopping_options['niter_success'], 'seed': basinhopping_options['seed']},
                           SLSQP_options={'jac': SLSQP_options['jac'], 'tol': SLSQP_options['tol'], 'callback': SLSQP_options['callback'],
                                          'func': SLSQP_options['func'], 'maxiter': SLSQP_options['maxiter'], 'ftol': SLSQP_options['ftol'],
                                          'iprint': SLSQP_options['iprint'], 'disp': SLSQP_options['disp'], 'eps': SLSQP_options['eps']})

    IB1_fit = IB1fit(x=x, b=b, x0=IB1_x0, weights=np.array([1]), bootstraps=IB1_bs, method=method,
                     return_parameters=True, return_gofs=True,
                     verbose_bootstrap=verbose_bootstrap, ci=ci, verbose=verbose, fit=fit, plot=plot,
                     plot_cosmetics={'bins': plot_cosmetics['bins'], 'col_fit': plot_cosmetics['blue'],
                                     'col_model': plot_cosmetics['col_model']},
                     basinhopping_options={'niter': basinhopping_options['niter'],
                                           'T': basinhopping_options['T'],
                                           'stepsize': basinhopping_options['stepsize'],
                                           'take_step': basinhopping_options['take_step'],
                                           'accept_test': basinhopping_options['accept_test'],
                                           'callback': basinhopping_options['callback'],
                                           'interval': basinhopping_options['interval'],
                                           'disp': basinhopping_options['disp'],
                                           'niter_success': basinhopping_options['niter_success'],
                                           'seed': basinhopping_options['seed']},
                     SLSQP_options={'jac': SLSQP_options['jac'], 'tol': SLSQP_options['tol'],
                                    'callback': SLSQP_options['callback'], 'func': SLSQP_options['func'],
                                    'maxiter': SLSQP_options['maxiter'], 'ftol': SLSQP_options['ftol'],
                                    'iprint': SLSQP_options['iprint'], 'disp': SLSQP_options['disp'],
                                    'eps': SLSQP_options['eps']})

    GB1_fit = GB1fit(x=x, b=b, x0=GB1_x0, weights=np.array([1]), bootstraps=GB1_bs, method=method,
                     return_parameters=True, return_gofs=True,
                     verbose_bootstrap=verbose_bootstrap, ci=ci, verbose=verbose, fit=fit, plot=plot,
                     plot_cosmetics={'bins': plot_cosmetics['bins'], 'col_fit': plot_cosmetics['blue'],
                                     'col_model': plot_cosmetics['col_model']},
                     basinhopping_options={'niter': basinhopping_options['niter'], 'T': basinhopping_options['T'],
                                           'stepsize': basinhopping_options['stepsize'],
                                           'take_step': basinhopping_options['take_step'],
                                           'accept_test': basinhopping_options['accept_test'],
                                           'callback': basinhopping_options['callback'],
                                           'interval': basinhopping_options['interval'],
                                           'disp': basinhopping_options['disp'],
                                           'niter_success': basinhopping_options['niter_success'],
                                           'seed': basinhopping_options['seed']},
                     SLSQP_options={'jac': SLSQP_options['jac'], 'tol': SLSQP_options['tol'],
                                    'callback': SLSQP_options['callback'],
                                    'func': SLSQP_options['func'],
                                    'maxiter': SLSQP_options['maxiter'],
                                    'ftol': SLSQP_options['ftol'],
                                    'iprint': SLSQP_options['iprint'],
                                    'disp': SLSQP_options['disp'],
                                    'eps': SLSQP_options['eps']})

    GB_fit = GBfit(x=x, b=b, x0=GB_x0, weights=np.array([1]), bootstraps=GB_bs, method=method,
                   return_parameters=True, return_gofs=True,
                   verbose_bootstrap=verbose_bootstrap, ci=ci, verbose=verbose, fit=fit, plot=plot,
                   plot_cosmetics={'bins': plot_cosmetics['bins'],
                                   'col_fit': plot_cosmetics['blue'],
                                   'col_model': plot_cosmetics['col_model']},
                   basinhopping_options={'niter': basinhopping_options['niter'],
                                         'T': basinhopping_options['T'],
                                         'stepsize': basinhopping_options['stepsize'],
                                         'take_step': basinhopping_options['take_step'],
                                         'accept_test': basinhopping_options['accept_test'],
                                         'callback': basinhopping_options['callback'],
                                         'interval': basinhopping_options['interval'],
                                         'disp': basinhopping_options['disp'],
                                         'niter_success': basinhopping_options['niter_success'],
                                         'seed': basinhopping_options['seed']},
                   SLSQP_options={'jac': SLSQP_options['jac'], 'tol': SLSQP_options['tol'],
                                  'callback': SLSQP_options['callback'], 'func': SLSQP_options['func'],
                                  'maxiter': SLSQP_options['maxiter'], 'ftol': SLSQP_options['ftol'],
                                  'iprint': SLSQP_options['iprint'], 'disp': SLSQP_options['disp'],
                                  'eps': SLSQP_options['eps']})
    # saving parameters
    # TODO: save parameters here

    # if rejecting_criteria == "LRtest":
    # TODO: LRtest

    #     # 1. LRtest Pareto vs IB1
    #     LRtest(Pareto(x=x, b=b, p=p_fit1).LL, IB1(x=x, b=b, p=p_fit2, q=q_fit2).LL, df=2)
    #     # 2. LRtest IB1 vs GB1
    #     LRtest(IB1(x=x, b=b, p=p_fit2, q=q_fit2).LL, GB1(x=x, b=b, a=a_fit3, p=p_fit3, q=q_fit3).LL, df=3)
    #     # 3. LRtest GB1 vs GB
    #     LRtest(GB1(x=x, b=b, a=a_fit3, p=p_fit3, q=q_fit3).LL, GB(x=x, b=b, a=a_fit4, c=c_fit4, p=p_fit4, q=q_fit4).LL, df=4)
    #
    # if rejecting_criteria == "AIC":

        # 1. LRtest Pareto vs IB1

        # 2. LRtest IB1 vs GB1

        # 3. LRtest GB1 vs GB


    # TODO: all fit functions in row
    # TODO: LRtest (AIC) decision
    # TODO: return_parameters
    # TODO: result summary

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
    if verbose: print("p: {}, se: {}".format(np.around(p, 4), np.around(p_se, 4)))
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
        print("b: {}, se: {}\np: {}, se: {}\nq: {}, se: {}".format(np.around(b,3), np.around(b_se,3),
                                                                               np.around(p,3), np.around(p_se,3),
                                                                               np.around(q,3), np.around(q_se,3)))
    if display_hessian == True:
        print("Hessian Matrix:", hess)
    return b_se, p_se, q_se
