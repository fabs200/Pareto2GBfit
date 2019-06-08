import numpy as np
import scipy.optimize as opt
from scipy import linalg
from scipy.misc import derivative
from scipy.special import digamma, gammaln
import matplotlib.pyplot as plt
import progressbar
from prettytable import PrettyTable
from .distributions import Pareto_pdf, IB1_pdf, GB1_pdf, GB_pdf, Pareto_icdf, IB1_icdf_ne, GB1_icdf_ne, GB_icdf_ne


class gof:
    """ Goodness of fit measures """
    def __init__(self, x, x_hat, parms, b):
        """
        :param x: model with same shape as data
        :param x_hat: fitted data
        :param parms: np.array []
        :param b: location parameter, fixed
        """
        self.n = n = len(x)
        self.e = e = np.array(x) - np.array(x_hat)
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

def GB_ll(parm, x, b):
    """
    :param parm: np.array [a, c, p, q] optimized
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :return: neg. log-likelihood of GB1
    """
    a = parm[0]
    c = parm[1]
    p = parm[2]
    q = parm[3]
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

def Paretofit(x, b, x0, bootstraps=500, method='SLSQP',
              verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, return_parameters=False,
              plot_cosmetics={'bins': 50, 'col_fit': 'blue', 'col_model': 'orange'},
              basinhoppin_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                                   'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
              SLSQP_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                             'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit Pareto distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [p] or simply (p)
    :param bootstraps: amount of bootstraps
    :param method: # default: SLSQP (local optimization, much faster), 'L-BFGS-B' (global optimization, but slower)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhoppin_options: dictionary with optimization options
    :param SLSQP_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """
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
        opts = basinhoppin_options

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
                                "bounds": (bnd,),
                                "constraints": constr}

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
            result = opt.minimize(Pareto_ll, x0,
                                  method='SLSQP',
                                  bounds=(bnd,),
                                  args=(boot_sample, b),
                                  jac=opts['jac'],
                                  tol=opts['tol'],
                                  callback=opts['callback'],
                                  options=({'maxiter': opts['maxiter'], 'disp': False, 'iprint': opts['iprint'],
                                            'fun': opts['func'], 'ftol': opts['ftol'], 'eps': opts['eps']}),
                                  constraints=constr)
            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))
            p_fit = result.x.item(0)
            p_fit_bs.append(p_fit)
            bar.update(bootstrapping)
            bootstrapping += 1
        bar.finish()
    if ci is False and verbose:
            tbl.field_names = ['parameter', 'value', 'se']
            tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), np.around(np.std(p_fit_bs), 4)])
            print(tbl)
    if ci and verbose:
            tbl.field_names = ['parameter', 'value', 'se', 'cilo_95', 'cihi_95', 'n']
            tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), np.around(np.std(p_fit_bs), 4),
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
    if fit:
        u = np.array(np.random.uniform(.0, 1., len(x)))
        model = Pareto_icdf(u=u, b=b, p=np.mean(p_fit_bs))
        mae = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).mae
        mse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).mse
        rmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).rmse
        rrmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).rrmse
        aic = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).aic
        bic = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).bic
        n = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).n
        ll = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'n']
            tbl_gof.add_row(['GOF', np.around(aic, 4), np.around(bic, 4), np.around(mae, 4), np.around(mse, 4),
                                    np.around(rmse, 4), np.around(rrmse, 4), np.around(ll, 4), np.around(n, 4)])
            print("\n", tbl_gof, "\n")
    if return_parameters:
        return np.mean(p_fit_bs), np.std(p_fit_bs)


def IB1fit(x, b, x0, bootstraps=500, method='SLSQP',
           verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, return_parameters=False,
           plot_cosmetics={'bins': 50, 'col_fit': 'blue', 'col_model': 'orange'}
              basinhoppin_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                                   'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
              SLSQP_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                             'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit the IB1 distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [p,q] or simply (p,q)
    :param bootstraps: amount of bootstraps
    :param method: # default: SLSQP (local optimization, much faster), 'L-BFGS-B' (global optimization, but slower)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhoppin_options: dictionary with optimization options
    :param SLSQP_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """
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
        opts = basinhoppin_options

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
                                "bounds": (bnds, bnds,),
                                "constraints": constr}

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
                                  method='SLSQP',
                                  bounds=(bnds, bnds,),
                                  args=(boot_sample, b),
                                  jac=opts['jac'],
                                  tol=opts['tol'],
                                  callback=opts['callback'],
                                  options=({'maxiter': opts['maxiter'], 'disp': False, 'iprint': opts['iprint'],
                                            'fun': opts['func'], 'ftol': opts['ftol'], 'eps': opts['eps']}),
                                  constraints=constr)

            if verbose_bootstrap: print("\nbootstrap: {}".format(bootstrapping))

            p_fit, q_fit = result.x.item(0), result.x.item(1)
            p_fit_bs.append(p_fit)
            q_fit_bs.append(q_fit)
            bar.update(bootstrapping)
            bootstrapping += 1

        bar.finish()

    if ci is False and verbose is True:
        tbl.field_names = ['parameter', 'value', 'se']
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), np.around(np.std(p_fit_bs), 4)])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), np.around(np.std(q_fit_bs), 4)])
        print(tbl)
    if ci and verbose:
        tbl.field_names = ['parameter', 'value', 'se', 'cilo_95', 'cihi_95', 'n']
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), np.around(np.std(p_fit_bs), 4),
                          np.around(np.mean(p_fit_bs)-np.std(p_fit_bs)*1.96, 4),
                          np.around(np.mean(p_fit_bs)+np.std(p_fit_bs)*1.96, 4), k])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), np.around(np.std(q_fit_bs), 4),
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

    if fit:
        if (.95<np.mean(q_fit_bs)<1.05) | (.95<q_fit<1.05):
            model = Pareto_icdf(u=np.array(np.random.uniform(.0, 1., len(x))), b=b, p=np.mean(p_fit_bs))
        else:
            model, u = IB1_icdf_ne(x=x, b=b, p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        mae = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=model, x_hat=x, b=b, parms=[np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'n']
            tbl_gof.add_row(['GOF', np.around(aic, 4), np.around(bic, 4), np.around(mae, 4), np.around(mse, 4),
                                    np.around(rmse, 4), np.around(rrmse, 4), np.around(ll, 4), np.around(n, 4)])
            print("\n", tbl_gof, "\n")

    if return_parameters:
        return np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs)


def GB1fit(x, b, x0, bootstraps=250, method='SLSQP',
           verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, return_parameters=False,
           plot_cosmetics={'bins': 50, 'col_fit': 'blue', 'col_model': 'orange'}
              basinhoppin_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                                   'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
              SLSQP_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                             'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit the GB1 distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [a,p,q] or simply (a,p,q)
    :param bootstraps: amount of bootstraps
    :param method: # default: SLSQP (local optimization, much faster), 'L-BFGS-B' (global optimization, but slower)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhoppin_options: dictionary with optimization options
    :param SLSQP_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """
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
        opts = basinhoppin_options

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
                                "bounds": (a_bnd, bnds, bnds,),
                                "constraints": constr}

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
                                  method='SLSQP',
                                  bounds=(a_bnd, bnds, bnds,),
                                  args=(boot_sample, b),
                                  jac=opts['jac'],
                                  tol=opts['tol'],
                                  callback=opts['callback'],
                                  options=({'maxiter': opts['maxiter'], 'disp': False, 'iprint': opts['iprint'],
                                            'fun': opts['func'], 'ftol': opts['ftol'], 'eps': opts['eps']}),
                                  constraints=constr)

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
        tbl.add_row(['a', np.around(np.mean(a_fit_bs), 4), np.around(np.std(a_fit_bs), 4)])
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), np.around(np.std(p_fit_bs), 4)])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), np.around(np.std(q_fit_bs), 4)])
        print(tbl)

    if ci and verbose:
        tbl.field_names = ['parameter', 'value', 'se', 'cilo_95', 'cihi_95', 'n']
        tbl.add_row(['a', np.around(np.mean(a_fit_bs), 4), np.around(np.std(a_fit_bs), 4),
                          np.around(np.mean(a_fit_bs)-np.std(a_fit_bs)*1.96, 4),
                          np.around(np.mean(a_fit_bs)+np.std(a_fit_bs)*1.96, 4), k])
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), np.around(np.std(p_fit_bs), 4),
                          np.around(np.mean(p_fit_bs)-np.std(p_fit_bs)*1.96, 4),
                          np.around(np.mean(p_fit_bs)+np.std(p_fit_bs)*1.96, 4), k])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), np.around(np.std(q_fit_bs), 4),
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

    if fit:
        if (.95<np.mean(q_fit_bs)<1.05) | (.95<q_fit<1.05) and (-1.05<np.mean(a_fit_bs)<-.95) | (-1.05<a_fit<-.95):
            model = Pareto_icdf(u=np.array(np.random.uniform(.0, 1., len(x))), b=b, p=np.mean(p_fit_bs))
        else:
            model, u = GB1_icdf_ne(x=x, b=b, a=np.mean(a_fit_bs), p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        mae = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'n']
            tbl_gof.add_row(['GOF', np.around(aic, 4), np.around(bic, 4), np.around(mae, 4), np.around(mse, 4),
                                    np.around(rmse, 4), np.around(rrmse, 4), np.around(ll, 4), np.around(n, 4)])
            print("\n", tbl_gof, "\n")

    if return_parameters:
        return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs)



def GBfit(x, b, x0, bootstraps=250, method='SLSQP',
          verbose_bootstrap=False, ci=True, verbose=True, fit=False, plot=False, return_parameters=False,
          plot_cosmetics={'bins': 50, 'col_fit': 'blue', 'col_model': 'orange'}
              basinhoppin_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                                   'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123},
              SLSQP_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
                             'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}):
    """
    Function to fit the GB distribution to data
    :param x: linspace or data, fixed
    :param b: location parameter, fixed
    :param x0: initial guess, np.array [a,c,p,q] or simply (q,c,p,q)
    :param bootstraps: amount of bootstraps
    :param method: # default: SLSQP (local optimization, much faster), 'L-BFGS-B' (global optimization, but slower)
    :param verbose_bootstrap: display each bootstrap
    :param ci: default ci displayed
    :param verbose: default true
    :param fit: gof measurements, default false
    :param plot: plot fit vs model
    :param return_parameters: default, parameters are not returned
    :param plot_cosmetics: dictionary, add some simple cosmetics, important for setting bins (default: bins=50)
    :param basinhoppin_options: dictionary with optimization options
    :param SLSQP_options: dictionary with optimization options
    :return: fitted parameters, gof, plot
    """
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
        opts = basinhoppin_options

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
                                "bounds": (a_bnd, c_bnd, bnds, bnds,),
                                "constraints": constr}

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
                                  method='SLSQP',
                                  bounds=(a_bnd, c_bnd, bnds, bnds,),
                                  args=(boot_sample, b),
                                  jac=opts['jac'],
                                  tol=opts['tol'],
                                  callback=opts['callback'],
                                  options=({'maxiter': opts['maxiter'], 'disp': False, 'iprint': opts['iprint'],
                                            'fun': opts['func'], 'ftol': opts['ftol'], 'eps': opts['eps']}),
                                  constraints=constr)

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
        tbl.add_row(['a', np.around(np.mean(a_fit_bs), 4), np.around(np.std(a_fit_bs), 4)])
        tbl.add_row(['c', np.around(np.mean(c_fit_bs), 4), np.around(np.std(c_fit_bs), 4)])
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), np.around(np.std(p_fit_bs), 4)])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), np.around(np.std(q_fit_bs), 4)])
        print(tbl)

    if ci and verbose:
        tbl.field_names = ['parameter', 'value', 'se', 'cilo_95', 'cihi_95', 'n']
        tbl.add_row(['a', np.around(np.mean(a_fit_bs), 4), np.around(np.std(a_fit_bs), 4),
                          np.around(np.mean(a_fit_bs)-np.std(a_fit_bs)*1.96, 4),
                          np.around(np.mean(a_fit_bs)+np.std(a_fit_bs)*1.96, 4), k])
        tbl.add_row(['c', np.around(np.mean(c_fit_bs), 4), np.around(np.std(c_fit_bs), 4),
                          np.around(np.mean(c_fit_bs)-np.std(c_fit_bs)*1.96, 4),
                          np.around(np.mean(c_fit_bs)+np.std(c_fit_bs)*1.96, 4), k])
        tbl.add_row(['p', np.around(np.mean(p_fit_bs), 4), np.around(np.std(p_fit_bs), 4),
                          np.around(np.mean(p_fit_bs)-np.std(p_fit_bs)*1.96, 4),
                          np.around(np.mean(p_fit_bs)+np.std(p_fit_bs)*1.96, 4), k])
        tbl.add_row(['q', np.around(np.mean(q_fit_bs), 4), np.around(np.std(q_fit_bs), 4),
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
    if fit:
        if (.95<np.mean(q_fit_bs)<1.05) | (.95<q_fit<1.05) \
                and (-1.05<np.mean(a_fit_bs)<-.95) | (-1.05<a_fit<-.95) \
                and (-.05<np.mean(c_fit_bs)<.05) | (-.05<c_fit<-.05):
            model = Pareto_icdf(u=np.array(np.random.uniform(.0, 1., len(x))), b=b, p=np.mean(p_fit_bs))
        else:
            model, u = GB_icdf_ne(x=x, b=b, a=np.mean(a_fit_bs), c=np.mean(c_fit_bs), p=np.mean(p_fit_bs), q=np.mean(q_fit_bs))
        mae = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mae
        mse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).mse
        rmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rmse
        rrmse = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).rrmse
        aic = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).aic
        bic = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).bic
        n = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).n
        ll = gof(x=model, x_hat=x, b=b, parms=[np.mean(a_fit_bs), np.mean(c_fit_bs), np.mean(p_fit_bs), np.mean(q_fit_bs)]).ll
        if verbose:
            tbl_gof.field_names = ['', 'AIC', 'BIC', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'LL', 'n']
            tbl_gof.add_row(['GOF', np.around(aic, 4), np.around(bic, 4), np.around(mae, 4), np.around(mse, 4),
                                    np.around(rmse, 4), np.around(rrmse, 4), np.around(ll, 4), np.around(n, 4)])
            print("\n", tbl_gof, "\n")
    if return_parameters:
        return np.mean(a_fit_bs), np.std(a_fit_bs), np.mean(c_fit_bs), np.std(c_fit_bs), np.mean(p_fit_bs), np.std(p_fit_bs), np.mean(q_fit_bs), np.std(q_fit_bs)


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
