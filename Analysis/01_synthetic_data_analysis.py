from Pareto2GBfit import *
import os, matplotlib
import numpy as np
import pandas as pd
from pandas import ExcelWriter
import openpyxl

random.seed(104891)

plot_data = True
run_descriptives = True
run_optimize = True

# windows paths
if os.name == 'nt':
    graphspath = 'D:/OneDrive/Studium/Masterarbeit/Python/graphs/'
    descriptivespath = 'D:/OneDrive/Studium/Masterarbeit/Python/descriptives/'

# mac paths
if os.name == 'posix':
    graphspath = '/Users/Fabian/OneDrive/Studium/Masterarbeit/Python/graphs/'
    descriptivespath = '/Users/Fabian/OneDrive/Studium/Masterarbeit/Python/descriptives/'

# set figsize
import matplotlib.pyplot as plt
print(plt.rcParams.get('figure.figsize'))
plt.rcParams['figure.figsize'] = 10, 8

"""
--------------------------------
Define Functions
--------------------------------
"""
def prep_fit_results_for_table(fit_result):
    """
    prepares the returned vector of the optimization for a simplified exporting to dataframe/Excel
    :param fit_result: result of Paretobranchfit, needs return_bestmodel=True
    :return: returns vector with same shape, doesn't matter which model is best
    """
    bestfit, fit_result, placeholder, list = fit_result[0], np.array(fit_result[1]).tolist(), ['--', '--'], []
    for el in fit_result[:-2]:
        list.append('{:.3f}'.format(el))
    for el in fit_result[-2:]:
        list.append('{:.3f}'.format(int(el)))
    if bestfit == "Pareto_best" or len(list) == 16:
        out = placeholder * 2 #a,c
        out = out + list[0:2] + placeholder #p,q
        out = out + list[2:] #q, rest
    if bestfit == "IB1_best" or len(list) == 18:
        out = placeholder * 2 #a,c
        out = out + list #p,q, rest
    if bestfit == "GB1_best" or len(list) == 20:
        out = list[0:2] + placeholder #c
        out = out + list[2:] #rest
    if bestfit == "GB_best" or len(list) == 22:
        out = list
    del out[15] # remove soe
    del out[13] # remove rrmse
    del out[10] # remove mae
    del out[9] # remove bic
    return out # returns: parameters, aic, mse, rrmse, ll, ... (always same structure)



""" 
--------------------------------
1. Data Generating Process 
--------------------------------
"""

# Pareto Parameters
b, p = 250, 2.5

# size of overall synthetic / noise data
n = 10000

# linspace
xmin = 0.1
xmax = 10000
x = linspace(xmin, xmax, n)

# random uniform
u = np.array(np.random.uniform(.0, 1., n))
u = np.sort(u)

# Pareto simulated data (no noise)
Pareto_data = Pareto_icdf(u, b, p)
# alternatively: simulate Pareto_data numerically evaluated (ne) along pdf
# Pareto_data_ne, u_ne = Pareto_icdf_ne(x[x > b], b, p)

# 1. Small Gaussian noise
mu = 0
sigma1 = 100
gauss_noise_1 = np.random.normal(mu, sigma1, size=n)

# 2. Large Gaussian noise
sigma2 = 200
gauss_noise_2 = np.random.normal(mu, sigma2, size=n)

# Pareto simulated data + het. Gaussian noise
Pareto_data_gauss_noise_1 = Pareto_data + gauss_noise_1
Pareto_data_gauss_noise_2 = Pareto_data + gauss_noise_2

# Define function that generates heteroscedastic noise
def het(x, sigma, s=1.):
    x = np.array(x)
    n = np.size(x)
    e = x*(s * (np.random.normal(0, sigma, n)))
    return x + e

# sensitivity
s = 15e-4

# 3. Small heteroscedastic Gaussian noise
Pareto_data_het_noise_1 = het(x=Pareto_data, sigma=sigma1, s=s)

# 4. Large heteroscedastic Gaussian noise
Pareto_data_het_noise_2 = het(x=Pareto_data, sigma=sigma2, s=s)

# 5. Robustness Check: Generate IB1 distrib. data which are NOT Pareto (i.e. q!=1)
IB1_data = IB1_icdf_ne(x=x, b=b, p=p, q=5)

# 6. Robustness Check: Generate GB1 distrib. data which are NOT Pareto (i.e. a!=-1)
GB1_data = GB1_icdf_ne(x=x, b=b, p=p, q=5, a=-5)



"""
--------------------------------
2. Plot data
--------------------------------
"""

if plot_data:
    # check gaussian noise data
    plt.scatter(u, Pareto_data_gauss_noise_2, marker="o", s=2, color='blue', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=N(0,200^2)$')
    plt.scatter(u, Pareto_data_gauss_noise_1, marker="o", s=2, color='orangered', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=N(0,100^2)$')
    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{Pareto}(b=250, p=2.5)=x$')
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x');
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'icdf_gauss_noise.' + type, dpi=300, format=type)
    plt.show()
    plt.close()

    # check heteroscedastic noise data
    plt.scatter(u, Pareto_data_het_noise_2, marker="o", s=2, color='blue', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=x\cdot s\cdot N(0,200^2)$')
    plt.scatter(u, Pareto_data_het_noise_1, marker="o", s=2, color='orangered', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=x\cdot s\cdot N(0,100^2)$')
    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{Pareto}(b=250, p=2.5)=x$')
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'icdf_het_noise.' + type, dpi=300, format=type)
    plt.show()
    plt.close()

    # check NON Pareto data (IB1, GB1)
    plt.scatter(IB1_data[1], IB1_data[0], marker="o", s=2, color='blue', alpha=.75, label='IB1 data with q=2')
    plt.scatter(IB1_data[1], GB1_data[0], marker="o", s=2, color='orangered', alpha=.75, label='GB1 data with a=-4')
    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{Pareto}(b=250, p=2.5)=x$')
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'NON_Pareto_IB1_GB1.' + type, dpi=300, format=type)
    plt.show()
    plt.close()


"""
--------------------------------
2. Descriptive Stats
--------------------------------
"""

if run_descriptives:
    # sort
    Pareto_data = np.sort(Pareto_data)
    Pareto_data_gauss_noise_1 = np.sort(Pareto_data_gauss_noise_1)
    Pareto_data_gauss_noise_2 = np.sort(Pareto_data_gauss_noise_2)
    Pareto_data_het_noise_1 = np.sort(Pareto_data_het_noise_1)
    Pareto_data_het_noise_2 = np.sort(Pareto_data_het_noise_2)

    # write descriptives to dataframe
    df_synthetic_data_descriptives = pd.DataFrame(np.array([['N', '{:d}'.format(np.size(Pareto_data)), '{:d}'.format(np.size(Pareto_data_gauss_noise_1)), '{:d}'.format(np.size(Pareto_data_gauss_noise_2)), '{:d}'.format(np.size(Pareto_data_het_noise_1)), '{:d}'.format(np.size(Pareto_data_het_noise_2))],
                                                            ['mean', '{:.2f}'.format(np.mean(Pareto_data)), '{:.2f}'.format(np.mean(Pareto_data_gauss_noise_1)), '{:.2f}'.format(np.mean(Pareto_data_gauss_noise_2)), '{:.2f}'.format(np.mean(Pareto_data_het_noise_1)), '{:.2f}'.format(np.mean(Pareto_data_het_noise_2))],
                                                            ['sd', '{:.2f}'.format(np.std(Pareto_data)), '{:.2f}'.format(np.std(Pareto_data_gauss_noise_1)), '{:.2f}'.format(np.std(Pareto_data_gauss_noise_2)), '{:.2f}'.format(np.std(Pareto_data_het_noise_1)), '{:.2f}'.format(np.std(Pareto_data_het_noise_2))],
                                                            ['lower bound b', b, b, b, b, b],
                                                            ['min', '{:.2f}'.format(np.min(Pareto_data)), '{:.2f}'.format(np.min(Pareto_data_gauss_noise_1)), '{:.2f}'.format(np.min(Pareto_data_gauss_noise_2)), '{:.2f}'.format(np.min(Pareto_data_het_noise_1)), '{:.2f}'.format(np.min(Pareto_data_het_noise_2))],
                                                            ['p50', '{:.2f}'.format(np.percentile(Pareto_data, q=.5)), '{:.2f}'.format(np.percentile(Pareto_data_gauss_noise_1, q=.5)), '{:.2f}'.format(np.percentile(Pareto_data_gauss_noise_2, q=.5)), '{:.2f}'.format(np.percentile(Pareto_data_het_noise_1, q=.5)), '{:.2f}'.format(np.percentile(Pareto_data_het_noise_2, q=.5))],
                                                            ['p75', '{:.2f}'.format(np.percentile(Pareto_data, q=.75)), '{:.2f}'.format(np.percentile(Pareto_data_gauss_noise_1, q=.75)), '{:.2f}'.format(np.percentile(Pareto_data_gauss_noise_2, q=.75)), '{:.2f}'.format(np.percentile(Pareto_data_het_noise_1, q=.75)), '{:.2f}'.format(np.percentile(Pareto_data_het_noise_2, q=.75))],
                                                            ['p90', '{:.2f}'.format(np.percentile(Pareto_data, q=.9)), '{:.2f}'.format(np.percentile(Pareto_data_gauss_noise_1, q=.9)), '{:.2f}'.format(np.percentile(Pareto_data_gauss_noise_2, q=.9)), '{:.2f}'.format(np.percentile(Pareto_data_het_noise_1, q=.9)), '{:.2f}'.format(np.percentile(Pareto_data_het_noise_2, q=.9))],
                                                            ['p99', '{:.2f}'.format(np.percentile(Pareto_data, q=.99)), '{:.2f}'.format(np.percentile(Pareto_data_gauss_noise_1, q=.99)), '{:.2f}'.format(np.percentile(Pareto_data_gauss_noise_2, q=.99)), '{:.2f}'.format(np.percentile(Pareto_data_het_noise_1, q=.99)), '{:.2f}'.format(np.percentile(Pareto_data_het_noise_2, q=.99))],
                                                            ['p99.9', '{:.2f}'.format(np.percentile(Pareto_data, q=.999)), '{:.2f}'.format(np.percentile(Pareto_data_gauss_noise_1, q=.999)), '{:.2f}'.format(np.percentile(Pareto_data_gauss_noise_2, q=.999)), '{:.2f}'.format(np.percentile(Pareto_data_het_noise_1, q=.999)), '{:.2f}'.format(np.percentile(Pareto_data_het_noise_2, q=.999))],
                                                            ['max', '{:.2f}'.format(np.max(Pareto_data)), '{:.2f}'.format(np.max(Pareto_data_gauss_noise_1)), '{:.2f}'.format(np.max(Pareto_data_gauss_noise_2)), '{:.2f}'.format(np.max(Pareto_data_het_noise_1)), '{:.2f}'.format(np.max(Pareto_data_het_noise_2))],
                                                            ]),
                                                 columns=['', 'Pareto_data', 'Pareto_data_gauss_noise_1', 'Pareto_data_gauss_noise_2', 'Pareto_data_het_noise_1', 'Pareto_data_het_noise_2'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'synthetic_data_descriptives.xlsx', mode='w') as writer:
        df_synthetic_data_descriptives.to_excel(writer, sheet_name='synthetic_data_descriptives', index=False)

    # sort
    Pareto_data = np.sort(Pareto_data)
    IB1_data = np.sort(IB1_data[0])
    GB1_data = np.sort(GB1_data[0])

    # write descriptives to dataframe
    df_synthetic_non_Pareto_descriptives = pd.DataFrame(np.array([['N', '{:d}'.format(np.size(IB1_data)), '{:d}'.format(np.size(GB1_data))],
                                                            ['mean', '{:.2f}'.format(np.mean(IB1_data)), '{:.2f}'.format(np.mean(GB1_data))],
                                                            ['sd', '{:.2f}'.format(np.std(IB1_data)), '{:.2f}'.format(np.std(GB1_data))],
                                                            ['lower bound b', b, b],
                                                            ['min', '{:.2f}'.format(np.min(IB1_data)), '{:.2f}'.format(np.min(GB1_data))],
                                                            ['p50', '{:.2f}'.format(np.percentile(IB1_data, q=.5)), '{:.2f}'.format(np.percentile(GB1_data, q=.5))],
                                                            ['p75', '{:.2f}'.format(np.percentile(IB1_data, q=.75)), '{:.2f}'.format(np.percentile(GB1_data, q=.75))],
                                                            ['p90', '{:.2f}'.format(np.percentile(IB1_data, q=.9)), '{:.2f}'.format(np.percentile(GB1_data, q=.9))],
                                                            ['p99', '{:.2f}'.format(np.percentile(IB1_data, q=.99)), '{:.2f}'.format(np.percentile(GB1_data, q=.99))],
                                                            ['p99.9', '{:.2f}'.format(np.percentile(IB1_data, q=.999)), '{:.2f}'.format(np.percentile(GB1_data, q=.999))],
                                                            ['max', '{:.2f}'.format(np.max(IB1_data)), '{:.2f}'.format(np.max(GB1_data))],
                                                            ]),
                                                 columns=['', 'IB1_data', 'GB1_data,'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'synthetic_non_Pareto_descriptives.xlsx', mode='w') as writer:
        df_synthetic_non_Pareto_descriptives.to_excel(writer, sheet_name='descriptives', index=False)



""" 
--------------------------------
3. Fitting
--------------------------------
"""

if run_optimize:

    ## with LRtest rejection criterion

    Pareto_data_parms_LR = Paretobranchfit(x=Pareto_data, x0=(-1, .5, 1, 1), b=250,
                                        bootstraps=(250, 250, 250, 250),
                                        return_bestmodel=True, rejection_criterion='LRtest', plot=True,
                                        plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'}) # best: GB

    Pareto_data_gauss_noise_1_parms_LR = Paretobranchfit(x=Pareto_data_gauss_noise_1, x0=(-1, .5, 1, 1), b=250,
                                                      bootstraps=(250, 250, 250, 250),
                                                      return_bestmodel=True, rejection_criterion='LRtest', plot=True,
                                                      plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'}) # best: GB1

    Pareto_data_gauss_noise_2_parms_LR = Paretobranchfit(x=Pareto_data_gauss_noise_2, x0=(-1, .5, 1, 1), b=250,
                                                      bootstraps=(250, 250, 250, 250),
                                                      return_bestmodel=True, rejection_criterion='LRtest', plot=True,
                                                      plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'}) # best: GB

    Pareto_data_het_noise_1_parms_LR = Paretobranchfit(x=Pareto_data_het_noise_1, x0=(-1, .5, 1, 1), b=250,
                                                    bootstraps=(250, 250, 250, 250),
                                                    return_bestmodel=True, rejection_criterion='LRtest', plot=True,
                                                    plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'}) # best: GB1

    Pareto_data_het_noise_2_parms_LR = Paretobranchfit(x=Pareto_data_het_noise_2, x0=(-1, .5, 1, 1), b=250,
                                                    bootstraps=(250, 250, 250, 250),
                                                    return_bestmodel=True, rejection_criterion='LRtest', plot=True,
                                                    plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'}) # best: GB1

    ## with AIC rejection criterion

    Pareto_data_parms_Aic = Paretobranchfit(x=Pareto_data, x0=(-1, .5, 1, 1), b=250,
                                        bootstraps=(250, 250, 250, 250),
                                        return_bestmodel=True, rejection_criterion='AIC', plot=True,
                                        plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'}) # best: IB1

    Pareto_data_gauss_noise_1_parms_Aic = Paretobranchfit(x=Pareto_data_gauss_noise_1, x0=(-1, .5, 1, 1), b=250,
                                                      bootstraps=(250, 250, 250, 250),
                                                      return_bestmodel=True, rejection_criterion='AIC', plot=True,
                                                      plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'}) # best: GB1

    Pareto_data_gauss_noise_2_parms_Aic = Paretobranchfit(x=Pareto_data_gauss_noise_2, x0=(-1, .5, 1, 1), b=250,
                                                      bootstraps=(250, 250, 250, 250),
                                                      return_bestmodel=True, rejection_criterion='AIC', plot=True,
                                                      plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'}) # best: GB

    Pareto_data_het_noise_1_parms_Aic = Paretobranchfit(x=Pareto_data_het_noise_1, x0=(-1, .5, 1, 1), b=250,
                                                    bootstraps=(250, 250, 250, 250),
                                                    return_bestmodel=True, rejection_criterion='AIC', plot=True,
                                                    plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'}) # best: GB1

    Pareto_data_het_noise_2_parms_Aic = Paretobranchfit(x=Pareto_data_het_noise_2, x0=(-1, .5, 1, 1), b=250,
                                                    bootstraps=(250, 250, 250, 250),
                                                    return_bestmodel=True, rejection_criterion='AIC', plot=True,
                                                    plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'}) # best: GB1

    # Robustness Check: NON Pareto data

    IB1_non_Pareto_parms_LR = Paretobranchfit(x=IB1_data, x0=(-1, .5, 1, 1), b=250,
                                              bootstraps=(250, 250, 250, 250),
                                              return_bestmodel=True, rejection_criterion='LRtest', plot=True,
                                              plot_cosmetics={'bins': 300, 'col_data': 'blue', 'col_fit': 'red'})

    IB1_non_Pareto_parms_AIC = Paretobranchfit(x=IB1_data, x0=(-1, .5, 1, 1), b=250,
                                               bootstraps=(250, 250, 250, 250),
                                               return_bestmodel=True, rejection_criterion='AIC', plot=True,
                                               plot_cosmetics={'bins': 300, 'col_data': 'blue', 'col_fit': 'red'})

    GB1_non_Pareto_parms_LR = Paretobranchfit(x=GB1_data, x0=(-1, .5, 1, 1), b=250,
                                               bootstraps=(250, 250, 250, 250),
                                               return_bestmodel=True, rejection_criterion='LRtest', plot=True,
                                               plot_cosmetics={'bins': 300, 'col_data': 'blue', 'col_fit': 'red'})

    GB1_non_Pareto_parms_AIC = Paretobranchfit(x=GB1_data, x0=(-1, .5, 1, 1), b=250,
                                               bootstraps=(250, 250, 250, 250),
                                               return_bestmodel=True, rejection_criterion='AIC', plot=True,
                                               plot_cosmetics={'bins': 300, 'col_data': 'blue', 'col_fit': 'red'})



"""
--------------------------------
4. Save fitted parms to table
--------------------------------
"""

# shorter names
parms1 = prep_fit_results_for_table(Pareto_data_parms_AIC)
parms2 = prep_fit_results_for_table(Pareto_data_gauss_noise_1_parms_AIC)
parms3 = prep_fit_results_for_table(Pareto_data_gauss_noise_2_parms_AIC)
parms4 = prep_fit_results_for_table(Pareto_data_het_noise_1_parms_AIC)
parms5 = prep_fit_results_for_table(Pareto_data_het_noise_2_parms_AIC)

if run_optimize:
    df_synthetic_fit_parms_AIC = pd.DataFrame(np.array([['best fitted model', '{}'.format(Pareto_data_parms[0]),  '{}'.format(Pareto_data_gauss_noise_1_parms[0]),  '{}'.format(Pareto_data_gauss_noise_2_parms[0]),  '{}'.format(Pareto_data_het_noise_1_parms[0]),  '{}'.format(Pareto_data_het_noise_2_parms[0])],
                                                    ['a',               '{}'.format(parms1[0]),     '{}'.format(parms2[0]),     '{}'.format(parms3[0]),     '{}'.format(parms4[0]),     '{}'.format(parms5[0])],
                                                    [' ',               '{}'.format(parms1[1]),     '{}'.format(parms2[1]),     '{}'.format(parms3[1]),     '{}'.format(parms4[1]),     '{}'.format(parms5[1])],
                                                    ['c',               '{}'.format(parms1[2]),     '{}'.format(parms2[2]),     '{}'.format(parms3[2]),     '{}'.format(parms4[2]),     '{}'.format(parms5[2])],
                                                    [' ',               '{}'.format(parms1[3]),     '{}'.format(parms2[3]),     '{}'.format(parms3[3]),     '{}'.format(parms4[3]),     '{}'.format(parms5[3])],
                                                    ['p',               '{}'.format(parms1[4]),     '{}'.format(parms2[4]),     '{}'.format(parms3[4]),     '{}'.format(parms4[4]),     '{}'.format(parms5[4])],
                                                    [' ',               '{}'.format(parms1[5]),     '{}'.format(parms2[5]),     '{}'.format(parms3[5]),     '{}'.format(parms4[5]),     '{}'.format(parms5[5])],
                                                    ['q',               '{}'.format(parms1[6]),     '{}'.format(parms2[6]),     '{}'.format(parms3[6]),     '{}'.format(parms4[6]),     '{}'.format(parms5[6])],
                                                    [' ',               '{}'.format(parms1[7]),     '{}'.format(parms2[7]),     '{}'.format(parms3[7]),     '{}'.format(parms4[7]),     '{}'.format(parms5[7])],
                                                    ['lower bound b',   '{}'.format(b),             '{}'.format(b),             '{}'.format(b),             '{}'.format(b),             '{}'.format(b)],
                                                    ['LL',              '{}'.format(parms1[11]),    '{}'.format(parms2[11]),    '{}'.format(parms3[11]),    '{}'.format(parms4[11]),    '{}'.format(parms5[11])],
                                                    ['AIC',             '{}'.format(parms1[8]),     '{}'.format(parms2[8]),     '{}'.format(parms3[8]),     '{}'.format(parms4[8]),     '{}'.format(parms5[8])],
                                                    ['MSE',             '{}'.format(parms1[9]),     '{}'.format(parms2[9]),     '{}'.format(parms3[9]),     '{}'.format(parms4[9]),     '{}'.format(parms5[9])],
                                                    ['RMSE',            '{}'.format(parms1[10]),    '{}'.format(parms2[10]),    '{}'.format(parms3[10]),    '{}'.format(parms4[10]),    '{}'.format(parms5[10])],
                                                    ['emp. mean',       '{}'.format(parms1[12]),    '{}'.format(parms2[12]),    '{}'.format(parms3[12]),    '{}'.format(parms4[12]),    '{}'.format(parms5[12])],
                                                    ['emp. var.',       '{}'.format(parms1[13]),    '{}'.format(parms2[13]),    '{}'.format(parms3[13]),    '{}'.format(parms4[13]),    '{}'.format(parms5[13])],
                                                    ['pred. mean',      '{}'.format(parms1[14]),    '{}'.format(parms2[14]),    '{}'.format(parms3[14]),    '{}'.format(parms4[14]),    '{}'.format(parms5[14])],
                                                    ['pred. var.',      '{}'.format(parms1[15]),    '{}'.format(parms2[15]),    '{}'.format(parms3[15]),    '{}'.format(parms4[15]),    '{}'.format(parms5[15])],
                                                    ['n',               '{}'.format(parms1[16]),    '{}'.format(parms2[16]),    '{}'.format(parms3[16]),    '{}'.format(parms4[16]),    '{}'.format(parms5[16])],
                                                    ]),
                                                 columns=['', 'Pareto_data', 'Pareto_data_gauss_noise_1', 'Pareto_data_gauss_noise_2', 'Pareto_data_het_noise_1', 'Pareto_data_het_noise_2'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'synthetic_data_fit_results.xlsx', engine='openpyxl', mode='w') as writer:
        df_synthetic_fit_parms_AIC.to_excel(writer, sheet_name='synthetic_data_fit_results_AIC', index=False)


# shorter names
parms10 = Pareto_data_parms_LR[1] #IB1
parms11 = Pareto_data_gauss_noise_1_parms_LR[1] #GB1
parms12 = Pareto_data_gauss_noise_2_parms_LR[1] #GB
parms13 = Pareto_data_het_noise_1_parms_LR[1] #IB1
parms14 = Pareto_data_het_noise_2_parms_LR[1] #IB1

if run_optimize:
    df_synthetic_fit_parms_LR = pd.DataFrame(np.array([['best fitted model', '{}'.format(Pareto_data_parms[0]),  '{}'.format(Pareto_data_gauss_noise_1_parms[0]),  '{}'.format(Pareto_data_gauss_noise_2_parms[0]),  '{}'.format(Pareto_data_het_noise_1_parms[0]),  '{}'.format(Pareto_data_het_noise_2_parms[0])],
                                                    ['a',               '{}'.format(parms10[0]),     '{}'.format(parms11[0]),     '{}'.format(parms12[0]),     '{}'.format(parms13[0]),     '{}'.format(parms14[0])],
                                                    [' ',               '{}'.format(parms10[1]),     '{}'.format(parms11[1]),     '{}'.format(parms12[1]),     '{}'.format(parms13[1]),     '{}'.format(parms14[1])],
                                                    ['c',               '{}'.format(parms10[2]),     '{}'.format(parms11[2]),     '{}'.format(parms12[2]),     '{}'.format(parms13[2]),     '{}'.format(parms14[2])],
                                                    [' ',               '{}'.format(parms10[3]),     '{}'.format(parms11[3]),     '{}'.format(parms12[3]),     '{}'.format(parms13[3]),     '{}'.format(parms14[3])],
                                                    ['p',               '{}'.format(parms10[4]),     '{}'.format(parms11[4]),     '{}'.format(parms12[4]),     '{}'.format(parms13[4]),     '{}'.format(parms14[4])],
                                                    [' ',               '{}'.format(parms10[5]),     '{}'.format(parms11[5]),     '{}'.format(parms12[5]),     '{}'.format(parms13[5]),     '{}'.format(parms14[5])],
                                                    ['q',               '{}'.format(parms10[6]),     '{}'.format(parms11[6]),     '{}'.format(parms12[6]),     '{}'.format(parms13[6]),     '{}'.format(parms14[6])],
                                                    [' ',               '{}'.format(parms10[7]),     '{}'.format(parms11[7]),     '{}'.format(parms12[7]),     '{}'.format(parms13[7]),     '{}'.format(parms14[7])],
                                                    ['lower bound b',   '{}'.format(b),             '{}'.format(b),             '{}'.format(b),             '{}'.format(b),             '{}'.format(b)],
                                                    ['LL',              '{}'.format(parms10[11]),    '{}'.format(parms11[11]),    '{}'.format(parms12[11]),    '{}'.format(parms13[11]),    '{}'.format(parms14[11])],
                                                    ['AIC',             '{}'.format(parms10[8]),     '{}'.format(parms11[8]),     '{}'.format(parms12[8]),     '{}'.format(parms13[8]),     '{}'.format(parms14[8])],
                                                    ['MSE',             '{}'.format(parms10[9]),     '{}'.format(parms11[9]),     '{}'.format(parms12[9]),     '{}'.format(parms13[9]),     '{}'.format(parms14[9])],
                                                    ['RMSE',            '{}'.format(parms10[10]),    '{}'.format(parms11[10]),    '{}'.format(parms12[10]),    '{}'.format(parms13[10]),    '{}'.format(parms14[10])],
                                                    ['emp. mean',       '{}'.format(parms10[12]),    '{}'.format(parms11[12]),    '{}'.format(parms12[12]),    '{}'.format(parms13[12]),    '{}'.format(parms14[12])],
                                                    ['emp. var.',       '{}'.format(parms10[13]),    '{}'.format(parms11[13]),    '{}'.format(parms12[13]),    '{}'.format(parms13[13]),    '{}'.format(parms14[13])],
                                                    ['pred. mean',      '{}'.format(parms10[14]),    '{}'.format(parms11[14]),    '{}'.format(parms12[14]),    '{}'.format(parms13[14]),    '{}'.format(parms14[14])],
                                                    ['pred. var.',      '{}'.format(parms10[15]),    '{}'.format(parms11[15]),    '{}'.format(parms12[15]),    '{}'.format(parms13[15]),    '{}'.format(parms14[15])],
                                                    ['n',               '{}'.format(parms10[16]),    '{}'.format(parms11[16]),    '{}'.format(parms12[16]),    '{}'.format(parms13[16]),    '{}'.format(parms14[16])],
                                                    ]),
                                                 columns=['', 'Pareto_data', 'Pareto_data_gauss_noise_1', 'Pareto_data_gauss_noise_2', 'Pareto_data_het_noise_1', 'Pareto_data_het_noise_2'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'synthetic_data_fit_results.xlsx', engine='openpyxl', mode='a') as writer:
        df_synthetic_fit_parms_LR.to_excel(writer, sheet_name='synthetic_data_fit_results_LR', index=False)


    # NON Pareto data: shorter names
    parms15 = prep_fit_results_for_table(IB1_non_Pareto_parms_LR)
    parms16 = prep_fit_results_for_table(IB1_non_Pareto_parms_AIC)
    parms17 = prep_fit_results_for_table(GB1_non_Pareto_parms_LR)
    parms18 = prep_fit_results_for_table(GB1_non_Pareto_parms_AIC)

    df_non_Pareto_fit_results = pd.DataFrame(np.array([['best fitted model', '{}'.format(IB1_non_Pareto_parms_LR[0]), '{}'.format(IB1_non_Pareto_parms_AIC[0]), '{}'.format(GB1_non_Pareto_parms_LR[0]), '{}'.format(GB1_non_Pareto_parms_AIC[0])],
                                           ['a',               '{}'.format(parms15[0]),     '{}'.format(parms16[0]),    '{}'.format(parms17[0]),    '{}'.format(parms18[0])],
                                           [' ',               '{}'.format(parms15[1]),     '{}'.format(parms16[1]),    '{}'.format(parms17[1]),    '{}'.format(parms18[1])],
                                           ['c',               '{}'.format(parms15[2]),     '{}'.format(parms16[2]),    '{}'.format(parms17[2]),    '{}'.format(parms18[2])],
                                           [' ',               '{}'.format(parms15[3]),     '{}'.format(parms16[3]),    '{}'.format(parms17[3]),    '{}'.format(parms18[3])],
                                           ['p',               '{}'.format(parms15[4]),     '{}'.format(parms16[4]),    '{}'.format(parms17[4]),    '{}'.format(parms18[4])],
                                           [' ',               '{}'.format(parms15[5]),     '{}'.format(parms16[5]),    '{}'.format(parms17[5]),    '{}'.format(parms18[5])],
                                           ['q',               '{}'.format(parms15[6]),     '{}'.format(parms16[6]),    '{}'.format(parms17[6]),    '{}'.format(parms18[6])],
                                           [' ',               '{}'.format(parms15[7]),     '{}'.format(parms16[7]),    '{}'.format(parms17[7]),    '{}'.format(parms18[7])],
                                           ['lower bound b',   '{}'.format(b),              '{}'.format(b),             '{}'.format(b),             '{}'.format(b)],
                                           ['LL',              '{}'.format(parms15[11]),    '{}'.format(parms16[11]),   '{}'.format(parms17[11]),   '{}'.format(parms18[11])],
                                           ['AIC',             '{}'.format(parms15[8]),     '{}'.format(parms16[8]),    '{}'.format(parms17[8]),    '{}'.format(parms18[8])],
                                           ['MSE',             '{}'.format(parms15[9]),     '{}'.format(parms16[9]),    '{}'.format(parms17[9]),    '{}'.format(parms18[9])],
                                           ['RMSE',            '{}'.format(parms15[10]),    '{}'.format(parms16[10]),   '{}'.format(parms17[10]),   '{}'.format(parms18[10])],
                                           ['emp. mean',       '{}'.format(parms15[12]),    '{}'.format(parms16[12]),   '{}'.format(parms17[12]),   '{}'.format(parms18[12])],
                                           ['emp. var.',       '{}'.format(parms15[13]),    '{}'.format(parms16[13]),   '{}'.format(parms17[13]),   '{}'.format(parms18[13])],
                                           ['pred. mean',      '{}'.format(parms15[14]),    '{}'.format(parms16[14]),   '{}'.format(parms17[14]),   '{}'.format(parms18[14])],
                                           ['pred. var.',      '{}'.format(parms15[15]),    '{}'.format(parms16[15]),   '{}'.format(parms17[15]),   '{}'.format(parms18[15])],
                                           ['n',               '{}'.format(parms15[16]),    '{}'.format(parms16[16]),   '{}'.format(parms17[16]),   '{}'.format(parms18[16])],
                                           ['N',               '{}'.format(parms15[17]),    '{}'.format(parms16[17]),   '{}'.format(parms17[17]),   '{}'.format(parms18[17])],
                                           ]),
                                 columns=['', 'IB1_non_Pareto_LR', 'IB1_non_Pareto_AIC', 'GB1_non_Pareto_LR', 'GB1_non_Pareto_AIC'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'non_pareto_fit_results.xlsx', engine='openpyxl', mode='w') as writer:
        df_non_Pareto_fit_results.to_excel(writer, sheet_name='descriptives', index=False)


""" 
--------------------------------
5. Plot Fit vs data
--------------------------------
"""

if run_optimize:
    ### fit of Pareto_data
    print('best fit for Pareto_data:', Pareto_data_parms[0])

    # generate new data based on fitted parms and best model
    Pareto_data_fit = Pareto_icdf(u=u, b=b, p=Pareto_data_parms[1][0])

    # Note: when using latex, doubling {{}} -> latex text, tripling {{{}}} -> use variables form .format()

    # plt.scatter(u, Pareto_data_gauss_noise_1, marker="o", s=2, color='orangered', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=N(0,100^2)$')
    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{Pareto}}(b=250, p={{{}}})$'.format(p))
    plt.plot(u, Pareto_data_fit, color='red', alpha=.75, label=r'$icdf_{{Pareto}}(b=250, \hat{{p}}={{{}}})$'.format(np.around(Pareto_data_parms[1][0],3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_Pareto_data.' + type, dpi=300, format=type)
    plt.show()
    plt.close()


    ### fit of Pareto_data_gauss_noise_1
    print('best fit for Pareto_data_gauss_noise_1:', Pareto_data_gauss_noise_1_parms[0])

    # generate new data based on fitted parms and best model
    Pareto_data_gauss_noise_1_fit, u_temp = IB1_icdf_ne(x=Pareto_data, b=b, p=Pareto_data_gauss_noise_1_parms[1][0], q=Pareto_data_gauss_noise_1_parms[1][2])

    plt.scatter(u, Pareto_data_het_noise_1, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{Pareto}}(b=250, p={{{}}}) + \epsilon\,\,with\,\,\epsilon=N(0,{{{}}}^2)$'.format(p, sigma1))
    plt.plot(u_temp, Pareto_data_gauss_noise_1_fit, color='red', alpha=.75, label=r'$icdf_{{IB1}}(b=250, \hat{{p}}={{{}}}, \hat{{q}}={{{}}})$'.format(np.around(Pareto_data_gauss_noise_1_parms[1][0],3), np.around(Pareto_data_gauss_noise_1_parms[1][2],3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_Pareto_data_gauss_noise_1.' + type, dpi=300, format=type)
    plt.show()
    plt.close()

    ### Pareto_data_gauss_noise_2
    print('best fit for Pareto_data_gauss_noise_2:', Pareto_data_gauss_noise_2_parms[0])

    # generate new data based on fitted parms and best model
    Pareto_data_gauss_noise_2_fit, u_temp = IB1_icdf_ne(x=Pareto_data, b=b, p=Pareto_data_gauss_noise_2_parms[1][0], q=Pareto_data_gauss_noise_2_parms[1][2])

    plt.scatter(u, Pareto_data_het_noise_2, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{Pareto}}(b=250, p={{{}}}) + \epsilon\,\,with\,\,\epsilon=N(0,200^2)$'.format(p))
    plt.plot(u_temp, Pareto_data_gauss_noise_2_fit, color='red', alpha=.75, label=r'$icdf_{{IB1}}(b=250, \hat{{p}}={{{}}}, \hat{{q}}={{{}}})$'.format(np.around(Pareto_data_gauss_noise_2_parms[1][0],3), np.around(Pareto_data_gauss_noise_2_parms[1][2],3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_Pareto_data_gauss_noise_2.' + type, dpi=300, format=type)
    plt.show()
    plt.close()

    ### Pareto_data_het_noise_1
    print('best fit for Pareto_data_het_noise_1_parms:', Pareto_data_het_noise_1_parms[0])

    # generate new data based on fitted parms and best model
    Pareto_data_het_noise_1_fit = Pareto_icdf(u=u, b=b, p=Pareto_data_het_noise_1_parms[1][0])

    plt.scatter(u, Pareto_data_het_noise_1, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{Pareto}}(b=250, p={{{}}}) +\epsilon\,\,with\,\,\epsilon=x\cdot s\cdot N(0,{{{}}}^2)$'.format(p,sigma1))
    plt.plot(u_temp, Pareto_data_het_noise_1_fit, color='red', alpha=.75, label=r'$icdf_{{Pareto}}(b=250, \hat{{p}}={{{}}})$'.format(np.around(Pareto_data_het_noise_1_parms[1][0],3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_Pareto_data_het_noise_1.' + type, dpi=300, format=type)
    plt.show()
    plt.close()

    ### Pareto_data_het_noise_2
    print('best fit for Pareto_data_het_noise_2_parms:', Pareto_data_het_noise_2_parms[0])

    # generate new data based on fitted parms and best model
    Pareto_data_het_noise_2_fit, u_temp = GB_icdf_ne(x=Pareto_data, b=b,
                                             a=Pareto_data_het_noise_2_parms[1][0], c=Pareto_data_het_noise_2_parms[1][2],
                                             p=Pareto_data_het_noise_2_parms[1][4], q=Pareto_data_het_noise_2_parms[1][6])

    plt.scatter(u, Pareto_data_het_noise_2, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{Pareto}}(b=250, p={{{}}}) +\epsilon\,\,with\,\,\epsilon=x\cdot s\cdot N(0,{{{}}}^2)$'.format(p,sigma2))
    plt.plot(u_temp, Pareto_data_het_noise_2_fit, color='red', alpha=.75, label=r'$icdf_{{GB}}(b=250, \hat{{a}}={{{}}}, \hat{{c}}={{{}}}, \hat{{p}}={{{}}}, \hat{{q}}={{{}}})$'.format(np.around(Pareto_data_het_noise_2_parms[1][0],3), np.around(Pareto_data_het_noise_2_parms[1][2],3), np.around(Pareto_data_het_noise_2_parms[1][4],3), np.around(Pareto_data_het_noise_2_parms[1][6],3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_Pareto_data_het_noise_2.' + type, dpi=300, format=type)
    plt.show()
    plt.close()
