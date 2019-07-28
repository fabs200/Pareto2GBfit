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

def prep_fit_results_for_table(result):
    """
    prepares the returned vector of the optimization for a simplified exporting to dataframe/Excel
    :param fit_result: result of Paretobranchfit, needs return_bestmodel=True
    :return: returns vector with same shape, doesn't matter which model is best
    """
    bestfit, fit_result, placeholder, list = result[0], np.array(result[1]).tolist(), ['--', '--'], []

    if bestfit == "Pareto_best":

        for el in fit_result[:14]:
            list.append('{:.3f}'.format(el))
        for el in fit_result[14:16]:
            list.append('{}'.format(int(el)))
        for el in fit_result[16:18]:
            list.append('{:.3f}'.format(el))
        out = placeholder * 4 #a,c
        out = out + list[0:2] + list[16:18] + placeholder * 2 + list[2:16] #p,q, rest

    if bestfit == "IB1_best":

        for el in fit_result[:16]:
            list.append('{:.3f}'.format(el))
        for el in fit_result[16:18]:
            list.append('{}'.format(int(el)))
        for el in fit_result[18:22]:
            list.append('{:.3f}'.format(el))
        out = placeholder * 4 #a,c
        out = out + list[0:2] + list[18:20] + list[2:4] + list[20:22] + list[4:18] #p,q, rest

    if bestfit == "GB1_best":

        for el in fit_result[:18]:
            list.append('{:.3f}'.format(el))
        for el in fit_result[18:20]:
            list.append('{}'.format(int(el)))
        for el in fit_result[20:]:
            list.append('{:.3f}'.format(el))
        out = list[0:2] + list[20:22] + placeholder*2 + list[2:4] + list[22:24] + list[4:6] + list[24:26] + list[6:20] #q, rest

    if bestfit == "GB_best":

        for el in fit_result[:20]:
            list.append('{:.3f}'.format(el))
        for el in fit_result[20:22]:
            list.append('{}'.format(int(el)))
        for el in fit_result[22:]:
            list.append('{:.3f}'.format(el))
        out = list[0:2] + list[22:24] + list[2:4] + list[24:26] + list[4:6] + list[26:28] + list[6:8] + list[28:30] + list[8:22] #q, rest

    del out[23] # remove soe
    del out[21] # remove rrmse
    del out[18] # remove mae
    del out[17] # remove bic
    return out # returns: parameters, cis, aic, mse, rrmse, ll, ... (always same structure)


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

# 5. Robustness Check: Generate GB1 distrib. data which are NOT Pareto (i.e. a!=-1)
GB1_data = GB1_icdf_ne(x=x, b=b, p=p, q=5, a=-2)

# 6. Robustness Check: Generate GB distrib. data which are NOT Pareto (i.e. q!=1)
GB_data = GB_icdf_ne(x=x, a=-2, c=.5, b=b, p=p, q=10)




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
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
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

    # check NON Pareto data (B1, GB)
    plt.scatter(GB_data[1], GB_data[0], marker="o", s=2, color='blue', alpha=.75, label='GB data with a=-2, c=.5, q=10, p=2.5')
    plt.scatter(GB1_data[1], GB1_data[0], marker="o", s=2, color='orangered', alpha=.75, label='GB1 data with a=-5, q=5')
    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{Pareto}(b=250, p=2.5)=x$')
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'NON_Pareto_GB_GB1.' + type, dpi=300, format=type)
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
    GB1_data = np.sort(GB1_data[0])
    GB_data = np.sort(GB_data[0])

    # write descriptives to dataframe
    df_synthetic_non_Pareto_descriptives = pd.DataFrame(np.array([['N', '{:d}'.format(np.size(GB1_data)), '{:d}'.format(np.size(GB_data))],
                                                            ['mean', '{:.2f}'.format(np.mean(GB1_data)), '{:.2f}'.format(np.mean(GB_data))],
                                                            ['sd', '{:.2f}'.format(np.std(GB1_data)), '{:.2f}'.format(np.std(GB_data))],
                                                            ['lower bound b', b, b],
                                                            ['min', '{:.2f}'.format(np.min(GB1_data)), '{:.2f}'.format(np.min(GB_data))],
                                                            ['p50', '{:.2f}'.format(np.percentile(GB1_data, q=.5)), '{:.2f}'.format(np.percentile(GB_data, q=.5))],
                                                            ['p75', '{:.2f}'.format(np.percentile(GB1_data, q=.75)), '{:.2f}'.format(np.percentile(GB_data, q=.75))],
                                                            ['p90', '{:.2f}'.format(np.percentile(GB1_data, q=.9)), '{:.2f}'.format(np.percentile(GB_data, q=.9))],
                                                            ['p99', '{:.2f}'.format(np.percentile(GB1_data, q=.99)), '{:.2f}'.format(np.percentile(GB_data, q=.99))],
                                                            ['p99.9', '{:.2f}'.format(np.percentile(GB1_data, q=.999)), '{:.2f}'.format(np.percentile(GB_data, q=.999))],
                                                            ['max', '{:.2f}'.format(np.max(GB1_data)), '{:.2f}'.format(np.max(GB_data))],
                                                            ]),
                                                 columns=['', 'GB1_data', 'GB_data'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'synthetic_non_Pareto_descriptives.xlsx', mode='w') as writer:
        df_synthetic_non_Pareto_descriptives.to_excel(writer, sheet_name='descriptives', index=False)


""" 
--------------------------------
3. Fitting
--------------------------------
"""

if run_optimize:

    x0=(-1, .5, 1, 1)
    bootstraps=(250, 250, 250, 250)
    rejection_criteria = ['LRtest', 'AIC', 'AIC_alternative']
    cosmetics = {'bins': 300, 'col_data': 'blue', 'col_fit': 'red'}
    plot = False

    ## rejection criterion: LRtest, AIC, AIC alternative (method #2)

    Pareto_data_parms = Paretobranchfit(x=Pareto_data, x0=x0, b=b, bootstraps=bootstraps, return_bestmodel=True,
                                        rejection_criterion=rejection_criteria, plot=plot, plot_cosmetics=cosmetics)

    Pareto_data_gauss_noise_1_parms = Paretobranchfit(x=Pareto_data_gauss_noise_1, x0=x0, b=b, bootstraps=bootstraps,
                                                      return_bestmodel=True, plot=plot,
                                                      rejection_criterion=rejection_criteria, plot_cosmetics=cosmetics)

    Pareto_data_gauss_noise_2_parms = Paretobranchfit(x=Pareto_data_gauss_noise_2, x0=x0, b=b, bootstraps=bootstraps,
                                                      return_bestmodel=True, plot=plot,
                                                      rejection_criterion=rejection_criteria, plot_cosmetics=cosmetics)

    Pareto_data_het_noise_1_parms = Paretobranchfit(x=Pareto_data_het_noise_1, x0=x0, b=b, bootstraps=bootstraps,
                                                    return_bestmodel=True, plot=plot,
                                                    rejection_criterion=rejection_criteria, plot_cosmetics=cosmetics)

    Pareto_data_het_noise_2_parms = Paretobranchfit(x=Pareto_data_het_noise_2, x0=x0, b=b, bootstraps=bootstraps,
                                                    return_bestmodel=True, plot=plot,
                                                    rejection_criterion=rejection_criteria, plot_cosmetics=cosmetics)


    # Robustness Check: NON Pareto data

    GB1_non_Pareto_parms = Paretobranchfit(x=GB1_data, x0=x0, b=b, bootstraps=bootstraps, return_bestmodel=True,
                                           plot=plot, rejection_criterion=rejection_criteria, plot_cosmetics=cosmetics)

    GB_non_Pareto_parms = Paretobranchfit(x=GB_data, x0=x0, b=b, bootstraps=bootstraps, return_bestmodel=True,
                                          plot=plot, rejection_criterion=rejection_criteria, plot_cosmetics=cosmetics)

"""
--------------------------------
4. Save fitted parms to table
--------------------------------
"""

# AIC results, shorter names
parms1 = prep_fit_results_for_table(Pareto_data_parms[1])
parms2 = prep_fit_results_for_table(Pareto_data_gauss_noise_1_parms[1])
parms3 = prep_fit_results_for_table(Pareto_data_gauss_noise_2_parms[1])
parms4 = prep_fit_results_for_table(Pareto_data_het_noise_1_parms[1])
parms5 = prep_fit_results_for_table(Pareto_data_het_noise_2_parms[1])

if run_optimize:
    df_synthetic_fit_parms_AIC = pd.DataFrame(np.array([['best fitted model', '{}'.format(Pareto_data_parms[1][0]),  '{}'.format(Pareto_data_gauss_noise_1_parms[1][0]),  '{}'.format(Pareto_data_gauss_noise_2_parms[1][0]),  '{}'.format(Pareto_data_het_noise_1_parms[1][0]),  '{}'.format(Pareto_data_het_noise_2_parms[1][0])],
                                                    ['a',               '{}'.format(parms1[0]),     '{}'.format(parms2[0]),     '{}'.format(parms3[0]),     '{}'.format(parms4[0]),     '{}'.format(parms5[0])],
                                                    [' ',               '[{}; {}]'.format(parms1[2], parms1[3]),     '[{}; {}]'.format(parms2[2], parms2[3]),     '[{}; {}]'.format(parms3[2], parms3[3]),     '[{}; {}]'.format(parms4[2], parms4[3]),     '[{}; {}]'.format(parms5[2], parms5[3])],
                                                    [' ',               '({})'.format(parms1[1]),     '({})'.format(parms2[1]),     '({})'.format(parms3[1]),     '({})'.format(parms4[1]),     '({})'.format(parms5[1])],
                                                    ['c',               '{}'.format(parms1[4]),     '{}'.format(parms2[4]),     '{}'.format(parms3[4]),     '{}'.format(parms4[4]),     '{}'.format(parms5[4])],
                                                    [' ',               '[{}; {}]'.format(parms1[6], parms1[7]),     '[{}; {}]'.format(parms2[6], parms2[7]),     '[{}; {}]'.format(parms3[6], parms3[7]),     '[{}; {}]'.format(parms4[6], parms4[7]),     '[{}; {}]'.format(parms5[6], parms5[7])],
                                                    [' ',               '({})'.format(parms1[5]),     '({})'.format(parms2[5]),     '({})'.format(parms3[5]),     '({})'.format(parms4[5]),     '({})'.format(parms5[5])],
                                                    ['p',               '{}'.format(parms1[8]),     '{}'.format(parms2[8]),     '{}'.format(parms3[8]),     '{}'.format(parms4[8]),     '{}'.format(parms5[8])],
                                                    [' ',               '[{}; {}]'.format(parms1[10], parms1[11]),     '[{}; {}]'.format(parms2[10], parms2[11]),     '[{}; {}]'.format(parms3[10], parms3[11]),     '[{}; {}]'.format(parms4[10], parms4[11]),     '[{}; {}]'.format(parms5[10], parms5[11])],
                                                    [' ',               '({})'.format(parms1[9]),     '({})'.format(parms2[9]),     '({})'.format(parms3[9]),     '({})'.format(parms4[9]),     '({})'.format(parms5[9])],
                                                    ['q',               '{}'.format(parms1[12]),     '{}'.format(parms2[12]),     '{}'.format(parms3[12]),     '{}'.format(parms4[12]),     '{}'.format(parms5[12])],
                                                    [' ',               '[{}; {}]'.format(parms1[14], parms1[15]),     '[{}; {}]'.format(parms2[14], parms2[15]),     '[{}; {}]'.format(parms3[14], parms3[15]),     '[{}; {}]'.format(parms4[14], parms4[15]),     '[{}; {}]'.format(parms5[14], parms5[15])],
                                                    [' ',               '({})'.format(parms1[13]),     '({})'.format(parms2[13]),     '({})'.format(parms3[13]),     '({})'.format(parms4[13]),     '({})'.format(parms5[13])],
                                                    ['lower bound b',   '{}'.format(b),             '{}'.format(b),             '{}'.format(b),             '{}'.format(b),             '{}'.format(b)],
                                                    ['LL',              '{}'.format(parms1[19]),    '{}'.format(parms2[19]),    '{}'.format(parms3[19]),    '{}'.format(parms4[19]),    '{}'.format(parms5[19])],
                                                    ['AIC',             '{}'.format(parms1[16]),     '{}'.format(parms2[16]),     '{}'.format(parms3[16]),     '{}'.format(parms4[16]),     '{}'.format(parms5[16])],
                                                    ['MSE',             '{}'.format(parms1[17]),     '{}'.format(parms2[17]),     '{}'.format(parms3[17]),     '{}'.format(parms4[17]),     '{}'.format(parms5[17])],
                                                    ['RMSE',            '{}'.format(parms1[18]),    '{}'.format(parms2[18]),    '{}'.format(parms3[18]),    '{}'.format(parms4[18]),    '{}'.format(parms5[18])],
                                                    ['emp. mean',       '{}'.format(parms1[20]),    '{}'.format(parms2[20]),    '{}'.format(parms3[20]),    '{}'.format(parms4[20]),    '{}'.format(parms5[20])],
                                                    ['emp. var.',       '{}'.format(parms1[21]),    '{}'.format(parms2[21]),    '{}'.format(parms3[21]),    '{}'.format(parms4[21]),    '{}'.format(parms5[21])],
                                                    ['pred. mean',      '{}'.format(parms1[22]),    '{}'.format(parms2[22]),    '{}'.format(parms3[22]),    '{}'.format(parms4[22]),    '{}'.format(parms5[22])],
                                                    ['pred. var.',      '{}'.format(parms1[23]),    '{}'.format(parms2[23]),    '{}'.format(parms3[23]),    '{}'.format(parms4[23]),    '{}'.format(parms5[23])],
                                                    ['n',               '{}'.format(parms1[24]),    '{}'.format(parms2[24]),    '{}'.format(parms3[24]),    '{}'.format(parms4[24]),    '{}'.format(parms5[24])],
                                                    ]),
                                                 columns=['', 'Pareto_data', 'Pareto_data_gauss_noise_1', 'Pareto_data_gauss_noise_2', 'Pareto_data_het_noise_1', 'Pareto_data_het_noise_2'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'synthetic_data_fit_results.xlsx', engine='openpyxl', mode='w') as writer:
        df_synthetic_fit_parms_AIC.to_excel(writer, sheet_name='synthetic_data_fit_results_AIC', index=False)


# shorter names
parms10 = prep_fit_results_for_table(Pareto_data_parms[0])
parms11 = prep_fit_results_for_table(Pareto_data_gauss_noise_1_parms[0])
parms12 = prep_fit_results_for_table(Pareto_data_gauss_noise_2_parms[0])
parms13 = prep_fit_results_for_table(Pareto_data_het_noise_1_parms[0])
parms14 = prep_fit_results_for_table(Pareto_data_het_noise_2_parms[0])

if run_optimize:
    df_synthetic_fit_parms_LR = pd.DataFrame(np.array([['best fitted model', '{}'.format(Pareto_data_parms[0][0]),  '{}'.format(Pareto_data_gauss_noise_1_parms[0][0]),  '{}'.format(Pareto_data_gauss_noise_2_parms[0][0]),  '{}'.format(Pareto_data_het_noise_1_parms[0][0]),  '{}'.format(Pareto_data_het_noise_2_parms[0][0])],
                                            ['a',               '{}'.format(parms10[0]),     '{}'.format(parms11[0]),     '{}'.format(parms12[0]),     '{}'.format(parms13[0]),     '{}'.format(parms14[0])],
                                            [' ',               '[{}; {}]'.format(parms10[2], parms10[3]),     '[{}; {}]'.format(parms11[2], parms11[3]),     '[{}; {}]'.format(parms12[2], parms12[3]),     '[{}; {}]'.format(parms13[2], parms13[3]),     '[{}; {}]'.format(parms14[2], parms14[3])],
                                            [' ',               '({})'.format(parms10[1]),     '({})'.format(parms11[1]),     '({})'.format(parms12[1]),     '({})'.format(parms13[1]),     '({})'.format(parms14[1])],
                                            ['c',               '{}'.format(parms10[4]),     '{}'.format(parms11[4]),     '{}'.format(parms12[4]),     '{}'.format(parms13[4]),     '{}'.format(parms14[4])],
                                            [' ',               '[{}; {}]'.format(parms10[6], parms10[7]),     '[{}; {}]'.format(parms11[6], parms11[7]),     '[{}; {}]'.format(parms12[6], parms12[7]),     '[{}; {}]'.format(parms13[6], parms13[7]),     '[{}; {}]'.format(parms14[6], parms14[7])],
                                            [' ',               '({})'.format(parms10[5]),     '({})'.format(parms11[5]),     '({})'.format(parms12[5]),     '({})'.format(parms13[5]),     '({})'.format(parms14[5])],
                                            ['p',               '{}'.format(parms10[8]),     '{}'.format(parms11[8]),     '{}'.format(parms12[8]),     '{}'.format(parms13[8]),     '{}'.format(parms14[8])],
                                            [' ',               '[{}; {}]'.format(parms10[10], parms10[11]),     '[{}; {}]'.format(parms11[10], parms11[11]),     '[{}; {}]'.format(parms12[10], parms12[11]),     '[{}; {}]'.format(parms13[10], parms13[11]),     '[{}; {}]'.format(parms14[10], parms14[11])],
                                            [' ',               '({})'.format(parms10[9]),     '({})'.format(parms11[9]),     '({})'.format(parms12[9]),     '({})'.format(parms13[9]),     '({})'.format(parms14[9])],
                                            ['q',               '{}'.format(parms10[12]),     '{}'.format(parms11[12]),     '{}'.format(parms12[12]),     '{}'.format(parms13[12]),     '{}'.format(parms14[12])],
                                            [' ',               '[{}; {}]'.format(parms10[14], parms10[15]),     '[{}; {}]'.format(parms11[14], parms11[15]),     '[{}; {}]'.format(parms12[14], parms12[15]),     '[{}; {}]'.format(parms13[14], parms13[15]),     '[{}; {}]'.format(parms14[14], parms14[15])],
                                            [' ',               '({})'.format(parms10[13]),     '({})'.format(parms11[13]),     '({})'.format(parms12[13]),     '({})'.format(parms13[13]),     '({})'.format(parms14[13])],
                                            ['lower bound b',   '{}'.format(b),             '{}'.format(b),             '{}'.format(b),             '{}'.format(b),             '{}'.format(b)],
                                            ['LL',              '{}'.format(parms10[19]),    '{}'.format(parms11[19]),    '{}'.format(parms12[19]),    '{}'.format(parms13[19]),    '{}'.format(parms14[19])],
                                            ['AIC',             '{}'.format(parms10[16]),     '{}'.format(parms11[16]),     '{}'.format(parms12[16]),     '{}'.format(parms13[16]),     '{}'.format(parms14[16])],
                                            ['MSE',             '{}'.format(parms10[17]),     '{}'.format(parms11[17]),     '{}'.format(parms12[17]),     '{}'.format(parms13[17]),     '{}'.format(parms14[17])],
                                            ['RMSE',            '{}'.format(parms10[18]),    '{}'.format(parms11[18]),    '{}'.format(parms12[18]),    '{}'.format(parms13[18]),    '{}'.format(parms14[18])],
                                            ['emp. mean',       '{}'.format(parms10[20]),    '{}'.format(parms11[20]),    '{}'.format(parms12[20]),    '{}'.format(parms13[20]),    '{}'.format(parms14[20])],
                                            ['emp. var.',       '{}'.format(parms10[21]),    '{}'.format(parms11[21]),    '{}'.format(parms12[21]),    '{}'.format(parms13[21]),    '{}'.format(parms14[21])],
                                            ['pred. mean',      '{}'.format(parms10[22]),    '{}'.format(parms11[22]),    '{}'.format(parms12[22]),    '{}'.format(parms13[22]),    '{}'.format(parms14[22])],
                                            ['pred. var.',      '{}'.format(parms10[23]),    '{}'.format(parms11[23]),    '{}'.format(parms12[23]),    '{}'.format(parms13[23]),    '{}'.format(parms14[23])],
                                            ['n',               '{}'.format(parms10[24]),    '{}'.format(parms11[24]),    '{}'.format(parms12[24]),    '{}'.format(parms13[24]),    '{}'.format(parms14[24])],
                                            ]),
                                     columns=['', 'Pareto_data', 'Pareto_data_gauss_noise_1', 'Pareto_data_gauss_noise_2', 'Pareto_data_het_noise_1', 'Pareto_data_het_noise_2'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'synthetic_data_fit_results.xlsx', engine='openpyxl', mode='a') as writer:
        df_synthetic_fit_parms_LR.to_excel(writer, sheet_name='synthetic_data_fit_results_LR', index=False)


    # NON Pareto data: shorter names
    parms15 = prep_fit_results_for_table(GB1_non_Pareto_parms[0])
    parms16 = prep_fit_results_for_table(GB1_non_Pareto_parms[1])
    parms17 = prep_fit_results_for_table(GB1_non_Pareto_parms[2])
    parms18 = prep_fit_results_for_table(GB_non_Pareto_parms[0])
    parms19 = prep_fit_results_for_table(GB_non_Pareto_parms[1])
    parms20 = prep_fit_results_for_table(GB_non_Pareto_parms[2])

    df_non_Pareto_fit_results = pd.DataFrame(np.array([['best fitted model', '{}'.format(GB1_non_Pareto_parms[0][0]), '{}'.format(GB1_non_Pareto_parms[1][0]), '{}'.format(GB1_non_Pareto_parms[2][0]), '{}'.format(GB_non_Pareto_parms[0][0]), '{}'.format(GB_non_Pareto_parms[1][0]), '{}'.format(GB_non_Pareto_parms[2][0])],
                                            ['a',               '{}'.format(parms15[0]),     '{}'.format(parms16[0]),     '{}'.format(parms17[0]),     '{}'.format(parms18[0]),     '{}'.format(parms19[0]),     '{}'.format(parms20[0])],
                                            [' ',               '[{}; {}]'.format(parms15[2], parms15[3]),     '[{}; {}]'.format(parms16[2], parms16[3]),     '[{}; {}]'.format(parms17[2], parms17[3]),     '[{}; {}]'.format(parms18[2], parms18[3]),     '[{}; {}]'.format(parms19[2], parms19[3]),     '[{}; {}]'.format(parms20[2], parms20[3])],
                                            [' ',               '({})'.format(parms15[1]),     '({})'.format(parms16[1]),     '({})'.format(parms17[1]),     '({})'.format(parms18[1]),     '({})'.format(parms19[1]),     '({})'.format(parms20[1])],
                                            ['c',               '{}'.format(parms15[4]),     '{}'.format(parms16[4]),     '{}'.format(parms17[4]),     '{}'.format(parms18[4]),     '{}'.format(parms19[4]),     '{}'.format(parms20[4])],
                                            [' ',               '[{}; {}]'.format(parms15[6], parms15[7]),     '[{}; {}]'.format(parms16[6], parms16[7]),     '[{}; {}]'.format(parms17[6], parms17[7]),     '[{}; {}]'.format(parms18[6], parms18[7]),     '[{}; {}]'.format(parms19[6], parms19[7]),     '[{}; {}]'.format(parms20[6], parms20[7])],
                                            [' ',               '({})'.format(parms15[5]),     '({})'.format(parms16[5]),     '({})'.format(parms17[5]),     '({})'.format(parms18[5]),     '({})'.format(parms19[5]),     '({})'.format(parms20[5])],
                                            ['p',               '{}'.format(parms15[8]),     '{}'.format(parms16[8]),     '{}'.format(parms17[8]),     '{}'.format(parms18[8]),     '{}'.format(parms19[8]),     '{}'.format(parms20[8])],
                                            [' ',               '[{}; {}]'.format(parms15[10], parms15[11]),     '[{}; {}]'.format(parms16[10], parms16[11]),     '[{}; {}]'.format(parms17[10], parms17[11]),     '[{}; {}]'.format(parms18[10], parms18[11]),     '[{}; {}]'.format(parms19[10], parms19[11]),     '[{}; {}]'.format(parms20[10], parms20[11])],
                                            [' ',               '({})'.format(parms15[9]),     '({})'.format(parms16[9]),     '({})'.format(parms17[9]),     '({})'.format(parms18[9]),     '({})'.format(parms19[9]),     '({})'.format(parms20[9])],
                                            ['q',               '{}'.format(parms15[12]),     '{}'.format(parms16[12]),     '{}'.format(parms17[12]),     '{}'.format(parms18[12]),     '{}'.format(parms19[12]),     '{}'.format(parms20[12])],
                                            [' ',               '[{}; {}]'.format(parms15[14], parms15[15]),     '[{}; {}]'.format(parms16[14], parms16[15]),     '[{}; {}]'.format(parms17[14], parms17[15]),     '[{}; {}]'.format(parms18[14], parms18[15]),     '[{}; {}]'.format(parms19[14], parms19[15]),     '[{}; {}]'.format(parms20[14], parms20[15])],
                                            [' ',               '({})'.format(parms15[13]),     '({})'.format(parms16[13]),     '({})'.format(parms17[13]),     '({})'.format(parms18[13]),     '({})'.format(parms19[13]),     '({})'.format(parms20[13])],
                                            ['lower bound b',   '{}'.format(b),             '{}'.format(b),             '{}'.format(b),             '{}'.format(b),             '{}'.format(b),             '{}'.format(b)],
                                            ['LL',              '{}'.format(parms15[19]),    '{}'.format(parms16[19]),    '{}'.format(parms17[19]),    '{}'.format(parms18[19]),    '{}'.format(parms19[19]),    '{}'.format(parms20[19])],
                                            ['AIC',             '{}'.format(parms15[16]),     '{}'.format(parms16[16]),     '{}'.format(parms17[16]),     '{}'.format(parms18[16]),     '{}'.format(parms19[16]),     '{}'.format(parms20[16])],
                                            ['MSE',             '{}'.format(parms15[17]),     '{}'.format(parms16[17]),     '{}'.format(parms17[17]),     '{}'.format(parms18[17]),     '{}'.format(parms19[17]),     '{}'.format(parms20[17])],
                                            ['RMSE',            '{}'.format(parms15[18]),    '{}'.format(parms16[18]),    '{}'.format(parms17[18]),    '{}'.format(parms18[18]),    '{}'.format(parms19[18]),    '{}'.format(parms20[18])],
                                            ['emp. mean',       '{}'.format(parms15[20]),    '{}'.format(parms16[20]),    '{}'.format(parms17[20]),    '{}'.format(parms18[20]),    '{}'.format(parms19[20]),    '{}'.format(parms20[20])],
                                            ['emp. var.',       '{}'.format(parms15[21]),    '{}'.format(parms16[21]),    '{}'.format(parms17[21]),    '{}'.format(parms18[21]),    '{}'.format(parms19[21]),    '{}'.format(parms20[21])],
                                            ['pred. mean',      '{}'.format(parms15[22]),    '{}'.format(parms16[22]),    '{}'.format(parms17[22]),    '{}'.format(parms18[22]),    '{}'.format(parms19[22]),    '{}'.format(parms20[22])],
                                            ['pred. var.',      '{}'.format(parms15[23]),    '{}'.format(parms16[23]),    '{}'.format(parms17[23]),    '{}'.format(parms18[23]),    '{}'.format(parms19[23]),    '{}'.format(parms20[23])],
                                            ['n',               '{}'.format(parms15[24]),    '{}'.format(parms16[24]),    '{}'.format(parms17[24]),    '{}'.format(parms18[24]),    '{}'.format(parms19[24]),    '{}'.format(parms20[24])],
                                            ]),
                                    columns=['', 'GB_non_Pareto_LR', 'GB_non_Pareto_AIC', 'GB_non_Pareto_AIC2', 'GB1_non_Pareto_LR', 'GB1_non_Pareto_AIC', 'GB1_non_Pareto_AIC2'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'non_pareto_fit_results.xlsx', engine='openpyxl', mode='w') as writer:
        df_non_Pareto_fit_results.to_excel(writer, sheet_name='descriptives', index=False)


""" 
--------------------------------
5. Plot Fit vs data
--------------------------------
"""

if run_optimize:

    ### fit of Pareto_data (LRtest)
    print('best fit for Pareto_data:', Pareto_data_parms[0][0])

    p_fit = Pareto_data_parms[0][1][0]

    # generate new data based on fitted parms and best model
    Pareto_data_fit = Pareto_icdf(u=u, b=b, p=p_fit)

    # Note: when using latex, doubling {{}} -> latex text, tripling {{{}}} -> use variables form .format()

    # data
    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{Pareto}}(b={{{}}}, p={{{}}})$'.format(b, p))
    # fit
    plt.plot(u, Pareto_data_fit, color='red', alpha=.75, label=r'$icdf_{{Pareto}}(b={{{}}}, \hat{{p}}={{{}}})$'.format(b, np.around(p_fit,3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_Pareto.' + type, dpi=300, format=type)
    plt.show()
    plt.close()


    ### fit of Pareto_data_gauss_noise_1
    print('best fit for Pareto_data_gauss_noise_1:', Pareto_data_gauss_noise_1_parms[0][0])

    a_fit, p_fit, q_fit = Pareto_data_gauss_noise_1_parms[0][1][0], Pareto_data_gauss_noise_1_parms[0][1][2], Pareto_data_gauss_noise_1_parms[0][1][4]

    # generate new data based on fitted parms and best model
    GB1_data_fit, u = GB1_icdf_ne(x=Pareto_data, b=b, a=a_fit, p=p_fit, q=q_fit)

    # Note: when using latex, doubling {{}} -> latex text, tripling {{{}}} -> use variables form .format()

    # data
    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{Pareto}}(b={{{}}}, p={{{}}})$'.format(b, p))
    # fit
    plt.plot(u, GB1_data_fit, color='red', alpha=.75, label=r'$icdf_{{GB1}}(b={{{}}}, \hat{{a}}={{{}}}, \hat{{p}}={{{}}}, \hat{{q}}={{{}}})$'.format(b, np.around(a_fit,3), np.around(p_fit,3), np.around(q_fit,3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_Pareto_Gauss1.' + type, dpi=300, format=type)
    plt.show()
    plt.close()

    ### Pareto_data_gauss_noise_2
    print('best fit for Pareto_data_gauss_noise_2:', Pareto_data_gauss_noise_2_parms[0][0])

    # generate new data based on fitted parms and best model
    a_fit, c_fit, p_fit, q_fit = Pareto_data_gauss_noise_2_parms[0][1][0], Pareto_data_gauss_noise_2_parms[0][1][2], Pareto_data_gauss_noise_2_parms[0][1][4], Pareto_data_gauss_noise_2_parms[0][1][6]

    # generate new data based on fitted parms and best model
    GB_data_fit, u = GB_icdf_ne(x=Pareto_data, b=b, a=a_fit, c=c_fit, p=p_fit, q=q_fit)

    # Note: when using latex, doubling {{}} -> latex text, tripling {{{}}} -> use variables form .format()

    # data
    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{Pareto}}(b={{{}}}, p={{{}}})$'.format(b, p))
    # fit
    plt.plot(u, GB_data_fit, color='red', alpha=.75, label=r'$icdf_{{GB}}(b={{{}}}, \hat{{a}}={{{}}}, \hat{{c}}={{{}}}, \hat{{p}}={{{}}}, \hat{{q}}={{{}}}, )$'.format(b, np.around(a_fit,3), np.around(c_fit,3), np.around(p_fit,3), np.around(q_fit,3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_Pareto_Gauss2.' + type, dpi=300, format=type)
    plt.show()
    plt.close()

    ### Pareto_data_het_noise_1
    print('best fit for Pareto_data_het_noise_1_parms:', Pareto_data_het_noise_1_parms[0][0])

    # generate new data based on fitted parms and best model
    a_fit, p_fit, q_fit = Pareto_data_het_noise_1_parms[0][1][0], Pareto_data_het_noise_1_parms[0][1][2], Pareto_data_het_noise_1_parms[0][1][4]

    # generate new data based on fitted parms and best model
    GB1_data_fit, u = GB1_icdf_ne(x=Pareto_data, b=b, a=a_fit, p=p_fit, q=q_fit)

    # Note: when using latex, doubling {{}} -> latex text, tripling {{{}}} -> use variables form .format()

    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{Pareto}}(b={{{}}}, p={{{}}})$'.format(b, p))
    plt.plot(u, GB1_data_fit, color='red', alpha=.75, label=r'$icdf_{{GB1}}(b={{{}}}, \hat{{a}}={{{}}}, \hat{{p}}={{{}}}, \hat{{q}}={{{}}})$'.format(b, np.around(a_fit,3), np.around(p_fit,3), np.around(q_fit,3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_Pareto_Het1.' + type, dpi=300, format=type)
    plt.show()
    plt.close()

    ### Pareto_data_het_noise_2 (LRtest)
    print('best fit for Pareto_data_het_noise_2_parms:', Pareto_data_het_noise_2_parms[0][0])

    # generate new data based on fitted parms and best model
    a_fit, p_fit, q_fit = Pareto_data_het_noise_2_parms[0][1][0], Pareto_data_het_noise_2_parms[0][1][2], Pareto_data_het_noise_2_parms[0][1][4]

    # generate new data based on fitted parms and best model
    GB1_data_fit, u = GB1_icdf_ne(x=Pareto_data, b=b, a=a_fit, p=p_fit, q=q_fit)

    # Note: when using latex, doubling {{}} -> latex text, tripling {{{}}} -> use variables form .format()

    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{Pareto}}(b={{{}}}, p={{{}}})$'.format(b, p))
    plt.plot(u, GB1_data_fit, color='red', alpha=.75, label=r'$icdf_{{GB1}}(b={{{}}}, \hat{{a}}={{{}}}, \hat{{p}}={{{}}}, \hat{{q}}={{{}}})$'.format(b, np.around(a_fit,3), np.around(p_fit,3), np.around(q_fit,3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_Pareto_Het2.' + type, dpi=300, format=type)
    plt.show()
    plt.close()


    ### nonParto GB1
    print('best fit for nonPareto_data_GB1_parms:', GB1_non_Pareto_parms[0][0])

    # generate new data based on fitted parms and best model
    a_fit, p_fit, q_fit = GB1_non_Pareto_parms[0][1][0], GB1_non_Pareto_parms[0][1][2], GB1_non_Pareto_parms[0][1][4]

    # generate new data based on fitted parms and best model
    GB1_data_fit, u = GB1_icdf_ne(x=GB1_data, b=b, a=a_fit, p=p_fit, q=q_fit)

    # Note: when using latex, doubling {{}} -> latex text, tripling {{{}}} -> use variables form .format()

    plt.scatter(u, GB1_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{GB1}}(b={{{}}}, a={{{}}}, p={{{}}}, q={{{}}})$'.format(b, -5, p, 5))
    plt.plot(u, GB1_data_fit, color='red', alpha=.75, label=r'$icdf_{{GB1}}(b={{{}}}, \hat{{a}}={{{}}}, \hat{{p}}={{{}}}, \hat{{q}}={{{}}})$'.format(b, np.around(a_fit,3), np.around(p_fit,3), np.around(q_fit,3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_nonPareto_GB1.' + type, dpi=300, format=type)
    plt.show()
    plt.close()


    ### nonPareto GB
    print('best fit for nonPareto_data_GB_parms:', GB_non_Pareto_parms[0][0])

    # generate new data based on fitted parms and best model
    a_fit, c_fit, p_fit, q_fit = GB_non_Pareto_parms[0][1][0], GB_non_Pareto_parms[0][1][2], GB_non_Pareto_parms[0][1][4], GB_non_Pareto_parms[0][1][6]

    # generate new data based on fitted parms and best model
    GB_data_fit, u = GB_icdf_ne(x=GB_data, b=b, a=a_fit, c=c_fit, p=p_fit, q=q_fit)

    # Note: when using latex, doubling {{}} -> latex text, tripling {{{}}} -> use variables form .format()

    plt.scatter(u, GB_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{GB}}(b={{{}}}, a={{{}}}, c={{{}}}, p={{{}}}, q={{{}}})$'.format(b, -2, .5, p, 10))
    plt.plot(u, GB_data_fit, color='red', alpha=.75, label=r'$icdf_{{GB}}(b={{{}}}, \hat{{a}}={{{}}}, \hat{{c}}={{{}}}, \hat{{p}}={{{}}}, \hat{{q}}={{{}}})$'.format(b, np.around(a_fit,3), np.around(c_fit,3), np.around(p_fit,3), np.around(q_fit,3)))
    plt.legend(loc='upper left'); plt.xlabel('quantiles'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'fit_vs_data_nonPareto_GB.' + type, dpi=300, format=type)
    plt.show()
    plt.close()

