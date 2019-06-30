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
    plt.legend(loc='upper left'); plt.xlabel('cdf'); plt.ylabel('x');
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'icdf_gauss_noise.' + type, dpi=300, format=type)
    plt.show()
    plt.close()

    # check heteroscedastic noise data
    plt.scatter(u, Pareto_data_het_noise_2, marker="o", s=2, color='blue', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=x\cdot s\cdot N(0,100^2)$')
    plt.scatter(u, Pareto_data_het_noise_1, marker="o", s=2, color='orangered', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=x\cdot s\cdot N(0,200^2)$')
    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{Pareto}(b=250, p=2.5)=x$')
    plt.legend(loc='upper left'); plt.xlabel('cdf'); plt.ylabel('x')
    for type in ['png', 'pdf']:
        plt.savefig(fname=graphspath + 'icdf_het_noise.' + type, dpi=300, format=type)
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
    df_synthetic_data_descriptives = pd.DataFrame(np.array([['N', np.around(np.size(Pareto_data), 2), np.around(np.size(Pareto_data_gauss_noise_1), 2), np.around(np.size(Pareto_data_gauss_noise_2), 2), np.around(np.size(Pareto_data_het_noise_1), 2), np.around(np.size(Pareto_data_het_noise_2), 2)],
                                                            ['mean', np.around(np.mean(Pareto_data), 2), np.around(np.mean(Pareto_data_gauss_noise_1), 2), np.around(np.mean(Pareto_data_gauss_noise_2), 2), np.around(np.mean(Pareto_data_het_noise_1), 2), np.around(np.mean(Pareto_data_het_noise_2), 2)],
                                                            ['sd', np.around(np.std(Pareto_data), 2), np.around(np.std(Pareto_data_gauss_noise_1), 2), np.around(np.std(Pareto_data_gauss_noise_2), 2), np.around(np.std(Pareto_data_het_noise_1), 2), np.around(np.std(Pareto_data_het_noise_2), 2)],
                                                            ['lower bound b', b, b, b, b, b],
                                                            ['min', np.around(np.min(Pareto_data), 2), np.around(np.min(Pareto_data_gauss_noise_1), 2), np.around(np.min(Pareto_data_gauss_noise_2), 2), np.around(np.min(Pareto_data_het_noise_1), 2), np.around(np.min(Pareto_data_het_noise_2), 2)],
                                                            ['p50', np.around(np.percentile(Pareto_data, q=.5), 2), np.around(np.percentile(Pareto_data_gauss_noise_1, q=.5), 2), np.around(np.percentile(Pareto_data_gauss_noise_2, q=.5), 2), np.around(np.percentile(Pareto_data_het_noise_1, q=.5), 2), np.around(np.percentile(Pareto_data_het_noise_2, q=.5), 2)],
                                                            ['p75', np.around(np.percentile(Pareto_data, q=.75), 2), np.around(np.percentile(Pareto_data_gauss_noise_1, q=.75), 2), np.around(np.percentile(Pareto_data_gauss_noise_2, q=.75), 2), np.around(np.percentile(Pareto_data_het_noise_1, q=.75), 2), np.around(np.percentile(Pareto_data_het_noise_2, q=.75), 2)],
                                                            ['p90', np.around(np.percentile(Pareto_data, q=.9), 2), np.around(np.percentile(Pareto_data_gauss_noise_1, q=.9), 2), np.around(np.percentile(Pareto_data_gauss_noise_2, q=.9), 2), np.around(np.percentile(Pareto_data_het_noise_1, q=.9), 2), np.around(np.percentile(Pareto_data_het_noise_2, q=.9), 2)],
                                                            ['p99', np.around(np.percentile(Pareto_data, q=.99), 2), np.around(np.percentile(Pareto_data_gauss_noise_1, q=.99), 2), np.around(np.percentile(Pareto_data_gauss_noise_2, q=.99), 2), np.around(np.percentile(Pareto_data_het_noise_1, q=.99), 2), np.around(np.percentile(Pareto_data_het_noise_2, q=.99), 2)],
                                                            ['p99.9', np.around(np.percentile(Pareto_data, q=.999), 2), np.around(np.percentile(Pareto_data_gauss_noise_1, q=.999), 2), np.around(np.percentile(Pareto_data_gauss_noise_2, q=.999), 2), np.around(np.percentile(Pareto_data_het_noise_1, q=.999), 2), np.around(np.percentile(Pareto_data_het_noise_2, q=.999), 2)],
                                                            ['max', np.around(np.max(Pareto_data), 2), np.around(np.max(Pareto_data_gauss_noise_1), 2), np.around(np.max(Pareto_data_gauss_noise_2), 2), np.around(np.max(Pareto_data_het_noise_1), 2), np.around(np.max(Pareto_data_het_noise_2), 2)],
                                                            ]),
                                                 columns=['', 'Pareto_data', 'Pareto_data_gauss_noise_1', 'Pareto_data_gauss_noise_2', 'Pareto_data_het_noise_1', 'Pareto_data_het_noise_2'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'synthetic_data_descriptives.xlsx', mode='w') as writer:
        df_synthetic_data_descriptives.to_excel(writer, sheet_name='synthetic_data_descriptives', index=False)

""" 
--------------------------------
3. Fitting
--------------------------------
"""

if run_optimize:

    ## with AIC rejection criterion

    Pareto_data_parms = Paretobranchfit(x=Pareto_data, x0=(-1, .5, 1, 1), b=250,
                                        bootstraps=(250, 250, 250, 250),
                                        return_bestmodel=True, rejection_criteria='AIC', plot=True,
                                        plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_gauss_noise_1_parms = Paretobranchfit(x=Pareto_data_gauss_noise_1, x0=(-1, .5, 1, 1), b=250,
                                                      bootstraps=(250, 250, 250, 250),
                                                      return_bestmodel=True, rejection_criteria='AIC', plot=True,
                                                      plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_gauss_noise_2_parms = Paretobranchfit(x=Pareto_data_gauss_noise_2, x0=(-1, .5, 1, 1), b=250,
                                                      bootstraps=(250, 250, 250, 250),
                                                      return_bestmodel=True, rejection_criteria='AIC', plot=True,
                                                      plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_het_noise_1_parms = Paretobranchfit(x=Pareto_data_het_noise_1, x0=(-1, .5, 1, 1), b=250,
                                                    bootstraps=(250, 250, 250, 250),
                                                    return_bestmodel=True, rejection_criteria='AIC', plot=True,
                                                    plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_het_noise_2_parms = Paretobranchfit(x=Pareto_data_het_noise_2, x0=(-1, .5, 1, 1), b=250,
                                                    bootstraps=(250, 250, 250, 250),
                                                    return_bestmodel=True, rejection_criteria='AIC', plot=True,
                                                    plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    ## with LRtest rejection criterion

    Pareto_data_parms_LR = Paretobranchfit(x=Pareto_data, x0=(-1, .5, 1, 1), b=250,
                                        bootstraps=(250, 250, 250, 250),
                                        return_bestmodel=True, rejection_criteria='LRtest', plot=True,
                                        plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_gauss_noise_1_parms_LR = Paretobranchfit(x=Pareto_data_gauss_noise_1, x0=(-1, .5, 1, 1), b=250,
                                                      bootstraps=(250, 250, 250, 250),
                                                      return_bestmodel=True, rejection_criteria='LRtest', plot=True,
                                                      plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_gauss_noise_2_parms_LR = Paretobranchfit(x=Pareto_data_gauss_noise_2, x0=(-1, .5, 1, 1), b=250,
                                                      bootstraps=(250, 250, 250, 250),
                                                      return_bestmodel=True, rejection_criteria='LRtest', plot=True,
                                                      plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_het_noise_1_parms_LR = Paretobranchfit(x=Pareto_data_het_noise_1, x0=(-1, .5, 1, 1), b=250,
                                                    bootstraps=(250, 250, 250, 250),
                                                    return_bestmodel=True, rejection_criteria='LRtest', plot=True,
                                                    plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_het_noise_2_parms_LR = Paretobranchfit(x=Pareto_data_het_noise_2, x0=(-1, .5, 1, 1), b=250,
                                                    bootstraps=(250, 250, 250, 250),
                                                    return_bestmodel=True, rejection_criteria='LRtest', plot=True,
                                                    plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

"""
--------------------------------
4. Save fitted parms to table
--------------------------------
"""

# shorter names
parms1 = Pareto_data_parms[1]
parms2 = Pareto_data_gauss_noise_1_parms[1]
parms3 = Pareto_data_gauss_noise_2_parms[1]
parms4 = Pareto_data_het_noise_1_parms[1]
parms5 = Pareto_data_het_noise_2_parms[1]

if run_optimize:
    df_synthetic_fit_parms = pd.DataFrame(np.array([['best fitted model','Pareto',                          'IB1',                           'IB1',                         'Pareto',                         'GB'],
                                                    ['a',               '--',                               '--',                            '--',                          '--',                           '{:.3f}'.format(parms5[0])],
                                                    [' ',               '--',                               '--',                            '--',                          '--',                           '({:.3f})'.format(parms5[1])],
                                                    ['c',               '--',                               '--',                            '--',                          '--',                           '{:.3f}'.format(parms5[2])],
                                                    [' ',               '--',                               '--',                            '--',                          '--',                           '({:.3f})'.format(parms5[3])],
                                                    ['p',               '{:.3f}'.format(parms1[0]),      '{:.3f}'.format(parms2[0]),   '{:.3f}'.format(parms3[0]), '{:.3f}'.format(parms4[0]),  '{:.3f}'.format(parms5[4])],
                                                    [' ',               '({:.3f})'.format(parms1[1]),    '({:.3f})'.format(parms2[1]), '({:.3f})'.format(parms3[1]),'({:.3f})'.format(parms4[1]),'({:.3f})'.format(parms5[5])],
                                                    ['q',               '--',                               '{:.3f}'.format(parms2[2]),   '{:.3f}'.format(parms3[2]),  '--',                           '{:.3f}'.format(parms5[6])],
                                                    [' ',               '--',                               '({:.3f})'.format(parms2[3]), '({:.3f})'.format(parms3[3]),'--',                           '({:.3f})'.format(parms5[7])],
                                                    ['lower bound b',   '{:.3f}'.format(b),                 '{:.3f}'.format(b),              '{:.3f}'.format(b),             '{:.3f}'.format(b),             '{:.3f}'.format(b)],
                                                    ['LL',              '{:.3f}'.format(parms1[8]),      '{:.3f}'.format(parms2[10]),  '{:.3f}'.format(parms3[10]), '{:.3f}'.format(parms4[8]),  '{:.3f}'.format(parms5[14])],
                                                    ['AIC',             '{:.3f}'.format(parms1[2]),      '{:.3f}'.format(parms2[4]),   '{:.3f}'.format(parms3[4]),  '{:.3f}'.format(parms4[2]),  '{:.3f}'.format(parms5[8])],
                                                    ['MSE',             '{:.3f}'.format(parms1[5]),      '{:.3f}'.format(parms2[7]),   '{:.3f}'.format(parms3[7]),  '{:.3f}'.format(parms4[5]),  '{:.3f}'.format(parms5[11])],
                                                    ['RMSE',            '{:.3f}'.format(parms1[6]),      '{:.3f}'.format(parms2[8]),   '{:.3f}'.format(parms3[8]),  '{:.3f}'.format(parms4[6]),  '{:.3f}'.format(parms5[12])],
                                                    ['emp. mean',       '{:.3f}'.format(parms1[10]),     '{:.3f}'.format(parms2[12]),  '{:.3f}'.format(parms3[12]), '{:.3f}'.format(parms4[10]), '{:.3f}'.format(parms5[16])],
                                                    ['emp. var.',       '{:.3f}'.format(parms1[11]),     '{:.3f}'.format(parms2[13]),  '{:.3f}'.format(parms3[13]), '{:.3f}'.format(parms4[11]), '{:.3f}'.format(parms5[17])],
                                                    ['pred. mean',      '{:.3f}'.format(parms1[12]),     '{:.3f}'.format(parms2[14]),  '{:.3f}'.format(parms3[14]), '{:.3f}'.format(parms4[12]), '{:.3f}'.format(parms5[18])],
                                                    ['pred. var.',      '{:.3f}'.format(parms1[13]),     '{:.3f}'.format(parms2[15]),  '{:.3f}'.format(parms3[15]), '{:.3f}'.format(parms4[13]), '{:.3f}'.format(parms5[19])],
                                                    ['n',               '{}'.format(parms1[14]),         '{}'.format(parms2[16]),      '{}'.format(parms3[16]),     '{}'.format(parms4[14]),     '{}'.format(parms5[20])],
                                                    ]),
                                                 columns=['', 'Pareto_data', 'Pareto_data_gauss_noise_1', 'Pareto_data_gauss_noise_2', 'Pareto_data_het_noise_1', 'Pareto_data_het_noise_2'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'synthetic_data_fit_results.xlsx', engine='openpyxl', mode='w') as writer:
        df_synthetic_fit_parms.to_excel(writer, sheet_name='synthetic_data_fit_results_AIC', index=False)


# shorter names
parms10 = Pareto_data_parms_LR[1] #IB1
parms11 = Pareto_data_gauss_noise_1_parms_LR[1] #GB1
parms12 = Pareto_data_gauss_noise_2_parms_LR[1] #GB
parms13 = Pareto_data_het_noise_1_parms_LR[1] #IB1
parms14 = Pareto_data_het_noise_2_parms_LR[1] #IB1

if run_optimize:
    df_synthetic_fit_parms_LR = pd.DataFrame(np.array([['best fitted model','IB1',                     'GB1',                         'GB',                           'IB1',                          'IB1'],
                                                    ['a',               '--',                          '{:.3f}'.format(parms11[0]),   '{:.3f}'.format(parms12[0]),     '--',                           '--'],
                                                    [' ',               '--',                          '({:.3f})'.format(parms11[1]), '({:.3f})'.format(parms12[1]),   '--',                           '--'],
                                                    ['c',               '--',                          '--',                          '{:.3f}'.format(parms12[2]),     '--',                           '--'],
                                                    [' ',               '--',                          '--',                          '({:.3f})'.format(parms12[3]),   '--',                           '--'],
                                                    ['p',               '{:.3f}'.format(parms10[0]),   '{:.3f}'.format(parms11[2]),   '{:.3f}'.format(parms12[4]),    '{:.3f}'.format(parms13[0]),    '{:.3f}'.format(parms14[0])],
                                                    [' ',               '({:.3f})'.format(parms10[1]), '({:.3f})'.format(parms11[3]), '({:.3f})'.format(parms12[5]),  '({:.3f})'.format(parms13[1]),  '({:.3f})'.format(parms14[1])],
                                                    ['q',               '{:.3f}'.format(parms10[2]),   '{:.3f}'.format(parms11[4]),   '{:.3f}'.format(parms12[6]),     '{:.3f}'.format(parms13[2]),    '{:.3f}'.format(parms14[2])],
                                                    [' ',               '({:.3f})'.format(parms10[3]), '({:.3f})'.format(parms11[5]), '({:.3f})'.format(parms12[7]),   '({:.3f})'.format(parms13[3]),  '({:.3f})'.format(parms14[3])],
                                                    ['lower bound b',   '{:.3f}'.format(b),            '{:.3f}'.format(b),            '{:.3f}'.format(b),             '{:.3f}'.format(b),             '{:.3f}'.format(b)],
                                                    ['LL',              '{:.3f}'.format(parms10[10]),  '{:.3f}'.format(parms11[12]),  '{:.3f}'.format(parms12[14]),   '{:.3f}'.format(parms13[10]),   '{:.3f}'.format(parms14[10])],
                                                    ['AIC',             '{:.3f}'.format(parms10[4]),   '{:.3f}'.format(parms11[6]),   '{:.3f}'.format(parms12[8]),    '{:.3f}'.format(parms13[4]),    '{:.3f}'.format(parms14[4])],
                                                    ['MSE',             '{:.3f}'.format(parms10[7]),   '{:.3f}'.format(parms11[9]),   '{:.3f}'.format(parms12[11]),   '{:.3f}'.format(parms13[7]),    '{:.3f}'.format(parms14[7])],
                                                    ['RMSE',            '{:.3f}'.format(parms10[8]),   '{:.3f}'.format(parms11[10]),  '{:.3f}'.format(parms12[12]),   '{:.3f}'.format(parms13[8]),    '{:.3f}'.format(parms14[8])],
                                                    ['emp. mean',       '{:.3f}'.format(parms10[12]),  '{:.3f}'.format(parms11[14]),  '{:.3f}'.format(parms12[16]),   '{:.3f}'.format(parms13[12]),   '{:.3f}'.format(parms14[12])],
                                                    ['emp. var.',       '{:.3f}'.format(parms10[13]),  '{:.3f}'.format(parms11[15]),  '{:.3f}'.format(parms12[17]),   '{:.3f}'.format(parms13[13]),   '{:.3f}'.format(parms14[13])],
                                                    ['pred. mean',      '{:.3f}'.format(parms10[14]),  '{:.3f}'.format(parms11[16]),  '{:.3f}'.format(parms12[18]),   '{:.3f}'.format(parms13[14]),   '{:.3f}'.format(parms14[14])],
                                                    ['pred. var.',      '{:.3f}'.format(parms10[15]),  '{:.3f}'.format(parms11[17]),  '{:.3f}'.format(parms12[19]),   '{:.3f}'.format(parms13[15]),   '{:.3f}'.format(parms14[15])],
                                                    ['n',               '{}'.format(parms10[16]),      '{}'.format(parms11[18]),      '{}'.format(parms12[20]),       '{}'.format(parms13[16]),       '{}'.format(parms14[16])],
                                                    ]),
                                                 columns=['', 'Pareto_data', 'Pareto_data_gauss_noise_1', 'Pareto_data_gauss_noise_2', 'Pareto_data_het_noise_1', 'Pareto_data_het_noise_2'])


    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'synthetic_data_fit_results.xlsx', engine='openpyxl', mode='a') as writer:
        df_synthetic_fit_parms_LR.to_excel(writer, sheet_name='synthetic_data_fit_results_LR', index=False)



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
