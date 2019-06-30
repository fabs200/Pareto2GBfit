from Pareto2GBfit import *
import os, matplotlib
import numpy as np
import pandas as pd
from pandas import ExcelWriter


random.seed(104891)

plot_data = False
run_descriptives = True
run_optimize = False

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
    Pareto_data_parms = Paretobranchfit(x=Pareto_data, x0=(-1, .5, 1, 1), b=250,
                                        bootstraps=(4, 4, 4, 4),
                                        return_bestmodel=True, rejection_criteria='AIC', plot=True,
                                        plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_gauss_noise_1_parms = Paretobranchfit(x=Pareto_data_gauss_noise_1, x0=(-1, .5, 1, 1), b=250,
                                                      bootstraps=(4, 4, 4, 4),
                                                      return_bestmodel=True, rejection_criteria='AIC', plot=True,
                                                      plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_gauss_noise_2_parms = Paretobranchfit(x=Pareto_data_gauss_noise_2, x0=(-1, .5, 1, 1), b=250,
                                                      bootstraps=(4, 4, 4, 4),
                                                      return_bestmodel=True, rejection_criteria='AIC', plot=True,
                                                      plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_het_noise_1_parms = Paretobranchfit(x=Pareto_data_het_noise_1, x0=(-1, .5, 1, 1), b=250,
                                                    bootstraps=(4, 4, 4, 4),
                                                    return_bestmodel=True, rejection_criteria='AIC', plot=True,
                                                    plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

    Pareto_data_het_noise_2_parms = Paretobranchfit(x=Pareto_data_het_noise_2, x0=(-1, .5, 1, 1), b=250,
                                                    bootstraps=(4, 4, 4, 4),
                                                    return_bestmodel=True, rejection_criteria='AIC', plot=True,
                                                    plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})


"""
--------------------------------
4. Save fitted parms to table
--------------------------------
"""

if run_optimize:
    df_synthetic_fit_parms = pd.DataFrame(np.array([['a',               '--', '--', '--', '--', '{}'.format(Pareto_data_het_noise_2_parms[1][0])],
                                                    [' ',               '--', '--', '--', '--', '({})'.format(Pareto_data_het_noise_2_parms[1][1])],
                                                    ['c',               '--', '--', '--', '--', '{}'.format(Pareto_data_het_noise_2_parms[1][2])],
                                                    [' ',               '--', '--', '--', '--', '({})'.format(Pareto_data_het_noise_2_parms[1][3])],
                                                    ['p',               '{}'.format(Pareto_data_parms[1][0]), '--'.format(Pareto_data_gauss_noise_1_parms[1][0]), '--'.format(Pareto_data_gauss_noise_2_parms[1][0]), '{}'.format(Pareto_data_het_noise_1_parms[1][0]), '{}'.format(Pareto_data_het_noise_2_parms[1][4])],
                                                    [' ',               '({})'.format(Pareto_data_parms[1][1]), '--'.format(Pareto_data_gauss_noise_1_parms[1][1]), '--'.format(Pareto_data_gauss_noise_2_parms[1][1]), '({})'.format(Pareto_data_het_noise_1_parms[1][1]), '({})'.format(Pareto_data_het_noise_2_parms[1][5])],
                                                    ['q',               '--', '{}'.format(Pareto_data_gauss_noise_1_parms[1][2]), '--'.format(Pareto_data_gauss_noise_2_parms[1][2]), '--', '{}'.format(Pareto_data_het_noise_2_parms[1][6])],
                                                    [' ',               '--', '({})'.format(Pareto_data_gauss_noise_1_parms[1][3]), '--'.format(Pareto_data_gauss_noise_2_parms[1][3]), '--', '({})'.format(Pareto_data_het_noise_2_parms[1][7])],
                                                    ['lower bound b',   '{}'.format(b), '{}'.format(b),'{}'.format(b),'{}'.format(b),'{}'.format(b)],
                                                    ['LL',              '{}'.format(Pareto_data_parms[1][8]), '{}'.format(Pareto_data_gauss_noise_1_parms[1][10]), '{}'.format(Pareto_data_gauss_noise_2_parms[1][10]), '{}'.format(Pareto_data_het_noise_1_parms[1][8]), '{}'.format(Pareto_data_het_noise_2_parms[1][14])],
                                                    ['AIC',             '{}'.format(Pareto_data_parms[1][2]), '{}'.format(Pareto_data_gauss_noise_1_parms[1][4]), '{}'.format(Pareto_data_gauss_noise_2_parms[1][4]), '{}'.format(Pareto_data_het_noise_1_parms[1][2]), '{}'.format(Pareto_data_het_noise_2_parms[1][8])],
                                                    ['MSE',             '{}'.format(Pareto_data_parms[1][5]), '{}'.format(Pareto_data_gauss_noise_1_parms[1][7]), '{}'.format(Pareto_data_gauss_noise_2_parms[1][7]), '{}'.format(Pareto_data_het_noise_1_parms[1][5]),'{}'.format(Pareto_data_het_noise_2_parms[1][8])],
                                                    ['RMSE',            '{}'.format(Pareto_data_parms[1][6]), '{}'.format(Pareto_data_gauss_noise_1_parms[1][8]), '{}'.format(Pareto_data_gauss_noise_2_parms[1][8]), '{}'.format(Pareto_data_het_noise_1_parms[1][6]),'{}'.format(Pareto_data_het_noise_2_parms[1][8])],
                                                    ['emp. mean',       '{}'.format(Pareto_data_parms[1][10]), '{}'.format(Pareto_data_gauss_noise_1_parms[1][12]), '{}'.format(Pareto_data_gauss_noise_2_parms[1][12]), '{}'.format(Pareto_data_het_noise_1_parms[1][10]), '{}'.format(Pareto_data_het_noise_2_parms[1][16])],
                                                    ['emp. var.',       '{}'.format(Pareto_data_parms[1][11]), '{}'.format(Pareto_data_gauss_noise_1_parms[1][13]), '{}'.format(Pareto_data_gauss_noise_2_parms[1][13]), '{}'.format(Pareto_data_het_noise_1_parms[1][11]), '{}'.format(Pareto_data_het_noise_2_parms[1][17])],
                                                    ['pred. mean',      '{}'.format(Pareto_data_parms[1][12]), '{}'.format(Pareto_data_gauss_noise_1_parms[1][14]), '{}'.format(Pareto_data_gauss_noise_2_parms[1][14]), '{}'.format(Pareto_data_het_noise_1_parms[1][12]), '{}'.format(Pareto_data_het_noise_2_parms[1][18])],
                                                    ['pred. var.',      '{}'.format(Pareto_data_parms[1][13]), '{}'.format(Pareto_data_gauss_noise_1_parms[1][15]), '{}'.format(Pareto_data_gauss_noise_2_parms[1][15]), '{}'.format(Pareto_data_het_noise_1_parms[1][13]), '{}'.format(Pareto_data_het_noise_2_parms[1][19])],
                                                    ['n',               '{}'.format(Pareto_data_parms[1][14]), '{}'.format(Pareto_data_gauss_noise_1_parms[1][16]), '{}'.format(Pareto_data_gauss_noise_2_parms[1][16]), '{}'.format(Pareto_data_het_noise_1_parms[1][14]), '{}'.format(Pareto_data_het_noise_2_parms[1][20])],
                                                    ]),
                                                 columns=['', 'Pareto_data', 'Pareto_data_gauss_noise_1', 'Pareto_data_gauss_noise_2', 'Pareto_data_het_noise_1', 'Pareto_data_het_noise_2'])

    # save dataframes to excel sheet
    with ExcelWriter(descriptivespath + 'synthetic_data_fit_results.xlsx', mode='w') as writer:
        df_synthetic_data_descriptives.to_excel(writer, sheet_name='synthetic_data_fit_results', index=False)



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
    plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{{Pareto}}(b=250, \hat{{p}}={{{}}})$'.format(p))
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
