from Pareto2GBfit.fitting import *
from Pareto2GBfit.testing import *
import numpy as np
from scipy.stats import describe

# test with actual data
netwealth = np.loadtxt("Examples/netwealth.csv", delimiter = ",")

# set lower bound
b = 100000

# describe data
print("netwealth\n", describe(netwealth))

# test Paretofit and IB1fit
Paretofit(x=netwealth, b=b, bootstraps=1000, method='SLSQP', fit=True, plot=True, plot_cosmetics={'bins': 500})
IB1fit(x=netwealth, b=b, bootstraps=1000, method='SLSQP', fit=True, plot=True, plot_cosmetics={'bins': 500})

# test GB1fit and GBfit
GB1fit(x=netwealth, b=b, bootstraps=100, method='SLSQP', plot=True)
GBfit(x=netwealth, b=b, bootstraps=100, method='SLSQP', plot=True)

# test IB1fit with global optimization basinhopping
IB1fit(x=netwealth, b=b, bootstraps=1000, method='basinhopping', fit=True, plot=True,
       plot_cosmetics={'bins': 500},
       basinhopping_options={'niter': 50, 'stepsize': .75})

# save fitted parameters
b, bs = 100000, 250
p_fit1, p_se1, q_fit1, q_se1 = IB1fit(x=netwealth, b=b, bootstraps=bs, method='SLSQP', verbose=False, return_parameters=True)
a_fit2, a_se2, p_fit2, p_se2, q_fit2, q_se2 = GB1fit(x=netwealth, b=b, bootstraps=bs, method='SLSQP', verbose=False, return_parameters=True)

# check fitted parameters
print("IB1fit: p={} sd=({}), \n\t\tq={} ({})".format(p_fit1, p_se1, q_fit1, q_se1))
print("GB1fit: a={} ({}), \n\t\tp={} ({}), \n\t\tq={} ({})".format(a_fit2, a_se2, p_fit2, p_se2, q_fit2, q_se2))

# testing fitted parameters with LR test
LRtest(LL1=IB1(x=netwealth, b=b, p=p_fit1, q=1).LL, LL2=IB1(x=netwealth, b=b, p=p_fit1, q=q_fit1).LL, df=1)
LRtest(LL1=GB1(x=netwealth, b=b, a=-1, p=p_fit2, q=q_fit2).LL, LL2=GB1(x=netwealth, b=b, a=a_fit2, p=p_fit2, q=q_fit2).LL, df=1)

# Run Paretobranchfit on netwealth
Paretobranchfit(x=netwealth, b=b, bootstraps=bs, rejection_criterion=['LRtest', 'AIC'], method='basinhopping', alpha=.025)

