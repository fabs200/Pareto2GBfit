
from Pareto2GBfit.fitting import *
from Pareto2GBfit.testing import *
import numpy as np
from scipy.stats import describe

# Pareto Parameters
b, p = 500, 2.5

# size of overall synthetic / noise data
n = 10000

# noise
mu = 0
sigma = 100
np.random.seed(123)
noise = np.random.normal(mu, sigma, size=n)

# linspace
xmin = 0.1
xmax = 10000
x = np.linspace(xmin, xmax, n)

# noise
mu = 0
sigma = 100
random.seed(123)
noise = np.random.normal(mu, sigma, size=n)
x_noise = x + noise

# random uniform
u = np.array(np.random.uniform(.0, 1., n)) # (1xn)

# Pareto simulated data
Pareto_data = Pareto_icdf(u, b, p)

# Pareto simulated data + noise
Pareto_data_noise = Pareto_icdf(u, b, p) + noise

# test fits with Pareto_data plots
Paretofit(x=Pareto_data, b=500, x0=2, bootstraps=10, verbose=True, method='L-BFGS-B', plot=True)
Paretofit(x=Pareto_data, b=500, x0=2, bootstraps=10, verbose=True, method='SLSQP', plot=True)
IB1fit(x=Pareto_data, b=500, x0=(2,1), bootstraps=10, verbose=True, method='L-BFGS-B', plot=True)
IB1fit(x=Pareto_data, b=500, x0=(2,1), bootstraps=10, verbose=True, method='SLSQP', plot=True)
GB1fit(x=Pareto_data, b=500, x0=(-.1,2,1), bootstraps=10, verbose=True, method='L-BFGS-B', plot=True)
GB1fit(x=Pareto_data, b=500, x0=(-.1,2,1), bootstraps=10, verbose=True, method='SLSQP', plot=True)
GBfit(x=Pareto_data, b=500, x0=(-.1,0,2,1), bootstraps=10, verbose=True, method='L-BFGS-B', plot=True)
GBfit(x=Pareto_data, b=500, x0=(-.1,0,2,1), bootstraps=10, verbose=True, method='SLSQP', plot=True)

# test fits with Pareto_data_noise
Paretofit(x=Pareto_data_noise, b=500, x0=2, bootstraps=10, method='L-BFGS-B', plot=True)
Paretofit(x=Pareto_data_noise, b=500, x0=2, bootstraps=1000, method='SLSQP', plot=True)
IB1fit(x=Pareto_data_noise, b=500, x0=(2,1), bootstraps=10, method='L-BFGS-B', plot=True)
IB1fit(x=Pareto_data_noise, b=500, x0=(2,1), bootstraps=10, method='SLSQP', plot=True)
GB1fit(x=Pareto_data_noise, b=500, x0=(-.1,2,1), bootstraps=10, method='L-BFGS-B', plot=True)
GB1fit(x=Pareto_data_noise, b=500, x0=(-.1,2,1), bootstraps=10, method='SLSQP', plot=True)
GBfit(x=Pareto_data_noise, b=500, x0=(-.1,0,2,1), bootstraps=10, method='L-BFGS-B', plot=True)
GBfit(x=Pareto_data_noise, b=500, x0=(-.1,0,2,1), bootstraps=10, method='SLSQP', plot=True)


# test with actual data
netwealth = np.loadtxt("netwealth.csv", delimiter = ",")

# set lower bound
b = 100000

# describe data
print("netwealth\n", describe(netwealth))
w = np.ones(len(netwealth)) * 5
Paretofit(x=netwealth, b=100000, x0=1, bootstraps=1000, method='SLSQP', fit=True, plot=True, plot_cosmetics={'bins': 500})
IB1fit(x=netwealth, b=100000, x0=(1,1), bootstraps=1000, method='SLSQP', fit=True, plot=True, plot_cosmetics={'bins': 500})
IB1fit(x=netwealth, b=100000, x0=(1,1), bootstraps=1000, method='L-BFGS-B', fit=True, plot=True, plot_cosmetics={'bins': 500}, basinhopping_options={'niter': 50, 'stepsize': .75})

GB1fit(x=netwealth, b=100000, x0=(-.5,1,1), weights=w, bootstraps=100, method='SLSQP', plot=True)
GBfit(x=netwealth, b=100000, x0=(-.5,.1,1,1), weights=w, bootstraps=100, method='SLSQP', plot=True)

basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                      'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123}


# save fitted parameters
p_fit1, p_se1 = Paretofit(x=netwealth, b=100000, x0=1, bootstraps=250, method='SLSQP', verbose=False, return_parameters=True)
p_fit2, p_se2, q_fit2, q_se2 = IB1fit(x=netwealth, b=100000, x0=(1,1), bootstraps=250, method='SLSQP', verbose=False, return_parameters=True)
a_fit3, a_se3, p_fit3, p_se3, q_fit3, q_se3 = GB1fit(x=netwealth, b=100000, x0=(-0.5,1,1), bootstraps=250, method='SLSQP', verbose=False, return_parameters=True)

# testing fitted parameters
LRtest(Pareto(x=netwealth, b=b, p=p_fit1).LL, IB1(x=netwealth, b=b, p=p_fit2, q=q_fit2).LL, df=2)
LRtest(Pareto(x=netwealth, b=b, p=p_fit1).LL, GB1(x=netwealth, b=b, a=a_fit3, p=p_fit3, q=q_fit3).LL, df=3)
LRtest(IB1(x=netwealth, b=b, p=p_fit2, q=q_fit2).LL, GB1(x=netwealth, b=b, a=a_fit3, p=p_fit3, q=q_fit3).LL, df=3)


Pareto_fit = Paretofit(x=netwealth, b=100000, x0=1, bootstraps=100, method='L-BFGS-B', verbose=False, return_parameters=True, return_gofs=True)
IB1_fit = IB1fit(x=netwealth, b=100000, x0=(1,1), bootstraps=50, method='L-BFGS-B', verbose=False, return_parameters=True, return_gofs=True)
GB1_fit = GB1fit(x=netwealth, b=100000, x0=(-.1,1,1), bootstraps=10, method='L-BFGS-B', verbose=False, return_parameters=True, return_gofs=True)
GB_fit = GBfit(x=netwealth, b=100000, x0=(-.1,.1,1,1), bootstraps=4, method='L-BFGS-B', verbose=False, return_parameters=True, return_gofs=True)
p_fit1, p_se1 = Pareto_fit[:2]
p_fit2, p_se2, q_fit2, q_se2 = IB1_fit[:4]
a_fit3, a_se3, p_fit3, p_se3, q_fit3, q_se3 = GB1_fit[:6]
a_fit4, a_se4, c_fit4, c_se4, p_fit4, p_se4, q_fit4, q_se4 = GB_fit[:8]

LRtest1v2 = LRtest(Pareto(x=netwealth, b=b, p=p_fit1).LL,
                   IB1(x=netwealth, b=b, p=p_fit2, q=q_fit2).LL,
                   df=2, verbose=False)
# 2. LRtest IB1 vs GB1
LRtest2v3 = LRtest(IB1(x=x, b=b, p=p_fit2, q=q_fit2).LL,
                   GB1(x=x, b=b, a=a_fit3, p=p_fit3, q=q_fit3).LL,
                   df=3, verbose=False)
# 3. LRtest GB1 vs GB
LRtest3v4 = LRtest(GB1(x=x, b=b, a=a_fit3, p=p_fit3, q=q_fit3).LL,
                   GB(x=x, b=b, a=a_fit4, c=c_fit4, p=p_fit4, q=q_fit4).LL,
                   df=4, verbose=False)

Paretobranchfit(x=netwealth, b=100000, x0=(-.1, .1, 1, 1), bootstraps=(100, 50, 10, 4), rejecting_criteria='LRtest')
Paretobranchfit(x=netwealth, b=100000, x0=(-.1, .1, 1, 1), bootstraps=(100, 50, 10, 4), rejecting_criteria='LRtest')
Paretobranchfit(x=Pareto_data, b=500, x0=(-.1, .1, 1, 1), bootstraps=(100, 50, 10, 4), rejecting_criteria='LRtest')
Paretobranchfit(x=Pareto_data_noise, b=500, x0=(-.1, .1, 1, 1), bootstraps=(100, 50, 10, 4), rejecting_criteria='LRtest')


