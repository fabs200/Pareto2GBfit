
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

Paretofit(x=netwealth, b=100000, x0=1, bootstraps=1000, method='SLSQP', fit=True, plot=True, plot_cosmetics={'bins': 500})
IB1fit(x=netwealth, b=100000, x0=(1,1), bootstraps=1000, method='SLSQP', fit=True, plot=True, plot_cosmetics={'bins': 500})
IB1fit(x=netwealth, b=100000, x0=(1,1), bootstraps=1000, method='L-BFGS-B', fit=True, plot=True, plot_cosmetics={'bins': 500}, basinhopping_options={'niter': 50, 'stepsize': .75})

GB1fit(x=netwealth, b=100000, x0=(-.5,1,1), bootstraps=100, method='SLSQP', plot=True)
GBfit(x=netwealth, b=100000, x0=(-.5,.1,1,1), bootstraps=100, method='SLSQP', plot=True)

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
