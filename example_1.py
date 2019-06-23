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

# Run Paretobranchfit on simulated Pareto_data
Paretobranchfit(x=Pareto_data, b=500, x0=(-.1, .1, 1, 1), bootstraps=(100, 50, 10, 4), rejection_criteria='LRtest', verbose=True)
Paretobranchfit(x=Pareto_data_noise, b=500, x0=(-.1, .1, 1, 1), bootstraps=(100, 50, 10, 4), rejection_criteria='LRtest', verbose=True)