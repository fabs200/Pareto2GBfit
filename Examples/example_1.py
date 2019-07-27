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

# run optimization (size of boostraps is default: n)
Paretofit(x=Pareto_data, x0=2, b=500, method='SLSQP')

# run optimization with less bootstraps
Paretofit(x=Pareto_data, x0=2, bootstraps=500, b=500, method='SLSQP')

# run optimization of noisy data
Paretofit(x=Pareto_data_noise, x0=2, b=500, method='SLSQP')

# test fitting of Pareto_data with basinhopping method and plot fit
Paretofit(x=Pareto_data, b=500, x0=2, bootstraps=100, verbose=True, method='basinhopping', plot=True)

# test fitting of Pareto_data_noise with basinhopping method and plot fit
Paretofit(x=Pareto_data_noise, b=500, x0=2, bootstraps=100, verbose=True, method='basinhopping', plot=True)

# Run Paretobranchfit on simulated Pareto_data
Paretobranchfit(x=Pareto_data, b=500, x0=(-.1, .1, 1, 1), bootstraps=10, rejection_criterion=['LRtest', 'AIC'])

# Run Paretobranchfit on simulated Pareto_data_noise
Paretobranchfit(x=Pareto_data_noise, b=500, bootstraps=10, rejection_criterion=['LRtest', 'AIC'])
