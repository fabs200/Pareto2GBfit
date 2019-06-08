
from Pareto2GBfit.distributions import *
from Pareto2GBfit.fitting import *
import numpy as np
from scipy.stats import describe

np.set_printoptions(precision=4)

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
x_noise = x + noise

# random uniform
u = np.array(np.random.uniform(.0, 1., n)) # (1xn)

# Pareto simulated data
Pareto_data = Pareto_icdf(u, b, p)


# test
Paretofit(x=Pareto_data, b=500, x0=2, bootstraps=10, verbose=True, method='L-BFGS-B', plot=True)
Paretofit(x=Pareto_data, b=500, x0=2, bootstraps=10, verbose=True, method='SLSQP', plot=True)
IB1fit(x=Pareto_data, b=500, x0=(2,1), bootstraps=10, verbose=True, method='L-BFGS-B', plot=True)
IB1fit(x=Pareto_data, b=500, x0=(2,1), bootstraps=10, verbose=True, method='SLSQP', plot=True)
GB1fit(x=Pareto_data, b=500, x0=(-.1,2,1), bootstraps=10, verbose=True, method='L-BFGS-B', plot=True)
GB1fit(x=Pareto_data, b=500, x0=(-.1,2,1), bootstraps=10, verbose=True, method='SLSQP', plot=True)
GBfit(x=Pareto_data, b=500, x0=(-.1,0,2,1), bootstraps=10, verbose=True, method='L-BFGS-B', plot=True)
GBfit(x=Pareto_data, b=500, x0=(-.1,0,2,1), bootstraps=10, verbose=True, method='SLSQP', plot=True)

Paretofit(x=Pareto_data, b=500, x0=2, bootstraps=10, method='L-BFGS-B')
Paretofit(x=Pareto_data, b=500, x0=2, bootstraps=10, method='SLSQP')
IB1fit(x=Pareto_data, b=500, x0=(2,1), bootstraps=10, method='L-BFGS-B')
IB1fit(x=Pareto_data, b=500, x0=(2,1), bootstraps=10, method='SLSQP')
GB1fit(x=Pareto_data, b=500, x0=(-.1,2,1), bootstraps=10, method='L-BFGS-B')
GB1fit(x=Pareto_data, b=500, x0=(-.1,2,1), bootstraps=10, method='SLSQP')
GBfit(x=Pareto_data, b=500, x0=(-.1,0,2,1), bootstraps=10, method='L-BFGS-B')
GBfit(x=Pareto_data, b=500, x0=(-.1,0,2,1), bootstraps=10, method='SLSQP')


# test with actual data
netwealth = np.loadtxt("netwealth.csv", delimiter = ",")

print("netwealth\n", describe(netwealth))

Paretofit(x=netwealth, b=100000, x0=2, bootstraps=100, method='SLSQP', plot=True)
IB1fit(x=netwealth, b=100000, x0=(1,1), bootstraps=100, method='SLSQP', plot=True)
GB1fit(x=netwealth, b=100000, x0=(-.5,1,1), bootstraps=100, method='SLSQP', plot=True)
GBfit(x=netwealth, b=100000, x0=(-.5,.1,1,1), bootstraps=100, method='SLSQP', plot=True)

