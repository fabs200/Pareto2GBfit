# x=Pareto_data
# weights=np.array([1])

b=2000000
x0=(-.1,1,1)
x0=(-.1,.1,1,1)
bootstraps=10
method='SLSQP'
omit_missings=True
verbose_bootstrap=False
ci=True
verbose=True
fit=False
plot=False
suppress_warnings=True
return_parameters=False
return_gofs=False
plot_cosmetics={'bins': 50, 'col_data': 'blue', 'col_fit': 'orange'}
basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                      'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123}
slsqp_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
               'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}

x=dfPSID['wealth_17']
weights=dfPSID['weight_17']

x=dfSOEP['wealth_17']
weights=dfSOEP['weight_17']

weighting = 'expand'
weighting = 'multiply'

weights=np.array([1])

###########################################

# Paretofit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=500000, x0=2, bootstraps=100, plot=True, weighting='expand')
Paretofit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=500000, x0=2, bootstraps=1000, plot=True, weighting='multiply')

# Paretofit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=500000, x0=2, bootstraps=25, plot=True, weighting='expand')
Paretofit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=1000000, x0=2, bootstraps=25, plot=True, weighting='multiply')  # OK!!

Paretofit(x=dfPSID['wealth_17'], b=500000, x0=2, bootstraps=100, plot=True)  # OK!!
Paretofit(x=dfSOEP['wealth_17'], b=500000, x0=2, bootstraps=100, plot=True, method='basinhopping')

###########################################

# IB1fit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=500000, x0=(2,1), bootstraps=100, plot=True, weighting='expand')
IB1fit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=500000, x0=(2,1), bootstraps=1000, plot=True, weighting='multiply')

#IB1fit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=500000, x0=(1,1), bootstraps=5, plot=True, weighting='expand')
IB1fit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=1000000, x0=(1,1), bootstraps=1000, plot=True, weighting='multiply')  # OK!!

IB1fit(x=dfPSID['wealth_17'], b=500000, x0=(2,1), bootstraps=100, plot=True)  # OK!!
IB1fit(x=dfSOEP['wealth_17'], b=500000, x0=(1,1), bootstraps=100, plot=True, method='basinhopping')

###########################################

# GB1fit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=500000, x0=(-.1,1,1), bootstraps=100, plot=True, weighting='expand')
GB1fit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=500000, x0=(-.1,1,1), bootstraps=10, plot=True, weighting='multiply')

#GB1fit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=500000, x0=(1,1,-.1), bootstraps=5, plot=True, weighting='expand')
GB1fit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=500000, x0=(-.1,1,1), bootstraps=100, plot=True, weighting='multiply')   # BAD

GB1fit(x=dfPSID['wealth_17'], b=500000, x0=(-.1,1,1), bootstraps=50, plot=True)  # OK!!
GB1fit(x=dfSOEP['wealth_17'], b=500000, x0=(-.1,1,1), bootstraps=100, plot=True, method='basinhopping')

###########################################

# GBfit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=500000, x0=(-.1,.1,1,1), bootstraps=100, plot=True, weighting='expand')
GBfit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=500000, x0=(-.1,.1,1,1), bootstraps=1000, plot=True, weighting='multiply')

#GBfit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=500000, x0=(1,.1,1,-.1), bootstraps=5, plot=True, weighting='expand')
GBfit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=500000, x0=(-.1,.1,1,1), bootstraps=100, plot=True, weighting='multiply')  # BAD

GBfit(x=dfPSID['wealth_17'], b=1000000, x0=(-.1,.1,1,1), bootstraps=50, plot=True, weighting='multiply')  # OK!!
GBfit(x=dfSOEP['wealth_17'], b=500000, x0=(-.1,.1,1,1), bootstraps=10, plot=True, method='basinhopping')
