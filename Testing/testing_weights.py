# x=Pareto_data
# weights=np.array([1])

b=1000000
# x0=(-.1,1,1)
x0=(-.1,.1,1,1)
bootstraps=20
method='SLSQP'
omit_missings=True
verbose_bootstrap=False
ci=True
verbose=True
fit=False
plot=True
suppress_warnings=True
return_parameters=False
return_gofs=False
plot_cosmetics={'bins': 200, 'col_data': 'blue', 'col_fit': 'orange'}
basinhopping_options={'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None,
                      'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123}
slsqp_options={'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14,
               'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}
num_bins=200
col_fit='blue'
col_model='red'
weighting = 'expand'

x=dfPSID['wealth_17']
weights=dfPSID['weight_17']

x=dfSOEP['wealth_17']
weights=dfSOEP['weight_17']
weighting = 'multiply'


weights=np.array([1])


weights = np.ones(len(x))*2
weights = np.ones(len(x))


###########################################

Paretofit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=2000000, x0=2, bootstraps=100, plot=True, weighting='multiply') # OK!!
Paretofit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=2000000, x0=2, bootstraps=100, plot=True, weighting='multiply') # OK!!

Paretofit(x=dfPSID['wealth_17'], weights=weights, b=2000000, x0=2, bootstraps=100, plot=True, weighting='multiply') # OK!!
Paretofit(x=dfSOEP['wealth_17'], weights=weights, b=2000000, x0=2, bootstraps=100, plot=True, weighting='multiply') # OK!!

Paretofit(x=dfPSID['wealth_17'], b=2000000, x0=2, bootstraps=100, plot=True) # OK!!
Paretofit(x=dfSOEP['wealth_17'], b=2000000, x0=2, bootstraps=100, plot=True) # OK!!

###########################################

IB1fit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=2000000, x0=(2,1), bootstraps=100, plot=True, weighting='multiply') # OK!!
IB1fit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=2000000, x0=(1,1), bootstraps=100, plot=True, weighting='multiply') # OK!!

IB1fit(x=dfPSID['wealth_17'], weights=weights, b=2000000, x0=(1,1), bootstraps=100, plot=True, weighting='multiply')  # OK!!
IB1fit(x=dfSOEP['wealth_17'], weights=weights, b=2000000, x0=(1,1), bootstraps=100, plot=True, weighting='multiply')  # OK!!

IB1fit(x=dfPSID['wealth_17'], b=1000000, x0=(1,1), bootstraps=100, plot=True)  # OK!!
IB1fit(x=dfSOEP['wealth_17'], b=1000000, x0=(1,1), bootstraps=100, plot=True)  # OK!!

###########################################

GB1fit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=1000000, x0=(-.1,2,1), bootstraps=100, plot=True, weighting='multiply', plot_cosmetics={'bins':500})
GB1fit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=1000000, x0=(-.1,1,1), bootstraps=100, plot=True, weighting='multiply', plot_cosmetics={'bins':500})

GB1fit(x=dfSOEP['wealth_17'], weights=weights, b=1000000, x0=(-.1,1,1), bootstraps=100, plot=True, weighting='multiply')   # BAD
GB1fit(x=dfSOEP['wealth_17'], weights=weights, b=1000000, x0=(-.1,1,1), bootstraps=100, plot=True, weighting='multiply')   # BAD

GB1fit(x=dfPSID['wealth_17'], b=1000000, x0=(-.1,2,1), bootstraps=10, plot=True)  #
GB1fit(x=dfSOEP['wealth_17'], b=1000000, x0=(-.1,1,1), bootstraps=10, plot=True)  #


# GB1fit(x=Pareto_data, b=250, x0=(-.1,1,1), bootstraps=10, plot=True, plot_cosmetics={'bins': 500})

###########################################

# GBfit(x=dfPSID['wealth_17'], weights=dfPSID['weight_17'], b=500000, x0=(-.1,.1,1,1), bootstraps=100, plot=True, weighting='expand')
GBfit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=500000, x0=(-.1,.1,1,1), bootstraps=100, plot=True, weighting='multiply')

GBfit(x=dfSOEP['wealth_17'], weights=weights, b=500000, x0=(-.1,.1,1,1), bootstraps=100, plot=True, weighting='multiply')  # BAD
GBfit(x=dfSOEP['wealth_17'], weights=weights, b=500000, x0=(-.1,.1,1,1), bootstraps=100, plot=True, weighting='multiply')  # BAD

GBfit(x=dfPSID['wealth_17'], b=1000000, x0=(-.1,.1,1,1), bootstraps=100, plot=True)  # OK !!
GBfit(x=dfSOEP['wealth_17'], b=1000000, x0=(-.1,.1,1,1), bootstraps=100, plot=True)  # OK !!

###########################################

Paretobranchfit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=2000000, x0=(-.1,.1,1,1),
                bootstraps=10, return_bestmodel=True, plot=True,
                rejection_criterion='LRtest',
                plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

fit = Paretobranchfit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=2000000, x0=(-.1,.1,1,1), bootstraps=10, return_bestmodel=True, plot=True,
                rejection_criterion='LRtest', plot_cosmetics={'bins': 250, 'col_data': 'blue', 'col_fit': 'red'})

