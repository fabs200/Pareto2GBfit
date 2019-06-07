from Pareto2GBfit import *
from scipy.optimize import basinhopping


# Pareto Parameters
b, p = 1, 2.5

# size of overall synthetic / noise data
n = 10000
u = np.array(np.random.uniform(.0, 1., n))
x = Pareto_icdf(u, b, p)

minimizer_kwargs = {"method": "BFGS", "args": (x,b)}

# Test ParetoFit with basinhopping

x0 = np.array([2])
ret = basinhopping(Pareto_ll, x0, minimizer_kwargs=minimizer_kwargs, niter=200, disp=False)
print(ret)
print(ret.fun)


# Test IB1Fit with basinhopping

x0 = np.array([2,1])
ret = basinhopping(IB1_ll, x0, minimizer_kwargs=minimizer_kwargs, niter=200, disp=False)
print(ret)
print(ret.fun)

# Test GB1Fit with basinhopping

x0 = np.array([-0.1,2,1])
ret = basinhopping(GB1_ll, x0, minimizer_kwargs=minimizer_kwargs, niter=200, disp=False)
print(ret)
print(ret.fun)

# Test GBFit with basinhopping

x0 = np.array([-1.1,-.1,2,1])
ret = basinhopping(GB_ll, x0, minimizer_kwargs=minimizer_kwargs, niter=200, disp=False)
print(ret)
print(ret.fun)



###### with bounds

# Pareto
x0=1
bnds = (1e-1, np.inf)
bounds = (bnds, )
minimizer_kwargs = {"method": "L-BFGS-B", "args": (x,b), "bounds": bounds}
ret = basinhopping(Pareto_ll, x0, minimizer_kwargs=minimizer_kwargs, niter=500, disp=False, seed=123)
print(ret)
print(ret.fun)

# IB1
x0 = (1,1)
bnds = (1e-14, np.inf)
bounds = (bnds, bnds, )
minimizer_kwargs = {"method": "L-BFGS-B", "args": (x,b), "bounds": bounds}
ret = basinhopping(IB1_ll, x0, minimizer_kwargs=minimizer_kwargs, niter=200, disp=False, seed=123, niter_success=20)
print(ret)
print(ret.fun)

# GB1
x0 = (-1.5, 1, .1)
bnds = (1e-14, np.inf)
bounds = ((-10, 10), bnds, bnds, )
minimizer_kwargs = {"method": "L-BFGS-B", "args": (x,b)}
ret = basinhopping(GB1_ll, x0, minimizer_kwargs=minimizer_kwargs, niter=200, disp=False, seed=123123, niter_success=20)
print(ret)
print(ret.fun)

# GB
x0 = (-1, .1, 1, 1)
a_bnd = (-10, 10)
c_bnd = (0,1)
bnds = (1e-14, np.inf)
bounds = (a_bnd, c_bnd, bnds, bnds, )
minimizer_kwargs = {"method": "L-BFGS-B", "args": (x,b), "bounds": bounds}
ret = basinhopping(GB_ll, x0, minimizer_kwargs=minimizer_kwargs, niter=200, disp=False, seed=123, niter_success=20)
print(ret)
print(ret.fun)


###### with bounds + constraints
# GB1
x0 = (-1.5, 1, .1)
bnds = (1e-14, np.inf)
bounds = ((-10, 10), bnds, bnds, )
def GB1_constraint(parms):
    a = parms[0]
    return (np.min(x)/b)**a
constr = {'type': 'ineq', 'fun': GB1_constraint}
minimizer_kwargs = {"method": "L-BFGS-B", "args": (x,b), "constraints": constr}
ret = basinhopping(GB1_ll, x0, minimizer_kwargs=minimizer_kwargs, niter=200, disp=False, seed=123123, niter_success=20)
print(ret)
print(ret.fun)

# GB
x0 = (-1, .1, 1, 1)
a_bnd = (-10, 10)
c_bnd = (0,1)
bnds = (1e-14, np.inf)
bounds = (a_bnd, c_bnd, bnds, bnds, )
def GB_constraint1(parms):
    a = parms[0]
    c = parms[1]
    return (b**a)/(1-c) - np.min(x)**a
def GB_constraint2(parms):
    a = parms[0]
    c = parms[1]
    return (b**a)/(1-c) - np.max(x)**a
def GB_constraint3(parms):
    c = parms[1]
    return 1-c
constr = ({'type': 'ineq', 'fun': GB_constraint1},
          {'type': 'ineq', 'fun': GB_constraint2})
def GB_constraint(parms):
    a = parms[0]
    c = parms[1]
    return (1-c)*(np.min(x)/b)**a
constr_2 = ({'type': 'ineq', 'fun': GB_constraint})

minimizer_kwargs = {"method": "L-BFGS-B", "args": (x,b), "bounds": bounds, "constraints": constr_2}
ret = basinhopping(GB_ll, x0, minimizer_kwargs=minimizer_kwargs, niter=200, disp=False, seed=123, niter_success=20)
print(ret)
print(ret.fun)
