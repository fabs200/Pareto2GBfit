from Pareto2GBfit import *

np.set_printoptions(precision=8)

plot_pdf_cdf = False
run_optimize = False
plot_fit = False # needs run_optimize=True


""" 1. Data Generating Process """
# test parameters
a, c, p, q = -1, 0, 2, 1 # to be optimized parms
b = 1000 # fixed parm

# size of overall synthetic / noise data
n = 10000

# noise
mu = 0
sigma = 100
random.seed(0)
noise = np.random.normal(mu, sigma, size=n)

# random uniform
u = np.array(np.random.uniform(.0, 1., n)) # (1xn)

# linspace
xmin = .01
xmax = 10000
x = linspace(xmin, xmax, n)
x_noise = x + noise

# simulated data + noise
GB_data, GB_u = GB_icdf_ne(x, a, b, c, p, q)
GB_data_noise, GB_u_noise = GB_icdf_ne(x_noise, a, b, c, p, q)
Pareto_data = Pareto_icdf(u, b, p)

# check data
plt.scatter(GB_u, GB_data, marker=".", s=.7, color='b')
plt.scatter(GB_u_noise, GB_data_noise, marker=".", s=.7, color='m')
#plt.scatter(np.sort(u), Pareto_data, marker=".", s=.7, color='r')
plt.show()

# check cdf (x>b)
# plt.scatter(x[x>b], GB_cdf_ne(x=x[x>b], a=-1, c=0, b=100, p=2, q=1), marker=".", s=.5, alpha=.75, color='r')
# plt.scatter(x_noise[x_noise>b], GB_cdf_ne(x=x_noise[x_noise>b], a=-1, c=0, b=100, p=2, q=1), marker=".", s=.75, alpha=.75, color='b')
# plt.show()

# check cdf (x<b)
# plt.scatter(x[x<=b], GB1_cdf(x=x[x<=b], a=1, b=1000, p=2, q=2), marker=".", s=.5, alpha=.75, color='k')
# plt.scatter(x_noise[x_noise<=b], GB1_cdf(x=x_noise[x_noise<=b], a=1, b=1000, p=2, q=1), marker=".", s=1.5, alpha=1, color='b')
# plt.show()

""" 2. Plot pdfs """
if plot_pdf_cdf is True:
    n = 5000
    xmin = .0001
    xmax = 100
    x = np.linspace(xmin, xmax, n)

    # Note: if a=-1: x>b, if a=1: a<b

    # 1: pareto (x>b), 2: IB1 (x>b), 3: power/uniform, 4: power/uniform, line 5: normal, 6: Rayleigh
    avals = [-1, -1, 1, .01, 2, 2, 1]
    bvals = [2, 4]*4
    cvals = [0]*7
    pvals = [1, 2, 1, 4, .5, 1, 1]
    qvals = [1, 2, 1, 1, 10, 10, 10]

    plt.figure()
    plt.xlabel("x"); plt.ylabel("pdf")
    plt.plot(x[x>bvals[0]], GB_pdf(x[x>bvals[0]], avals[0], bvals[0], cvals[0], pvals[0], qvals[0]), color='r', label="a={}, b={}, c={}, p={}, q={} (Pareto)".format(avals[0], bvals[0], cvals[0], pvals[0], qvals[0]))
    plt.plot(x[x>bvals[1]], GB_pdf(x[x>bvals[1]], avals[1], bvals[1], cvals[1], pvals[1], qvals[1]), color='b', label="a={}, b={}, c={}, p={}, q={} (Inverse Beta 1)".format(avals[1], bvals[1], cvals[1], pvals[1], qvals[1]))
    plt.plot(x[x<=bvals[2]], GB_pdf(x[x<=bvals[2]], avals[2], bvals[2], cvals[2], pvals[2], qvals[2]), color='g', label="a={}, b={}, c={}, p={}, q={} (Uniform)".format(avals[2], bvals[2], cvals[2], pvals[2], qvals[2]))
    plt.plot(x[x<=bvals[3]], GB_pdf(x[x<=bvals[3]], avals[3], bvals[3], cvals[3], pvals[3], qvals[3]), color='y', label="a={}, b={}, c={}, p={}, q={} (Power)".format(avals[3], bvals[3], cvals[3], pvals[3], qvals[3]))
    plt.plot(x[x<=bvals[4]], GB_pdf(x[x<=bvals[4]], avals[4], bvals[4], cvals[4], pvals[4], qvals[4]), color='k', label="a={}, b={}, c={}, p={}, q={} (Half Normal)".format(avals[4], bvals[4], cvals[4], pvals[4], qvals[4]))
    plt.plot(x[x<=bvals[5]], GB_pdf(x[x<=bvals[5]], avals[5], bvals[5], cvals[5], pvals[5], qvals[5]), color='m', label="a={}, b={}, c={}, p={}, q={} (Rayleigh)".format(avals[5], bvals[5], cvals[5], pvals[5], qvals[5]))
    plt.plot(x[x<=bvals[6]], GB_pdf(x[x<=bvals[6]], avals[6], bvals[6], cvals[6], pvals[6], qvals[6]), color='orange', label="a={}, b={}, c={}, p={}, q={} (Exponential)".format(avals[6], bvals[6], cvals[6], pvals[6], qvals[6]))
    plt.axvline(x=bvals[0], ymin=0, ymax=4, color='black', linewidth=1, alpha=.5, label="b (lower/upper bounds)", linestyle='dashed')
    plt.axvline(x=bvals[1], ymin=0, ymax=4, color='black', linewidth=1, alpha=.5, linestyle='dashed')
    plt.legend(loc='upper right', fontsize = 'x-small'); plt.ylim(bottom=0, top=2);plt.xlim(left=0, right=10)
    plt.savefig('graphs/GB_simulated_pdf.png', dpi=300)
    plt.show()


    """ 4. Plot cdfs """
    plt.figure()
    plt.xlabel("x"); plt.ylabel("cdf")
    plt.plot(x[x>bvals[0]], GB_cdf_ne(x[x>bvals[0]], avals[0], bvals[0], cvals[0], pvals[0], qvals[0]), color='r', label="a={}, b={}, c={}, p={}, q={} (Pareto)".format(avals[0], bvals[0], cvals[0], pvals[0], qvals[0]))
    plt.plot(x[x>bvals[1]], GB_cdf_ne(x[x>bvals[1]], avals[1], bvals[1], cvals[1], pvals[1], qvals[1]), color='b', label="a={}, b={}, c={}, p={}, q={} (Inverse Beta 1)".format(avals[1], bvals[1], cvals[1], pvals[1], qvals[1]))
    plt.plot(x[x<=bvals[2]], GB_cdf_ne(x[x<=bvals[2]], avals[2], bvals[2], cvals[2], pvals[2], qvals[2]), color='g', label="a={}, b={}, c={}, p={}, q={} (Uniform)".format(avals[2], bvals[2], cvals[2], pvals[2], qvals[2]))
    plt.plot(x[x<=bvals[3]], GB_cdf_ne(x[x<=bvals[3]], avals[3], bvals[3], cvals[3], pvals[3], qvals[3]), color='y', label="a={}, b={}, c={}, p={}, q={} (Power)".format(avals[3], bvals[3], cvals[3], pvals[3], qvals[3]))
    plt.plot(x[x<=bvals[4]], GB_cdf_ne(x[x<=bvals[4]], avals[4], bvals[4], cvals[4], pvals[4], qvals[4]), color='k', label="a={}, b={}, c={}, p={}, q={} (Half Normal)".format(avals[4], bvals[4], cvals[4], pvals[4], qvals[4]))
    plt.plot(x[x<=bvals[5]], GB_cdf_ne(x[x<=bvals[5]], avals[5], bvals[5], cvals[5], pvals[5], qvals[5]), color='m', label="a={}, b={}, c={}, p={}, q={} (Rayleigh)".format(avals[5], bvals[5], cvals[5], pvals[5], qvals[5]))
    plt.plot(x[x<=bvals[6]], GB_cdf_ne(x[x<=bvals[6]], avals[6], bvals[6], cvals[6], pvals[6], qvals[6]), color='orange', label="a={}, b={}, c={}, p={}, q={} (Exponential)".format(avals[6], bvals[6], cvals[6], pvals[6], qvals[6]))
    plt.axvline(x=bvals[0], ymin=0, ymax=4, color='black', linewidth=1, alpha=.5, label="b (lower/upper bounds)", linestyle='dashed')
    plt.axvline(x=bvals[1], ymin=0, ymax=4, color='black', linewidth=1, alpha=.5, linestyle='dashed')
    plt.legend(loc='upper right', fontsize = 'x-small'); plt.ylim(bottom=0, top=1.1); plt.xlim(left=0, right=15)
    plt.savefig('graphs/GB_simulated_cdf.png', dpi=300)
    plt.show()



""" 3a. Constrained Optimization: SLSQP a<0 """
# if run_optimize is True and a<0:
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
          {'type': 'ineq', 'fun': GB_constraint2},
          {'type': 'eq', 'fun': GB_constraint3})

def GB_constraint(parms):
    a = parms[0]
    c = parms[1]
    return (1-c)*(np.min(x)/b)**a
constr_2 = ({'type': 'ineq', 'fun': GB_constraint},
            {'type': 'ineq', 'fun': GB_constraint3})

# bounds for parameters a, c, p, q
# NOTE: It turns out that for GB1 one need to set narrow bounds
bounds = ((-1.25,-.75), (0,1), (1.5,2.5), (1.5, 2.5))
bounds = ((-2,-.1), (.1,2), (.1, 3))

# initial guess
x0 = np.array([-0.8, 0.5, 2.1, 1.9])
x0 = np.array([-.1, .1, .9])

print("true: a: {}, c: {}, p: {}, q: {}, b (fix): {}".format(a, c, p, q, b))

### 5a.1 minimize -ll with GB1_data
x = Pareto_data
result = opt.minimize(GB_ll, x0, method='SLSQP', bounds=bounds, tol=1e-14, args=(x, b),
                      options=({'maxiter': 350, 'disp': True}), constraints=constr)
print(result)

