from Pareto2GBfit import *
import matplotlib
import os

random.seed(104891)

plot_pdf_cdf = False
run_optimize = True
plot_fit = False # needs run_optimize=True

if os.name == 'nt':
    graphspath = 'D:/OneDrive/Studium/Masterarbeit/Python/graphs/'
if os.name == 'mac':
    graphspath = '/Users/Fabian/OneDrive/Studium/Masterarbeit/Python/graphs/'

""" 
--------------------------------
1. Data Generating Process 
--------------------------------
"""

# Pareto Parameters
b, p = 250, 2.5

# size of overall synthetic / noise data
n = 10000

# linspace
xmin = 0.1
xmax = 10000
x = linspace(xmin, xmax, n)

# random uniform
u = np.array(np.random.uniform(.0, 1., n))
u = np.sort(u)

# Pareto simulated data (no noise)
Pareto_data = Pareto_icdf(u, b, p)
# alternatively: simulate Pareto_data numerically evaluated (ne) along pdf
# Pareto_data_ne, u_ne = Pareto_icdf_ne(x[x > b], b, p)

# 1. Small Gaussian noise
mu = 0
sigma1 = 100
gauss_noise_1 = np.random.normal(mu, sigma1, size=n)

# 2. Large Gaussian noise
sigma2 = 200
gauss_noise_2 = np.random.normal(mu, sigma2, size=n)

# Pareto simulated data + het. Gaussian noise
Pareto_data_gauss_noise_1 = Pareto_data + gauss_noise_1
Pareto_data_gauss_noise_2 = Pareto_data + gauss_noise_2

# Define function that generates heteroscedastic noise
def het(x, sigma, s=1.):
    x = np.array(x)
    n = np.size(x)
    e = x*(s * (np.random.normal(0, sigma, n)))
    return x + e

# sensitivity
s = 15e-4

# 3. Small heteroscedastic Gaussian noise
Pareto_data_het_noise_1 = het(x=Pareto_data, sigma=sigma1, s=s)

# 4. Large heteroscedastic Gaussian noise
Pareto_data_het_noise_2 = het(x=Pareto_data, sigma=sigma2, s=s)

# check gaussian noise data
plt.scatter(u, Pareto_data_gauss_noise_2, marker="o", s=2, color='blue', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=N(0,200^2)$')
plt.scatter(u, Pareto_data_gauss_noise_1, marker="o", s=2, color='orangered', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=N(0,100^2)$')
plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{Pareto}(b=250, p=2.5)=x$')
plt.legend(loc='upper left'); plt.xlabel('cdf'); plt.ylabel('x');
for type in ['png', 'pdf']:
    plt.savefig(fname=graphspath + 'icdf_gauss_noise.' + type, dpi=300, format=type)
plt.show()
plt.close()

# check heteroscedastic noise data
plt.scatter(u, Pareto_data_het_noise_2, marker="o", s=2, color='blue', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=x\cdot s\cdot N(0,100^2)$')
plt.scatter(u, Pareto_data_het_noise_1, marker="o", s=2, color='orangered', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=x\cdot s\cdot N(0,200^2)$')
plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{Pareto}(b=250, p=2.5)=x$')
plt.legend(loc='upper left'); plt.xlabel('cdf'); plt.ylabel('x')
for type in ['png', 'pdf']:
    plt.savefig(fname=graphspath + 'icdf_het_noise.' + type, dpi=300, format=type)
plt.show()
plt.close()

""" 
--------------------------------
2. Fitting
--------------------------------
"""

# TODO: ParetoBranchFit() -> get parameters of best model
# TODO: 1x set parameters of fit no_noise: Pareto_data
# TODO: 4x set parameters of fit noised_data: Pareto_data_gauss_noise_1, Pareto_data_gauss_noise_2, Pareto_data_het_noise_1, Pareto_data_het_noise_2

Pareto_data_parms = Paretobranchfit(x=Pareto_data, x0=(-1, .5, 1, 1), b=250, bootstraps=(250, 250, 100, 50),
                                    return_bestmodel=True, rejection_criteria='AIC', plot=True, save_all_plots=True)

Pareto_data_gauss_noise_1_parms = Paretobranchfit(x=Pareto_data_gauss_noise_1, x0=(-1, .5, 1, 1),
                                                  bootstraps=(250, 250, 100, 50), b=250,
                                                  return_bestmodel=True, rejection_criteria='AIC', plot=True)

Pareto_data_gauss_noise_2_parms = Paretobranchfit(x=Pareto_data_gauss_noise_2, x0=(-1, .5, 1, 1),
                                                  bootstraps=(250, 250, 100, 50), b=250,
                                                  return_bestmodel=True, rejection_criteria='AIC', plot=True)

Pareto_data_het_noise_1_parms = Paretobranchfit(x=Pareto_data_het_noise_1, x0=(-1, .5, 1, 1),
                                                bootstraps=(250, 250, 100, 50), b=250,
                                                return_bestmodel=True, rejection_criteria='AIC', plot=True)

Pareto_data_het_noise_2_parms = Paretobranchfit(x=Pareto_data_het_noise_2, x0=(-1, .5, 1, 1),
                                                bootstraps=(250, 250, 100, 50), b=250,
                                                return_bestmodel=True, rejection_criteria='AIC', plot=True)


""" 
--------------------------------
3. Plot Fit vs data
--------------------------------
"""

# TODO: 1x no_noise + fit
# TODO: 4x no_noise + noised_data + fit

### Pareto_data_gauss_noise_1
print(Pareto_data_gauss_noise_1_parms[0])

Pareto_data_gauss_noise_1_fit = Pareto_icdf(u=u, b=b, p=Pareto_data_gauss_noise_1_parms[1][0])

plt.scatter(u, Pareto_data_gauss_noise_1, marker="o", s=2, color='orangered', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=N(0,100^2)$')
plt.scatter(u, Pareto_data_gauss_noise_1_fit, marker="o", s=2, color='red', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=N(0,100^2)$')
plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{Pareto}(b=250, p=2.5)=x$')
plt.legend(loc='upper left'); plt.xlabel('cdf'); plt.ylabel('x');
# for type in ['png', 'pdf']:
#     plt.savefig(fname=graphspath + 'fit_gauss_noise1.' + type, dpi=300, format=type)
plt.show()
plt.close()

### Pareto_data_gauss_noise_2
print(Pareto_data_gauss_noise_2_parms[0])

plt.scatter(u, Pareto_data_gauss_noise_2, marker="o", s=2, color='blue', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=N(0,200^2)$')
plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{Pareto}(b=250, p=2.5)=x$')
plt.legend(loc='upper left'); plt.xlabel('cdf'); plt.ylabel('x');
for type in ['png', 'pdf']:
    plt.savefig(fname=graphspath + 'fit_gauss_noise2.' + type, dpi=300, format=type)
plt.show()
plt.close()

### Pareto_data_het_noise_1
print(Pareto_data_het_noise_1_parms[0])

plt.scatter(u, Pareto_data_het_noise_1, marker="o", s=2, color='orangered', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=N(0,100^2)$')
plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{Pareto}(b=250, p=2.5)=x$')
plt.legend(loc='upper left'); plt.xlabel('cdf'); plt.ylabel('x');
for type in ['png', 'pdf']:
    plt.savefig(fname=graphspath + 'fit_gauss_noise1.' + type, dpi=300, format=type)
plt.show()
plt.close()

### Pareto_data_het_noise_2
print(Pareto_data_het_noise_2_parms[0])

plt.scatter(u, Pareto_data_het_noise_2, marker="o", s=2, color='blue', alpha=.75, label=r'$x+\epsilon$ with $\epsilon=N(0,200^2)$')
plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.75, label=r'$icdf_{Pareto}(b=250, p=2.5)=x$')
plt.legend(loc='upper left'); plt.xlabel('cdf'); plt.ylabel('x');
for type in ['png', 'pdf']:
    plt.savefig(fname=graphspath + 'fit_gauss_noise2.' + type, dpi=300, format=type)
plt.show()
plt.close()




""" 
----------------------------------------------
4. Save results of true and fitted parameters
----------------------------------------------
"""

# TODO



























