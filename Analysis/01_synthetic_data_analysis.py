from Pareto2GBfit import *
import matplotlib

random.seed(104891)

plot_pdf_cdf = False
run_optimize = True
plot_fit = False # needs run_optimize=True

""" 
--------------------------------
1. Data Generating Process 
--------------------------------
"""

# Pareto Parameters
b, p = 250, 2.5

# size of overall synthetic / noise data
n = 10000

# 1. Small Gaussian noise
mu = 0
sigma1 = 100
gauss_noise_1 = np.random.normal(mu, sigma1, size=n)

# 2. Large Gaussian noise
sigma2 = 400
gauss_noise_2 = np.random.normal(mu, sigma2, size=n)

# 3. Small heteroscedastic noise
# Note: to save computational time, we append 1000 obs in each loop to the list het_noise_1
# het_noise_1, sigmalinspace = [], np.int(n/1000)
# for sig in linspace(10, 100, sigmalinspace):
#     het_noise_1.append(np.random.normal(mu, sig, size=n)[0:1000])
# het_noise_1 = [val for sublist in het_noise_1 for val in sublist]

# 4. Large heteroscedastic noise
# het_noise_2 = []
# for sig in linspace(10, 600, sigmalinspace):
#     het_noise_2.append(np.random.normal(mu, sig, size=n)[0:1000])
# het_noise_2 = [val for sublist in het_noise_2 for val in sublist]

# Define function that generates heteroscedastic noise
# def het(sigma, n, s=1., t=1.):
#     """
#     :param sigma: initial sigma value
#     :param n: size of np.array
#     :param s: growth rate
#     :param t: power
#     :return: np.array() with het.sced. variance
#     """
#     het, m = [sigma, ], 1
#     sig = sigma + s*sigma**t
#     while m<n:
#         het.append(sig)
#         sig = sig + s*sigma**t
#         m += 1
#     return np.array(het)

# linspace
xmin = 0.1
xmax = 10000
x = linspace(xmin, xmax, n)
x_gauss_noise_1 = x + gauss_noise_1
x_gauss_noise_2 = x + gauss_noise_2

# random uniform
u = np.array(np.random.uniform(.0, 1., n))
u = np.sort(u)

# Pareto simulated data (no noise)
Pareto_data = Pareto_icdf(u, b, p)
# alternatively: simulate Pareto_data numerically evaluated (ne) along pdf
# Pareto_data_ne, u_ne = Pareto_icdf_ne(x[x > b], b, p)

# Pareto simulated data + noise
Pareto_data_gauss_noise_1 = Pareto_data + gauss_noise_1
Pareto_data_gauss_noise_2 = Pareto_data + gauss_noise_2

# Define function that generates heteroscedastic noise
def het(x, sigma, s=1.):
    x = np.array(x)
    n = np.size(x)
    e = x*(s * (np.random.normal(0, sigma, n)))
    return x + e

# 3. Small heteroscedastic noise
het_noise_1 = np.random.normal(np.zeros(n), het(sigma=sigma1, s=.0001), size=n)

# 4. Large heteroscedastic noise
het_noise_2 = np.random.normal(np.zeros(n), het(sigma=sigma1, n=n, s=.0005), size=n)

Pareto_data_het_noise_1 = het(x=Pareto_data, sigma=sigma1, s=5e-4)
Pareto_data_het_noise_2 = het(x=Pareto_data, sigma=sigma2, s=5e-4)

# check gaussian noise data
plt.scatter(u, Pareto_data_gauss_noise_2, marker="o", s=2, color='blue', alpha=1)# TODO, label=r'$icdf_{Pareto}(b=250, p=2.5)+\mu=${}, $\sigma_2={}$'.format(mu, sigma2))
plt.scatter(u, Pareto_data_gauss_noise_1, marker="o", s=2, color='orangered', alpha=.25)#TODO, label=r'$icdf_{Pareto}(b=250, p=2.5)+\mu=${}, $\sigma_1={}$'.format(mu, sigma1))
plt.scatter(u, Pareto_data, marker="o", s=2, color='black', alpha=.3, label=r'$icdf_{Pareto}(b=250, p=2.5)$')
plt.legend(loc='upper right')
plt.show()

# check heteroscedastic noise data
plt.scatter(u, Pareto_data_het_noise_2, marker="o", s=1.3, color='blue', alpha=1, label=r'$\mu=${}, $h(x)={}+0.0005_t$'.format(mu, sigma1))
plt.scatter(u, Pareto_data_het_noise_1, marker="o", s=1, color='orangered', alpha=.7, label=r'$\mu=${}, $h(x)={}+0.0001_t$'.format(mu, sigma1))
plt.scatter(u, Pareto_data, marker="o", s=.5, color='black', alpha=.3, label=r'$icdf_{Pareto}(b=250, p=2.5)$')
plt.show()

# check cdfs
# plt.scatter(x[x > b], Pareto_cdf(p=p, b=b, x[x > b]), marker=".", s=.1, alpha=.3, color='r')
# plt.scatter(x_gauss_noise_1[x_gauss_noise_1 > b], Pareto_cdf(p=p, b=b, x_gauss_noise_1[x_gauss_noise_1 > b]), marker=".", s=.1, alpha=.3, color='dodgerblue')
# plt.scatter(x_gauss_noise_2[x_gauss_noise_2 > b], Pareto_cdf(p=p, b=b, x_gauss_noise_2[x_gauss_noise_2 > b]), marker=".", s=.1, alpha=.3, color='royalblue')
# plt.scatter(x_het_noise_1[x_het_noise_1 > b], Pareto_cdf(p=p, b=b, x_het_noise_1[x_het_noise_1 > b]), marker=".", s=.1, alpha=.3, color='mediumaquamarine')
# plt.scatter(x_het_noise_2[x_het_noise_2 > b], Pareto_cdf(p=p, b=b, x_het_noise_2[x_het_noise_2 > b]), marker=".", s=.1, alpha=.3, color='seagreen')
# plt.show()

""" 
--------------------------------
2. Fitting
--------------------------------
"""







""" 
--------------------------------
3. Plot Fit vs data
--------------------------------
"""



































