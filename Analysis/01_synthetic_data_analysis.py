from Pareto2GBfit import *

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
#TODO: het_noise_1

# 4. Large heteroscedastic noise
#TODO: het_noise_2

# linspace
xmin = 0.1
xmax = 10000
x = linspace(xmin, xmax, n)
x_gauss_noise_1 = x + gauss_noise_1
x_gauss_noise_2 = x + gauss_noise_2
#TODO: x_het_noise_1 = x + het_noise_1
# x_het_noise_2 = x + het_noise_2

# random uniform
u = np.array(np.random.uniform(.0, 1., n))
u = np.sort(u)

# Pareto simulated data (no noise)
Pareto_data = Pareto_icdf(u, b, p)
# alternatively: simulate Pareto_data numerically evaluated (ne) along pdf
# Pareto_data_ne, u_ne = Pareto_icdf_ne(x[x > b], b, p)

# Pareto simulated data + noise
Pareto_data_gauss_noise_1 = Pareto_icdf(x_gauss_noise_1[x_gauss_noise_1 > b], b, p)
Pareto_data_gauss_noise_2 = Pareto_icdf(x_gauss_noise_2[x_gauss_noise_2 > b], b, p)
#TODO: Pareto_data_het_noise_1 = Pareto_icdf(x_het_noise_1[x_het_noise_1 > b], b, p)
# Pareto_data_het_noise_2 = Pareto_icdf(x_het_noise_2[x_het_noise_2 > b], b, p)

# check data
# plt.scatter(u, Pareto_data, marker=".", s=.5, color='r')
# plt.scatter(u, Pareto_data_gauss_noise_1, marker=".", s=.5, alpha=.3, color='dodgerblue')
# plt.scatter(u, Pareto_data_gauss_noise_2, marker=".", s=.5, alpha=.3, color='royalblue')
# plt.scatter(u, Pareto_data_het_noise_1, marker=".", s=.5, alpha=.3, color='mediumaquamarine')
# plt.scatter(u, Pareto_data_het_noise_2, marker=".", s=.5, alpha=.3, color='seagreen')
# plt.show()

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



































