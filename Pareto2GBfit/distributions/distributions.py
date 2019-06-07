from numpy import linspace, random
import numpy as np

import scipy.optimize as opt
import scipy.integrate as integrate
from scipy import linalg  # invert matrix
from scipy.misc import derivative
from scipy.special import beta, betainc, digamma, gamma, gammaln, hyp2f1

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

import pandas as pd
import progressbar
from prettytable import PrettyTable


""" Pareto """
def Pareto_pdf(x, b, p):
    return (p*b**p) / (x**(p+1))

def Pareto_cdf(x, b, p):
    if np.min(x)<b:
        return 0
    elif np.min(x)>=b:
        return 1-(b/x)**p

def Pareto_icdf(u, b, p):
    u = np.sort(u)
    if np.max(u) <= 1.0 and np.min(u) >= 0.0:
        pareto_data = b*(1-u)**(-1/p)
        if np.min(pareto_data) >= b:
            return pareto_data
        else:
            raise Exception("error - Parameter b misspecified! b={} is larger than the min. of x={}".format(b, np.min(pareto_data)))
    else:
        raise Exception("error - provide random data between 0.0 to 1.0")

def Pareto_cdf_ne(x, b, p):  # feed x with a linspace (-> range, size); Note: size needs to be sufficiently large!
    F_temp = []
    if np.min(x)>b:
        pass
    else:
        raise Exception('error: xmin should not be lower than b!. xmin={}<b={}'.format(np.min(x),b))
    # Generate cdf via numeric evaluation of pdf
    for i in x:
        F_temp.append(integrate.quad(lambda x: Pareto_pdf(x, b, p), b, i)[0])
    return F_temp

def Pareto_icdf_ne(x, b, p): # feed x with a linspace (-> range, size); Note: size needs to be sufficiently large!
    x = x[x>b]
    k = len(x)
    # Generate cdf via numeric evaluation of pdf
    F_ne = Pareto_cdf_ne(x, b, p)
    u = np.array(np.random.uniform(.0, 1., k)) # (1xn)
    u = np.sort(u)
    # Generate synthetic data via interpolation of F_ne, x_ne based on u
    def icdf(u):
        icdf_F = []
        # interpolate F
        for el in u:
            icdf_F.append(np.interp(el, F_ne, x))
        return icdf_F
    return icdf(u), u

""" IB1 """
def IB1_pdf(x, b, p, q):
    return ((b**p)*((1-(b/x))**(q-1))) / ((x**(p+1))*beta(p,q))

def IB1_cdf(x, b, p, q):
    z = (b/x)
    return 1 - betainc(p, q, z)

def IB1_icdf_ne(x, b, p, q): # Note: this gives two lists: x and the corresponding F
    x = np.array(x)
    x_temp = x[x>b]
    k = len(x_temp)
    # Generate cdf via numeric evaluating pdf
    F_ne = IB1_cdf(x_temp, b, p, q)
    u = np.array(np.random.uniform(.0, 1., k)) # (1xn)
    u = np.sort(u)
    # Generate synthetic data via interpolation of F_ne, x_ne based on u
    def icdf(u):
        icdf_F = []
        # interpolate F
        for el in u:
            icdf_F.append(np.interp(el, F_ne, x_temp))
        return icdf_F
    return icdf(u), u


""" GB1 """
def GB1_pdf(x, a, b, p, q):
    x = np.array(x)
    pdf = (np.abs(a)*(x**(a*p-1))*((1-((x/b)**a))**(q-1))) / (b**(a*p)*beta(p,q))
    return pdf

def GB1_cdf(x, a, b, p, q):
    x = np.array(x)
    z = (x/b)**a
    cdf = betainc(p, q, z)
    if a == -1:
        return 1-cdf
    if a == 1:
        return cdf

def GB1_cdf_ne(x, a, b, p, q):
    x = np.array(x)
    F = []
    widgets = ['GB_cdf_ne  ', progressbar.Percentage(), progressbar.Bar(marker='>'), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(x)).start()
    i = 0
    if a<0:
        x = x[x>b]
        for el in x:
            F.append(integrate.quad(lambda x: GB1_pdf(x, a, b, p, q), b, el)[0])
            bar.update(i)
            i += 1
    elif a>0:
        x = x[x<=b]
        for el in x:
            F.append(integrate.quad(lambda x: GB1_pdf(x, a, b, p, q), x[0], el)[0])
            bar.update(i)
            i += 1
    bar.finish()
    return F

def GB1_icdf_ne(x, a, b, p, q): # Note: this gives two lists: x and the corresponding F
    x = np.array(x)
    F_ne = GB1_cdf_ne(x, a, b, p, q)
    if a<0:
        x = x[x>b]
    elif a>0:
        x = x[x<=b]
    k = len(x) # TODO: check, right position or 4 lines above?
    u = np.array(np.random.uniform(.0, 1., k)) # (1xn)
    u = np.sort(u)
    widgets = ['GB1_icdf_ne ', progressbar.Percentage(), progressbar.Bar(marker='>'), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=k).start()
    # Generate synthetic data via interpolation of F_ne, x_ne based on u
    def icdf(u):
        icdf_F, i = [], 1
        # interpolate F
        for el in u:
            icdf_F.append(np.interp(el, F_ne, x))
            bar.update(i)
            i += 1
        return icdf_F
        bar.finish()
    return icdf(u), u


""" GB """
def GB_pdf(x, a, b, c, p, q):
    x = np.array(x) #data
    ## Pareto, IB1
    if (c==0) & (a==-1):
        # print("Pareto pdf")
        x = x[x>b]
    ## Power, Uniform, B1, GA, chi2, Exponential
    if (c==0) & (a==1):
        x = x[x<=b]
    ## UG ?? TODO
    if (c==0) & (0<a<1):
        x = x[x<=b]
    ## Normal
    if (c==0) & (a==2) & (p==0.5):
        x = x[x<=b]
    ## Rayleigh
    if (c==0) & (p==1) & (a==2):
        x = x[x<=b]
    ## ?? UG, LN, GG, W
    # TODO
    # return (np.abs(a)*(x**(a*p-1))*((1-(1-c)*((x/b)**a))**(q-1))) / ((b**(a*p))*beta(b,q)*((1+c*((x/b)**a))**(p+q)))
    # bta = (gamma(p) + gamma(q)) / (gamma(p+q))
    pdf = abs(a)*x**(a*p-1)*(1-(1-c)*(x/b)**a)**(q-1) / (b**(a*p)*beta(p,q)*(1+c*(x/b)**a)**(p+q))
    return pdf

def GB_cdf_ne(x, a, b, c, p, q):
    x = np.array(x)
    F = []
    # distributions with x>b:
    # Pareto, IB1: c==0, a==1
    if (c==0) and (a==-1):
        x = x[x>b]
        widgets = ['GB_cdf_ne  ', progressbar.Percentage(), progressbar.Bar(marker='>'), progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(x)).start()
        for idx, i in enumerate(x):
            F.append(integrate.quad(lambda x: GB_pdf(x, a, b, c, p, q), b, i)[0])
            bar.update(idx+1)
    # distributions with x<=b:
    # Power, Uniform, B1, GA, chi2, Exponential: c==0, a==1
    # UG: c==0, 0<a<1; Half Normal: c==0, a==2, p==.5; Rayleigh: c==0,p==1, a==2
    if (c==0) and ((a==1) or (0<a<1) or ((a==2) and (p==.5)) or ((a==2) and (p==1))):
        x = x[x<=b]
        widgets = ['GB_cdf_ne  ', progressbar.Percentage(), progressbar.Bar(marker='>'), progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(x)).start()
        for idx, i in enumerate(x):
            F.append(integrate.quad(lambda x: GB_pdf(x, a, b, c, p, q), 0, i)[0])
            bar.update(idx)
    return F

def GB_icdf_ne(x, a, b, c, p, q): # Note: this gives two lists: x and the corresponding F
    x = np.array(x)
    # distributions with x>b:
    # Pareto, IB1: c==0, a==1
    if (c==0) & (a==-1):
        x = x[x>b]
    # distributions with x<=b:
    # Power, Uniform, B1, GA, chi2, Exponential: c==0, a==1
    # UG: c==0, 0<a<1; Half Normal: c==0, a==2, p==.5; Rayleigh: c==0,p==1, a==2
    if (c==0) and ((a==1) or (0<a<1) or ((a==2) and (a==.5)) or ((a==2) and (p==1))):
        x = x[x<=b]
    widgets = ['GB_icdf_ne ', progressbar.Percentage(), progressbar.Bar(marker='>'), progressbar.ETA()]
    bar = progressbar.ProgressBar(widgets=widgets, maxval=len(x)).start()
    # Generate cdf via numeric evaluating pdf
    F_ne = GB_cdf_ne(x, a, b, c, p, q)
    u = np.array(np.random.uniform(.0, 1., len(x))) # (1xn)
    u = np.sort(u)
    # Generate synthetic data via interpolation of F_ne, x_ne based on u
    def icdf(u):
        icdf_F = []
        # interpolate F
        for idx, i in enumerate(u):
            icdf_F.append(np.interp(i, F_ne, x))
            bar.update(idx+1)
        return icdf_F
    return icdf(u), u

