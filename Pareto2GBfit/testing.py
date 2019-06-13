from scipy.stats.distributions import chi2
from .distributions import *
from .fitting import *
from prettytable import PrettyTable

# def LRtest(llmin, llmax):
#     return (2*(llmax-llmin))


class Pareto:
    def __init__(self, x, b, p):
        x = np.array(x)
        x = x[x>b]
        n, self.df = len(x), 1
        sum = np.sum(np.log(x))
        self.LL = n*np.log(p) + p*n*np.log(b) - (p+1)*sum

class IB1:
    def __init__(self, x, b, p, q):
        x = np.array(x)
        x = x[x>b]
        n, self.df = len(x), 2
        lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
        self.LL = p*n*np.log(b) - n*lnb + (q-1)*np.sum(np.log(1-b/x)) - (p+1)*np.sum(np.log(x))

class GB1:
    def __init__(self, x, b, a, p, q):
        x = np.array(x)
        x = x[x>b]
        n, self.df = len(x), 3
        lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
        self.LL = n*np.log(abs(a)) + (a*p-1)*np.sum(np.log(x)) + (q-1)*np.sum(np.log(1-(x/b)**a)) - n*a*p*np.log(b) - n*lnb

class GB:
    def __init__(self, x, a, b, c, p, q):
        x = np.array(x)
        x = x[x>b]
        n, self.df = len(x), 4
        sum1 = np.sum(np.log(x))
        sum2 = np.sum(np.log(1-(1-c)*(x/b)**a))
        sum3 = np.sum(np.log(1+c*((x/b)**a)))
        lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
        self.LL = n*(np.log(np.abs(a)) - a*p*np.log(b) - lnb) + (a*p-1)*sum1 + (q-1)*sum2 - (p+q)*sum3

class LRtest:
    def __init__(self, LL1, LL2, df, verbose=True, return_parameters=False):
        self.w = w = 2*(LL1 - LL2)
        self.pval = pval = chi2.sf(w, df=df)
        tbl = PrettyTable()
        tbl.field_names = ['LR test', '']
        tbl.add_row(['chi2({}) = '.format(df), '{:.4f}'.format(w)])
        tbl.add_row(['Prob > chi2', '{:.4f}'.format(pval)])
        if verbose: print(tbl)
        if return_parameters:
            return w, pval
