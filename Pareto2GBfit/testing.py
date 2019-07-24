from scipy.stats.distributions import chi2
from .distributions import *
from prettytable import PrettyTable

class Pareto:
    def __init__(self, x, W, b, p):
        x = np.array(x)
        x = x[x>b]
        n, self.df = np.sum(W), 1
        sum = np.sum(np.log(x)*W)
        self.LL = n*np.log(p) + p*n*np.log(b) - (p+1)*sum

class IB1:
    def __init__(self, x, W, b, p, q):
        x = np.array(x)
        x = x[x>b]
        n, self.df = np.sum(W), 2
        lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
        sum1 = np.sum(np.log(1-b/x)*W)
        sum2 = np.sum(np.log(x)*W)
        self.LL = p*n*np.log(b) - n*lnb + (q-1)*sum1 - (p+1)*sum2

class GB1:
    def __init__(self, x, W, b, a, p, q):
        x = np.array(x)
        x = x[x>b]
        n, self.df = np.sum(W), 3
        lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
        sum1 = np.sum(np.log(x)*W)
        sum2 = np.sum(np.log(1-(x/b)**a)*W)
        self.LL = n*np.log(abs(a)) + (a*p-1)*sum1 + (q-1)*sum2 - n*a*p*np.log(b) - n*lnb

class GB:
    def __init__(self, x, W, a, b, c, p, q):
        x = np.array(x)
        x = x[x>b]
        n, self.df = np.sum(W), 4
        sum1 = np.sum(np.log(x)*W)
        sum2 = np.sum(np.log(1-(1-c)*(x/b)**a)*W)
        sum3 = np.sum(np.log(1+c*((x/b)**a))*W)
        lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
        self.LL = n*(np.log(np.abs(a)) - a*p*np.log(b) - lnb) + (a*p-1)*sum1 + (q-1)*sum2 - (p+q)*sum3

class LRtest:
    def __init__(self, LL1, LL2, df, verbose=True):
        """
        :param LL1: log-likelihood with H0
        :param LL2: log-likelihood with H1/fitted parameters
        :param df: specify dfs, # of tested params
        :param verbose: display results in table
        """
        self.LR = LR = 2*(LL2- LL1)
        self.pval = pval = chi2.sf(LR, df=df)
        tbl = PrettyTable()
        tbl.field_names = ['LR test', '']
        tbl.add_row(['chi2({}) = '.format(df), '{:.4f}'.format(LR)])
        tbl.add_row(['Prob > chi2', '{:.4f}'.format(pval)])
        if verbose: print(tbl)
