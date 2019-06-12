from scipy.stats.distributions import chi2
from .distributions import *
from .fitting import *
from prettytable import PrettyTable

def LRtest(llmin, llmax):
    return (2*(llmax-llmin))

class LRtest:
    """ LR test """
    def __init__(self, x, a, b, p, q, test, return_values=False, verbose=True):
        """
        :param x:
        :param parms:
        :param b:
        :param test:
        """
        a_fix, c_fix, q_fix = -1, 0, 1
        if test == "Pareto_vs_IB1":
            p_fit = p
            df = 1
            self.w = w = 2*(-Pareto_ll(parms=[p_fit], x=x, b=b) + IB1_ll(parms=[p_fit, q_fix], x=x, b=b))
        if test == "Pareto_vs_GB1":
            p_fit = p
            df = 1
            self.w = w = 2*(-Pareto_ll(parms=[p_fit], x=x, b=b) + GB1_ll(parms=[a_fix, p_fit, q_fix], x=x, b=b))
        if test == "Pareto_vs_GB":
            p_fit = p
            df = 1
            self.w = w = 2*(-Pareto_ll(parms=[p_fit], x=x, b=b) + GB1_ll(parms=[a_fix, p_fit, q_fix], x=x, b=b))
        if test == "IB1_vs_GB1":
            p_fit, q_fit = p, q
            df = 2
            self.w = w = 2*(-IB1_ll(parms=[p_fit, q_fit], x=x, b=b) + GB1_ll(parms=[a_fix, p_fit, q_fit], x=x, b=b))
        if test == "GB1_vs_GB":
            a_fit, p_fit, q_fit = a, p, q
            df = 3
            self.w = w = 2*(-GB1_ll(parms=[a_fit, p_fit, q_fit], x=x, b=b) + GB_ll(parms=[a_fit, c_fix, p_fit, q_fit], x=x, b=b))
        pval = chi2.sf(w, df)
        if verbose:
            tbl = PrettyTable()
            tbl.add_row('chi2({}) = '.format(df), '{:.4f}'.format(w))
            tbl.add_row('Prob > chi2', '{:.4f}'.format(pval))
        if return_values:
            return w, pval, df

