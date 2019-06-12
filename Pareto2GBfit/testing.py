from scipy.stats.distributions import chi2
from .distributions import *
from .fitting import *
from prettytable import PrettyTable

def LRtest(llmin, llmax):
    return (2*(llmax-llmin))


class LRtest:
    """ LR test """
    def __init__(self, x, a, b, p, q, test):
        """
        :param x:
        :param parms:
        :param b:
        :param test:
        """
        a_fix, c_fix, q_fix = -1, 0, 1
        if test == "Pareto_vs_IB1":
            p_fit = p
            # df = 1 # TODO: df
            self.w = 2*(-Pareto_ll(parms=[p_fit], x=x, b=b) + IB1_ll(parms=[p_fit, q_fix], x=x, b=b))
        if test == "Pareto_vs_GB1":
            p_fit = p
            self.w = 2*(-Pareto_ll(parms=[p_fit], x=x, b=b) + GB1_ll(parms=[a_fix, p_fit, q_fix], x=x, b=b))
        if test == "Pareto_vs_GB":
            p_fit = p
            self.w = 2*(-Pareto_ll(parms=[p_fit], x=x, b=b) + GB1_ll(parms=[a_fix, p_fit, q_fix], x=x, b=b))
        if test == "IB1_vs_GB1":
            p_fit, q_fit = p, q
            self.w = 2*(-IB1_ll(parms=[p_fit, q_fit], x=x, b=b) + GB1_ll(parms=[a_fix, p_fit, q_fit], x=x, b=b))
        if test == "GB1_vs_GB":
            a_fit, p_fit, q_fit = a, p, q
            self.w = 2*(-GB1_ll(parms=[a_fit, p_fit, q_fit], x=x, b=b) + GB_ll(parms=[a_fit, c_fix, p_fit, q_fit], x=x, b=b))
        # chi2.sf(LR, df) # TODO: chi2
        # pval # TODO: pval
        # return # TODO: return pval, chi2

# def LR_Pareto_vs_IB1(x, b, p, return_LR=False, return_pvalue=False, verbose=True):
#     """
#     LR-test
#     w=2*(Pareto_ll([p_fit], x, b) - IB1_ll([p_fit, q=1], x, b))
#     :param x: data
#     :param b: lower bound, fixed
#     :param p: fitted parameter p
#     :return:
#     """
#     # parm = [p]
#     # full_model = (-1)*Pareto_ll(parms=parm, x=x, b=b)
#     # null_model = (-1)*Pareto_ll(parms=[1], x=x, b=b)
#     # df = 1
#     # LR = LRtest(null_model, full_model)
#     # pval = chi2.sf(LR, df)
#     # if verbose:
#     #     tbl = PrettyTable()
#     #     tbl.add_row('chi2({}) = '.format(df), LR)
#     #     tbl.add_row('Prob > chi2', '{:.4f}'.format(pval))
#     # if return_LR:
#     #     return LR
#     # if return_pvalue:
#     #     return pval
