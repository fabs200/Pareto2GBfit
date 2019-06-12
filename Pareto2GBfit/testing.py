from scipy.stats.distributions import chi2
from .distributions import *
from .fitting import *
from prettytable import PrettyTable

# def LRtest(llmin, llmax):
#     return (2*(llmax-llmin))

class LRtest:
    """ LR test """
    def __init__(self, x, b, p, a=1, q=1, test='Pareto_vs_IB1', return_values=False, verbose=True):
        """
        :param x: data
        :param b: lower bound, fixed
        :param test: 'Pareto_vs_IB1' or 'Pareto_vs_GB1', etc.
        """
        x = np.array(x)
        x = x[x>b]

        def Pareto_ll_temp(x, b, p):
            n = len(x)
            sum = np.sum(np.log(x))
            ll = n*np.log(p) + p*n*np.log(b) - (p+1)*sum
            return ll

        def IB1_ll_temp(x, b, p, q):
            n = len(x)
            lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
            ll = p*n*np.log(b) - n*lnb + (q-1)*np.sum(np.log(1-b/x)) - (p+1)*np.sum(np.log(x))
            return ll

        def GB1_ll_temp(x, a, b, p, q):
            n = len(x)
            lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
            ll = n*np.log(abs(a)) + (a*p-1)*np.sum(np.log(x)) + (q-1)*np.sum(np.log(1-(x/b)**a)) - n*a*p*np.log(b) - n*lnb
            return ll

        def GB_ll_temp(x, a, b, c, p, q):
            n = len(x)
            sum1 = np.sum(np.log(x))
            sum2 = np.sum(np.log(1-(1-c)*(x/b)**a))
            sum3 = np.sum(np.log(1+c*((x/b)**a)))
            lnb = gammaln(p) + gammaln(q) - gammaln(p+q)
            ll = n*(np.log(np.abs(a)) - a*p*np.log(b) - lnb) + (a*p-1)*sum1 + (q-1)*sum2 - (p+q)*sum3
            return ll

        # GB tree restrictions: a, c, q = -1, 0, 1
        if test == "Pareto_vs_IB1":
            p_fit, df = p, 1
            self.w = w = 2*(IB1_ll_temp(x=x, b=b, p=p_fit, q=q_fit) - IB1_ll_temp(x=x, b=b, p=p_fit, q=1))
        # TODO: correct test hypotheses
        # if test == "Pareto_vs_GB1":
        #     p_fit, df = p, 1
        #     self.w = w = 2*(Pareto_ll_temp(x=x, b=b, p=p_fit) - GB1_ll_temp(x=x, a=-1, b=b, p=p_fit, q=1))
        # if test == "Pareto_vs_GB":
        #     p_fit, df = p, 1
        #     self.w = w = 2*(Pareto_ll_temp(x=x, b=b, p=p_fit) - GB1_ll_temp(x=x, a=-1, b=b, p=p_fit, q=1))
        # if test == "IB1_vs_GB1":
        #     p_fit, q_fit, df = p, q, 2
        #     self.w = w = 2*(IB1_ll_temp(x=x, b=b, p=p_fit, q=q_fit) - GB1_ll_temp(x=x, a=-1, b=b, p=p_fit, q=q_fit))
        # if test == "IB1_vs_GB":
        #     p_fit, q_fit, df = p, q, 2
        #     self.w = w = 2*(IB1_ll_temp(x=x, b=b, p=p_fit, q=q_fit) - GB_ll_temp(x=x, a=-1, b=b, c=0, p=p_fit, q=q_fit))
        # if test == "GB1_vs_GB":
        #     a_fit, p_fit, q_fit, df = a, p, q, 3
        #     self.w = w = 2*(GB1_ll_temp(x=x, a=a_fit, b=b, p=p_fit, q=q_fit) - GB_ll_temp(x=x, a=a_fit, b=b, c=0, p=p_fit, q=q_fit))
        pval = chi2.sf(w, df)
        if verbose:
            tbl = PrettyTable()
            tbl.field_names = ['LR test {}\t'.format(test), '']
            tbl.add_row(['chi2({}) = '.format(df), '{:.4f}'.format(w)])
            tbl.add_row(['Prob > chi2', '{:.4f}'.format(pval)])
            print(tbl)
        if return_values:
            return w, pval, df










