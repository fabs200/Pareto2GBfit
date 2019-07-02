from Pareto2GBfit.fitting import *
import numpy as np
import pandas as pd
from pandas import ExcelWriter
import os

# diw path
if os.getlogin() == "fnemeczek":
    descriptivespath = 'H:/Meine Dateien/Masterarbeit/Python/descriptives'
    data_PSID = 'H:/Meine Dateien/Masterarbeit/J261520/J261520.csv'
    data_SOEP = 'H:/Meine Dateien/Masterarbeit/DATA/SOEP/'

# windows paths
if os.getlogin() == 'Fabian' and os.name == 'nt':
    descriptivespath = 'D:/OneDrive/Studium/Masterarbeit/Python/descriptives/'
    data_PSID = 'D:/OneDrive/Studium/Masterarbeit/data/J261520/'
    data_SOEP = 'C:/Users/fabia/Documents/DATA/SOEP_v34/stata_de+en/'

# mac paths
if os.getlogin() == 'Fabian' and os.name == 'POSIX':
    descriptivespath = '/Users/Fabian/OneDrive/Studium/Masterarbeit/Python/descriptives/'
    data_PSID = "/Users/Fabian/OneDrive/Studium/Masterarbeit/data/J261520/"
    data_SOEP = '/Users/Fabian/Documents/DATA/STATA/SOEP_v34/stata_de+en/'

"""
-------------------------
define functions
-------------------------
"""
def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    source: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    NOTE: quantiles should be in [0, 1]
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def prep_fit_results_for_table(fit_result):
    """
    prepares the returned vector of the optimization for a simplified exporting to dataframe/Excel
    :param fit_result: result of Paretobranchfit, needs return_bestmodel=True
    :return: returns vector with same shape, doesn't matter which model is best
    """
    bestfit, fit_result, placeholder, list = fit_result[0], np.array(fit_result[1]).tolist(), ['--', '--'], []
    for el in fit_result[:-2]:
        list.append('{:.3f}'.format(el))
    for el in fit_result[-2:]:
        list.append('{:.3f}'.format(int(el)))
    if bestfit == "Pareto_best" or len(list) == 16:
        out = placeholder * 2 #a,c
        out = out + list[0:2] + placeholder #p,q
        out = out + list[2:] #q, rest
    if bestfit == "IB1_best" or len(list) == 18:
        out = placeholder * 2 #a,c
        out = out + list #p,q, rest
    if bestfit == "GB1_best" or len(list) == 20:
        out = list[0:2] + placeholder #c
        out = out + list[2:] #rest
    if bestfit == "GB_best" or len(list) == 22:
        out = list
    del out[15] # remove soe
    del out[13] # remove rrmse
    del out[10] # remove mae
    del out[9] # remove bic
    return out # returns: parameters, aic, mse, rrmse, ll, ...

"""
------------------------
PSID data preparation
------------------------
"""

# load dataset PSID
dfPSID = pd.read_csv(data_PSID + 'J261520.csv', delimiter=";", skiprows=False, decimal=',')

# renaming
columns={"ER17001": "release_01", "ER17002": "famid_01", "ER20394": "weight1_01", "ER20459": "weight2_01",
           "S500": "wrelease_01", "S516": "wealth1_01", "S517": "wealth2_01", "S516A": "wealthA1_01", "S517A": "wealthA2_01",

           "ER21001": "release_03", "ER21002": "famid_03", "ER24179": "weight1_03", "ER24180": "weight2_03",
           "S600": "wrelease_03", "S616": "wealth1_03", "S617": "wealth2_03", "S616A": "wealthA1_03", "S617A": "wealthA2_03",

           "ER25001": "release_05", "ER25002": "famid_05", "ER28078": "weight1_05",
           "S700": "wrelease_05", "S716": "wealth1_05", "S717": "wealth2_05", "S716A": "wealthA1_05", "S717A": "wealthA2_05",

           "ER36001": "release_07", "ER36002": "famid_07", "ER41069": "weight1_07",
           "S800": "wrelease_07", "S816": "wealth1_07", "S817": "wealth2_07", "S816A": "wealthA1_07", "S817A": "wealthA2_07",

           "ER42001": "release_09", "ER42002": "famid_09", "ER47012": "weight1_09",
           "ER46968": "wealth1_09", "ER46970": "wealth2_09", "ER46969": "wealthA1_09", "ER46971": "wealthA2_09",

           "ER47301": "release_11", "ER47302": "famid_11", "ER52436": "weight1_11",
           "ER52392": "wealth1_11", "ER52394": "wealth2_11", "ER52393": "wealthA1_11", "ER52395": "wealthA2_11",

           "ER53001": "release_13", "ER53002": "famid_13", "ER58257": "weight1_13",
           "ER58209": "wealth1_13", "ER58211": "wealth2_13", "ER58210": "wealthA1_13", "ER58212": "wealthA2_13",

           "ER60001": "release_15", "ER60002": "famid_15", "ER65492": "weight1_15",
           "ER65406": "wealth1_15", "ER65408": "wealth2_15", "ER65407": "wealthA1_15", "ER65409": "wealthA2_15",

           "ER66001": "release_17", "ER66002": "famid_17", "ER71570": "weight1_17",
           "ER71483": "wealth1_17", "ER71485": "wealth2_17", "ER71484": "wealthA1_17", "ER71486": "wealthA2_17"}

dfPSID = dfPSID.rename(index = str, columns=columns)

# multiply longitudinal weights by 1000 as described in the documentation of PSID
psid_longi_weights = ['weight1_01', 'weight1_03', 'weight1_05', 'weight1_07', 'weight1_09', 'weight1_11', 'weight1_13', 'weight1_15', 'weight1_17']
for weight in psid_longi_weights:
    dfPSID[weight] = dfPSID[weight].apply(lambda x: x*1000) # compare https://psidonline.isr.umich.edu/data/weights/cross_sec_weights_13.pdf
    # print(weight[-2:], dfPSID[weight].sum()) #TODO: correct family N?

# check
# dfPSID.head()

"""
-----------------------
SOEP data preparation
-----------------------
"""

data_SOEPwealth = data_SOEP + 'hwealth2.dta'
data_SOEPHHweight = data_SOEP + 'raw/hhrf.dta'

# read in data
dfSOEP_wealth = pd.read_stata(data_SOEPwealth, columns=['syear', 'hid', 'w011ha', 'w011hb', 'w011hc', 'w011hd', 'w011he'])
dfSOEP_hhweights = pd.read_stata(data_SOEPHHweight, columns=['hid', 'shhrf', 'xhhrf', 'bchhrf', 'bhhhrf'])

# mean imputed wealths
dfSOEP_wealth['wealth'] = dfSOEP_wealth[['w011ha', 'w011hb', 'w011hc', 'w011hd', 'w011he']].mean(axis=1)

# reshape long to wide
dfSOEP_wealth = dfSOEP_wealth.pivot(index='hid', columns='syear', values='wealth')

# merge datasets
dfSOEP = dfSOEP_wealth.merge(dfSOEP_hhweights, left_on='hid', right_on='hid')

# rename weights
dfSOEP = dfSOEP.rename(index=str, columns={2002: 'wealth_02', 2007: 'wealth_07', 2012: 'wealth_12', 2017: 'wealth_17',
                                           'shhrf': 'weight_02', 'xhhrf': 'weight_07', 'bchhrf': 'weight_12', 'bhhhrf': 'weight_17'})

# check
# dfSOEP_wealth.head()
# dfSOEP_hhweights.head()
# dfSOEP.head()

"""
-----------------------
Descriptive Statistics
-----------------------
"""

## unweighted

# unweighted, psid: wealth1, soep: wealth
df_unweighted_descriptives_w1 = pd.DataFrame(np.array([['N', dfPSID['wealth1_01'].count(), dfPSID['wealth1_03'].count(), dfPSID['wealth1_05'].count(), dfPSID['wealth1_07'].count(), dfPSID['wealth1_09'].count(), dfPSID['wealth1_11'].count(), dfPSID['wealth1_13'].count(), dfPSID['wealth1_15'].count(), dfPSID['wealth1_17'].count(), dfSOEP['wealth_02'].count(), dfSOEP['wealth_07'].count(), dfSOEP['wealth_12'].count(), dfSOEP['wealth_17'].count()],
                                                       ['mean', int(dfPSID['wealth1_01'].mean()), int(dfPSID['wealth1_03'].mean()), int(dfPSID['wealth1_05'].mean()), int(dfPSID['wealth1_07'].mean()), int(dfPSID['wealth1_09'].mean()), int(dfPSID['wealth1_11'].mean()), int(dfPSID['wealth1_13'].mean()), int(dfPSID['wealth1_15'].mean()), int(dfPSID['wealth1_17'].mean()), int(dfSOEP['wealth_02'].mean()), int(dfSOEP['wealth_07'].mean()), int(dfSOEP['wealth_12'].mean()), int(dfSOEP['wealth_17'].mean())],
                                                       ['sd', int(dfPSID['wealth1_01'].std()), int(dfPSID['wealth1_03'].std()), int(dfPSID['wealth1_05'].std()), int(dfPSID['wealth1_07'].std()), int(dfPSID['wealth1_09'].std()), int(dfPSID['wealth1_11'].std()), int(dfPSID['wealth1_13'].std()), int(dfPSID['wealth1_15'].std()), int(dfPSID['wealth1_17'].std()), int(dfSOEP['wealth_02'].std()), int(dfSOEP['wealth_07'].std()), int(dfSOEP['wealth_12'].std()), int(dfSOEP['wealth_17'].std())],
                                                       ['min', int(dfPSID['wealth1_01'].min()), int(dfPSID['wealth1_03'].min()), int(dfPSID['wealth1_05'].min()), int(dfPSID['wealth1_07'].min()), int(dfPSID['wealth1_09'].min()), int(dfPSID['wealth1_11'].min()), int(dfPSID['wealth1_13'].min()), int(dfPSID['wealth1_15'].min()), int(dfPSID['wealth1_17'].min()), int(dfSOEP['wealth_02'].min()), int(dfSOEP['wealth_07'].min()), int(dfSOEP['wealth_12'].min()), int(dfSOEP['wealth_17'].min())],
                                                       ['p50', int(dfPSID['wealth1_01'].quantile()), int(dfPSID['wealth1_03'].quantile()), int(dfPSID['wealth1_05'].quantile()), int(dfPSID['wealth1_07'].quantile()), int(dfPSID['wealth1_09'].quantile()), int(dfPSID['wealth1_11'].quantile()), int(dfPSID['wealth1_13'].quantile()), int(dfPSID['wealth1_15'].quantile()), int(dfPSID['wealth1_17'].quantile()), int(dfSOEP['wealth_02'].quantile()), int(dfSOEP['wealth_07'].quantile()), int(dfSOEP['wealth_12'].quantile()), int(dfSOEP['wealth_17'].quantile())],
                                                       ['p75', int(dfPSID['wealth1_01'].quantile(.75)), int(dfPSID['wealth1_03'].quantile(.75)), int(dfPSID['wealth1_05'].quantile(.75)), int(dfPSID['wealth1_07'].quantile(.75)), int(dfPSID['wealth1_09'].quantile(.75)), int(dfPSID['wealth1_11'].quantile(.75)), int(dfPSID['wealth1_13'].quantile(.75)), int(dfPSID['wealth1_15'].quantile(.75)), int(dfPSID['wealth1_17'].quantile(.75)), int(dfSOEP['wealth_02'].quantile(.75)), int(dfSOEP['wealth_07'].quantile(.75)), int(dfSOEP['wealth_12'].quantile(.75)), int(dfSOEP['wealth_17'].quantile(.75))],
                                                       ['p90', int(dfPSID['wealth1_01'].quantile(.9)), int(dfPSID['wealth1_03'].quantile(.9)), int(dfPSID['wealth1_05'].quantile(.9)), int(dfPSID['wealth1_07'].quantile(.9)), int(dfPSID['wealth1_09'].quantile(.9)), int(dfPSID['wealth1_11'].quantile(.9)), int(dfPSID['wealth1_13'].quantile(.9)), int(dfPSID['wealth1_15'].quantile(.9)), int(dfPSID['wealth1_17'].quantile(.9)), int(dfSOEP['wealth_02'].quantile(.9)), int(dfSOEP['wealth_07'].quantile(.9)), int(dfSOEP['wealth_12'].quantile(.9)), int(dfSOEP['wealth_17'].quantile(.9))],
                                                       ['p99', int(dfPSID['wealth1_01'].quantile(.99)), int(dfPSID['wealth1_03'].quantile(.99)), int(dfPSID['wealth1_05'].quantile(.99)), int(dfPSID['wealth1_07'].quantile(.99)), int(dfPSID['wealth1_09'].quantile(.99)), int(dfPSID['wealth1_11'].quantile(.99)), int(dfPSID['wealth1_13'].quantile(.99)), int(dfPSID['wealth1_15'].quantile(.99)), int(dfPSID['wealth1_17'].quantile(.99)), int(dfSOEP['wealth_02'].quantile(.99)), int(dfSOEP['wealth_07'].quantile(.99)), int(dfSOEP['wealth_12'].quantile(.99)), int(dfSOEP['wealth_17'].quantile(.99))],
                                                       ['p99.9', int(dfPSID['wealth1_01'].quantile(.999)), int(dfPSID['wealth1_03'].quantile(.999)), int(dfPSID['wealth1_05'].quantile(.999)), int(dfPSID['wealth1_07'].quantile(.999)), int(dfPSID['wealth1_09'].quantile(.999)), int(dfPSID['wealth1_11'].quantile(.999)), int(dfPSID['wealth1_13'].quantile(.999)), int(dfPSID['wealth1_15'].quantile(.999)), int(dfPSID['wealth1_17'].quantile(.999)), int(dfSOEP['wealth_02'].quantile(.999)), int(dfSOEP['wealth_07'].quantile(.999)), int(dfSOEP['wealth_12'].quantile(.999)), int(dfSOEP['wealth_17'].quantile(.999))],
                                                       ['max', int(dfPSID['wealth1_01'].max()), int(dfPSID['wealth1_03'].max()), int(dfPSID['wealth1_05'].max()), int(dfPSID['wealth1_07'].max()), int(dfPSID['wealth1_09'].max()), int(dfPSID['wealth1_11'].max()), int(dfPSID['wealth1_13'].max()), int(dfPSID['wealth1_15'].max()), int(dfPSID['wealth1_17'].max()), int(dfSOEP['wealth_02'].max()), int(dfSOEP['wealth_07'].max()), int(dfSOEP['wealth_12'].max()), int(dfSOEP['wealth_17'].max())],
                                                       ]),
                                             columns=['', '2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015', '2017', '2002SOEP', '2007SOEP', '2012SOEP', '2017SOEP'])


# unweighted, psid: wealth2, soep: wealth
df_unweighted_descriptives_w2 = pd.DataFrame(np.array([['N', dfPSID['wealth2_01'].count(), dfPSID['wealth2_03'].count(), dfPSID['wealth2_05'].count(), dfPSID['wealth2_07'].count(), dfPSID['wealth2_09'].count(), dfPSID['wealth2_11'].count(), dfPSID['wealth2_13'].count(), dfPSID['wealth2_15'].count(), dfPSID['wealth2_17'].count(), dfSOEP['wealth_02'].count(), dfSOEP['wealth_07'].count(), dfSOEP['wealth_12'].count(), dfSOEP['wealth_17'].count()],
                                                       ['mean', int(dfPSID['wealth2_01'].mean()), int(dfPSID['wealth2_03'].mean()), int(dfPSID['wealth2_05'].mean()), int(dfPSID['wealth2_07'].mean()), int(dfPSID['wealth2_09'].mean()), int(dfPSID['wealth2_11'].mean()), int(dfPSID['wealth2_13'].mean()), int(dfPSID['wealth2_15'].mean()), int(dfPSID['wealth2_17'].mean()), int(dfSOEP['wealth_02'].mean()), int(dfSOEP['wealth_07'].mean()), int(dfSOEP['wealth_12'].mean()), int(dfSOEP['wealth_17'].mean())],
                                                       ['sd', int(dfPSID['wealth2_01'].std()), int(dfPSID['wealth2_03'].std()), int(dfPSID['wealth2_05'].std()), int(dfPSID['wealth2_07'].std()), int(dfPSID['wealth2_09'].std()), int(dfPSID['wealth2_11'].std()), int(dfPSID['wealth2_13'].std()), int(dfPSID['wealth2_15'].std()), int(dfPSID['wealth2_17'].std()), int(dfSOEP['wealth_02'].std()), int(dfSOEP['wealth_07'].std()), int(dfSOEP['wealth_12'].std()), int(dfSOEP['wealth_17'].std())],
                                                       ['min', int(dfPSID['wealth2_01'].min()), int(dfPSID['wealth2_03'].min()), int(dfPSID['wealth2_05'].min()), int(dfPSID['wealth2_07'].min()), int(dfPSID['wealth2_09'].min()), int(dfPSID['wealth2_11'].min()), int(dfPSID['wealth2_13'].min()), int(dfPSID['wealth2_15'].min()), int(dfPSID['wealth2_17'].min()), int(dfSOEP['wealth_02'].min()), int(dfSOEP['wealth_07'].min()), int(dfSOEP['wealth_12'].min()), int(dfSOEP['wealth_17'].min())],
                                                       ['p50', int(dfPSID['wealth2_01'].quantile()), int(dfPSID['wealth2_03'].quantile()), int(dfPSID['wealth2_05'].quantile()), int(dfPSID['wealth2_07'].quantile()), int(dfPSID['wealth2_09'].quantile()), int(dfPSID['wealth2_11'].quantile()), int(dfPSID['wealth2_13'].quantile()), int(dfPSID['wealth2_15'].quantile()), int(dfPSID['wealth2_17'].quantile()), int(dfSOEP['wealth_02'].quantile()), int(dfSOEP['wealth_07'].quantile()), int(dfSOEP['wealth_12'].quantile()), int(dfSOEP['wealth_17'].quantile())],
                                                       ['p75', int(dfPSID['wealth2_01'].quantile(.75)), int(dfPSID['wealth2_03'].quantile(.75)), int(dfPSID['wealth2_05'].quantile(.75)), int(dfPSID['wealth2_07'].quantile(.75)), int(dfPSID['wealth2_09'].quantile(.75)), int(dfPSID['wealth2_11'].quantile(.75)), int(dfPSID['wealth2_13'].quantile(.75)), int(dfPSID['wealth2_15'].quantile(.75)), int(dfPSID['wealth2_17'].quantile(.75)), int(dfSOEP['wealth_02'].quantile(.75)), int(dfSOEP['wealth_07'].quantile(.75)), int(dfSOEP['wealth_12'].quantile(.75)), int(dfSOEP['wealth_17'].quantile(.75))],
                                                       ['p90', int(dfPSID['wealth2_01'].quantile(.9)), int(dfPSID['wealth2_03'].quantile(.9)), int(dfPSID['wealth2_05'].quantile(.9)), int(dfPSID['wealth2_07'].quantile(.9)), int(dfPSID['wealth2_09'].quantile(.9)), int(dfPSID['wealth2_11'].quantile(.9)), int(dfPSID['wealth2_13'].quantile(.9)), int(dfPSID['wealth2_15'].quantile(.9)), int(dfPSID['wealth2_17'].quantile(.9)), int(dfSOEP['wealth_02'].quantile(.9)), int(dfSOEP['wealth_07'].quantile(.9)), int(dfSOEP['wealth_12'].quantile(.9)), int(dfSOEP['wealth_17'].quantile(.9))],
                                                       ['p99', int(dfPSID['wealth2_01'].quantile(.99)), int(dfPSID['wealth2_03'].quantile(.99)), int(dfPSID['wealth2_05'].quantile(.99)), int(dfPSID['wealth2_07'].quantile(.99)), int(dfPSID['wealth2_09'].quantile(.99)), int(dfPSID['wealth2_11'].quantile(.99)), int(dfPSID['wealth2_13'].quantile(.99)), int(dfPSID['wealth2_15'].quantile(.99)), int(dfPSID['wealth2_17'].quantile(.99)), int(dfSOEP['wealth_02'].quantile(.99)), int(dfSOEP['wealth_07'].quantile(.99)), int(dfSOEP['wealth_12'].quantile(.99)), int(dfSOEP['wealth_17'].quantile(.99))],
                                                       ['p99.9', int(dfPSID['wealth2_01'].quantile(.999)), int(dfPSID['wealth2_03'].quantile(.999)), int(dfPSID['wealth2_05'].quantile(.999)), int(dfPSID['wealth2_07'].quantile(.999)), int(dfPSID['wealth2_09'].quantile(.999)), int(dfPSID['wealth2_11'].quantile(.999)), int(dfPSID['wealth2_13'].quantile(.999)), int(dfPSID['wealth2_15'].quantile(.999)), int(dfPSID['wealth2_17'].quantile(.999)), int(dfSOEP['wealth_02'].quantile(.999)), int(dfSOEP['wealth_07'].quantile(.999)), int(dfSOEP['wealth_12'].quantile(.999)), int(dfSOEP['wealth_17'].quantile(.999))],
                                                       ['max', int(dfPSID['wealth2_01'].max()), int(dfPSID['wealth2_03'].max()), int(dfPSID['wealth2_05'].max()), int(dfPSID['wealth2_07'].max()), int(dfPSID['wealth2_09'].max()), int(dfPSID['wealth2_11'].max()), int(dfPSID['wealth2_13'].max()), int(dfPSID['wealth2_15'].max()), int(dfPSID['wealth2_17'].max()), int(dfSOEP['wealth_02'].max()), int(dfSOEP['wealth_07'].max()), int(dfSOEP['wealth_12'].max()), int(dfSOEP['wealth_17'].max())],
                                                       ]),
                                             columns=['', '2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015', '2017', '2002SOEP', '2007SOEP', '2012SOEP', '2017SOEP'])

## weighted

# weighted, soep: wealth
for year in ['02', '07', '12', '17']:
    # weighted pcts
    globals()['w_wgt_pcts_soep_{}'.format(year)] = weighted_quantile(dfSOEP['wealth_{}'.format(year)][~np.isnan(dfSOEP['wealth_{}'.format(year)])], quantiles= [.5, .75, .9, .99, .999], sample_weight=dfSOEP['weight_{}'.format(year)][~np.isnan(dfSOEP['wealth_{}'.format(year)])])
    # weighted sd
    globals()['w_wgt_sd_soep_{}'.format(year)] = np.sqrt(np.cov(dfSOEP['wealth_{}'.format(year)][~np.isnan(dfSOEP['wealth_{}'.format(year)])], aweights=dfSOEP['weight_{}'.format(year)][~np.isnan(dfSOEP['wealth_{}'.format(year)])]))
    # # weighted avg
    globals()['w_wgt_mean_soep_{}'.format(year)] = np.average(dfSOEP['wealth_{}'.format(year)][~np.isnan(dfSOEP['wealth_{}'.format(year)])], weights=dfSOEP['weight_{}'.format(year)][~np.isnan(dfSOEP['wealth_{}'.format(year)])])


# weighted, psid: wealth1
for year in ['01', '03', '05', '07', '09', '11', '13', '15', '17']:
    # weighted pcts
    globals()['w1_wgt_pcts_psid_{}'.format(year)] = weighted_quantile(dfPSID['wealth1_{}'.format(year)][~np.isnan(dfPSID['wealth1_{}'.format(year)])], quantiles= [.5, .75, .9, .99, .999], sample_weight=dfPSID['weight1_{}'.format(year)][~np.isnan(dfPSID['weight1_{}'.format(year)])])
    # weighted sd
    globals()['w1_wgt_sd_psid_{}'.format(year)] = np.sqrt(np.cov(dfPSID['wealth1_{}'.format(year)][~np.isnan(dfPSID['wealth1_{}'.format(year)])], aweights=dfPSID['weight1_{}'.format(year)][~np.isnan(dfPSID['weight1_{}'.format(year)])]))
    # weighted avg
    globals()['w1_wgt_mean_psid_{}'.format(year)] = np.average(dfPSID['wealth1_{}'.format(year)][~np.isnan(dfPSID['wealth1_{}'.format(year)])], weights=dfPSID['weight1_{}'.format(year)][~np.isnan(dfPSID['weight1_{}'.format(year)])])

# weighted, psid: wealth2
for year in ['01', '03', '05', '07', '09', '11', '13', '15', '17']:
    # weighted pcts
    globals()['w2_wgt_pcts_psid_{}'.format(year)] = weighted_quantile(dfPSID['wealth2_{}'.format(year)][~np.isnan(dfPSID['wealth2_{}'.format(year)])], quantiles= [.5, .75, .9, .99, .999], sample_weight=dfPSID['weight1_{}'.format(year)][~np.isnan(dfPSID['weight1_{}'.format(year)])])
    # weighted sd
    globals()['w2_wgt_sd_psid_{}'.format(year)] = np.sqrt(np.cov(dfPSID['wealth2_{}'.format(year)][~np.isnan(dfPSID['wealth2_{}'.format(year)])], aweights=dfPSID['weight1_{}'.format(year)][~np.isnan(dfPSID['weight1_{}'.format(year)])]))
    # weighted avg
    globals()['w2_wgt_mean_psid_{}'.format(year)] = np.average(dfPSID['wealth2_{}'.format(year)][~np.isnan(dfPSID['wealth2_{}'.format(year)])], weights=dfPSID['weight1_{}'.format(year)][~np.isnan(dfPSID['weight1_{}'.format(year)])])

# write statistics to dataframe, psid: wealth1, soep: wealth
df_weighted_descriptives_w1 = pd.DataFrame(np.array([['N', int(dfPSID['weight1_01'].sum()), int(dfPSID['weight1_03'].sum()), int(dfPSID['weight1_05'].sum()), int(dfPSID['weight1_07'].sum()), int(dfPSID['weight1_09'].sum()), int(dfPSID['weight1_11'].sum()), int(dfPSID['weight1_13'].sum()), int(dfPSID['weight1_15'].sum()), int(dfPSID['weight1_17'].sum()), int(dfSOEP['weight_02'].sum()), int(dfSOEP['weight_07'].sum()), int(dfSOEP['weight_12'].sum()), int(dfSOEP['weight_17'].sum())],
                                                       ['mean', int(w1_wgt_mean_psid_01), int(w1_wgt_mean_psid_03), int(w1_wgt_mean_psid_05), int(w1_wgt_mean_psid_07), int(w1_wgt_mean_psid_09), int(w1_wgt_mean_psid_11), int(w1_wgt_mean_psid_13), int(w1_wgt_mean_psid_15), int(w1_wgt_mean_psid_17), int(w_wgt_mean_soep_02), int(w_wgt_mean_soep_07), int(w_wgt_mean_soep_12), int(w_wgt_mean_soep_17)],
                                                       ['sd', int(w1_wgt_sd_psid_01), int(w1_wgt_sd_psid_03), int(w1_wgt_sd_psid_05), int(w1_wgt_sd_psid_07), int(w1_wgt_sd_psid_09), int(w1_wgt_sd_psid_11), int(w1_wgt_sd_psid_13), int(w1_wgt_sd_psid_15), int(w1_wgt_sd_psid_17), int(w_wgt_sd_soep_02), int(w_wgt_sd_soep_07), int(w_wgt_sd_soep_12), int(w_wgt_sd_soep_17)],
                                                       ['min', int(dfPSID['wealth1_01'].min()), int(dfPSID['wealth1_03'].min()), int(dfPSID['wealth1_05'].min()), int(dfPSID['wealth1_07'].min()), int(dfPSID['wealth1_09'].min()), int(dfPSID['wealth1_11'].min()), int(dfPSID['wealth1_13'].min()), int(dfPSID['wealth1_15'].min()), int(dfPSID['wealth1_17'].min()), int(dfSOEP['wealth_02'].min()), int(dfSOEP['wealth_07'].min()), int(dfSOEP['wealth_12'].min()), int(dfSOEP['wealth_17'].min())],
                                                       ['p50', int(w1_wgt_pcts_psid_01[0]), int(w1_wgt_pcts_psid_03[0]), int(w1_wgt_pcts_psid_05[0]), int(w1_wgt_pcts_psid_07[0]), int(w1_wgt_pcts_psid_09[0]), int(w1_wgt_pcts_psid_11[0]), int(w1_wgt_pcts_psid_13[0]), int(w1_wgt_pcts_psid_15[0]), int(w1_wgt_pcts_psid_17[0]), int(w_wgt_pcts_soep_02[0]), int(w_wgt_pcts_soep_07[0]), int(w_wgt_pcts_soep_12[0]), int(w_wgt_pcts_soep_17[0])],
                                                       ['p75', int(w1_wgt_pcts_psid_01[1]), int(w1_wgt_pcts_psid_03[1]), int(w1_wgt_pcts_psid_05[1]), int(w1_wgt_pcts_psid_07[1]), int(w1_wgt_pcts_psid_09[1]), int(w1_wgt_pcts_psid_11[1]), int(w1_wgt_pcts_psid_13[1]), int(w1_wgt_pcts_psid_15[1]), int(w1_wgt_pcts_psid_17[1]), int(w_wgt_pcts_soep_02[1]), int(w_wgt_pcts_soep_07[1]), int(w_wgt_pcts_soep_12[1]), int(w_wgt_pcts_soep_17[1])],
                                                       ['p90', int(w1_wgt_pcts_psid_01[2]), int(w1_wgt_pcts_psid_03[2]), int(w1_wgt_pcts_psid_05[2]), int(w1_wgt_pcts_psid_07[2]), int(w1_wgt_pcts_psid_09[2]), int(w1_wgt_pcts_psid_11[2]), int(w1_wgt_pcts_psid_13[2]), int(w1_wgt_pcts_psid_15[2]), int(w1_wgt_pcts_psid_17[2]), int(w_wgt_pcts_soep_02[2]), int(w_wgt_pcts_soep_07[2]), int(w_wgt_pcts_soep_12[2]), int(w_wgt_pcts_soep_17[2])],
                                                       ['p99', int(w1_wgt_pcts_psid_01[3]), int(w1_wgt_pcts_psid_03[3]), int(w1_wgt_pcts_psid_05[3]), int(w1_wgt_pcts_psid_07[3]), int(w1_wgt_pcts_psid_09[3]), int(w1_wgt_pcts_psid_11[3]), int(w1_wgt_pcts_psid_13[3]), int(w1_wgt_pcts_psid_15[3]), int(w1_wgt_pcts_psid_17[3]), int(w_wgt_pcts_soep_02[3]), int(w_wgt_pcts_soep_07[3]), int(w_wgt_pcts_soep_12[3]), int(w_wgt_pcts_soep_17[3])],
                                                       ['p99.9', int(w1_wgt_pcts_psid_01[4]), int(w1_wgt_pcts_psid_03[4]), int(w1_wgt_pcts_psid_05[4]), int(w1_wgt_pcts_psid_07[4]), int(w1_wgt_pcts_psid_09[4]), int(w1_wgt_pcts_psid_11[4]), int(w1_wgt_pcts_psid_13[4]), int(w1_wgt_pcts_psid_15[4]), int(w1_wgt_pcts_psid_17[4]), int(w_wgt_pcts_soep_02[4]), int(w_wgt_pcts_soep_07[4]), int(w_wgt_pcts_soep_12[4]), int(w_wgt_pcts_soep_17[4])],
                                                       ['max', int(dfPSID['wealth1_01'].max()), int(dfPSID['wealth1_03'].max()), int(dfPSID['wealth1_05'].max()), int(dfPSID['wealth1_07'].max()), int(dfPSID['wealth1_09'].max()), int(dfPSID['wealth1_11'].max()), int(dfPSID['wealth1_13'].max()), int(dfPSID['wealth1_15'].max()), int(dfPSID['wealth1_17'].max()), int(dfSOEP['wealth_02'].max()), int(dfSOEP['wealth_07'].max()), int(dfSOEP['wealth_12'].max()), int(dfSOEP['wealth_17'].max())],
                                                       ]),
                                             columns=['', '2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015', '2017', '2002SOEP', '2007SOEP', '2012SOEP', '2017SOEP'])

# write statistics to dataframe, psid: wealth2, soep: wealth
df_weighted_descriptives_w2 = pd.DataFrame(np.array([['N', int(dfPSID['weight1_01'].sum()), int(dfPSID['weight1_03'].sum()), int(dfPSID['weight1_05'].sum()), int(dfPSID['weight1_07'].sum()), int(dfPSID['weight1_09'].sum()), int(dfPSID['weight1_11'].sum()), int(dfPSID['weight1_13'].sum()), int(dfPSID['weight1_15'].sum()), int(dfPSID['weight1_17'].sum()), int(dfSOEP['weight_02'].sum()), int(dfSOEP['weight_07'].sum()), int(dfSOEP['weight_12'].sum())],
                                                       ['mean', int(w2_wgt_mean_psid_01), int(w2_wgt_mean_psid_03), int(w2_wgt_mean_psid_05), int(w2_wgt_mean_psid_07), int(w2_wgt_mean_psid_09), int(w2_wgt_mean_psid_11), int(w2_wgt_mean_psid_13), int(w2_wgt_mean_psid_15), int(w2_wgt_mean_psid_17), int(w_wgt_mean_soep_02), int(w_wgt_mean_soep_07), int(w_wgt_mean_soep_12)],
                                                       ['sd', int(w2_wgt_sd_psid_01), int(w2_wgt_sd_psid_03), int(w2_wgt_sd_psid_05), int(w2_wgt_sd_psid_07), int(w2_wgt_sd_psid_09), int(w2_wgt_sd_psid_11), int(w2_wgt_sd_psid_13), int(w2_wgt_sd_psid_15), int(w2_wgt_sd_psid_17), int(w_wgt_sd_soep_02), int(w_wgt_sd_soep_07), int(w_wgt_sd_soep_12)],
                                                       ['min', int(dfPSID['wealth2_01'].min()), int(dfPSID['wealth2_03'].min()), int(dfPSID['wealth2_05'].min()), int(dfPSID['wealth2_07'].min()), int(dfPSID['wealth2_09'].min()), int(dfPSID['wealth2_11'].min()), int(dfPSID['wealth2_13'].min()), int(dfPSID['wealth2_15'].min()), int(dfPSID['wealth2_17'].min()), int(dfSOEP['wealth_02'].min()), int(dfSOEP['wealth_07'].min()), int(dfSOEP['wealth_12'].min())],
                                                       ['p50', int(w2_wgt_pcts_psid_01[0]), int(w2_wgt_pcts_psid_03[0]), int(w2_wgt_pcts_psid_05[0]), int(w2_wgt_pcts_psid_07[0]), int(w2_wgt_pcts_psid_09[0]), int(w2_wgt_pcts_psid_11[0]), int(w2_wgt_pcts_psid_13[0]), int(w2_wgt_pcts_psid_15[0]), int(w2_wgt_pcts_psid_17[0]), int(w_wgt_pcts_soep_02[0]), int(w_wgt_pcts_soep_07[0]), int(w_wgt_pcts_soep_12[0])],
                                                       ['p75', int(w2_wgt_pcts_psid_01[1]), int(w2_wgt_pcts_psid_03[1]), int(w2_wgt_pcts_psid_05[1]), int(w2_wgt_pcts_psid_07[1]), int(w2_wgt_pcts_psid_09[1]), int(w2_wgt_pcts_psid_11[1]), int(w2_wgt_pcts_psid_13[1]), int(w2_wgt_pcts_psid_15[1]), int(w2_wgt_pcts_psid_17[1]), int(w_wgt_pcts_soep_02[1]), int(w_wgt_pcts_soep_07[1]), int(w_wgt_pcts_soep_12[1])],
                                                       ['p90', int(w2_wgt_pcts_psid_01[2]), int(w2_wgt_pcts_psid_03[2]), int(w2_wgt_pcts_psid_05[2]), int(w2_wgt_pcts_psid_07[2]), int(w2_wgt_pcts_psid_09[2]), int(w2_wgt_pcts_psid_11[2]), int(w2_wgt_pcts_psid_13[2]), int(w2_wgt_pcts_psid_15[2]), int(w2_wgt_pcts_psid_17[2]), int(w_wgt_pcts_soep_02[2]), int(w_wgt_pcts_soep_07[2]), int(w_wgt_pcts_soep_12[2])],
                                                       ['p99', int(w2_wgt_pcts_psid_01[3]), int(w2_wgt_pcts_psid_03[3]), int(w2_wgt_pcts_psid_05[3]), int(w2_wgt_pcts_psid_07[3]), int(w2_wgt_pcts_psid_09[3]), int(w2_wgt_pcts_psid_11[3]), int(w2_wgt_pcts_psid_13[3]), int(w2_wgt_pcts_psid_15[3]), int(w2_wgt_pcts_psid_17[3]), int(w_wgt_pcts_soep_02[3]), int(w_wgt_pcts_soep_07[3]), int(w_wgt_pcts_soep_12[3])],
                                                       ['p99.9', int(w2_wgt_pcts_psid_01[4]), int(w2_wgt_pcts_psid_03[4]), int(w2_wgt_pcts_psid_05[4]), int(w2_wgt_pcts_psid_07[4]), int(w2_wgt_pcts_psid_09[4]), int(w2_wgt_pcts_psid_11[4]), int(w2_wgt_pcts_psid_13[4]), int(w2_wgt_pcts_psid_15[4]), int(w2_wgt_pcts_psid_17[4]), int(w_wgt_pcts_soep_02[4]), int(w_wgt_pcts_soep_07[4]), int(w_wgt_pcts_soep_12[4])],
                                                       ['max', int(dfPSID['wealth2_01'].max()), int(dfPSID['wealth2_03'].max()), int(dfPSID['wealth2_05'].max()), int(dfPSID['wealth2_07'].max()), int(dfPSID['wealth2_09'].max()), int(dfPSID['wealth2_11'].max()), int(dfPSID['wealth2_13'].max()), int(dfPSID['wealth2_15'].max()), int(dfPSID['wealth2_17'].max()), int(dfSOEP['wealth_02'].max()), int(dfSOEP['wealth_07'].max()), int(dfSOEP['wealth_12'].max())],
                                                       ]),
                                             columns=['', '2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015', '2017', '2002SOEP', '2007SOEP', '2012SOEP'])

# save dataframes to excel sheet
with ExcelWriter(descriptivespath + 'wealth_descriptives.xlsx', mode='w') as writer:
    df_unweighted_descriptives_w1.to_excel(writer, sheet_name='unweighted_descriptives_w1', index=False)
    df_unweighted_descriptives_w2.to_excel(writer, sheet_name='unweighted_descriptives_w2', index=False)
    df_weighted_descriptives_w1.to_excel(writer, sheet_name='weighted_descriptives_w1', index=False)
    df_weighted_descriptives_w2.to_excel(writer, sheet_name='weighted_descriptives_w2', index=False)


"""
-----------------------------
Fit data
-----------------------------
"""

b = 500000
x0 = (-1, .5, 1, 1)
bootstraps = (100, 100, 100, 100)

### PSID
for test in ['LRtest', 'AIC']:
    for year in ['01', '03', '05', '07', '09', '11', '13', '15', '17']:

        print('PSID:', year, test)

        # write temp variables names
        wealth = 'wealth1_' + year
        weight = 'weight1_' + year
        data = dfPSID[wealth]
        # wgt = int(dfPSID[weight])
        wgt = pd.to_numeric(dfPSID[weight], downcast='signed')

        result = Paretobranchfit(x=data, weights=wgt, b=b, x0=x0, bootstraps=bootstraps,
                                 return_bestmodel=True, rejection_criteria=test)
        globals()['fit_result_psid_{}_{}'.format(year, test)] = result
        globals()['fit_psid_{}_{}'.format(year, test)] = prep_fit_results_for_table(result)


### SOEP
for test in ['LRtest', 'AIC']:
    for year in ['02', '07', '12', '17']:

        print('SOEP:', year, test)

        # write temp variables names
        wealth = 'wealth_' + year
        weight = 'weight_' + year
        data = dfSOEP[wealth]
        # wgt = int(dfSOEP[weight])
        wgt = pd.to_numeric(dfSOEP[weight], downcast='signed')

        result = Paretobranchfit(x=data, weights=wgt, b=b, x0=x0, bootstraps=bootstraps,
                                 return_bestmodel=True, rejection_criteria=test)
        globals()['fit_result_soep_{}_{}'.format(year, test)] = result
        globals()['fit_soep_{}_{}'.format(year, test)] = prep_fit_results_for_table(result)


"""
-----------------------------
Fit results to table
-----------------------------
"""

df_wealth_results_LR = pd.DataFrame(np.array([['best fitted model', '{}'.format(fit_result_psid_01_LRtest[0]), '{}'.format(fit_result_psid_03_LRtest[0]), '{}'.format(fit_result_psid_05_LRtest[0]), '{}'.format(fit_result_psid_07_LRtest[0]), '{}'.format(fit_result_psid_09_LRtest[0]), '{}'.format(fit_result_psid_11_LRtest[0]), '{}'.format(fit_result_psid_13_LRtest[0]), '{}'.format(fit_result_psid_15_LRtest[0]), '{}'.format(fit_result_psid_17_LRtest[0]), '{}'.format(fit_result_soep_02_LRtest[0]), '{}'.format(fit_result_soep_07_LRtest[0]), '{}'.format(fit_result_soep_12_LRtest[0]), '{}'.format(fit_result_soep_17_LRtest[0])],
                                           ['a',               '{}'.format(fit_psid_01_LRtest[0]),  '{}'.format(fit_psid_03_LRtest[0]),  '{}'.format(fit_psid_05_LRtest[0]),  '{}'.format(fit_psid_07_LRtest[0]),  '{}'.format(fit_psid_09_LRtest[0]),  '{}'.format(fit_psid_11_LRtest[0]),  '{}'.format(fit_psid_13_LRtest[0]),  '{}'.format(fit_psid_15_LRtest[0]),  '{}'.format(fit_psid_17_LRtest[0]),  '{}'.format(fit_soep_02_LRtest[0]),  '{}'.format(fit_soep_07_LRtest[0]),  '{}'.format(fit_soep_12_LRtest[0]),  '{}'.format(fit_soep_17_LRtest[0])],
                                           [' ',               '{}'.format(fit_psid_01_LRtest[1]),  '{}'.format(fit_psid_03_LRtest[1]),  '{}'.format(fit_psid_05_LRtest[1]),  '{}'.format(fit_psid_07_LRtest[1]),  '{}'.format(fit_psid_09_LRtest[1]),  '{}'.format(fit_psid_11_LRtest[1]),  '{}'.format(fit_psid_13_LRtest[1]),  '{}'.format(fit_psid_15_LRtest[1]),  '{}'.format(fit_psid_17_LRtest[1]),  '{}'.format(fit_soep_02_LRtest[1]),  '{}'.format(fit_soep_07_LRtest[1]),  '{}'.format(fit_soep_12_LRtest[1]),  '{}'.format(fit_soep_17_LRtest[1])],
                                           ['c',               '{}'.format(fit_psid_01_LRtest[2]),  '{}'.format(fit_psid_03_LRtest[2]),  '{}'.format(fit_psid_05_LRtest[2]),  '{}'.format(fit_psid_07_LRtest[2]),  '{}'.format(fit_psid_09_LRtest[2]),  '{}'.format(fit_psid_11_LRtest[2]),  '{}'.format(fit_psid_13_LRtest[2]),  '{}'.format(fit_psid_15_LRtest[2]),  '{}'.format(fit_psid_17_LRtest[2]),  '{}'.format(fit_soep_02_LRtest[2]),  '{}'.format(fit_soep_07_LRtest[2]),  '{}'.format(fit_soep_12_LRtest[2]),  '{}'.format(fit_soep_17_LRtest[2])],
                                           [' ',               '{}'.format(fit_psid_01_LRtest[3]),  '{}'.format(fit_psid_03_LRtest[3]),  '{}'.format(fit_psid_05_LRtest[3]),  '{}'.format(fit_psid_07_LRtest[3]),  '{}'.format(fit_psid_09_LRtest[3]),  '{}'.format(fit_psid_11_LRtest[3]),  '{}'.format(fit_psid_13_LRtest[3]),  '{}'.format(fit_psid_15_LRtest[3]),  '{}'.format(fit_psid_17_LRtest[3]),  '{}'.format(fit_soep_02_LRtest[3]),  '{}'.format(fit_soep_07_LRtest[3]),  '{}'.format(fit_soep_12_LRtest[3]),  '{}'.format(fit_soep_17_LRtest[3])],
                                           ['p',               '{}'.format(fit_psid_01_LRtest[4]),  '{}'.format(fit_psid_03_LRtest[4]),  '{}'.format(fit_psid_05_LRtest[4]),  '{}'.format(fit_psid_07_LRtest[4]),  '{}'.format(fit_psid_09_LRtest[4]),  '{}'.format(fit_psid_11_LRtest[4]),  '{}'.format(fit_psid_13_LRtest[4]),  '{}'.format(fit_psid_15_LRtest[4]),  '{}'.format(fit_psid_17_LRtest[4]),  '{}'.format(fit_soep_02_LRtest[4]),  '{}'.format(fit_soep_07_LRtest[4]),  '{}'.format(fit_soep_12_LRtest[4]),  '{}'.format(fit_soep_17_LRtest[4])],
                                           [' ',               '{}'.format(fit_psid_01_LRtest[5]),  '{}'.format(fit_psid_03_LRtest[5]),  '{}'.format(fit_psid_05_LRtest[5]),  '{}'.format(fit_psid_07_LRtest[5]),  '{}'.format(fit_psid_09_LRtest[5]),  '{}'.format(fit_psid_11_LRtest[5]),  '{}'.format(fit_psid_13_LRtest[5]),  '{}'.format(fit_psid_15_LRtest[5]),  '{}'.format(fit_psid_17_LRtest[5]),  '{}'.format(fit_soep_02_LRtest[5]),  '{}'.format(fit_soep_07_LRtest[5]),  '{}'.format(fit_soep_12_LRtest[5]),  '{}'.format(fit_soep_17_LRtest[5])],
                                           ['q',               '{}'.format(fit_psid_01_LRtest[6]),  '{}'.format(fit_psid_03_LRtest[6]),  '{}'.format(fit_psid_05_LRtest[6]),  '{}'.format(fit_psid_07_LRtest[6]),  '{}'.format(fit_psid_09_LRtest[6]),  '{}'.format(fit_psid_11_LRtest[6]),  '{}'.format(fit_psid_13_LRtest[6]),  '{}'.format(fit_psid_15_LRtest[6]),  '{}'.format(fit_psid_17_LRtest[6]),  '{}'.format(fit_soep_02_LRtest[6]),  '{}'.format(fit_soep_07_LRtest[6]),  '{}'.format(fit_soep_12_LRtest[6]),  '{}'.format(fit_soep_17_LRtest[6])],
                                           [' ',               '{}'.format(fit_psid_01_LRtest[7]),  '{}'.format(fit_psid_03_LRtest[7]),  '{}'.format(fit_psid_05_LRtest[7]),  '{}'.format(fit_psid_07_LRtest[7]),  '{}'.format(fit_psid_09_LRtest[7]),  '{}'.format(fit_psid_11_LRtest[7]),  '{}'.format(fit_psid_13_LRtest[7]),  '{}'.format(fit_psid_15_LRtest[7]),  '{}'.format(fit_psid_17_LRtest[7]),  '{}'.format(fit_soep_02_LRtest[7]),  '{}'.format(fit_soep_07_LRtest[7]),  '{}'.format(fit_soep_12_LRtest[7]),  '{}'.format(fit_soep_17_LRtest[7])],
                                           ['lower bound b',   '{}'.format(b),                      '{}'.format(b),                     '{}'.format(b),                       '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b)],
                                           ['LL',              '{}'.format(fit_psid_01_LRtest[11]) , '{}'.format(fit_psid_03_LRtest[11]) , '{}'.format(fit_psid_05_LRtest[11]) , '{}'.format(fit_psid_07_LRtest[11]) , '{}'.format(fit_psid_09_LRtest[11]) , '{}'.format(fit_psid_11_LRtest[11]) , '{}'.format(fit_psid_13_LRtest[11]) , '{}'.format(fit_psid_15_LRtest[11]) , '{}'.format(fit_psid_17_LRtest[11]) , '{}'.format(fit_soep_02_LRtest[11]) , '{}'.format(fit_soep_07_LRtest[11]) , '{}'.format(fit_soep_12_LRtest[11]) , '{}'.format(fit_soep_17_LRtest[11])],
                                           ['AIC',             '{}'.format(fit_psid_01_LRtest[8]) , '{}'.format(fit_psid_03_LRtest[8]) , '{}'.format(fit_psid_05_LRtest[8]) , '{}'.format(fit_psid_07_LRtest[8]) , '{}'.format(fit_psid_09_LRtest[8]) , '{}'.format(fit_psid_11_LRtest[8]) , '{}'.format(fit_psid_13_LRtest[8]) , '{}'.format(fit_psid_15_LRtest[8]) , '{}'.format(fit_psid_17_LRtest[8]) , '{}'.format(fit_soep_02_LRtest[8]) , '{}'.format(fit_soep_07_LRtest[8]) , '{}'.format(fit_soep_12_LRtest[8]) , '{}'.format(fit_soep_17_LRtest[8])],
                                           ['MSE',             '{}'.format(fit_psid_01_LRtest[9]), '{}'.format(fit_psid_03_LRtest[9]), '{}'.format(fit_psid_05_LRtest[9]), '{}'.format(fit_psid_07_LRtest[9]), '{}'.format(fit_psid_09_LRtest[9]), '{}'.format(fit_psid_11_LRtest[9]), '{}'.format(fit_psid_13_LRtest[9]), '{}'.format(fit_psid_15_LRtest[9]), '{}'.format(fit_psid_17_LRtest[9]), '{}'.format(fit_soep_02_LRtest[9]), '{}'.format(fit_soep_07_LRtest[9]), '{}'.format(fit_soep_12_LRtest[9]), '{}'.format(fit_soep_17_LRtest[9])],
                                           ['RMSE',            '{}'.format(fit_psid_01_LRtest[10]), '{}'.format(fit_psid_03_LRtest[10]), '{}'.format(fit_psid_05_LRtest[10]), '{}'.format(fit_psid_07_LRtest[10]), '{}'.format(fit_psid_09_LRtest[10]), '{}'.format(fit_psid_11_LRtest[10]), '{}'.format(fit_psid_13_LRtest[10]), '{}'.format(fit_psid_15_LRtest[10]), '{}'.format(fit_psid_17_LRtest[10]), '{}'.format(fit_soep_02_LRtest[10]), '{}'.format(fit_soep_07_LRtest[10]), '{}'.format(fit_soep_12_LRtest[10]), '{}'.format(fit_soep_17_LRtest[10])],
                                           ['emp. mean',       '{}'.format(fit_psid_01_LRtest[12]), '{}'.format(fit_psid_03_LRtest[12]), '{}'.format(fit_psid_05_LRtest[12]), '{}'.format(fit_psid_07_LRtest[12]), '{}'.format(fit_psid_09_LRtest[12]), '{}'.format(fit_psid_11_LRtest[12]), '{}'.format(fit_psid_13_LRtest[12]), '{}'.format(fit_psid_15_LRtest[12]), '{}'.format(fit_psid_17_LRtest[12]), '{}'.format(fit_soep_02_LRtest[12]), '{}'.format(fit_soep_07_LRtest[12]), '{}'.format(fit_soep_12_LRtest[12]), '{}'.format(fit_soep_17_LRtest[12])],
                                           ['emp. var.',       '{}'.format(fit_psid_01_LRtest[13]), '{}'.format(fit_psid_03_LRtest[13]), '{}'.format(fit_psid_05_LRtest[13]), '{}'.format(fit_psid_07_LRtest[13]), '{}'.format(fit_psid_09_LRtest[13]), '{}'.format(fit_psid_11_LRtest[13]), '{}'.format(fit_psid_13_LRtest[13]), '{}'.format(fit_psid_15_LRtest[13]), '{}'.format(fit_psid_17_LRtest[13]), '{}'.format(fit_soep_02_LRtest[13]), '{}'.format(fit_soep_07_LRtest[13]), '{}'.format(fit_soep_12_LRtest[13]), '{}'.format(fit_soep_17_LRtest[13])],
                                           ['pred. mean',      '{}'.format(fit_psid_01_LRtest[14]), '{}'.format(fit_psid_03_LRtest[14]), '{}'.format(fit_psid_05_LRtest[14]), '{}'.format(fit_psid_07_LRtest[14]), '{}'.format(fit_psid_09_LRtest[14]), '{}'.format(fit_psid_11_LRtest[14]), '{}'.format(fit_psid_13_LRtest[14]), '{}'.format(fit_psid_15_LRtest[14]), '{}'.format(fit_psid_17_LRtest[14]), '{}'.format(fit_soep_02_LRtest[14]), '{}'.format(fit_soep_07_LRtest[14]), '{}'.format(fit_soep_12_LRtest[14]), '{}'.format(fit_soep_17_LRtest[14])],
                                           ['pred. var.',      '{}'.format(fit_psid_01_LRtest[15]), '{}'.format(fit_psid_03_LRtest[15]), '{}'.format(fit_psid_05_LRtest[15]), '{}'.format(fit_psid_07_LRtest[15]), '{}'.format(fit_psid_09_LRtest[15]), '{}'.format(fit_psid_11_LRtest[15]), '{}'.format(fit_psid_13_LRtest[15]), '{}'.format(fit_psid_15_LRtest[15]), '{}'.format(fit_psid_17_LRtest[15]), '{}'.format(fit_soep_02_LRtest[15]), '{}'.format(fit_soep_07_LRtest[15]), '{}'.format(fit_soep_12_LRtest[15]), '{}'.format(fit_soep_17_LRtest[15])],
                                           ['n',               '{}'.format(fit_psid_01_LRtest[16]), '{}'.format(fit_psid_03_LRtest[16]), '{}'.format(fit_psid_05_LRtest[16]), '{}'.format(fit_psid_07_LRtest[16]), '{}'.format(fit_psid_09_LRtest[16]), '{}'.format(fit_psid_11_LRtest[16]), '{}'.format(fit_psid_13_LRtest[16]), '{}'.format(fit_psid_15_LRtest[16]), '{}'.format(fit_psid_17_LRtest[16]), '{}'.format(fit_soep_02_LRtest[16]), '{}'.format(fit_soep_07_LRtest[16]), '{}'.format(fit_soep_12_LRtest[16]), '{}'.format(fit_soep_17_LRtest[16])],
                                           ['N',               '{}'.format(fit_psid_01_LRtest[17]), '{}'.format(fit_psid_03_LRtest[17]), '{}'.format(fit_psid_05_LRtest[17]), '{}'.format(fit_psid_07_LRtest[17]), '{}'.format(fit_psid_09_LRtest[17]), '{}'.format(fit_psid_11_LRtest[17]), '{}'.format(fit_psid_13_LRtest[17]), '{}'.format(fit_psid_15_LRtest[17]), '{}'.format(fit_psid_17_LRtest[17]), '{}'.format(fit_soep_02_LRtest[17]), '{}'.format(fit_soep_07_LRtest[17]), '{}'.format(fit_soep_12_LRtest[17]), '{}'.format(fit_soep_17_LRtest[17])]
                                           ]),
                                 columns=['', '01', '03', '05', '07', '09', '11', '13', '15', '17', 'SOEP02', 'SOEP07', 'SOEP12', 'SOEP17'])

df_wealth_results_AIC = pd.DataFrame(np.array([['best fitted model', '{}'.format(fit_result_psid_01_AIC[0]), '{}'.format(fit_result_psid_03_AIC[0]), '{}'.format(fit_result_psid_05_AIC[0]), '{}'.format(fit_result_psid_07_AIC[0]), '{}'.format(fit_result_psid_09_AIC[0]), '{}'.format(fit_result_psid_11_AIC[0]), '{}'.format(fit_result_psid_13_AIC[0]), '{}'.format(fit_result_psid_15_AIC[0]), '{}'.format(fit_result_psid_17_AIC[0]), '{}'.format(fit_result_soep_02_AIC[0]), '{}'.format(fit_result_soep_07_AIC[0]), '{}'.format(fit_result_soep_12_AIC[0]), '{}'.format(fit_result_soep_17_AIC[0])],
                                           ['a',               '{}'.format(fit_psid_01_AIC[0]),  '{}'.format(fit_psid_03_AIC[0]),  '{}'.format(fit_psid_05_AIC[0]),  '{}'.format(fit_psid_07_AIC[0]),  '{}'.format(fit_psid_09_AIC[0]),  '{}'.format(fit_psid_11_AIC[0]),  '{}'.format(fit_psid_13_AIC[0]),  '{}'.format(fit_psid_15_AIC[0]),  '{}'.format(fit_psid_17_AIC[0]),  '{}'.format(fit_soep_02_AIC[0]),  '{}'.format(fit_soep_07_AIC[0]),  '{}'.format(fit_soep_12_AIC[0]),  '{}'.format(fit_soep_17_AIC[0])],
                                           [' ',               '{}'.format(fit_psid_01_AIC[1]),  '{}'.format(fit_psid_03_AIC[1]),  '{}'.format(fit_psid_05_AIC[1]),  '{}'.format(fit_psid_07_AIC[1]),  '{}'.format(fit_psid_09_AIC[1]),  '{}'.format(fit_psid_11_AIC[1]),  '{}'.format(fit_psid_13_AIC[1]),  '{}'.format(fit_psid_15_AIC[1]),  '{}'.format(fit_psid_17_AIC[1]),  '{}'.format(fit_soep_02_AIC[1]),  '{}'.format(fit_soep_07_AIC[1]),  '{}'.format(fit_soep_12_AIC[1]),  '{}'.format(fit_soep_17_AIC[1])],
                                           ['c',               '{}'.format(fit_psid_01_AIC[2]),  '{}'.format(fit_psid_03_AIC[2]),  '{}'.format(fit_psid_05_AIC[2]),  '{}'.format(fit_psid_07_AIC[2]),  '{}'.format(fit_psid_09_AIC[2]),  '{}'.format(fit_psid_11_AIC[2]),  '{}'.format(fit_psid_13_AIC[2]),  '{}'.format(fit_psid_15_AIC[2]),  '{}'.format(fit_psid_17_AIC[2]),  '{}'.format(fit_soep_02_AIC[2]),  '{}'.format(fit_soep_07_AIC[2]),  '{}'.format(fit_soep_12_AIC[2]),  '{}'.format(fit_soep_17_AIC[2])],
                                           [' ',               '{}'.format(fit_psid_01_AIC[3]),  '{}'.format(fit_psid_03_AIC[3]),  '{}'.format(fit_psid_05_AIC[3]),  '{}'.format(fit_psid_07_AIC[3]),  '{}'.format(fit_psid_09_AIC[3]),  '{}'.format(fit_psid_11_AIC[3]),  '{}'.format(fit_psid_13_AIC[3]),  '{}'.format(fit_psid_15_AIC[3]),  '{}'.format(fit_psid_17_AIC[3]),  '{}'.format(fit_soep_02_AIC[3]),  '{}'.format(fit_soep_07_AIC[3]),  '{}'.format(fit_soep_12_AIC[3]),  '{}'.format(fit_soep_17_AIC[3])],
                                           ['p',               '{}'.format(fit_psid_01_AIC[4]),  '{}'.format(fit_psid_03_AIC[4]),  '{}'.format(fit_psid_05_AIC[4]),  '{}'.format(fit_psid_07_AIC[4]),  '{}'.format(fit_psid_09_AIC[4]),  '{}'.format(fit_psid_11_AIC[4]),  '{}'.format(fit_psid_13_AIC[4]),  '{}'.format(fit_psid_15_AIC[4]),  '{}'.format(fit_psid_17_AIC[4]),  '{}'.format(fit_soep_02_AIC[4]),  '{}'.format(fit_soep_07_AIC[4]),  '{}'.format(fit_soep_12_AIC[4]),  '{}'.format(fit_soep_17_AIC[4])],
                                           [' ',               '{}'.format(fit_psid_01_AIC[5]),  '{}'.format(fit_psid_03_AIC[5]),  '{}'.format(fit_psid_05_AIC[5]),  '{}'.format(fit_psid_07_AIC[5]),  '{}'.format(fit_psid_09_AIC[5]),  '{}'.format(fit_psid_11_AIC[5]),  '{}'.format(fit_psid_13_AIC[5]),  '{}'.format(fit_psid_15_AIC[5]),  '{}'.format(fit_psid_17_AIC[5]),  '{}'.format(fit_soep_02_AIC[5]),  '{}'.format(fit_soep_07_AIC[5]),  '{}'.format(fit_soep_12_AIC[5]),  '{}'.format(fit_soep_17_AIC[5])],
                                           ['q',               '{}'.format(fit_psid_01_AIC[6]),  '{}'.format(fit_psid_03_AIC[6]),  '{}'.format(fit_psid_05_AIC[6]),  '{}'.format(fit_psid_07_AIC[6]),  '{}'.format(fit_psid_09_AIC[6]),  '{}'.format(fit_psid_11_AIC[6]),  '{}'.format(fit_psid_13_AIC[6]),  '{}'.format(fit_psid_15_AIC[6]),  '{}'.format(fit_psid_17_AIC[6]),  '{}'.format(fit_soep_02_AIC[6]),  '{}'.format(fit_soep_07_AIC[6]),  '{}'.format(fit_soep_12_AIC[6]),  '{}'.format(fit_soep_17_AIC[6])],
                                           [' ',               '{}'.format(fit_psid_01_AIC[7]),  '{}'.format(fit_psid_03_AIC[7]),  '{}'.format(fit_psid_05_AIC[7]),  '{}'.format(fit_psid_07_AIC[7]),  '{}'.format(fit_psid_09_AIC[7]),  '{}'.format(fit_psid_11_AIC[7]),  '{}'.format(fit_psid_13_AIC[7]),  '{}'.format(fit_psid_15_AIC[7]),  '{}'.format(fit_psid_17_AIC[7]),  '{}'.format(fit_soep_02_AIC[7]),  '{}'.format(fit_soep_07_AIC[7]),  '{}'.format(fit_soep_12_AIC[7]),  '{}'.format(fit_soep_17_AIC[7])],
                                           ['lower bound b',   '{}'.format(b),                      '{}'.format(b),                     '{}'.format(b),                       '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b),                      '{}'.format(b)],
                                           ['LL',              '{}'.format(fit_psid_01_AIC[11]) , '{}'.format(fit_psid_03_AIC[11]) , '{}'.format(fit_psid_05_AIC[11]) , '{}'.format(fit_psid_07_AIC[11]) , '{}'.format(fit_psid_09_AIC[11]) , '{}'.format(fit_psid_11_AIC[11]) , '{}'.format(fit_psid_13_AIC[11]) , '{}'.format(fit_psid_15_AIC[11]) , '{}'.format(fit_psid_17_AIC[11]) , '{}'.format(fit_soep_02_AIC[11]) , '{}'.format(fit_soep_07_AIC[11]) , '{}'.format(fit_soep_12_AIC[11]) , '{}'.format(fit_soep_17_AIC[11])],
                                           ['AIC',             '{}'.format(fit_psid_01_AIC[8]) , '{}'.format(fit_psid_03_AIC[8]) , '{}'.format(fit_psid_05_AIC[8]) , '{}'.format(fit_psid_07_AIC[8]) , '{}'.format(fit_psid_09_AIC[8]) , '{}'.format(fit_psid_11_AIC[8]) , '{}'.format(fit_psid_13_AIC[8]) , '{}'.format(fit_psid_15_AIC[8]) , '{}'.format(fit_psid_17_AIC[8]) , '{}'.format(fit_soep_02_AIC[8]) , '{}'.format(fit_soep_07_AIC[8]) , '{}'.format(fit_soep_12_AIC[8]) , '{}'.format(fit_soep_17_AIC[8])],
                                           ['MSE',             '{}'.format(fit_psid_01_AIC[9]), '{}'.format(fit_psid_03_AIC[9]), '{}'.format(fit_psid_05_AIC[9]), '{}'.format(fit_psid_07_AIC[9]), '{}'.format(fit_psid_09_AIC[9]), '{}'.format(fit_psid_11_AIC[9]), '{}'.format(fit_psid_13_AIC[9]), '{}'.format(fit_psid_15_AIC[9]), '{}'.format(fit_psid_17_AIC[9]), '{}'.format(fit_soep_02_AIC[9]), '{}'.format(fit_soep_07_AIC[9]), '{}'.format(fit_soep_12_AIC[9]), '{}'.format(fit_soep_17_AIC[9])],
                                           ['RMSE',            '{}'.format(fit_psid_01_AIC[10]), '{}'.format(fit_psid_03_AIC[10]), '{}'.format(fit_psid_05_AIC[10]), '{}'.format(fit_psid_07_AIC[10]), '{}'.format(fit_psid_09_AIC[10]), '{}'.format(fit_psid_11_AIC[10]), '{}'.format(fit_psid_13_AIC[10]), '{}'.format(fit_psid_15_AIC[10]), '{}'.format(fit_psid_17_AIC[10]), '{}'.format(fit_soep_02_AIC[10]), '{}'.format(fit_soep_07_AIC[10]), '{}'.format(fit_soep_12_AIC[10]), '{}'.format(fit_soep_17_AIC[10])],
                                           ['emp. mean',       '{}'.format(fit_psid_01_AIC[12]), '{}'.format(fit_psid_03_AIC[12]), '{}'.format(fit_psid_05_AIC[12]), '{}'.format(fit_psid_07_AIC[12]), '{}'.format(fit_psid_09_AIC[12]), '{}'.format(fit_psid_11_AIC[12]), '{}'.format(fit_psid_13_AIC[12]), '{}'.format(fit_psid_15_AIC[12]), '{}'.format(fit_psid_17_AIC[12]), '{}'.format(fit_soep_02_AIC[12]), '{}'.format(fit_soep_07_AIC[12]), '{}'.format(fit_soep_12_AIC[12]), '{}'.format(fit_soep_17_AIC[12])],
                                           ['emp. var.',       '{}'.format(fit_psid_01_AIC[13]), '{}'.format(fit_psid_03_AIC[13]), '{}'.format(fit_psid_05_AIC[13]), '{}'.format(fit_psid_07_AIC[13]), '{}'.format(fit_psid_09_AIC[13]), '{}'.format(fit_psid_11_AIC[13]), '{}'.format(fit_psid_13_AIC[13]), '{}'.format(fit_psid_15_AIC[13]), '{}'.format(fit_psid_17_AIC[13]), '{}'.format(fit_soep_02_AIC[13]), '{}'.format(fit_soep_07_AIC[13]), '{}'.format(fit_soep_12_AIC[13]), '{}'.format(fit_soep_17_AIC[13])],
                                           ['pred. mean',      '{}'.format(fit_psid_01_AIC[14]), '{}'.format(fit_psid_03_AIC[14]), '{}'.format(fit_psid_05_AIC[14]), '{}'.format(fit_psid_07_AIC[14]), '{}'.format(fit_psid_09_AIC[14]), '{}'.format(fit_psid_11_AIC[14]), '{}'.format(fit_psid_13_AIC[14]), '{}'.format(fit_psid_15_AIC[14]), '{}'.format(fit_psid_17_AIC[14]), '{}'.format(fit_soep_02_AIC[14]), '{}'.format(fit_soep_07_AIC[14]), '{}'.format(fit_soep_12_AIC[14]), '{}'.format(fit_soep_17_AIC[14])],
                                           ['pred. var.',      '{}'.format(fit_psid_01_AIC[15]), '{}'.format(fit_psid_03_AIC[15]), '{}'.format(fit_psid_05_AIC[15]), '{}'.format(fit_psid_07_AIC[15]), '{}'.format(fit_psid_09_AIC[15]), '{}'.format(fit_psid_11_AIC[15]), '{}'.format(fit_psid_13_AIC[15]), '{}'.format(fit_psid_15_AIC[15]), '{}'.format(fit_psid_17_AIC[15]), '{}'.format(fit_soep_02_AIC[15]), '{}'.format(fit_soep_07_AIC[15]), '{}'.format(fit_soep_12_AIC[15]), '{}'.format(fit_soep_17_AIC[15])],
                                           ['n',               '{}'.format(fit_psid_01_AIC[16]), '{}'.format(fit_psid_03_AIC[16]), '{}'.format(fit_psid_05_AIC[16]), '{}'.format(fit_psid_07_AIC[16]), '{}'.format(fit_psid_09_AIC[16]), '{}'.format(fit_psid_11_AIC[16]), '{}'.format(fit_psid_13_AIC[16]), '{}'.format(fit_psid_15_AIC[16]), '{}'.format(fit_psid_17_AIC[16]), '{}'.format(fit_soep_02_AIC[16]), '{}'.format(fit_soep_07_AIC[16]), '{}'.format(fit_soep_12_AIC[16]), '{}'.format(fit_soep_17_AIC[16])],
                                           ['N',               '{}'.format(fit_psid_01_AIC[17]), '{}'.format(fit_psid_03_AIC[17]), '{}'.format(fit_psid_05_AIC[17]), '{}'.format(fit_psid_07_AIC[17]), '{}'.format(fit_psid_09_AIC[17]), '{}'.format(fit_psid_11_AIC[17]), '{}'.format(fit_psid_13_AIC[17]), '{}'.format(fit_psid_15_AIC[17]), '{}'.format(fit_psid_17_AIC[17]), '{}'.format(fit_soep_02_AIC[17]), '{}'.format(fit_soep_07_AIC[17]), '{}'.format(fit_soep_12_AIC[17]), '{}'.format(fit_soep_17_AIC[17])]
                                           ]),
                                 columns=['', '01', '03', '05', '07', '09', '11', '13', '15', '17', 'SOEP02', 'SOEP07', 'SOEP12', 'SOEP17'])

# save dataframes to excel sheet
with ExcelWriter(descriptivespath + 'wealth_data_fit_results_b{}.xlsx'.format(b), engine='openpyxl', mode='w') as writer:
    df_wealth_results_LR.to_excel(writer, sheet_name='wealth_fit_results_LR_{}'.format(b), index=False)
    df_wealth_results_AIC.to_excel(writer, sheet_name='wealth_fit_results_AIC_{}'.format(b), index=False)


"""
-----------------------------
Plot fit vs data
-----------------------------
"""
## PSID 2017

# set figsize
import matplotlib.pyplot as plt
print(plt.rcParams.get('figure.figsize'))
plt.rcParams['figure.figsize'] = 10, 8

# LRtest
Paretobranchfit(x=dfPSID['wealth1_17'], weights=dfPSID['weight1_17'], b=b, x0=x0,
                bootstraps=bootstraps, return_bestmodel=False, plot=True,
                rejection_criteria='LRtest',
                plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})
# AIC
Paretobranchfit(x=dfPSID['wealth1_17'], weights=dfPSID['weight1_17'], b=b, x0=x0,
                bootstraps=bootstraps, return_bestmodel=True, plot=True,
                rejection_criteria='LRtest',
                plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

## SOEP 2017

# LRtest
Paretobranchfit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=b, x0=x0,
                bootstraps=bootstraps, return_bestmodel=True, plot=True,
                rejection_criteria='LRtest',
                plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})

# AIC
Paretobranchfit(x=dfSOEP['wealth_17'], weights=dfSOEP['weight_17'], b=b, x0=x0,
                bootstraps=bootstraps, return_bestmodel=True, plot=True,
                rejection_criteria='LRtest',
                plot_cosmetics={'bins': 500, 'col_data': 'blue', 'col_fit': 'red'})
