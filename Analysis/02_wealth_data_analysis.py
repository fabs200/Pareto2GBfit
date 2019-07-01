from Pareto2GBfit.fitting import *
import numpy as np
from scipy.stats import describe
import pandas as pd
from pandas import ExcelWriter
import xlsxwriter
import os

# TODO: SOEP 2017

# windows paths
if os.name == 'nt':
    descriptivespath = 'D:/OneDrive/Studium/Masterarbeit//Python/descriptives/'
    data_PSID = 'D:/OneDrive/Studium/Masterarbeit/data/J261520/'
    data_SOEP = 'C:/Users/fabia/Documents/DATA/SOEP_v33.1/SOEP-CORE_v33.1_stata_bilingual/'

# mac paths
if os.name == 'posix':
    descriptivespath = '/Users/Fabian/OneDrive/Studium/Masterarbeit/Python/descriptives/'
    data_PSID = "/Users/Fabian/OneDrive/Studium/Masterarbeit/data/J261520/"
    data_SOEP = '/Users/Fabian/Documents/DATA/STATA/SOEP_v33.1/SOEP_wide/'

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
dfPSID.head()

"""
-----------------------
SOEP data preparation
-----------------------
"""

data_SOEPwealth = data_SOEP + 'hwealth.dta'
data_SOEPHHweight = data_SOEP + 'hhrf.dta'

# read in data
dfSOEP_wealth = pd.read_stata(data_SOEPwealth, columns=["syear", "hhnrakt", "w011ha", "w011hb", "w011hc", "w011hd", "w011he"])
dfSOEP_hhweights = pd.read_stata(data_SOEPHHweight, columns=["hhnrakt", "shhrf", "xhhrf", "bchhrf"])

# mean imputed wealths
dfSOEP_wealth['wealth'] = dfSOEP_wealth[["w011ha", "w011hb", "w011hc", "w011hd", "w011he"]].mean(axis=1)

# reshape long to wide
dfSOEP_wealth = dfSOEP_wealth.pivot(index='hhnrakt', columns='syear', values='wealth')

# merge datasets
dfSOEP = dfSOEP_wealth.merge(dfSOEP_hhweights, left_on='hhnrakt', right_on='hhnrakt')

# rename weights
dfSOEP = dfSOEP.rename(index=str, columns={2002: 'wealth_02', 2007: 'wealth_07', 2012: 'wealth_12',
                                           'shhrf': 'weight_02', 'xhhrf': 'weight_07', 'bchhrf': 'weight_12'})

# check
dfSOEP_wealth.head()
dfSOEP_hhweights.head()
dfSOEP.head()

"""
-----------------------
Descriptive Statistics
-----------------------
"""

# unweighted, psid: wealth1, soep: wealth
df_unweighted_descriptives_w1 = pd.DataFrame(np.array([['N', dfPSID['wealth1_01'].count(), dfPSID['wealth1_03'].count(), dfPSID['wealth1_05'].count(), dfPSID['wealth1_07'].count(), dfPSID['wealth1_09'].count(), dfPSID['wealth1_11'].count(), dfPSID['wealth1_13'].count(), dfPSID['wealth1_15'].count(), dfPSID['wealth1_17'].count(), dfSOEP['wealth_02'].count(), dfSOEP['wealth_07'].count(), dfSOEP['wealth_12'].count()],
                                                       ['mean', int(dfPSID['wealth1_01'].mean()), int(dfPSID['wealth1_03'].mean()), int(dfPSID['wealth1_05'].mean()), int(dfPSID['wealth1_07'].mean()), int(dfPSID['wealth1_09'].mean()), int(dfPSID['wealth1_11'].mean()), int(dfPSID['wealth1_13'].mean()), int(dfPSID['wealth1_15'].mean()), int(dfPSID['wealth1_17'].mean()), int(dfSOEP['wealth_02'].mean()), int(dfSOEP['wealth_07'].mean()), int(dfSOEP['wealth_12'].mean())],
                                                       ['sd', int(dfPSID['wealth1_01'].std()), int(dfPSID['wealth1_03'].std()), int(dfPSID['wealth1_05'].std()), int(dfPSID['wealth1_07'].std()), int(dfPSID['wealth1_09'].std()), int(dfPSID['wealth1_11'].std()), int(dfPSID['wealth1_13'].std()), int(dfPSID['wealth1_15'].std()), int(dfPSID['wealth1_17'].std()), int(dfSOEP['wealth_02'].std()), int(dfSOEP['wealth_07'].std()), int(dfSOEP['wealth_12'].std())],
                                                       ['min', int(dfPSID['wealth1_01'].min()), int(dfPSID['wealth1_03'].min()), int(dfPSID['wealth1_05'].min()), int(dfPSID['wealth1_07'].min()), int(dfPSID['wealth1_09'].min()), int(dfPSID['wealth1_11'].min()), int(dfPSID['wealth1_13'].min()), int(dfPSID['wealth1_15'].min()), int(dfPSID['wealth1_17'].min()), int(dfSOEP['wealth_02'].min()), int(dfSOEP['wealth_07'].min()), int(dfSOEP['wealth_12'].min())],
                                                       ['p50', int(dfPSID['wealth1_01'].quantile()), int(dfPSID['wealth1_03'].quantile()), int(dfPSID['wealth1_05'].quantile()), int(dfPSID['wealth1_07'].quantile()), int(dfPSID['wealth1_09'].quantile()), int(dfPSID['wealth1_11'].quantile()), int(dfPSID['wealth1_13'].quantile()), int(dfPSID['wealth1_15'].quantile()), int(dfPSID['wealth1_17'].quantile()), int(dfSOEP['wealth_02'].quantile()), int(dfSOEP['wealth_07'].quantile()), int(dfSOEP['wealth_12'].quantile())],
                                                       ['p75', int(dfPSID['wealth1_01'].quantile(.75)), int(dfPSID['wealth1_03'].quantile(.75)), int(dfPSID['wealth1_05'].quantile(.75)), int(dfPSID['wealth1_07'].quantile(.75)), int(dfPSID['wealth1_09'].quantile(.75)), int(dfPSID['wealth1_11'].quantile(.75)), int(dfPSID['wealth1_13'].quantile(.75)), int(dfPSID['wealth1_15'].quantile(.75)), int(dfPSID['wealth1_17'].quantile(.75)), int(dfSOEP['wealth_02'].quantile(.75)), int(dfSOEP['wealth_07'].quantile(.75)), int(dfSOEP['wealth_12'].quantile(.75))],
                                                       ['p90', int(dfPSID['wealth1_01'].quantile(.9)), int(dfPSID['wealth1_03'].quantile(.9)), int(dfPSID['wealth1_05'].quantile(.9)), int(dfPSID['wealth1_07'].quantile(.9)), int(dfPSID['wealth1_09'].quantile(.9)), int(dfPSID['wealth1_11'].quantile(.9)), int(dfPSID['wealth1_13'].quantile(.9)), int(dfPSID['wealth1_15'].quantile(.9)), int(dfPSID['wealth1_17'].quantile(.9)), int(dfSOEP['wealth_02'].quantile(.9)), int(dfSOEP['wealth_07'].quantile(.9)), int(dfSOEP['wealth_12'].quantile(.9))],
                                                       ['p99', int(dfPSID['wealth1_01'].quantile(.99)), int(dfPSID['wealth1_03'].quantile(.99)), int(dfPSID['wealth1_05'].quantile(.99)), int(dfPSID['wealth1_07'].quantile(.99)), int(dfPSID['wealth1_09'].quantile(.99)), int(dfPSID['wealth1_11'].quantile(.99)), int(dfPSID['wealth1_13'].quantile(.99)), int(dfPSID['wealth1_15'].quantile(.99)), int(dfPSID['wealth1_17'].quantile(.99)), int(dfSOEP['wealth_02'].quantile(.99)), int(dfSOEP['wealth_07'].quantile(.99)), int(dfSOEP['wealth_12'].quantile(.99))],
                                                       ['p99.9', int(dfPSID['wealth1_01'].quantile(.999)), int(dfPSID['wealth1_03'].quantile(.999)), int(dfPSID['wealth1_05'].quantile(.999)), int(dfPSID['wealth1_07'].quantile(.999)), int(dfPSID['wealth1_09'].quantile(.999)), int(dfPSID['wealth1_11'].quantile(.999)), int(dfPSID['wealth1_13'].quantile(.999)), int(dfPSID['wealth1_15'].quantile(.999)), int(dfPSID['wealth1_17'].quantile(.999)), int(dfSOEP['wealth_02'].quantile(.999)), int(dfSOEP['wealth_07'].quantile(.999)), int(dfSOEP['wealth_12'].quantile(.999))],
                                                       ['max', int(dfPSID['wealth1_01'].max()), int(dfPSID['wealth1_03'].max()), int(dfPSID['wealth1_05'].max()), int(dfPSID['wealth1_07'].max()), int(dfPSID['wealth1_09'].max()), int(dfPSID['wealth1_11'].max()), int(dfPSID['wealth1_13'].max()), int(dfPSID['wealth1_15'].max()), int(dfPSID['wealth1_17'].max()), int(dfSOEP['wealth_02'].max()), int(dfSOEP['wealth_07'].max()), int(dfSOEP['wealth_12'].max())],
                                                       ]),
                                             columns=['', '2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015', '2017', '2002SOEP', '2007SOEP', '2012SOEP'])


# unweighted, psid: wealth2, soep: wealth
df_unweighted_descriptives_w2 = pd.DataFrame(np.array([['N', dfPSID['wealth2_01'].count(), dfPSID['wealth2_03'].count(), dfPSID['wealth2_05'].count(), dfPSID['wealth2_07'].count(), dfPSID['wealth2_09'].count(), dfPSID['wealth2_11'].count(), dfPSID['wealth2_13'].count(), dfPSID['wealth2_15'].count(), dfPSID['wealth2_17'].count(), dfSOEP['wealth_02'].count(), dfSOEP['wealth_07'].count(), dfSOEP['wealth_12'].count()],
                                                       ['mean', int(dfPSID['wealth2_01'].mean()), int(dfPSID['wealth2_03'].mean()), int(dfPSID['wealth2_05'].mean()), int(dfPSID['wealth2_07'].mean()), int(dfPSID['wealth2_09'].mean()), int(dfPSID['wealth2_11'].mean()), int(dfPSID['wealth2_13'].mean()), int(dfPSID['wealth2_15'].mean()), int(dfPSID['wealth2_17'].mean()), int(dfSOEP['wealth_02'].mean()), int(dfSOEP['wealth_07'].mean()), int(dfSOEP['wealth_12'].mean())],
                                                       ['sd', int(dfPSID['wealth2_01'].std()), int(dfPSID['wealth2_03'].std()), int(dfPSID['wealth2_05'].std()), int(dfPSID['wealth2_07'].std()), int(dfPSID['wealth2_09'].std()), int(dfPSID['wealth2_11'].std()), int(dfPSID['wealth2_13'].std()), int(dfPSID['wealth2_15'].std()), int(dfPSID['wealth2_17'].std()), int(dfSOEP['wealth_02'].std()), int(dfSOEP['wealth_07'].std()), int(dfSOEP['wealth_12'].std())],
                                                       ['min', int(dfPSID['wealth2_01'].min()), int(dfPSID['wealth2_03'].min()), int(dfPSID['wealth2_05'].min()), int(dfPSID['wealth2_07'].min()), int(dfPSID['wealth2_09'].min()), int(dfPSID['wealth2_11'].min()), int(dfPSID['wealth2_13'].min()), int(dfPSID['wealth2_15'].min()), int(dfPSID['wealth2_17'].min()), int(dfSOEP['wealth_02'].min()), int(dfSOEP['wealth_07'].min()), int(dfSOEP['wealth_12'].min())],
                                                       ['p50', int(dfPSID['wealth2_01'].quantile()), int(dfPSID['wealth2_03'].quantile()), int(dfPSID['wealth2_05'].quantile()), int(dfPSID['wealth2_07'].quantile()), int(dfPSID['wealth2_09'].quantile()), int(dfPSID['wealth2_11'].quantile()), int(dfPSID['wealth2_13'].quantile()), int(dfPSID['wealth2_15'].quantile()), int(dfPSID['wealth2_17'].quantile()), int(dfSOEP['wealth_02'].quantile()), int(dfSOEP['wealth_07'].quantile()), int(dfSOEP['wealth_12'].quantile())],
                                                       ['p75', int(dfPSID['wealth2_01'].quantile(.75)), int(dfPSID['wealth2_03'].quantile(.75)), int(dfPSID['wealth2_05'].quantile(.75)), int(dfPSID['wealth2_07'].quantile(.75)), int(dfPSID['wealth2_09'].quantile(.75)), int(dfPSID['wealth2_11'].quantile(.75)), int(dfPSID['wealth2_13'].quantile(.75)), int(dfPSID['wealth2_15'].quantile(.75)), int(dfPSID['wealth2_17'].quantile(.75)), int(dfSOEP['wealth_02'].quantile(.75)), int(dfSOEP['wealth_07'].quantile(.75)), int(dfSOEP['wealth_12'].quantile(.75))],
                                                       ['p90', int(dfPSID['wealth2_01'].quantile(.9)), int(dfPSID['wealth2_03'].quantile(.9)), int(dfPSID['wealth2_05'].quantile(.9)), int(dfPSID['wealth2_07'].quantile(.9)), int(dfPSID['wealth2_09'].quantile(.9)), int(dfPSID['wealth2_11'].quantile(.9)), int(dfPSID['wealth2_13'].quantile(.9)), int(dfPSID['wealth2_15'].quantile(.9)), int(dfPSID['wealth2_17'].quantile(.9)), int(dfSOEP['wealth_02'].quantile(.9)), int(dfSOEP['wealth_07'].quantile(.9)), int(dfSOEP['wealth_12'].quantile(.9))],
                                                       ['p99', int(dfPSID['wealth2_01'].quantile(.99)), int(dfPSID['wealth2_03'].quantile(.99)), int(dfPSID['wealth2_05'].quantile(.99)), int(dfPSID['wealth2_07'].quantile(.99)), int(dfPSID['wealth2_09'].quantile(.99)), int(dfPSID['wealth2_11'].quantile(.99)), int(dfPSID['wealth2_13'].quantile(.99)), int(dfPSID['wealth2_15'].quantile(.99)), int(dfPSID['wealth2_17'].quantile(.99)), int(dfSOEP['wealth_02'].quantile(.99)), int(dfSOEP['wealth_07'].quantile(.99)), int(dfSOEP['wealth_12'].quantile(.99))],
                                                       ['p99.9', int(dfPSID['wealth2_01'].quantile(.999)), int(dfPSID['wealth2_03'].quantile(.999)), int(dfPSID['wealth2_05'].quantile(.999)), int(dfPSID['wealth2_07'].quantile(.999)), int(dfPSID['wealth2_09'].quantile(.999)), int(dfPSID['wealth2_11'].quantile(.999)), int(dfPSID['wealth2_13'].quantile(.999)), int(dfPSID['wealth2_15'].quantile(.999)), int(dfPSID['wealth2_17'].quantile(.999)), int(dfSOEP['wealth_02'].quantile(.999)), int(dfSOEP['wealth_07'].quantile(.999)), int(dfSOEP['wealth_12'].quantile(.999))],
                                                       ['max', int(dfPSID['wealth2_01'].max()), int(dfPSID['wealth2_03'].max()), int(dfPSID['wealth2_05'].max()), int(dfPSID['wealth2_07'].max()), int(dfPSID['wealth2_09'].max()), int(dfPSID['wealth2_11'].max()), int(dfPSID['wealth2_13'].max()), int(dfPSID['wealth2_15'].max()), int(dfPSID['wealth2_17'].max()), int(dfSOEP['wealth_02'].max()), int(dfSOEP['wealth_07'].max()), int(dfSOEP['wealth_12'].max())],
                                                       ]),
                                             columns=['', '2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015', '2017', '2002SOEP', '2007SOEP', '2012SOEP'])



# weighted, psid: wealth1, soep: wealth

# weighted avg
w1_wgt_mean_psid_01 = np.average(a=dfPSID['wealth1_01'][~np.isnan(dfPSID['wealth1_01'])], weights=dfPSID['weight1_01'][~np.isnan(dfPSID['weight1_01'])])
w1_wgt_mean_psid_03 = np.average(a=dfPSID['wealth1_03'][~np.isnan(dfPSID['wealth1_03'])], weights=dfPSID['weight1_03'][~np.isnan(dfPSID['weight1_03'])])
w1_wgt_mean_psid_05 = np.average(a=dfPSID['wealth1_05'][~np.isnan(dfPSID['wealth1_05'])], weights=dfPSID['weight1_05'][~np.isnan(dfPSID['weight1_05'])])
w1_wgt_mean_psid_07 = np.average(a=dfPSID['wealth1_07'][~np.isnan(dfPSID['wealth1_07'])], weights=dfPSID['weight1_07'][~np.isnan(dfPSID['weight1_07'])])
w1_wgt_mean_psid_09 = np.average(a=dfPSID['wealth1_09'][~np.isnan(dfPSID['wealth1_09'])], weights=dfPSID['weight1_09'][~np.isnan(dfPSID['weight1_09'])])
w1_wgt_mean_psid_11 = np.average(a=dfPSID['wealth1_11'][~np.isnan(dfPSID['wealth1_11'])], weights=dfPSID['weight1_11'][~np.isnan(dfPSID['weight1_11'])])
w1_wgt_mean_psid_13 = np.average(a=dfPSID['wealth1_13'][~np.isnan(dfPSID['wealth1_13'])], weights=dfPSID['weight1_13'][~np.isnan(dfPSID['weight1_13'])])
w1_wgt_mean_psid_15 = np.average(a=dfPSID['wealth1_15'][~np.isnan(dfPSID['wealth1_15'])], weights=dfPSID['weight1_15'][~np.isnan(dfPSID['weight1_15'])])
w1_wgt_mean_psid_17 = np.average(a=dfPSID['wealth1_17'][~np.isnan(dfPSID['wealth1_17'])], weights=dfPSID['weight1_17'][~np.isnan(dfPSID['weight1_17'])])
w_wgt_mean_soep_02 = np.average(a=dfSOEP['wealth_02'][~np.isnan(dfSOEP['wealth_02'])], weights=dfSOEP['weight_02'][~np.isnan(dfSOEP['wealth_02'])])
w_wgt_mean_soep_07 = np.average(a=dfSOEP['wealth_07'][~np.isnan(dfSOEP['wealth_07'])], weights=dfSOEP['weight_07'][~np.isnan(dfSOEP['wealth_07'])])
w_wgt_mean_soep_12 = np.average(a=dfSOEP['wealth_12'][~np.isnan(dfSOEP['wealth_12'])], weights=dfSOEP['weight_12'][~np.isnan(dfSOEP['wealth_12'])])
# w_wgt_mean_soep_17 = np.average(a=dfPSID['wealth_17'][~np.isnan(dfPSID['wealth_17'])], weights=dfPSID['weight_17'][~np.isnan(dfPSID['wealth_17'])])

# weighted sd
w1_wgt_sd_psid_01 = np.sqrt(np.cov(dfPSID['wealth1_01'][~np.isnan(dfPSID['wealth1_01'])], aweights=dfPSID['weight1_01'][~np.isnan(dfPSID['weight1_01'])]))
w1_wgt_sd_psid_03 = np.sqrt(np.cov(dfPSID['wealth1_03'][~np.isnan(dfPSID['wealth1_03'])], aweights=dfPSID['weight1_03'][~np.isnan(dfPSID['weight1_03'])]))
w1_wgt_sd_psid_05 = np.sqrt(np.cov(dfPSID['wealth1_05'][~np.isnan(dfPSID['wealth1_05'])], aweights=dfPSID['weight1_05'][~np.isnan(dfPSID['weight1_05'])]))
w1_wgt_sd_psid_07 = np.sqrt(np.cov(dfPSID['wealth1_07'][~np.isnan(dfPSID['wealth1_07'])], aweights=dfPSID['weight1_07'][~np.isnan(dfPSID['weight1_07'])]))
w1_wgt_sd_psid_09 = np.sqrt(np.cov(dfPSID['wealth1_09'][~np.isnan(dfPSID['wealth1_09'])], aweights=dfPSID['weight1_09'][~np.isnan(dfPSID['weight1_09'])]))
w1_wgt_sd_psid_11 = np.sqrt(np.cov(dfPSID['wealth1_11'][~np.isnan(dfPSID['wealth1_11'])], aweights=dfPSID['weight1_11'][~np.isnan(dfPSID['weight1_11'])]))
w1_wgt_sd_psid_13 = np.sqrt(np.cov(dfPSID['wealth1_13'][~np.isnan(dfPSID['wealth1_13'])], aweights=dfPSID['weight1_13'][~np.isnan(dfPSID['weight1_13'])]))
w1_wgt_sd_psid_15 = np.sqrt(np.cov(dfPSID['wealth1_15'][~np.isnan(dfPSID['wealth1_15'])], aweights=dfPSID['weight1_15'][~np.isnan(dfPSID['weight1_15'])]))
w1_wgt_sd_psid_17 = np.sqrt(np.cov(dfPSID['wealth1_17'][~np.isnan(dfPSID['wealth1_17'])], aweights=dfPSID['weight1_17'][~np.isnan(dfPSID['weight1_17'])]))
w_wgt_sd_soep_07 = np.sqrt(np.cov(dfSOEP['wealth_07'][~np.isnan(dfSOEP['wealth_07'])], aweights=dfSOEP['weight_07'][~np.isnan(dfSOEP['wealth_07'])]))
w_wgt_sd_soep_02 = np.sqrt(np.cov(dfSOEP['wealth_02'][~np.isnan(dfSOEP['wealth_02'])], aweights=dfSOEP['weight_02'][~np.isnan(dfSOEP['wealth_02'])]))
w_wgt_sd_soep_12 = np.sqrt(np.cov(dfSOEP['wealth_12'][~np.isnan(dfSOEP['wealth_12'])], aweights=dfSOEP['weight_12'][~np.isnan(dfSOEP['wealth_12'])]))
# w_wgt_sd_soep_17 = np.sqrt(np.cov(dfSOEP['wealth_17'][~np.isnan(dfSOEP['wealth_17'])], aweights=dfSOEP['weight_17'][~np.isnan(dfSOEP['wealth_17'])]))

# weighted pcts
w1_wgt_pcts_psid_01 = weighted_quantile(dfPSID['wealth1_01'][~np.isnan(dfPSID['wealth1_01'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_01'][~np.isnan(dfPSID['weight1_01'])])
w1_wgt_pcts_psid_03 = weighted_quantile(dfPSID['wealth1_03'][~np.isnan(dfPSID['wealth1_03'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_03'][~np.isnan(dfPSID['weight1_03'])])
w1_wgt_pcts_psid_05 = weighted_quantile(dfPSID['wealth1_05'][~np.isnan(dfPSID['wealth1_05'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_05'][~np.isnan(dfPSID['weight1_05'])])
w1_wgt_pcts_psid_07 = weighted_quantile(dfPSID['wealth1_07'][~np.isnan(dfPSID['wealth1_07'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_07'][~np.isnan(dfPSID['weight1_07'])])
w1_wgt_pcts_psid_09 = weighted_quantile(dfPSID['wealth1_09'][~np.isnan(dfPSID['wealth1_09'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_09'][~np.isnan(dfPSID['weight1_09'])])
w1_wgt_pcts_psid_11 = weighted_quantile(dfPSID['wealth1_11'][~np.isnan(dfPSID['wealth1_11'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_11'][~np.isnan(dfPSID['weight1_11'])])
w1_wgt_pcts_psid_13 = weighted_quantile(dfPSID['wealth1_13'][~np.isnan(dfPSID['wealth1_13'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_13'][~np.isnan(dfPSID['weight1_13'])])
w1_wgt_pcts_psid_15 = weighted_quantile(dfPSID['wealth1_15'][~np.isnan(dfPSID['wealth1_15'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_15'][~np.isnan(dfPSID['weight1_15'])])
w1_wgt_pcts_psid_17 = weighted_quantile(dfPSID['wealth1_17'][~np.isnan(dfPSID['wealth1_17'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_17'][~np.isnan(dfPSID['weight1_17'])])
w_wgt_pcts_soep_02 = weighted_quantile(dfSOEP['wealth_02'][~np.isnan(dfSOEP['wealth_02'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfSOEP['weight_02'][~np.isnan(dfSOEP['wealth_02'])])
w_wgt_pcts_soep_07 = weighted_quantile(dfSOEP['wealth_07'][~np.isnan(dfSOEP['wealth_07'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfSOEP['weight_07'][~np.isnan(dfSOEP['wealth_07'])])
w_wgt_pcts_soep_12 = weighted_quantile(dfSOEP['wealth_12'][~np.isnan(dfSOEP['wealth_12'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfSOEP['weight_12'][~np.isnan(dfSOEP['wealth_12'])])
# w_wgt_pcts_soep_17 = weighted_quantile(dfSOEP['wealth_17'][~np.isnan(dfSOEP['wealth_17'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfSOEP['weight_17'][~np.isnan(dfSOEP['wealth_17'])])


df_weighted_descriptives_w1 = pd.DataFrame(np.array([['N', int(dfPSID['weight1_01'].sum()), int(dfPSID['weight1_03'].sum()), int(dfPSID['weight1_05'].sum()), int(dfPSID['weight1_07'].sum()), int(dfPSID['weight1_09'].sum()), int(dfPSID['weight1_11'].sum()), int(dfPSID['weight1_13'].sum()), int(dfPSID['weight1_15'].sum()), int(dfPSID['weight1_17'].sum()), int(dfSOEP['weight_02'].sum()), int(dfSOEP['weight_07'].sum()), int(dfSOEP['weight_12'].sum())],
                                                       ['mean', int(w1_wgt_mean_psid_01), int(w1_wgt_mean_psid_03), int(w1_wgt_mean_psid_05), int(w1_wgt_mean_psid_07), int(w1_wgt_mean_psid_09), int(w1_wgt_mean_psid_11), int(w1_wgt_mean_psid_13), int(w1_wgt_mean_psid_15), int(w1_wgt_mean_psid_17), int(w_wgt_mean_soep_02), int(w_wgt_mean_soep_07), int(w_wgt_mean_soep_12)],
                                                       ['sd', int(w1_wgt_sd_psid_01), int(w1_wgt_sd_psid_03), int(w1_wgt_sd_psid_05), int(w1_wgt_sd_psid_07), int(w1_wgt_sd_psid_09), int(w1_wgt_sd_psid_11), int(w1_wgt_sd_psid_13), int(w1_wgt_sd_psid_15), int(w1_wgt_sd_psid_17), int(w_wgt_sd_soep_02), int(w_wgt_sd_soep_07), int(w_wgt_sd_soep_12)],
                                                       ['min', int(dfPSID['wealth1_01'].min()), int(dfPSID['wealth1_03'].min()), int(dfPSID['wealth1_05'].min()), int(dfPSID['wealth1_07'].min()), int(dfPSID['wealth1_09'].min()), int(dfPSID['wealth1_11'].min()), int(dfPSID['wealth1_13'].min()), int(dfPSID['wealth1_15'].min()), int(dfPSID['wealth1_17'].min()), int(dfSOEP['wealth_02'].min()), int(dfSOEP['wealth_07'].min()), int(dfSOEP['wealth_12'].min())],
                                                       ['p50', int(w1_wgt_pcts_psid_01[0]), int(w1_wgt_pcts_psid_03[0]), int(w1_wgt_pcts_psid_05[0]), int(w1_wgt_pcts_psid_07[0]), int(w1_wgt_pcts_psid_09[0]), int(w1_wgt_pcts_psid_11[0]), int(w1_wgt_pcts_psid_13[0]), int(w1_wgt_pcts_psid_15[0]), int(w1_wgt_pcts_psid_17[0]), int(w_wgt_pcts_soep_02[0]), int(w_wgt_pcts_soep_07[0]), int(w_wgt_pcts_soep_12[0])],
                                                       ['p75', int(w1_wgt_pcts_psid_01[1]), int(w1_wgt_pcts_psid_03[1]), int(w1_wgt_pcts_psid_05[1]), int(w1_wgt_pcts_psid_07[1]), int(w1_wgt_pcts_psid_09[1]), int(w1_wgt_pcts_psid_11[1]), int(w1_wgt_pcts_psid_13[1]), int(w1_wgt_pcts_psid_15[1]), int(w1_wgt_pcts_psid_17[1]), int(w_wgt_pcts_soep_02[1]), int(w_wgt_pcts_soep_07[1]), int(w_wgt_pcts_soep_12[1])],
                                                       ['p90', int(w1_wgt_pcts_psid_01[2]), int(w1_wgt_pcts_psid_03[2]), int(w1_wgt_pcts_psid_05[2]), int(w1_wgt_pcts_psid_07[2]), int(w1_wgt_pcts_psid_09[2]), int(w1_wgt_pcts_psid_11[2]), int(w1_wgt_pcts_psid_13[2]), int(w1_wgt_pcts_psid_15[2]), int(w1_wgt_pcts_psid_17[2]), int(w_wgt_pcts_soep_02[2]), int(w_wgt_pcts_soep_07[2]), int(w_wgt_pcts_soep_12[2])],
                                                       ['p99', int(w1_wgt_pcts_psid_01[3]), int(w1_wgt_pcts_psid_03[3]), int(w1_wgt_pcts_psid_05[3]), int(w1_wgt_pcts_psid_07[3]), int(w1_wgt_pcts_psid_09[3]), int(w1_wgt_pcts_psid_11[3]), int(w1_wgt_pcts_psid_13[3]), int(w1_wgt_pcts_psid_15[3]), int(w1_wgt_pcts_psid_17[3]), int(w_wgt_pcts_soep_02[3]), int(w_wgt_pcts_soep_07[3]), int(w_wgt_pcts_soep_12[3])],
                                                       ['p99.9', int(w1_wgt_pcts_psid_01[4]), int(w1_wgt_pcts_psid_03[4]), int(w1_wgt_pcts_psid_05[4]), int(w1_wgt_pcts_psid_07[4]), int(w1_wgt_pcts_psid_09[4]), int(w1_wgt_pcts_psid_11[4]), int(w1_wgt_pcts_psid_13[4]), int(w1_wgt_pcts_psid_15[4]), int(w1_wgt_pcts_psid_17[4]), int(w_wgt_pcts_soep_02[4]), int(w_wgt_pcts_soep_07[4]), int(w_wgt_pcts_soep_12[4])],
                                                       ['max', int(dfPSID['wealth1_01'].max()), int(dfPSID['wealth1_03'].max()), int(dfPSID['wealth1_05'].max()), int(dfPSID['wealth1_07'].max()), int(dfPSID['wealth1_09'].max()), int(dfPSID['wealth1_11'].max()), int(dfPSID['wealth1_13'].max()), int(dfPSID['wealth1_15'].max()), int(dfPSID['wealth1_17'].max()), int(dfSOEP['wealth_02'].max()), int(dfSOEP['wealth_07'].max()), int(dfSOEP['wealth_12'].max())],
                                                       ]),
                                             columns=['', '2001', '2003', '2005', '2007', '2009', '2011', '2013', '2015', '2017', '2002SOEP', '2007SOEP', '2012SOEP'])



# weighted, psid: wealth1, soep: wealth

# weighted avg
w2_wgt_mean_psid_01 = np.average(a=dfPSID['wealth2_01'][~np.isnan(dfPSID['wealth2_01'])], weights=dfPSID['weight1_01'][~np.isnan(dfPSID['weight1_01'])])
w2_wgt_mean_psid_03 = np.average(a=dfPSID['wealth2_03'][~np.isnan(dfPSID['wealth2_03'])], weights=dfPSID['weight1_03'][~np.isnan(dfPSID['weight1_03'])])
w2_wgt_mean_psid_05 = np.average(a=dfPSID['wealth2_05'][~np.isnan(dfPSID['wealth2_05'])], weights=dfPSID['weight1_05'][~np.isnan(dfPSID['weight1_05'])])
w2_wgt_mean_psid_07 = np.average(a=dfPSID['wealth2_07'][~np.isnan(dfPSID['wealth2_07'])], weights=dfPSID['weight1_07'][~np.isnan(dfPSID['weight1_07'])])
w2_wgt_mean_psid_09 = np.average(a=dfPSID['wealth2_09'][~np.isnan(dfPSID['wealth2_09'])], weights=dfPSID['weight1_09'][~np.isnan(dfPSID['weight1_09'])])
w2_wgt_mean_psid_11 = np.average(a=dfPSID['wealth2_11'][~np.isnan(dfPSID['wealth2_11'])], weights=dfPSID['weight1_11'][~np.isnan(dfPSID['weight1_11'])])
w2_wgt_mean_psid_13 = np.average(a=dfPSID['wealth2_13'][~np.isnan(dfPSID['wealth2_13'])], weights=dfPSID['weight1_13'][~np.isnan(dfPSID['weight1_13'])])
w2_wgt_mean_psid_15 = np.average(a=dfPSID['wealth2_15'][~np.isnan(dfPSID['wealth2_15'])], weights=dfPSID['weight1_15'][~np.isnan(dfPSID['weight1_15'])])
w2_wgt_mean_psid_17 = np.average(a=dfPSID['wealth2_17'][~np.isnan(dfPSID['wealth2_17'])], weights=dfPSID['weight1_17'][~np.isnan(dfPSID['weight1_17'])])

# weighted sd
w2_wgt_sd_psid_01 = np.sqrt(np.cov(dfPSID['wealth2_01'][~np.isnan(dfPSID['wealth2_01'])], aweights=dfPSID['weight1_01'][~np.isnan(dfPSID['weight1_01'])]))
w2_wgt_sd_psid_03 = np.sqrt(np.cov(dfPSID['wealth2_03'][~np.isnan(dfPSID['wealth2_03'])], aweights=dfPSID['weight1_03'][~np.isnan(dfPSID['weight1_03'])]))
w2_wgt_sd_psid_05 = np.sqrt(np.cov(dfPSID['wealth2_05'][~np.isnan(dfPSID['wealth2_05'])], aweights=dfPSID['weight1_05'][~np.isnan(dfPSID['weight1_05'])]))
w2_wgt_sd_psid_07 = np.sqrt(np.cov(dfPSID['wealth2_07'][~np.isnan(dfPSID['wealth2_07'])], aweights=dfPSID['weight1_07'][~np.isnan(dfPSID['weight1_07'])]))
w2_wgt_sd_psid_09 = np.sqrt(np.cov(dfPSID['wealth2_09'][~np.isnan(dfPSID['wealth2_09'])], aweights=dfPSID['weight1_09'][~np.isnan(dfPSID['weight1_09'])]))
w2_wgt_sd_psid_11 = np.sqrt(np.cov(dfPSID['wealth2_11'][~np.isnan(dfPSID['wealth2_11'])], aweights=dfPSID['weight1_11'][~np.isnan(dfPSID['weight1_11'])]))
w2_wgt_sd_psid_13 = np.sqrt(np.cov(dfPSID['wealth2_13'][~np.isnan(dfPSID['wealth2_13'])], aweights=dfPSID['weight1_13'][~np.isnan(dfPSID['weight1_13'])]))
w2_wgt_sd_psid_15 = np.sqrt(np.cov(dfPSID['wealth2_15'][~np.isnan(dfPSID['wealth2_15'])], aweights=dfPSID['weight1_15'][~np.isnan(dfPSID['weight1_15'])]))
w2_wgt_sd_psid_17 = np.sqrt(np.cov(dfPSID['wealth2_17'][~np.isnan(dfPSID['wealth2_17'])], aweights=dfPSID['weight1_17'][~np.isnan(dfPSID['weight1_17'])]))

# weighted pcts
w2_wgt_pcts_psid_01 = weighted_quantile(dfPSID['wealth2_01'][~np.isnan(dfPSID['wealth2_01'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_01'][~np.isnan(dfPSID['weight1_01'])])
w2_wgt_pcts_psid_03 = weighted_quantile(dfPSID['wealth2_03'][~np.isnan(dfPSID['wealth2_03'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_03'][~np.isnan(dfPSID['weight1_03'])])
w2_wgt_pcts_psid_05 = weighted_quantile(dfPSID['wealth2_05'][~np.isnan(dfPSID['wealth2_05'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_05'][~np.isnan(dfPSID['weight1_05'])])
w2_wgt_pcts_psid_07 = weighted_quantile(dfPSID['wealth2_07'][~np.isnan(dfPSID['wealth2_07'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_07'][~np.isnan(dfPSID['weight1_07'])])
w2_wgt_pcts_psid_09 = weighted_quantile(dfPSID['wealth2_09'][~np.isnan(dfPSID['wealth2_09'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_09'][~np.isnan(dfPSID['weight1_09'])])
w2_wgt_pcts_psid_11 = weighted_quantile(dfPSID['wealth2_11'][~np.isnan(dfPSID['wealth2_11'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_11'][~np.isnan(dfPSID['weight1_11'])])
w2_wgt_pcts_psid_13 = weighted_quantile(dfPSID['wealth2_13'][~np.isnan(dfPSID['wealth2_13'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_13'][~np.isnan(dfPSID['weight1_13'])])
w2_wgt_pcts_psid_15 = weighted_quantile(dfPSID['wealth2_15'][~np.isnan(dfPSID['wealth2_15'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_15'][~np.isnan(dfPSID['weight1_15'])])
w2_wgt_pcts_psid_17 = weighted_quantile(dfPSID['wealth2_17'][~np.isnan(dfPSID['wealth2_17'])], quantiles= [.5, .75, .9, .99, .999],sample_weight=dfPSID['weight1_17'][~np.isnan(dfPSID['weight1_17'])])

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

#test
Paretobranchfit(x=dfPSID['wealth1_01'], weights=dfPSID['weight1_01'],
                b=1000000, x0=(-1, .5, 1, 1), bootstraps=(10, 10, 10, 10))


### PSID
for year in ['13', '15', '17']:

    # write temp variables names
    wealth = 'wealth1_' + year
    weight = 'weight1_' + year
    print(dfPSID[wealth].size)
    print(dfPSID[weight].size)
    data = dfPSID[wealth]
    # wgt = int(dfPSID[weight])
    wgt = pd.to_numeric(dfPSID[weight], downcast='signed')
    print(wgt)

    globals()['fit_results_psid_%s' % year] = Paretobranchfit(x=data, weights=wgt, b=1000000, x0=(-1, .5, 1, 1), bootstraps=(10, 10, 10, 10))



### SOEP
for year in ['12', '17']:

    # write temp variables names
    wealth = 'wealth_' + year
    weight = 'weight_' + year
    print(dfSOEP[wealth].size)
    print(dfSOEP[weight].size)
    data = dfSOEP[wealth]
    # wgt = int(dfSOEP[weight])
    wgt = pd.to_numeric(dfSOEP[weight], downcast='signed')
    print(wgt)

    globals()['fit_results_psid_%s' % year] = Paretobranchfit(x=data, weights=wgt, b=1000000, x0=(-1, .5, 1, 1), bootstraps=(10, 10, 10, 10))



"""
-----------------------------
Plot fit vs data
-----------------------------
"""


