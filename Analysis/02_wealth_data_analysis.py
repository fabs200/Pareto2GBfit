from Pareto2GBfit.fitting import *
import numpy as np
from scipy.stats import describe
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

# TODO: SOEP 2017

"""
-------------------------
define functions
-------------------------
"""
def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights. (source: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy)
    NOTE: quantiles should be in [0, 1]!
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
data_PSID = "/Users/Fabian/OneDrive/Studium/Masterarbeit/data/J261520/J261520.csv"
dfPSID = pd.read_csv(data_PSID, delimiter = ";", skiprows=False, decimal=',')

# renaming
columns = {"ER17001": "release_01", "ER17002": "famid_01", "ER20394": "weight1_01", "ER20459": "weight2_01",
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

dfPSID = dfPSID.rename(index=str, columns=columns)

# multiply longitudinal weights by 1000 as described in the documentation of PSID
psid_longi_weights = ['weight1_01', 'weight1_03', 'weight1_05', 'weight1_07', 'weight1_09', 'weight1_11', 'weight1_13', 'weight1_15', 'weight1_17']
for weight in psid_longi_weights:
    dfPSID[weight] = dfPSID[weight].apply(lambda x: x*1000)
    # print(weight[-2:], dfPSID[weight].sum()) #TODO: correct family N?


"""
-----------------------
SOEP data preparation
-----------------------
"""

data_SOEPwealth = '/Users/Fabian/Documents/DATA/STATA/SOEP_v33.1/SOEP_wide/hwealth.dta'
data_SOEPHHweight = '/Users/Fabian/Documents/DATA/STATA/SOEP_v33.1/SOEP_wide/hhrf.dta'

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
# dfSOEP_wealth.head()
# dfSOEP_hhweights.head()
# dfSOEP.head()


