# Pareto2GBfit

This small package provides distributions and functions to fit 4 of the
Generalized Beta distribution family. The theoretical framework bases on
the paper by McDonald, J. B. and Xu, Y. J. (1995) ‘A generalization of
the beta distribution with applications’ (Journal of Econometrics,
66(1), pp. 133–152). With this package one can test the equality of the
parameters in the GB tree when we focus on the Pareto branch.

GB tree: 

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/GBtree.jpg/1280px-GBtree.jpg" width="600">

(Source: Wikipedia)

## Requirements
Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `scipy`
- `matplotlib`
- `progressbar`
- `prettytable`


## Distributions

Following functions are implemented:

|        	| pdf                        	| cdf                                                  	| icdf                                             	| Jacobian              	| Hessian                	|
|--------	|----------------------------	|------------------------------------------------------	|--------------------------------------------------	|-----------------------	|------------------------	|
| Pareto 	| `Pareto_pdf(x, b, p)`      	| `Pareto_cdf(x, b, p)` `Pareto_cdf_ne(x, b, p)`       	| `Pareto_icdf(u, b, p)` `Pareto_icdf_ne(x, b, p)` 	| `Pareto_jac(x, b, p)` 	| `Pareto_hess(x, b, p)` 	|
| IB1    	| `IB1_pdf(x, b, p, q)`      	| `IB1_cdf(x, b, p, q)`                                	| `IB1_icdf_ne(x, b, p, q)`                        	| `IB1_jac(x, b, p, q)` 	| --                     	|
| GB1    	| `GB1_pdf(x, a, b, p, q)`   	| `GB1_cdf(x, a, b, p, q)` `GB1_cdf_ne(x, a, b, p, q)` 	| `GB1_icdf_ne(x, a, b, p, q)`                     	| --                    	| --                     	|
| GB     	| `GB_pdf(x, a, b, c, p, q)` 	| `GB_cdf_ne(x, a, b, c, p, q)`                        	| `GB_icdf_ne(x, a, b, c, p, q)`                   	| --                    	| --                     	|


## Fitting

To fit the distributions, the package provides following functions:

|        	| fit  &nbsp; &nbsp; &nbsp; |
|--------	|--------------------------	|
| Pareto 	| `Paretofit(x, b, x0, ...)`|
| IB1    	| `IB1fit(x, b, x0, ...)`   |
| GB1    	| `GB1fit(x, b, x0, ...)`   |
| GB     	| `GBfit(x, b, x0, ...)`    |

with following options:

| arg                    | default                                                                                                                                                                 | description                                                                                                                                                                                                                                                                                                                                                         |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|           `x`          |                                                                                    --                                                                                   | actual data, will be converted to a numpy.array                                                                                                                                                                                                                                                                                                                     |
|           `b`          |                                                                                    --                                                                                   | lower bound                                                                                                                                                                                                                                                                                                                                                         |
|        `weights`       |                                                                              `array([1., 1.,...])`                                                                      | array with frequency weights, default is a `numpy.ones` array of same shape as `x`. Note that that each observation in x is "inflated" w-times. Thus, all elements in `weights` needs to be integer, and, ideally `numpy.int64` to prevent loss of precision.                                                                                                                    |
|      `bootstraps`      |                                                                              `500` (`250` for GB1, GB)                                                                  | number of bootstraps, for Pareto and IB1 default `=500`, for GB1, GB `=250`, the more parameters one need to optimize, to more time-consuming will the optimization take. Thus, less bootstraps are set as default.                                                                                                                                                 |
|        `method`        |                                                                                 `SLSQP`                                                                                 | either run `SLSQP` (= local optimization with bounds, constraints, much faster), or chose 'L-BFGS-B' which bases on Basin-Hopping (=global optimization) . Note that depending on the selected method different optimization options are available (see [SciPy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)) |
|   `verbose_bootstrap`  |                                                                                 `False`                                                                                 | display each bootstrap round                                                                                                                                                                                                                                                                                                                                        |
|          `ci`          |                                                                                  `True`                                                                                 | display fitted parameters with 95th confidence intervals                                                                                                                                                                                                                                                                                                            |
|        `verbose`       |                                                                                  `True`                                                                                 | display any results, if set `False`, no output is printed                                                                                                                                                                                                                                                                                                           |
|          `fit`         |                                                                                 `False`                                                                                 | display goodness of fit measures in a table (aic, bic, mae, mse, rmse, rrmse, ll, n)                                                                                                                                                                                                                                                                                |
|         `plot`         |                                                                                 `False`                                                                                 | If `True`, a graph of the fit will be plottet, graphical cosmetics can be adjusted with the dictionary `plot_cosmetics`                                                                                                                                                                                                                                             |
|   `return_parameters`  |                                                                                 `False`                                                                                 | If `True`, fitted parameters with standard errors are returned. E.g. `Paretofit(...)` would return the 1x2-array: `=[p_fit, p_se]`, `IB1fit(...)` 1x4-array: `=[p_fit, p_se, q_fit, p_se]`, etc.                                                                                                                                                                    |
|      `return_gof`      |                                                                                 `False`                                                                                 | If `True`, goodness of fit measures are returned. 1x8-array: `=[aic, bic, mae, mse, rmse, rrmse, ll, n]`                                                                                                                                                                                                                                                            |
|    `plot_cosmetics`    | `{'bins': 50, 'col_fit': 'blue', 'col_model': 'orange'}`                                                                                                                | Specify bins by adding the dictionary `plot_cosmetics={'bins': 250}`                                                                                                                                                                                                                                                                                                |
| `basinhopping_options` | `{'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None, 'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123}` | if `method='basinhopping'`, the user can specify arguments to the optimizer which are then passed to `scipy.optimize.basinhopping`. For further information, refer to [SciPy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping).                                                   |
|     `SLSQP_options`    | `{'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}`                  | if `method='SLSQP'`, the user can specify arguments to the optimizer which are then passed to `scipy.optimize.minimize(method='SLSQP', ...)`. For further information, refer to [SciPy's documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).                                                                                   |

Also you can fit the whole Pareto branch with following function:

|            	| fit &nbsp; &nbsp; &nbsp; &nbsp; |
|---------------|---------------------------------|
| Pareto to GB 	| `Paretobranchfit(x, b, x0, ...)`|

This is a wrapper that applies all fit functions above in a row (from Paretofit to GBfit), saves all parameters and GOFs and finally evaluates the fit according the specified rejection criteria (rejection_criteria='LRtest' or `rejection_criteria='AIC'`).
Options:

| arg                    | default                                                                                                                                                                 | description                                                         |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| `x`                    | --                                                                                                                                                                      | see above                                                           |
| `b`                    | --                                                                                                                                                                      | see above                                                           |
| `weights`              | `array([1., 1.,...])`                                                                                                                                                   | see above                                                           |
| `rejection_criteria`   | `LRtest`                                                                                                                                                                |                                                                     |
| `alpha`                | `.05`                                                                                                                                                                   | significance level of LR test                                       |
| `bootstraps`           | `250`                                                                                                                                                                   | all distribution have the default value of `boostraps=250`          |
| `method`               | `SLSQP`                                                                                                                                                                 | see above                                                           |
| `verbose_bootstrap`    | `False`                                                                                                                                                                 | see above                                                           |
| `verbose_single`       | `False`                                                                                                                                                                 | if `True` the output of each fit is displayed                       |
| `verbose`              | `True`                                                                                                                                                                  | rejection summary, table with fitted parameters and gofs are listed |
| `fit`                  | `False`                                                                                                                                                                 | see above                                                           |
| `plot`                 | `False`                                                                                                                                                                 | see above                                                           |
| `return_bestmodel`     | `False`                                                                                                                                                                 | if `True` best model's parameters and gofs are returned             |
| `return_all`           | `False`                                                                                                                                                                 | if `True` all model's parameters and gofs are returned              |
| `return_gof`           | `False`                                                                                                                                                                 | not used                                                            |
| `plot_cosmetics`       | `{'bins': 50, 'col_fit': 'blue', 'col_model': 'orange'}`                                                                                                                | see above                                                           |
| `basinhopping_options` | `{'niter': 20, 'T': 1.0, 'stepsize': 0.5, 'take_step': None, 'accept_test': None, 'callback': None, 'interval': 50, 'disp': False, 'niter_success': None, 'seed': 123}` | see above                                                           |
| `SLSQP_options`        | {'jac': None, 'tol': None, 'callback': None, 'func': None, 'maxiter': 300, 'ftol': 1e-14, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08}                    | see above                                                           |


## Examples

### Example 1

Lets generate a pareto distributed dataset for which we know the true parameters. Then, we run the `Paretofit()`-function and expect that the model with the true parameters will be fitted to the data.

1. Load packages
```
import numpy as np
from Pareto2GBfit.fitting import *
```
2. Specify parameters for synthetic dataset
```
b, p = 500, 2.5
```
3. Linspace
```
n = 10000
xmin = 0.1
xmax = 10000
x = np.linspace(xmin, xmax, n)
```
4. Noise
```
mu = 0
sigma = 100
random.seed(123)
noise = np.random.normal(mu, sigma, size=n)
```
5. Generate synthetic (noised) dataset, e.g. that is pareto distributed
```
data = Pareto_icdf(u, b, p)
data_noised = Pareto_icdf(u, b, p) + noise
```
6. Run optimization for data
```
Paretofit(x=data, b=500, x0=2, bootstraps=1000, method='SLSQP')
```
this returns following output:
```
Bootstrapping 100%|##############################################|Time: 0:00:04
+-----------+--------+--------+---------+--------+---------+---------+-------+
| parameter | value  |   se   |    z    | P>|z|  | cilo_95 | cihi_95 |   n   |
+-----------+--------+--------+---------+--------+---------+---------+-------+
|     p     | 2.4764 | 0.0247 | 100.457 | 0.0000 |  2.4281 |  2.5247 | 10000 |
+-----------+--------+--------+---------+--------+---------+---------+-------+
```
7. Run optimization for data_noised
```
Paretofit(x=data_noise, b=500, x0=2, bootstraps=1000, method='SLSQP')
```
Output:
```
Bootstrapping 100%|##############################################|Time: 0:00:03
+-----------+--------+--------+----------+--------+---------+---------+------+
| parameter | value  |   se   |    z     | P>|z|  | cilo_95 | cihi_95 |  n   |
+-----------+--------+--------+----------+--------+---------+---------+------+
|     p     | 2.0992 | 0.0183 | 114.4548 | 0.0000 |  2.0633 |  2.1352 | 8678 |
+-----------+--------+--------+----------+--------+---------+---------+------+
```
Note that die observations of `data_noise` reduced from n=10000 to
n=8678. This is due to the gaussian noise and Pareto constraint x>b.

### Example 2

Lets load the netwealth-dataset and fit the Pareto- and the IB1-distribution to the data.
Then, we test the equality of the shared parameter p.

1. Load packages
```
from Pareto2GBfit.fitting import *
import numpy as np
from scipy.stats import describe
```
2. Load dataset
```
netwealth = np.loadtxt("netwealth.csv", delimiter = ",")
```
3. Describe dataset
```
describe(netwealth)
```
this returns following:
```
DescribeResult(nobs=28072, minmax=(-4434000.0, 207020000.0), mean=142245.32003419776, variance=6263377257629.95, skewness=69.73674629459225, kurtosis=5340.710623236435)
```

4. Lets fit the Pareto distribution to the data
```
Paretofit(x=netwealth, b=100000, x0=1, bootstraps=1000, method='SLSQP', fit=True, plot=True, plot_cosmetics={'bins': 500})
```
Output:
```
Bootstrapping 100%|##############################################|Time: 0:00:03
+-----------+--------+--------+---------+--------+---------+---------+------+
| parameter | value  |   se   |    z    | P>|z|  | cilo_95 | cihi_95 |  n   |
+-----------+--------+--------+---------+--------+---------+---------+------+
|     p     | 1.2003 | 0.0127 | 94.5339 | 0.0000 |  1.1755 |  1.2252 | 7063 |
+-----------+--------+--------+---------+--------+---------+---------+------+
+-----+-------------+-------------+------------+--------------------+---------------+--------+-------------+------+
|     |     AIC     |     BIC     |    MAE     |        MSE         |      RMSE     | RRMSE  |      LL     |  n   |
+-----+-------------+-------------+------------+--------------------+---------------+--------+-------------+------+
| GOF | 185948.2486 | 185955.1112 | 793147.028 | 551367829496230.94 | 23481222.9131 | 4.6038 | -92973.1243 | 7063 |
+-----+-------------+-------------+------------+--------------------+---------------+--------+-------------+------+

```
<img src="Figure_1.png" width="400">
Note: In the plot window, you have the option to zoom in.

5. Lets go one parameter level upwards in the GB-tree and fit the IB1 distribution
```
IB1fit(x=netwealth, b=100000, x0=(1,1), bootstraps=1000, method='SLSQP', fit=True, plot=True, plot_cosmetics={'bins': 500})
```
Output:
```
Bootstrapping 100%|##############################################|Time: 0:00:14
+-----------+--------+--------+---------+--------+---------+---------+------+
| parameter | value  |   se   |    z    | P>|z|  | cilo_95 | cihi_95 |  n   |
+-----------+--------+--------+---------+--------+---------+---------+------+
|     p     | 1.4464 | 0.0254 | 57.0174 | 0.0000 |  1.3966 |  1.4961 | 7063 |
|     q     | 1.3022 | 0.0219 | 59.4884 | 0.0000 |  1.2593 |  1.3451 | 7063 |
+-----------+--------+--------+---------+--------+---------+---------+------+
+-----+-------------+-------------+-------------+--------------------+--------------+--------+------------+------+
|     |     AIC     |     BIC     |     MAE     |        MSE         |     RMSE     | RRMSE  |     LL     |  n   |
+-----+-------------+-------------+-------------+--------------------+--------------+--------+------------+------+
| GOF | 185686.9459 | 185700.6712 | 549988.3777 | 22899226860951.406 | 4785313.6638 | 4.0645 | -92841.473 | 7063 |
+-----+-------------+-------------+-------------+--------------------+--------------+--------+------------+------+

```
<img src="Figure_2.png" width="400">

6. Lets run the global optimization of the `IB1fit()` and compare this result to the local optimization in step 5.
```
IB1fit(x=netwealth, b=100000, x0=(1,1), bootstraps=1000, method='L-BFGS-B', fit=True, plot=True, plot_cosmetics={'bins': 500}, basinhopping_options={'niter': 50, 'stepsize': .75})
```
Output:
```
Bootstrapping 100%|##############################################|Time: 0:05:09
+-----------+--------+--------+---------+--------+---------+---------+------+
| parameter | value  |   se   |    z    | P>|z|  | cilo_95 | cihi_95 |  n   |
+-----------+--------+--------+---------+--------+---------+---------+------+
|     p     | 1.4479 | 0.0251 | 57.6374 | 0.0000 |  1.3987 |  1.4971 | 7063 |
|     q     | 1.3038 | 0.0223 | 58.5595 | 0.0000 |  1.2602 |  1.3475 | 7063 |
+-----------+--------+--------+---------+--------+---------+---------+------+
+-----+-------------+-------------+-------------+--------------------+--------------+--------+-------------+------+
|     |     AIC     |     BIC     |     MAE     |        MSE         |     RMSE     | RRMSE  |      LL     |  n   |
+-----+-------------+-------------+-------------+--------------------+--------------+--------+-------------+------+
| GOF | 185686.9583 | 185700.6836 | 541010.4802 | 22794194873280.457 | 4774326.6408 | 4.0531 | -92841.4792 | 7063 |
+-----+-------------+-------------+-------------+--------------------+--------------+--------+-------------+------+
```
Note that the global optimization process took about 5 mins compared to the local optimization with 14 seconds.
Indeed, both optimizations result in the same parameters.

<img src="Figure_3.png" width="400">

7. Save the fitted parameters, e.g. for Pareto, IB1, GB1
```
p_fit1, p_se1 = Paretofit(x=netwealth, b=100000, x0=1, bootstraps=250, method='SLSQP', verbose=False, return_parameters=True)
p_fit2, p_se2, q_fit2, q_se2 = IB1fit(x=netwealth, b=100000, x0=(1,1), bootstraps=250, method='SLSQP', verbose=False, return_parameters=True)
a_fit3, a_se3, p_fit3, p_se3, q_fit3, q_se3 = GB1fit(x=netwealth, b=100000, x0=(-0.5,1,1), bootstraps=250, method='SLSQP', verbose=False, return_parameters=True)
```

Output:
```
Bootstrapping 100%|##############################################|Time: 0:00:00
Bootstrapping 100%|##############################################|Time: 0:00:04
Bootstrapping 100%|##############################################|Time: 0:01:08
```

8. Finally, lets test the validity of the GB tree restriction
```
LRtest(Pareto(x=netwealth, b=b, p=p_fit1).LL, IB1(x=netwealth, b=b, p=p_fit2, q=q_fit2).LL, df=2)
```
Output:
```
+-------------+-----------+
|   LR test   |           |
+-------------+-----------+
|  chi2(2) =  | -263.3104 |
| Prob > chi2 |   1.0000  |
+-------------+-----------+
```
We highly cannot reject the null that the GB tree restriction q=1 is not valid.

```
LRtest(Pareto(x=netwealth, b=b, p=p_fit1).LL, GB1(x=netwealth, b=b, a=a_fit3, p=p_fit3, q=q_fit3).LL, df=3)
```
Output:
```
+-------------+----------+
|   LR test   |          |
+-------------+----------+
|  chi2(3) =  | 461.9817 |
| Prob > chi2 |  0.0000  |
+-------------+----------+
```
We can highly reject the null that the GB tree restrictions a=-1 and q=1 are valid.


```
LRtest(IB1(x=netwealth, b=b, p=p_fit2, q=q_fit2).LL, GB1(x=netwealth, b=b, a=a_fit3, p=p_fit3, q=q_fit3).LL, df=3)
```
Output:
```
+-------------+----------+
|   LR test   |          |
+-------------+----------+
|  chi2(3) =  | 725.2921 |
| Prob > chi2 |  0.0000  |
+-------------+----------+
```
We can highly reject the null that the GB tree restriction a=-1 is valid.

### Example 3

Lets test the validity of the Pareto branch with the `Paretobranchfit()` function.
1. Load packages
```
from Pareto2GBfit.fitting import *
import numpy as np
```
2. Load dataset
```
netwealth = np.loadtxt("netwealth.csv", delimiter = ",")
```
3. Apply the Paretobranchfit with the rejection criteria LR test by simply specifying `rejection_criteria='LRtest'`. (Note that the significance level: alpha=.05)
```
Paretobranchfit(x=netwealth, b=100000, x0=(-.1, .1, 1, 1), bootstraps=(100, 50, 10, 4), rejection_criteria='LRtest')
```
Output:
```
Bootstrapping 100%|##############################################|Time: 0:00:00
Bootstrapping 100%|##############################################|Time: 0:00:00
Bootstrapping 100%|##############################################|Time: 0:00:02
GB_cdf_ne  100%|>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>|Time: 0:00:18
Bootstrapping 100%|##############################################|Time: 0:00:04
GB_icdf_ne  99%|>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> |ETA:  0:00:00
+---------------+------+-------------+-----------+------+------------+
|   comparison  |  H0  |   LR test   |           | stop | best model |
+---------------+------+-------------+-----------+------+------------+
| Pareto vs IB1 | q=1  |  chi2(2) =  | -263.3015 |  --  |     --     |
|               |      | Prob > chi2 |   1.0000  |  --  |     --     |
|   IB1 vs GB1  | a=-1 |  chi2(3) =  |  282.1934 |  XX  |    IB1     |
|               |      | Prob > chi2 |   0.0000  |  XX  |    IB1     |
|   GB1 vs GB   | c=0  |  chi2(4) =  | -280.1505 |  --  |     --     |
|               |      | Prob > chi2 |   1.0000  |  --  |     --     |
+---------------+------+-------------+-----------+------+------------+
+-----------+---------+---------+---------+----------+
| parameter |  Pareto |   IB1   |   GB1   |    GB    |
+-----------+---------+---------+---------+----------+
|     a     |    -    |    -    |  -0.858 |  -1.126  |
|           |         |         | (0.344) | (-1.126) |
|     c     |    -    |    -    |    -    |  0.001   |
|           |         |         |         | (0.001)  |
|     p     |  1.202  |  1.445  |  0.910  |  1.305   |
|           | (0.012) | (0.024) | (1.307) | (0.183)  |
|     q     |    -    |  1.302  |  1.307  |  1.325   |
|           |         | (0.022) | (0.020) | (0.025)  |
+-----------+---------+---------+---------+----------+
+--------+------------+------------+------------+--------------------+-------------+-------+------------+----------------+------------+--------------------+------------+-------------------+----+------+------+
|        |    AIC     |    BIC     |    MAE     |        MSE         |     RMSE    | RRMSE |     LL     | sum of errors  | emp. mean  |     emp. var.      | pred. mean |     pred. var.    | df |  n   |  N   |
+--------+------------+------------+------------+--------------------+-------------+-------+------------+----------------+------------+--------------------+------------+-------------------+----+------+------+
| Pareto | 185948.258 | 185955.121 | 562589.445 | 39326144855435.81  |  6271056.12 | 4.578 | -92973.129 | -147115182.966 | 495360.824 | 21711786227141.043 | 516189.818 | 24693433986030.39 | 1  | 7063 | 7063 |
|  IB1   | 185686.957 | 185700.682 | 549969.043 | 22887341069249.562 |  4784071.6  | 4.091 | -92841.478 | -140210160.517 | 496338.458 |  753115151267.842  | 516189.818 | 24693433986030.39 | 2  | 7063 | 7063 |
|  GB1   | 185971.15  | 185991.738 | 472794.361 | 23400446738451.49  | 4837400.825 | 4.579 | -92982.575 | -912433299.364 | 387004.869 |  469527614265.387  | 516189.818 | 24693433986030.39 | 3  | 7063 | 7063 |
|   GB   |  185693.0  | 185720.45  | 534535.937 | 22940534834454.793 | 4789627.839 | 4.077 |  -92842.5  | -284552298.345 | 475902.08  |  630238966114.576  | 516189.818 | 24693433986030.39 | 4  | 7063 | 7063 |
+--------+------------+------------+------------+--------------------+-------------+-------+------------+----------------+------------+--------------------+------------+-------------------+----+------+------+
```
4. Lets check the AIC criteria by specifying `rejection_criteria='AIC'`
```
Paretobranchfit(x=netwealth, b=100000, x0=(-.1, .1, 1, 1), bootstraps=(100, 50, 10, 4), rejection_criteria='AIC', verbose=False)
```
Output:
```
Bootstrapping 100%|##############################################|Time: 0:00:00
Bootstrapping 100%|##############################################|Time: 0:00:00
Bootstrapping 100%|##############################################|Time: 0:00:02
GB_cdf_ne  100%|>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>|Time: 0:00:17
Bootstrapping 100%|##############################################|Time: 0:00:03
GB_icdf_ne  99%|>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> |ETA:  0:00:00
+---------------+------------+------------+------+------------+
|   comparison  |    AIC1    |    AIC2    | stop | best model |
+---------------+------------+------------+------+------------+
| Pareto vs IB1 | 185948.255 | 185686.985 |  --  |     --     |
|   IB1 vs GB1  | 185686.985 | 187289.702 |  XX  |    IB1     |
|   GB1 vs GB   | 187289.702 | 189292.461 |  --  |     --     |
+---------------+------------+------------+------+------------+
```

Here we see that the results are both equivalent - the IB1 fits the data best.


### Author
Fabian Nemeczek, Freie Universität Berlin

*Acknowledgment*
A big thanks to Dr. Johannes König (DIW Berlin) for the great support.
