## Description

This small package provides distributions and functions to fit 4 of the
Generalized Beta distribution family. The theoretical framework bases on
the paper by McDonald, J. B. and Xu, Y. J. (1995) ‘A generalization of
the beta distribution with applications’ (Journal of Econometrics,
66(1), pp. 133–152). With this package one can test the equality of the
parameters in the GB tree when we focus on the Pareto branch.

GB tree: 

<img src="GBtree.jpg" width="400"></img> 

(Source: Wikipedia)

## Requirements
Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `scipy`
- `matplotlib`
- `progressbar`
- `prettytable`


## Distributions

I implemented following functions:

|        	| pdf &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   	| cdf &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | icdf &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Jacobian &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Hessian &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 	|
|--------	|--------------------------	|--------------------------------------------------	|----------------------------------------------	|---------------------	|----------------------	|
| Pareto 	| Pareto_pdf(x, b, p)      	| Pareto_cdf(x, b, p) Pareto_cdf_ne(x, b, p)       	| Pareto_icdf(u, b, p) Pareto_icdf_ne(x, b, p) 	| Pareto_jac(x, b, p) 	| Pareto_hess(x, b, p) 	|
| IB1    	| IB1_pdf(x, b, p, q)      	| IB1_cdf(x, b, p, q)                              	| IB1_icdf_ne(x, b, p, q)                      	| IB1_jac(x, b, p, q) 	|                      	|
| GB1    	| GB1_pdf(x, a, b, p, q)   	| GB1_cdf(x, a, b, p, q) GB1_cdf_ne(x, a, b, p, q) 	| GB1_icdf_ne(x, a, b, p, q)                   	|                     	|                      	|
| GB     	| GB_pdf(x, a, b, c, p, q) 	| GB_cdf_ne(x, a, b, c, p, q)                      	| GB_icdf_ne(x, a, b, c, p, q)                 	|                     	|                      	|

## Fitting

To fit the distributions, I provide following functions:

|        	| fit  &nbsp; &nbsp; &nbsp; |
|--------	|--------------------------	|
| Pareto 	| Paretofit(x, b, x0, ...) 	|
| IB1    	| IB1fit(x, b, x0, ...)    	|
| GB1    	| GB1fit(x, b, x0, ...)    	|
| GB     	| GBfit(x, b, x0, ...)     	|

with following options:
