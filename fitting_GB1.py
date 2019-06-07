from Pareto2GBfit import *

np.set_printoptions(precision=8)

plot_pdf_cdf = False
run_optimize = True
plot_fit = True # needs run_optimize=True


""" 1. Data Generating Process """
# test parameters
a, p, q = 1, 2, 2 # to be optimized parms
b = 2000 # fixed parm
# Note: When setting parmameters, bounds need to be adjusted:
# bounds = ((-2,-.1), (.1, 3), (.1, 2)) #TRUE, REAL PARAMETERS if a<0
# bounds = ((.1,2), (.1, 3), (.1, 2)) # TRUE PARAMETERS if a>0

# size of overall synthetic / noise data
n = 20000

# noise
mu = 0
sigma = 50
random.seed(0)
noise = np.random.normal(mu, sigma, size=n)

# random uniform
u = np.array(np.random.uniform(.0, 1., n)) # (1xn)

# linspace
xmin = .01
xmax = 100000
x = linspace(xmin, xmax, n)
x_noise = x + noise

# simulated data + noise
GB1_data, GB1_u = GB1_icdf_ne(x, a, b, p, q)
GB1_data_noise, GB1_u_noise = GB1_icdf_ne(x_noise, a, b, p, q)
Pareto_data = Pareto_icdf(np.sort(u), b, p)

# check data
# plt.scatter(GB1_u, GB1_data, marker=".", s=.5, alpha=.25, color='b')
# plt.scatter(GB1_u_noise, GB1_data_noise, marker=".", s=.25, alpha=.75, color='m')
# plt.scatter(np.sort(u), Pareto_data, marker=".", s=.5, alpha=.25, color='r')
# plt.show()

# check cdf (x>b)
# plt.scatter(x[x>b], GB1_cdf(x=x[x>b], a=-1, b=100, p=2, q=1), marker=".", s=.5, alpha=.75, color='k')
# plt.scatter(x_noise[x_noise>b], GB1_cdf(x=x_noise[x_noise>b], a=-1, b=100, p=2, q=1), marker=".", s=1.5, alpha=1, color='b')
# plt.scatter(x[x>b], Pareto_cdf(x[x>b], b=100, p=2), marker=".", s=.5, alpha=.25, color='r')
# plt.show()

# check cdf (x<b)
# plt.scatter(x[x<=b], GB1_cdf(x=x[x<=b], a=1, b=1000, p=2, q=2), marker=".", s=.5, alpha=.75, color='k')
# plt.scatter(x_noise[x_noise<=b], GB1_cdf(x=x_noise[x_noise<=b], a=1, b=1000, p=2, q=1), marker=".", s=1.5, alpha=1, color='b')
# plt.show()



""" 2. Plot pdfs """
if plot_pdf_cdf is True:
    n = 100000
    xmin = .0001
    xmax = 5000
    x = np.linspace(xmin, xmax, n)

    # Note: if a=-1: x>b, if a=1: a<b

    ## a=-1
    a=-1
    b=500

    plt.figure()
    # title, label
    #plt.title("Pareto Distribution\npdf with different Parameters b, p");
    plt.xlabel("x"); plt.ylabel("pdf")
    plt.plot(x[x>b], GB1_pdf(x[x>b], a=a, b=b, p=1, q=1), color='r', label="a={}, b={}, p=1, q=1".format(a,b))
    plt.plot(x[x>b], GB1_pdf(x[x>b], a=a, b=b, p=2, q=1), color='g', label="a={}, b={}, p=2, q=2".format(a,b))
    plt.plot(x[x>b], GB1_pdf(x[x>b], a=a, b=b, p=2, q=2), color='b', label="a={}, b={}, p=2, q=2".format(a,b))
    plt.plot(x[x>b], GB1_pdf(x[x>b], a=a, b=b, p=3, q=4), color='y', label="a={}, b={}, p=3, q=4".format(a,b))
    plt.axvline(x=b, ymin=0, ymax=1, color='black', linewidth=1, alpha=.5, label="b (lower bound)", linestyle='dashed')
    #plt.axvline(x=2, ymin=0, ymax=3, color='black', linewidth=1, alpha=.5, label="b (lower bound)", linestyle='dashed')
    # legend, xlim, ylim
    plt.legend(loc='upper right') #; plt.ylim(bottom=0, top=3)#;plt.xlim(left=0, right=100)
    plt.savefig('graphs/GB1_simulated_pdf_a={}.png'.format(a), dpi=300)
    plt.show()


    ## a=1
    a=1
    b=500

    plt.figure()
    # title, label
    #plt.title("Pareto Distribution\npdf with different Parameters b, p");
    plt.xlabel("x"); plt.ylabel("pdf")
    plt.plot(x[x<=b], GB1_pdf(x[x<=b], a=a, b=b, p=1, q=1), color='r', label="a={}, b={}, p=1, q=1".format(a,b))
    plt.plot(x[x<=b], GB1_pdf(x[x<=b], a=a, b=b, p=2, q=1), color='g', label="a={}, b={}, p=2, q=2".format(a,b))
    plt.plot(x[x<=b], GB1_pdf(x[x<=b], a=a, b=b, p=2, q=2), color='b', label="a={}, b={}, p=2, q=2".format(a,b))
    plt.plot(x[x<=b], GB1_pdf(x[x<=b], a=a, b=b, p=3, q=4), color='y', label="a={}, b={}, p=3, q=4".format(a,b))
    plt.axvline(x=b, ymin=0, ymax=1, color='black', linewidth=1, alpha=.5, label="b (upper bound)", linestyle='dashed')
    plt.axvline(x=2, ymin=0, ymax=3, color='black', linewidth=1, alpha=.5, label="b (lower bound)", linestyle='dashed')
    # legend, xlim, ylim
    plt.legend(loc='upper right') #; plt.ylim(bottom=0, top=3)#;plt.xlim(left=0, right=100)
    plt.savefig('graphs/GB1_simulated_pdf_a={}.png'.format(a), dpi=300)
    plt.show()


""" 3. Plot cdfs """
if plot_pdf_cdf is True:
    # Note: if a=-1: x>b, if a=1: a<b

    ## a=-1
    a=-1
    b=500

    plt.figure()
    plt.xlabel("x"); plt.ylabel("cdf")
    plt.plot(x[x>b], GB1_cdf(x[x>b], a=a, b=b, p=1, q=1), color='r', label="a={}, b={}, p=1, q=1".format(a,b))
    plt.plot(x[x>b], GB1_cdf(x[x>b], a=a, b=b, p=2, q=1), color='g', label="a={}, b={}, p=2, q=2".format(a,b))
    plt.plot(x[x>b], GB1_cdf(x[x>b], a=a, b=b, p=2, q=2), color='b', label="a={}, b={}, p=2, q=2".format(a,b))
    plt.plot(x[x>b], GB1_cdf(x[x>b], a=a, b=b, p=3, q=4), color='y', label="a={}, b={}, p=3, q=4".format(a,b))
    plt.legend(loc='upper left'); #plt.ylim(bottom=0,top=1)
    plt.savefig('graphs/GB1_simulated_cdf_a={}.png'.format(a), dpi=300)
    plt.show()

    ## a=1
    a=1
    b=500

    plt.figure()
    plt.xlabel("x"); plt.ylabel("cdf")
    plt.plot(x[x<b], GB1_cdf(x[x<b], a=a, b=b, p=1, q=1), color='r', label="a={}, b={}, p=1, q=1".format(a,b))
    plt.plot(x[x<b], GB1_cdf(x[x<b], a=a, b=b, p=2, q=1), color='g', label="a={}, b={}, p=2, q=2".format(a,b))
    plt.plot(x[x<b], GB1_cdf(x[x<b], a=a, b=b, p=2, q=2), color='b', label="a={}, b={}, p=2, q=2".format(a,b))
    plt.plot(x[x<b], GB1_cdf(x[x<b], a=a, b=b, p=3, q=4), color='y', label="a={}, b={}, p=3, q=4".format(a,b))
    plt.legend(loc='upper left'); #plt.ylim(bottom=0,top=1)
    plt.savefig('graphs/GB1_simulated_cdf_a={}.png'.format(a), dpi=300)
    plt.show()


""" 4a. Constrained Optimization: SLSQP a<0 """
if run_optimize is True and a<0:
    def GB1_constraint1(parms):
        a = parms[0]
        return b**a - np.min(x)**a
    def GB1_constraint2(parms):
        a = parms[0]
        return b**a - np.max(x)**a
    constr = ({'type': 'ineq', 'fun': GB1_constraint1}, {'type': 'ineq', 'fun': GB1_constraint2})

    def GB1_constraint(parms):
        a = parms[0]
        return (np.min(x)/b)**a
    constr_2 = {'type': 'ineq', 'fun': GB1_constraint}

    # bounds for parameters a, p, q
    # NOTE: It turns out that for GB1 one need to set narrow bounds
    #bounds = ((-2,-.1), (.1, 3), (.1, 2)) #TRUE, REAL PARAMETERS
    bounds = ((-2,-.1), (.1, 3), (.1, 2))

    # initial guess
    #x0 = np.array([-1, 20, 1, 1]) #TRUE, FALSE PARAMETERS
    #x0 = np.array([-1.1, 1.1, 2.1]) #TRUE, REAL PARAMETERS
    x0 = np.array([-1.1, 1.1, 2.1]) #TRUE, REAL PARAMETERS

    print("true: a: {}, p: {}, q: {}, b (fix): {}".format(a, p, q, b))


    ### 5a.1 minimize -ll with GB1_data
    x = GB1_data
    result = opt.minimize(GB1_ll, x0, method='SLSQP', bounds=bounds, tol=1e-16, args=(x, b),
                          options=({'maxiter': 350, 'disp': True}), constraints=constr_2)
    print(result)

    # save results
    a_fit_GB1_data, p_fit_GB1_data, q_fit_GB1_data = result.x

    # bootstrap
    GB1_bs_parms = GB1_a1_bootstrap(x=GB1_data, b=b, x0=x0, bounds=bounds, option=2)

    # save bs results
    a_fit_GB1_data_bs, p_fit_GB1_data_bs, q_fit_GB1_data_bs = GB1_bs_parms[0], GB1_bs_parms[2], GB1_bs_parms[4]
    a_fit_sd_GB1_data_bs, p_fit_sd_GB1_data_bs, q_fit_sd_GB1_data_bs = GB1_bs_parms[1], GB1_bs_parms[3], GB1_bs_parms[5]

    ### 5a.2 minimize -ll with GB1_data_noise
    x = GB1_data_noise
    result = opt.minimize(GB1_ll, x0, method='SLSQP', bounds=bounds, tol=1e-14, args=(x, b),
                          options=({'maxiter': 350, 'disp': True}), constraints=constr_2)
    print(result)

    # save results
    a_fit_GB1_data_noise, p_fit_GB1_data_noise, q_fit_GB1_data_noise = result.x

    # bootstrap
    GB1_noise_bs_parms = GB1_a1_bootstrap(x=GB1_data_noise, b=b, x0=x0, bounds=bounds, option=2)

    # save bs results
    a_fit_GB1_data_noise_bs, p_fit_GB1_data_noise_bs, q_fit_GB1_data_noise_bs = GB1_noise_bs_parms[0], GB1_noise_bs_parms[2], GB1_noise_bs_parms[4]
    a_fit_sd_GB1_data_noise_bs, p_fit_sd_GB1_data_noise_bs, q_fit_sd_GB1_data_noise_bs = GB1_noise_bs_parms[1], GB1_noise_bs_parms[3], GB1_noise_bs_parms[5]

    ### 5a.3 minimize -ll with Pareto_data
    x = Pareto_data
    result = opt.minimize(GB1_ll, x0, method='SLSQP', bounds=bounds, tol=1e-16, args=(x, b),
                          options=({'maxiter': 350, 'disp': True}), constraints=constr_2)
    print(result)

    # save results
    a_fit_Pareto_data, p_fit_Pareto_data, q_fit_Pareto_data = result.x

    # bootstrap
    Pareto_bs_parms = GB1_a1_bootstrap(x=Pareto_data, b=b, x0=x0, bounds=bounds, option=2)

    # save results
    a_fit_Pareto_data_bs, p_fit_Pareto_data_bs, q_fit_Pareto_data_bs = Pareto_bs_parms[0], Pareto_bs_parms[2], Pareto_bs_parms[4]
    a_fit_sd_Pareto_data_bs, p_fit_sd_Pareto_data_bs, q_fit_sd_Pareto_data_bs = Pareto_bs_parms[1], Pareto_bs_parms[3], Pareto_bs_parms[5]

""" 4b. Constrained Optimization: SLSQP a>0 """
if run_optimize is True and a>0:
    def GB1_constraint1(parms): # TODO: constraint checken
        a = parms[0]
        return b**a - np.min(x)**a
    def GB1_constraint2(parms):
        a = parms[0]
        return b**a - np.max(x)**a
    constr = ({'type': 'ineq', 'fun': GB1_constraint1}, {'type': 'ineq', 'fun': GB1_constraint2})

    def GB1_constraint(parms):
        a = parms[0]
        return (np.min(x)/b)**a
    constr_2 = {'type': 'ineq', 'fun': GB1_constraint}

    # bounds for parameters a, b, p, q
    # NOTE: It turns out that for GB1 one need to set narrow bounds
    # bounds = ((.1,2), (.1, 3), (.1, 2)) TRUE PARAMETERS
    bounds = ((.1,2), (.1, 3), (.1, 2))

    # initial guess
    #x0 = np.array([-1.1, 1.1, 2.1]) #TRUE, REAL PARAMETERS
    x0 = np.array([1.1, 1.1, 2.1]) #TRUE, REAL PARAMETERS

    print("true: a: {}, p: {}, q: {}, b (fix): {}".format(a, p, q, b))


    ### 5b.1 minimize -ll with GB1_data
    x = GB1_data
    result = opt.minimize(GB1_ll, x0, method='SLSQP', bounds=bounds, tol=1e-16, args=(x, b),
                          options=({'maxiter': 350, 'disp': True}), constraints=constr_2)
    print(result)

    # save results
    a_fit_GB1_data, p_fit_GB1_data, q_fit_GB1_data = result.x

    # bootstrap
    GB1_bs_parms = GB1_a2_bootstrap(x=GB1_data, b=b, x0=x0, bounds=bounds, option=2)

    # save bs results
    a_fit_GB1_data_bs, p_fit_GB1_data_bs, q_fit_GB1_data_bs = GB1_bs_parms[0], GB1_bs_parms[2], GB1_bs_parms[4]
    a_fit_sd_GB1_data_bs, p_fit_sd_GB1_data_bs, q_fit_sd_GB1_data_bs = GB1_bs_parms[1], GB1_bs_parms[3], GB1_bs_parms[5]

    ### 5b.2 minimize -ll with GB1_data_noise
    x = GB1_data_noise
    result = opt.minimize(GB1_ll, x0, method='SLSQP', bounds=bounds, tol=1e-14, args=(x, b),
                          options=({'maxiter': 350, 'disp': True}), constraints=constr_2)
    print(result)

    # save results
    a_fit_GB1_data_noise, p_fit_GB1_data_noise, q_fit_GB1_data_noise = result.x

    # bootstrap
    GB1_noise_bs_parms = GB1_a2_bootstrap(x=GB1_data_noise, b=b, x0=x0, bounds=bounds, option=2)

    # save bs results
    a_fit_GB1_data_noise_bs, p_fit_GB1_data_noise_bs, q_fit_GB1_data_noise_bs = GB1_noise_bs_parms[0], GB1_noise_bs_parms[2], GB1_noise_bs_parms[4]
    a_fit_sd_GB1_data_noise_bs, p_fit_sd_GB1_data_noise_bs, q_fit_sd_GB1_data_noise_bs = GB1_noise_bs_parms[1], GB1_noise_bs_parms[3], GB1_noise_bs_parms[5]





""" 5a. Plot simulated GB1_data (a<0) + fit (+ bootstrapped) """
if plot_fit is True and a<0:
    # true parms
    parms_true = a, b, p, q
    b_fix = b

    ### simulated + fitted data
    simul_data, u_simul         = GB1_icdf_ne(x=x, a=parms_true[0], b=parms_true[1], p=parms_true[2], q=parms_true[3])
    fit_simul_data, u_fit       = GB1_icdf_ne(x=x, a=a_fit_GB1_data, b=b_fix, p=p_fit_GB1_data, q=q_fit_GB1_data)
    fit_simul_data_bs, u_fit_bs = GB1_icdf_ne(x=x, a=a_fit_GB1_data_bs, b=b_fix, p=p_fit_GB1_data_bs, q=q_fit_GB1_data_bs)

    plt.figure()
    plt.scatter(u_simul, simul_data, color='blue', label="simulated data", marker='o', s=.3)
    plt.plot(u_fit, fit_simul_data, color='red', label="fit", marker='', linestyle='dashed')
    plt.plot(u_fit_bs, fit_simul_data_bs, color='green', label="fit (bootstrapped)", marker='', linestyle='dashed')
    plt.xlabel("u")
    plt.legend()
    plt.savefig('graphs/GB1_simulateddata_vs_fit_a1.png', dpi=300)
    plt.show()


""" 6a. Plot simulated GB1_data_noise (a<0) + fit (+ bootstrapped) """
if plot_fit is True and a<0:

    # simulated + fitted noise data
    simul_data_noise, u_simul_noise         = GB1_icdf_ne(x=x_noise, a=parms_true[0], b=parms_true[1], p=parms_true[2], q=parms_true[3])
    fit_simul_data_noise, u_fit_noise       = GB1_icdf_ne(x=x_noise, a=a_fit_GB1_data_noise, b=b_fix, p=p_fit_GB1_data_noise, q=q_fit_GB1_data_noise)
    fit_simul_data_noise_bs, u_fit_noise_bs = GB1_icdf_ne(x=x_noise, a=a_fit_GB1_data_noise_bs, b=b_fix, p=p_fit_GB1_data_noise_bs, q=q_fit_GB1_data_noise_bs)

    plt.figure()
    plt.scatter(u_simul_noise, simul_data_noise, color='blue', label="simulated data + noise", marker='o', s=.3)
    plt.plot(u_fit_noise, fit_simul_data_noise, color='red', label="fit", marker='', linestyle='dashed')
    plt.plot(u_fit_noise_bs, fit_simul_data_noise_bs, color='green', label="fit (bootstrapped)", marker='', linestyle='dashed')
    plt.xlabel("u")
    plt.legend()
    plt.savefig('graphs/GB1_simulateddata_noise_vs_fit_a1.png', dpi=300)
    plt.show()


""" 7a. Plot simulated Pareto_data (a<0) + fit (+ bootstrapped) """
if plot_fit is True and a<0:

    # simulated + fitted data via GB1(), Pareto()
    simul_data_data_viaGB1, u_simul_data_viaGB1 = GB1_icdf_ne(x=x, a=parms_true[0], b=parms_true[1], p=parms_true[2], q=parms_true[3])
    simul_data_data_viaPareto                   = Pareto_icdf(u=u_simul_data_viaGB1, b=parms_true[1], p=parms_true[2])
    fit_simul_data, u_fit                       = GB1_icdf_ne(x=x, a=a_fit_Pareto_data, b=b_fix, p=p_fit_Pareto_data, q=q_fit_Pareto_data)
    fit_simul_data_bs, u_fit_bs                 = GB1_icdf_ne(x=x, a=a_fit_Pareto_data_bs, b=b_fix, p=p_fit_Pareto_data_bs, q=q_fit_Pareto_data_bs)

    plt.figure()
    plt.scatter(u_simul_data_viaGB1, simul_data_data_viaGB1, color='blue', label="simulated data (GB1)", marker='o', s=.3)
    plt.scatter(u_simul_data_viaGB1, simul_data_data_viaPareto, color='m', label="simulated data (Pareto)", marker='o', s=.3)
    plt.plot(u_fit, fit_simul_data, color='red', label="fit", marker='', linestyle='dashed')
    plt.plot(u_fit_bs, fit_simul_data_bs, color='green', label="fit (bootstrapped)", marker='', linestyle='dashed')
    plt.xlabel("u")
    plt.legend()
    plt.savefig('graphs/GB1_Pareto_simulateddata_vs_fit_a1.png', dpi=300)
    plt.show()

""" 5b. Plot simulated GB1_data (a>0) + fit (+ bootstrapped) """
if plot_fit is True and a>0:
    # true parms
    parms_true = a, b, p, q
    b_fix = b

    x = np.array(x)
    x = x[x<=b]

    ### simulated + fitted data
    simul_data, u_simul         = GB1_icdf_ne(x=x, a=parms_true[0], b=parms_true[1], p=parms_true[2], q=parms_true[3])
    fit_simul_data, u_fit       = GB1_icdf_ne(x=x, a=a_fit_GB1_data, b=b_fix, p=p_fit_GB1_data, q=q_fit_GB1_data)
    fit_simul_data_bs, u_fit_bs = GB1_icdf_ne(x=x, a=a_fit_GB1_data_bs, b=b_fix, p=p_fit_GB1_data_bs, q=q_fit_GB1_data_bs)

    plt.figure()
    plt.scatter(u_simul, simul_data, color='blue', label="simulated data", marker='o', s=.3)
    plt.plot(u_fit, fit_simul_data, color='red', label="fit", marker='', linestyle='dashed')
    plt.plot(u_fit_bs, fit_simul_data_bs, color='green', label="fit (bootstrapped)", marker='', linestyle='dashed')
    plt.xlabel("u")
    plt.legend()
    plt.savefig('graphs/GB1_simulateddata_vs_fit_a2.png', dpi=300)
    plt.show()


""" 6b. Plot simulated GB1_data_noise + fit (+ bootstrapped) """
if plot_fit is True and a>0:

    # simulated + fitted noise data
    simul_data_noise, u_simul_noise         = GB1_icdf_ne(x=x_noise, a=parms_true[0], b=parms_true[1], p=parms_true[2], q=parms_true[3])
    fit_simul_data_noise, u_fit_noise       = GB1_icdf_ne(x=x_noise, a=a_fit_GB1_data_noise, b=b_fix, p=p_fit_GB1_data_noise, q=q_fit_GB1_data_noise)
    fit_simul_data_noise_bs, u_fit_noise_bs = GB1_icdf_ne(x=x_noise, a=a_fit_GB1_data_noise_bs, b=b_fix, p=p_fit_GB1_data_noise_bs, q=q_fit_GB1_data_noise_bs)

    plt.figure()
    plt.scatter(u_simul_noise, simul_data_noise, color='blue', label="simulated data + noise", marker='o', s=.3)
    plt.plot(u_fit_noise, fit_simul_data_noise, color='red', label="fit", marker='', linestyle='dashed')
    plt.plot(u_fit_noise_bs, fit_simul_data_noise_bs, color='green', label="fit (bootstrapped)", marker='', linestyle='dashed')
    plt.xlabel("u")
    plt.legend()
    plt.savefig('graphs/GB1_simulateddata_noise_vs_fit_a2.png', dpi=300)
    plt.show()

