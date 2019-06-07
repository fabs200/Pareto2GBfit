from Pareto2GBfit import *

np.set_printoptions(precision=10)

plot_pdf_cdf = False
run_optimize = False
plot_fit = True # needs run_optimize=True
plot_ll = False


""" 1. Data Generating Process """
# test parameters
b, p, q = 250, 2, 1

# size of overall synthetic / noise data
n = 100000

# noise
mu = 0
sigma = 50
random.seed(223)
noise = np.random.normal(mu, sigma, size=n)

# random uniform
u = np.array(np.random.uniform(.0, 1., n)) # (1xn)

# linspace
xmin = .1
xmax = 10000
x = linspace(xmin, xmax, n)
x_noise = x + noise

# simulated data + noise
IB1_data, IB1_u = IB1_icdf_ne(x, b, p, q)
IB1_data_noise, IB1_u_noise = IB1_icdf_ne(x_noise, b, p, q)
Pareto_data = Pareto_icdf(np.sort(u), b, p)

# check data
# plt.scatter(IB1_u, IB1_data, marker=".", s=.5, alpha=.25, color='b')
# plt.scatter(IB1_u_noise, IB1_data_noise, marker=".", s=.5, alpha=.75, color='m')
# plt.scatter(np.sort(u), Pareto_data, marker=".", s=.5, alpha=.25, color='r')
# plt.show()

# check cdf
# plt.scatter(x[x>b], IB1_cdf(parms, x[x>b]), marker=".", s=.5, alpha=.1, color='b')
# plt.scatter(x[x>b], Pareto_cdf(parms, x[x>b]), marker=".", s=.5, alpha=.1, color='r')
# plt.show()



""" 2. Plot pdfs """
if plot_pdf_cdf is True:
    n = 100000
    xmin = .0001
    xmax = 5000
    x = np.linspace(xmin, xmax, n)

    # define parameter set
    bvalues = [500,500,500,500]
    pvalues = [1,2,1,2]
    qvalues = [1,1,2,2]

    plt.figure()
    # title, label
    plt.xlabel("x"); plt.ylabel("pdf")
    plt.plot(x[x>bvalues[0]], IB1_pdf(x[x>bvalues[0]], bvalues[0],pvalues[0],qvalues[0]), color='r', label="b={}, p={}, q={}".format(bvalues[0],pvalues[0],qvalues[0]))
    plt.plot(x[x>bvalues[0]], IB1_pdf(x[x>bvalues[1]], bvalues[1],pvalues[1],qvalues[1]), color='g', label="b={}, p={}, q={}".format(bvalues[1],pvalues[1],qvalues[1]))
    plt.plot(x[x>bvalues[0]], IB1_pdf(x[x>bvalues[2]], bvalues[2],pvalues[2],qvalues[2]), color='b', label="b={}, p={}, q={}".format(bvalues[2],pvalues[2],qvalues[2]))
    plt.plot(x[x>bvalues[0]], IB1_pdf(x[x>bvalues[3]], bvalues[3],pvalues[3],qvalues[3]), color='y', label="b={}, p={}, q={}".format(bvalues[3],pvalues[3],qvalues[3]))
    plt.axvline(x=bvalues[0], ymin=0, ymax=3, color='black', linewidth=1, alpha=.5, label="b (lower bound)", linestyle='dashed')
    # legend, xlim, ylim
    plt.legend(loc='upper right'); #plt.ylim(bottom=0, top=3)
    plt.savefig('graphs/IB1_simulated_pdf.png', dpi=300)
    plt.show()


""" 3. Plot cdfs """
if plot_pdf_cdf is True:
    plt.figure()
    plt.xlabel("x"); plt.ylabel("cdf")
    plt.plot(x[x>bvalues[0]], IB1_cdf(x[x>bvalues[0]], bvalues[0],pvalues[0],qvalues[0]), color='r', label="b={}, p={}, q={}".format(bvalues[0],pvalues[0],qvalues[0]))
    plt.plot(x[x>bvalues[0]], IB1_cdf(x[x>bvalues[1]], bvalues[1],pvalues[1],qvalues[1]), color='g', label="b={}, p={}, q={}".format(bvalues[1],pvalues[1],qvalues[1]))
    plt.plot(x[x>bvalues[0]], IB1_cdf(x[x>bvalues[2]], bvalues[2],pvalues[2],qvalues[2]), color='b', label="b={}, p={}, q={}".format(bvalues[2],pvalues[2],qvalues[2]))
    plt.plot(x[x>bvalues[0]], IB1_cdf(x[x>bvalues[3]], bvalues[3],pvalues[3],qvalues[3]), color='y', label="b={}, p={}, q={}".format(bvalues[3],pvalues[3],qvalues[3]))
    plt.legend(loc='upper left')#; plt.ylim(bottom=0,top=1)
    plt.savefig('graphs/IB1_simulated_cdf.png', dpi=300)
    plt.show()


""" 4. Plot icdf """
if plot_pdf_cdf is True:
    IB_data1, u1 = IB1_icdf_ne(x[x>b], bvalues[0],pvalues[0],qvalues[0])
    IB_data2, u2 = IB1_icdf_ne(x[x>b], bvalues[1],pvalues[1],qvalues[1])
    IB_data3, u3 = IB1_icdf_ne(x[x>b], bvalues[2],pvalues[2],qvalues[2])
    IB_data4, u4 = IB1_icdf_ne(x[x>b], bvalues[3],pvalues[3],qvalues[3])

    plt.figure()
    plt.xlabel("F"); plt.ylabel("x")
    plt.scatter(u1, IB_data1, color='r', marker='o', s=.6, label="b={}, p={}, q={}".format(bvalues[0],pvalues[0],qvalues[0]))
    plt.scatter(u2, IB_data2, color='g', marker='o', s=.6, label="b={}, p={}, q={}".format(bvalues[1],pvalues[1],qvalues[1]))
    plt.scatter(u3, IB_data3, color='b', marker='o', s=.6, label="b={}, p={}, q={}".format(bvalues[2],pvalues[2],qvalues[2]))
    plt.scatter(u4, IB_data4, color='y', marker='o', s=.6, label="b={}, p={}, q={}".format(bvalues[3],pvalues[3],qvalues[3]))
    plt.legend(loc='upper left')
    plt.savefig('graphs/IB1_simulated_data_ne.png', dpi=300)
    plt.show()



""" 5. Constrained Optimization: SLSQP """
if run_optimize is True:
    # define constraints
    def IB1_constraint(b):
        return np.min(x) - b

    # specify parms-array which has to be optimized, prepare 'data', bounds for parameters p, q, prep. constraint dictionary for minimize()
    x = IB1_data
    parm_bounds = (10**-14, np.inf)
    bounds = (parm_bounds, parm_bounds,)
    constr = {'type': 'ineq', 'fun': IB1_constraint} # inequality means that it is to be non-negative

    # initial guess
    x0 = np.array([1, 1])
    b = 250

    ### minimize -ll of simulated IB1 data
    result = opt.minimize(IB1_ll, x0=x0, method='SLSQP', bounds=bounds, tol=1e-16, args=(x,b),
                          options=({'maxiter': 100, 'disp': True}), constraints=constr)
    print(result)

    # save results
    p_fit, q_fit = result.x
    p_fit_bs, p_fit_se_bs, q_fit_bs, q_fit_se_bs = IB1Fit(x, x0=x0, bootstraps=100, b=250, verbose=False)


    ### minimize -ll of simulated IB1 data + noise
    x = IB1_data_noise
    result = opt.minimize(IB1_ll, x0=x0, method='SLSQP', bounds=bounds, tol=1e-12, args=(x,b),
                          options=({'maxiter': 1000, 'disp': True}), constraints=constr)
    print(result)

    # save results
    p_fit_noise, q_fit_noise = result.x
    p_fit_noise_bs, q_fit_noise_se_bs, q_fit_noise_bs, q_fit_noise_se_bs = IB1Fit(x, x0=x0, bootstraps=100, b=250, verbose=False)



""" 6. Plot simulated data + fit """
if plot_fit is True:
    # parms
    parms_true = p, q
    b_true = b
    parms_fit = [p_fit, q_fit]
    parms_fit_bs = [p_fit_bs, q_fit_bs]

    ### simulated data
    simul_data, u_simul     = IB1_icdf_ne(x=x, p=parms_true[0], q=parms_true[1], b=b_true)
    fit_simul_data, u_fit   = IB1_icdf_ne(x=x, p=p_fit, q=q_fit, b=b_true)

    plt.figure()
    plt.plot(u_simul, simul_data, color='blue', label="simulated data", marker='o', markersize=.8, linewidth=.01)
    plt.plot(u_fit, fit_simul_data, color='red', label="fit", marker='')
    plt.xlabel("u")
    plt.legend()
    plt.savefig('graphs/IB1_simulateddata_vs_fit.png', dpi=300)
    plt.show()

    ### simulated data + noise
    simul_data_noise, u_simul_noise     = IB1_icdf_ne(x=x_noise, p=parms_true[0], q=parms_true[1], b=b_true)
    fit_simul_data_noise, u_fit_noise   = IB1_icdf_ne(x=x_noise, p=p_fit_noise, q=q_fit_noise, b=b_true)

    plt.figure()
    plt.plot(u_simul_noise, simul_data_noise, color='blue', label="simulated data + noise", marker='o', markersize=.8, linewidth=.01)
    plt.plot(u_fit_noise, fit_simul_data_noise, color='red', label="fit", marker='')
    plt.xlabel("u")
    plt.legend()
    plt.savefig('graphs/IB1_simulateddata_noise_vs_fit.png', dpi=300)
    plt.show()


""" 7. Plot simulated data + fit (bootstrapped)"""
if plot_fit is True:
    ### simulated data
    simul_data, u_simul     = IB1_icdf_ne(x=x, p=parms_true[0], q=parms_true[1], b=b_true)
    fit_simul_data, u_fit   = IB1_icdf_ne(x=x, p=p_fit_bs, q=q_fit_bs, b=b_true)

    plt.figure()
    plt.plot(u_simul, simul_data, color='blue', label="simulated data", marker='o', markersize=.8, linewidth=.01)
    plt.plot(u_fit, fit_simul_data, color='red', label="fit (bootstrapped)", marker='')
    plt.xlabel("u")
    plt.legend()
    plt.savefig('graphs/IB1_simulateddata_vs_fit_bs.png', dpi=300)
    plt.show()

    ### simulated data + noise
    simul_data_noise, u_simul_noise     = IB1_icdf_ne(x=x_noise, p=parms_true[0], q=parms_true[1], b=b_true)
    fit_simul_data_noise, u_fit_noise   = IB1_icdf_ne(x=x_noise, p=p_fit_noise_bs, q=q_fit_noise_bs, b=b_true)

    plt.figure()
    plt.plot(u_simul_noise, simul_data_noise, color='blue', label="simulated data + noise", marker='o', markersize=.8, linewidth=.01)
    plt.plot(u_fit_noise, fit_simul_data_noise, color='red', label="fit (bootstrapped)", marker='')
    plt.xlabel("u")
    plt.legend()
    plt.savefig('graphs/IB1_simulateddata_noise_vs_fit.png', dpi=300)
    plt.show()



""" 8. 2d plot: LL vs. b (p, q is fix) """
if plot_ll is True:
    k=500
    parms_true = p, q
    b_true = b
    p_fix = p
    q_fix = q
    parms = p_fix, q_fix
    data = IB1_data
    bmin, bmax = 0.001, np.min(data)
    B = np.linspace(bmin, bmax, k)
    LL = []
    for i in B:
            LL.append(IB1_ll(parms=parms, x=data, b=i))
            print(str(i), IB1_ll(parms=parms, x=data, b=i))

    # convert to lists and sort according to P
    LL = np.array(LL)
    LL_b_sorted = LL[B.argsort()]

    # plot
    plt.figure()
    plt.gca().set_position((.15, .35, .55, .55)) # to make a bit of room for extra text
    plt.title("Grid search\n p fix, q fix, b varied"); plt.xlabel("Paramter b"); plt.ylabel("LL")
    plt.plot(B, LL_b_sorted)
    plt.plot(b, IB1_ll(parms=parms_true, x=data, b=b_true), linestyle='None', marker='o', color='r', label="True Parameter")
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
    plot_text = "True Parameters: p={} (fix), q={} (fix),; n={}\nGrid search: b(min, max): ({}, {}), steps={}".format(p_fix, q_fix, n, bmin, bmax, k)
    side_text = plt.figtext(.25, .1, plot_text, bbox=dict(facecolor='white'), fontsize=8)
    plt.savefig('graphs/IB1_gridsearch_LL_b_pfix_qfix.png', dpi=300)
    plt.show()


""" 9. 2d plot: LL vs. p (b, q is fix) """
if plot_ll is True:
    k=250
    b_fix = b
    q_fix = q
    pmin, pmax = .1, 10
    P = np.linspace(pmin, pmax, k)
    LL = []
    for i in P:
            LL.append(IB1_ll(parms=[i, q_fix], x=data, b=b_fix))
            print(str(i), IB1_ll(parms=[i, q_fix], x=data, b=b_fix))

    # convert to lists and sort according to P
    LL = np.array(LL)
    LL_p_sorted = LL[P.argsort()]

    # plot
    plt.figure()
    plt.gca().set_position((.15, .35, .55, .55)) # to make a bit of room for extra text
    plt.title("Grid search\n b fix, q fix, p varied"); plt.xlabel("Paramter p"); plt.ylabel("LL")
    plt.plot(P, LL_p_sorted)
    plt.plot(p, IB1_ll(parms=parms_true, x=data, b=b_true), linestyle='None', marker='o', color='r', label="True Parameter")
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
    plot_text = "True Parameters: b={} (fix), q={} (fix),; n={}\nGrid search: p(min, max): ({}, {}), steps={}".format(b_fix, q_fix, n, pmin, pmax, k)
    side_text = plt.figtext(.25, .1, plot_text, bbox=dict(facecolor='white'), fontsize=8)
    plt.savefig('graphs/IB1_gridsearch_LL_bfix_p_qfix.png', dpi=300)
    plt.show()


""" 10. 2d plot: LL vs. q (b, p is fix) """
if plot_ll is True:
    k=250
    b_fix = b
    p_fix = p
    qmin, qmax = .1, 10
    Q = np.linspace(qmin, qmax, k)
    LL = []
    for i in Q:
            LL.append(IB1_ll(parms=[p_fix, i], x=data, b=b_fix))
            print(str(i), IB1_ll(parms=[p_fix, i], x=data, b=b_fix))

    # convert to lists and sort according to P
    LL = np.array(LL)
    LL_q_sorted = LL[Q.argsort()]

    # plot
    plt.figure()
    plt.gca().set_position((.15, .35, .55, .55)) # to make a bit of room for extra text
    plt.title("Grid search\n b fix, p fix, q varied"); plt.xlabel("Paramter q"); plt.ylabel("LL")
    plt.plot(Q, LL_q_sorted)
    plt.plot(q, IB1_ll(parms=parms_true, x=data, b=b_true), linestyle='None', marker='o', color='r', label="True Parameter")
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))
    plot_text = "True Parameters: b={} (fix), p={} (fix),; n={}\nGrid search: q(min, max): ({}, {}), steps={}".format(b_fix, p_fix, n, qmin, qmax, k)
    side_text = plt.figtext(.25, .1, plot_text, bbox=dict(facecolor='white'), fontsize=8)
    plt.savefig('graphs/IB1_gridsearch_LL_bfix_pfix_q.png', dpi=300)
    plt.show()




""" 11. 3d contour plot """
if plot_ll is True:
    data = IB1_data[0:625]
    k=25
    b_fix = b
    pmin, pmax = .01, 10
    qmin, qmax = .01, 10
    p_data = np.linspace(pmin, pmax, k)
    q_data = np.linspace(bmin, bmax, k)
    P, Q, LL = [], [], []
    for i in p_data:
        for j in q_data:
            P.append(i)
            Q.append(j)
            LL.append(IB1_ll(parms=[i,j], x=data, b=b_fix))
            print(str(i), str(j), IB1_ll(parms=[i,j], x=data, b=b_fix))

    # convert to arrays
    P, Q, LL = np.array(P), np.array(Q), np.array(LL)

    # data as meshgrids, needed for 3d plot
    P_plot, Q_plot = np.meshgrid(P, Q)
    LL_plot = IB1_ll(parms=[P_plot, Q_plot], x=data, b=b_fix)

    # True Parameter
    LL_true = IB1_ll(parms=parms_true, x=data, b=b_true)

    from mpl_toolkits.mplot3d import axes3d, Axes3D

    # 3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(P_plot, Q_plot, LL_plot, cmap=cm.coolwarm, linewidth=2, antialiased=True)
    ax.scatter(parms_true[0], parms_true[1], LL_true, s=100, c="r", alpha=1, label='True Parameter')
    # title, label
    plt.title("Grid search\nfor Parameters p and q; b is fix")
    ax.set_xlabel("b Parameter"); ax.set_ylabel("p Parameter"); ax.set_zlabel("LL")
    # legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    # perspective
    ax.view_init(elev=3, azim=-170)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=.75, aspect=7)
    #plt.savefig('graphs/pareto_gridsearch_LL_b_p.png', dpi=300)
    plt.show()
