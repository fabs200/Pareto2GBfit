from Pareto2GBfit import *

np.set_printoptions(precision=4)

plot_pdf_cdf = False
run_optimize = True
plot_fit = False # needs run_optimize=True
plot_ll = False


""" 1. Data Generating Process """

# Pareto Parameters
b, p = 250, 2.5

# size of overall synthetic / noise data
n = 10000

# noise
mu = 0
sigma = 100
random.seed(123)
noise = np.random.normal(mu, sigma, size=n)

# linspace
xmin = 0.1
xmax = 10000
x = linspace(xmin, xmax, n)
x_noise = x + noise

# random uniform
u = np.array(np.random.uniform(.0, 1., n)) # (1xn)

# Pareto simulated data
Pareto_data = Pareto_icdf(u, b, p)
Pareto_u = np.sort(u)

# simulated data + noise
Pareto_data_ne, Pareto_u_ne = Pareto_icdf_ne(x[x>b], b, p)
Pareto_data_noise, Pareto_u_noise = Pareto_icdf_ne(x_noise[x_noise>b], b, p)

# check data
# plt.scatter(Pareto_u_noise, Pareto_data_noise, marker=".", s=.5, alpha=.3, color='m')
# plt.scatter(Pareto_u_ne, Pareto_data_ne, marker=".", s=.5, color='b')
# plt.scatter(Pareto_u, Pareto_data, marker=".", s=.5, color='r')
# plt.show()

# check cdfs
# plt.scatter(x[x>b], Pareto_cdf(parms, x[x>b]), marker=".", s=.1, color='r')
# plt.scatter(x[x>b], Pareto_cdf_ne(parms, x[x>b]), marker=".", s=.1, color='b')
# plt.show()



""" 2. Plot pdfs """
if plot_pdf_cdf is True:
    n = 1000
    xmin = 0.1
    xmax = 5
    x = linspace(xmin, xmax, n)

    plt.figure()
    # title, label
    #plt.title("Pareto Distribution\npdf with different Parameters b, p");
    # x[x>1] means that x>b is selected out of x
    plt.xlabel("x"); plt.ylabel("pdf")
    plt.plot(x[x>1], Pareto_pdf(x[x>1], b=1, p=1), color='r', label="b=1, p=1")
    plt.plot(x[x>1], Pareto_pdf(x[x>1], b=1, p=2), color='g', label="b=1, p=2")
    plt.plot(x[x>1], Pareto_pdf(x[x>1], b=1, p=3), color='b', label="b=1, p=3")
    plt.axvline(x=1, ymin=0, ymax=3, color='black', linewidth=1, alpha=.5, label="b (lower bound)", linestyle='dashed')
    # legend, xlim, ylim
    plt.legend(loc='upper right'); plt.ylim(bottom=0, top=3)#;plt.xlim(left=0, right=100)
    plt.savefig('graphs/pareto_simulated_pdf.png', dpi=300)
    plt.show()

""" 3. Plot cdfs """
if plot_pdf_cdf is True:
    plt.figure()
    # title, label
    #plt.title("Pareto Distribution\npdf with different Parameters b, p");
    plt.xlabel("x"); plt.ylabel("cdf")
    plt.plot(x[x>1], Pareto_cdf(x[x>1], b=1, p=1), color='r', label="b=1, p=1")
    plt.plot(x[x>1], Pareto_cdf(x[x>1], b=1, p=2), color='g', label="b=1, p=2")
    plt.plot(x[x>1], Pareto_cdf(x[x>1], b=1, p=3), color='b', label="b=1, p=3")
    # legend, xlim, ylim
    plt.legend(loc='upper left'); plt.ylim(bottom=0,top=1)
    plt.savefig('graphs/pareto_simulated_cdf.png', dpi=300)
    plt.show()


""" 4. Constrained Optimization: SLSQP """
if run_optimize is True:
    # define constraints
    def Pareto_constraint(b):
        return np.min(x) - b # inequality means that to be non-negative, NOTE: x, data, etc. should be same as specified in opt.minimize's args

    # specify parms-array which has to be optimized, prepare 'data', bounds for parameters b, p, prep. constraint dictionary for minimize()
    parms = p
    x = Pareto_data
    parm_bounds = (10**-14, np.inf)
    bounds = (parm_bounds,)
    constr = {'type': 'ineq', 'fun': Pareto_constraint}

    # initial guess
    x0 = np.array([2])
    # set lower bound b
    b = 250

    ### minimize -ll with Pareto_data
    result = opt.minimize(Pareto_ll, x0, method='SLSQP', bounds=bounds, args=(x, b), tol=1e-12,
                          options=({'maxiter': 1000, 'disp': True, 'ftol': 1e-06}), constraints=constr)
    print(result)

    # save results
    p_fit = result.x
    p_fit_se = Pareto_extract_se(x=Pareto_data, b=b, p_fitted=p_fit)
    p_fit_bs, p_fit_se_bs = Paretofit(x=x, b=250, x0=x0, bootstraps=500, verbose=False, method='L-BFGS-B',
                                      return_parameters=True, plot=True)

    ### minimize -ll with Pareto_data_noise
    x = Pareto_data_noise
    parm_bounds = (10**-14, np.inf)
    bounds = (parm_bounds,)
    result = opt.minimize(Pareto_ll, x0, method='SLSQP', bounds=bounds, args=(x, b), tol=1e-12,
                          options=({'maxiter': 1000, 'disp': True, 'ftol': 1e-06}), constraints=constr)
    print(result)

    # save results
    p_fit_noise = result.x
    p_fit_noise_bs, p_fit_noise_se_bs = Paretofit(x=x, b=250, x0=x0, bootstraps=1000, verbose=False,
                                                  return_parameters=True, plot=True)


""" 5. Plot simulated data + fit """
if plot_fit is True:
    # write pandas dataframe
    df_simul_fit = pd.DataFrame({'u': np.sort(u),
                                'simul_data': Pareto_icdf(np.sort(u), b, p),
                                'simul_noise_data': Pareto_icdf(np.sort(u), b, p) + noise,
                                'fit_simul_data': Pareto_icdf(np.sort(u), b, p),
                                'fit_simul_noise_data': Pareto_icdf(np.sort(u), b, p)})

    plt.figure()
    plt.xlabel("u")
    plt.plot(np.sort(u), 'simul_data', data=df_simul_fit, color='blue', label="simulated data", marker='o', markersize=.8, linewidth=.01)
    plt.plot(np.sort(u), 'fit_simul_data', data=df_simul_fit, color='red', label="fit", marker='')
    plt.legend()
    plt.savefig('graphs/pareto_simulateddata_vs_fit.png', dpi=300)
    plt.show()


""" 6. Plot simulated data with noise + fit """
if plot_fit is True:
    plt.figure()
    plt.xlabel("u")
    plt.plot(np.sort(u), 'simul_noise_data', data=df_simul_fit, color='blue', label="simulated data + noise", marker='o', markersize=.8, linewidth=.01)
    plt.plot(np.sort(u), 'fit_simul_noise_data', data=df_simul_fit, color='red', label="fit", marker='')
    plt.ylim(bottom=0, top=df_simul_fit['simul_noise_data'].max()+df_simul_fit['simul_noise_data'].max()*0.1)
    plt.legend()
    plt.savefig('graphs/pareto_simulateddatawithnoise_vs_fit.png', dpi=300)
    plt.show()


""" 7. 2d plot: LL vs. p (b is fix) """
if plot_ll is True:
    k=500
    b_fix = b
    pmin, pmax = .1, 10
    P = np.linspace(pmin, pmax, k)
    LL = []
    for i in P:
            LL.append(Pareto_ll([i], x=Pareto_data, b=b_fix))
            print(str(i), Pareto_ll([i], x=Pareto_data, b=b_fix))

    # convert to lists and sort according to P
    LL = np.array(LL)
    LL_Psorted = LL[P.argsort()]

    # plot
    plt.figure()
    plt.gca().set_position((.15, .35, .55, .55)) # to make a bit of room for extra text

    # title, label
    plt.title("Grid search of Pareto Distribution\nfor Parameter p (b is fix)"); plt.xlabel("Paramter p"); plt.ylabel("LL")

    plt.plot(P, LL_Psorted)
    plt.plot(p, Pareto_ll([p], x=Pareto_data, b=b_fix), linestyle='None', marker='o', color='r', label="True Parameter")

    # legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

    plot_text = "True Parameters: b={} (fix), p={}; n={}\nGrid search: p(min, max): ({}, {}), steps={}".format(b_fix, p, n, pmin, pmax, k)
    side_text = plt.figtext(.25, .1, plot_text, bbox=dict(facecolor='white'), fontsize=8)
    plt.savefig('graphs/pareto_gridsearch_LL_p_bfix.png', dpi=300)
    plt.show()



""" 8. 2d plot: LL vs. b (p is fix) """
if plot_ll is True:
    k=2000
    p_fix = p
    bmin, bmax = 0.1, np.min(Pareto_data) # Note: y>=b should be always valid. Also, b>0. Thus, we can only simulate LL between these 2 bounds.
    B = np.linspace(bmin, bmax, k)
    LL = []
    for j in B:
            LL.append(Pareto_ll([p_fix], x=Pareto_data, b=j))
            print(str(j), Pareto_ll([p_fix], x=Pareto_data, b=j))

    # convert to lists and sort according to P
    LL = np.array(LL)
    LL_Bsorted = LL[B.argsort()]

    # plot
    plt.figure()
    plt.gca().set_position((.15, .35, .55, .55)) # to make a bit of room for extra text

    # plt.figtext(.95, .9, "example text 123", rotation='vertical')
    # plt.figtext(.02, .02, "example text aabbcc")

    # title, label
    plt.title("Grid search of Pareto Distribution\nfor Parameter b (p is fix)"); plt.xlabel("Paramter b"); plt.ylabel("LL")

    plt.plot(B, LL_Bsorted)
    plt.plot(b, Pareto_ll([p_fix], x=Pareto_data, b=b), linestyle='None', marker='o', color='r', label="True Parameter")

    # legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

    plot_text = "True Parameters: b={}, p={} (fix); n={}\nGrid search: b(min; max)=({}; {}) steps={}".format(b, p_fix, n, bmin, bmax, k)
    side_text = plt.figtext(.25, .1, plot_text, bbox=dict(facecolor='white'), fontsize=8)
    #fig.subplots_adjust(top=0.9)
    plt.savefig('graphs/pareto_gridsearch_LL_b_pfix.png', dpi=300)
    plt.show()



""" 9. 3d contour plot """
if plot_ll is True:
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    import matplotlib.cm as cm
    k=50
    bmin, bmax = 0.1, np.min(Pareto_data) # Note: y>=b should be always valid. Also, b>0. Thus, we can only simulate LL between these 2 bounds.
    pmin, pmax = .01, 10
    b_data = np.linspace(bmin, bmax, k)
    p_data = np.linspace(pmin, pmax, k)
    B, P, LL = [], [], []
    for i in b_data:
        for j in p_data:
            B.append(i)
            P.append(j)
            LL.append(Pareto_ll([j], x=Pareto_data, b=i))
            print(str(i), str(j), Pareto_ll([j], x=Pareto_data, b=i))

    # convert to lists and sort according to P
    B, P, LL = np.array(B), np.array(P), np.array(LL)

    # x as meshgrids, needed for 3d plot
    B_plot, P_plot = np.meshgrid(B, P)
    LL_plot = Pareto_ll([P_plot], x=Pareto_data, b=B_plot)

    # True Parameter
    LL_true = Pareto_ll([p], x=Pareto_data, b=b)

    # 3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    ax.scatter(b, p, LL_true, s=100, c="r", alpha=1, label='True Parameter')
    #surf = ax.plot_wireframe(B_plot, P_plot, LL_plot, color='blue')
    surf = ax.plot_surface(B_plot, P_plot, LL_plot, cmap='summer', linewidth=.1, antialiased=False)
    # title, label, legend
    plt.title("Grid search of Pareto Distribution\nfor Parameters b, p"); ax.set_xlabel("b Parameter");
    ax.set_ylabel("p Parameter"); ax.set_zlabel("LL"); plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=.5, aspect=7)
    # perspective
    ax.view_init(elev=0, azim=40)
    #plt.savefig('graphs/pareto_gridsearch_LL_b_p.png', dpi=300)
    plt.show()
