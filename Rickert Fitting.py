
from TheoreticalModels.HopDiffusion import HopDiffusion
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

DELTA_T = 0.001
DIMENSION = 2
R = 1
L = 1

def equation_free(x, D, LOCALIZATION_PRECISION):
    TERM_1 = 2*DIMENSION*D*DELTA_T*(x-(2*R))
    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

def equation_hop(x, DM, DU, LOCALIZATION_PRECISION, L_HOP):
    TERM_1_1_1 = (DU-DM)/DU
    TERM_1_1_2 = (0.26*(L_HOP**2))/(2*DIMENSION*x*DELTA_T)
    TERM_1_1_3 = 1 - (np.exp(-((2*DIMENSION*x*DELTA_T*DU)/(0.52*(L**2)))))

    TERM_1_1 = DM + (TERM_1_1_1*TERM_1_1_2*TERM_1_1_3)

    TERM_1_2 = DELTA_T * (x-(2*R))
    TERM_1_3 = 2*DIMENSION
    TERM_1 = TERM_1_1 * TERM_1_2 * TERM_1_3

    TERM_2 = 2*DIMENSION*(LOCALIZATION_PRECISION**2)
    return TERM_1 + TERM_2

msd_results = []
c = 0
for _ in tqdm.tqdm(list(range(100))):
    model = HopDiffusion.create_random_instance()
    trajectory = model.simulate_trajectory(1000,None)
    #trajectory.animate_plot()

    def eq_9_obj_raw(x, y, dm, du, delta, l_hop): return np.sum((y - equation_hop(x, dm, du, delta, l_hop))**2)
    def eq_4_obj_raw(x, y, d, delta): return np.sum((y - equation_free(x, d, delta))**2)

    _, y = trajectory.calculate_msd_curve(bin_width=DELTA_T)
    Y = np.array(y[:int(len(y)*0.20)])
    msd_results.append(Y)
    X = np.array(range(1,len(Y)+1))

    eq_9_obj = lambda coeffs: eq_9_obj_raw(X, Y, *coeffs)
    res_eq_9s = []

    for _ in range(99):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(100, 100000), np.random.uniform(1, 100), np.random.uniform(1, 1000)]
        res_eq_9 = minimize(eq_9_obj, x0=x0, bounds=[(0, None), (0, None), (0, None), (0, None)], constraints=[{'type':'eq', 'fun': lambda t: - (t[1]/t[0]) + 5}])
        res_eq_9s.append(res_eq_9)

    res_eq_9 = min(res_eq_9s, key=lambda r: r.fun)
    
    eq_4_obj = lambda coeffs: eq_4_obj_raw(X, Y, *coeffs)
    res_eq_4s = []

    for _ in range(99):        
        x0=[np.random.uniform(100, 100000), np.random.uniform(1, 100)]
        res_eq_4 = minimize(eq_4_obj, x0=x0, bounds=[(0, None), (0, None)])
        res_eq_4s.append(res_eq_4)

    res_eq_4 = min(res_eq_4s, key=lambda r: r.fun)

    n = len(y)
    print(res_eq_4, res_eq_9)
    BIC_4 = n * np.log(res_eq_4.fun/n) + 2 * np.log(n)
    BIC_9 = n * np.log(res_eq_9.fun/n) + 4 * np.log(n)
    #print(BIC_9 < BIC_4)

    plt.plot(X, Y, color='black')
    plt.plot(X, equation_hop(X, *res_eq_9.x), color='red')
    plt.plot(X, equation_free(X, *res_eq_4.x), color='green')
    plt.show()

    # plt.plot(X, equation_hop(X, 1000, 10000, 0, 50))
    # plt.plot(X, equation_hop(X, 2000, 10000, 0, 50))
    # plt.plot(X, equation_hop(X, 3000, 10000, 0, 50))
    # plt.plot(X, equation_hop(X, 4000, 10000, 0, 50))
    # plt.show()
    
    # plt.plot(X, equation_hop(X, 1000, 10000, 0, 50))
    # plt.plot(X, equation_hop(X, 1000, 20000, 0, 50))
    # plt.plot(X, equation_hop(X, 1000, 30000, 0, 50))
    # plt.plot(X, equation_hop(X, 1000, 40000, 0, 50))
    # plt.show()

    # plt.plot(X, equation_hop(X, 1000, 10000, 0, 50))
    # plt.plot(X, equation_hop(X, 1000, 10000, 0, 100))
    # plt.plot(X, equation_hop(X, 1000, 10000, 0, 150))
    # plt.plot(X, equation_hop(X, 1000, 10000, 0, 200))
    # plt.show()

    if BIC_9 < BIC_4:
        c += 1
print(c)
#msd_results = np.vstack(msd_results)
#plt.plot(msd_results)
#plt.show()