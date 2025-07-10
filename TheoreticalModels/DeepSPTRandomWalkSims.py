import numpy as np
from tqdm import tqdm
from fbm import fgn, times
import matplotlib.pyplot as plt
import random
from scipy.stats import loguniform


def SquareDist(x0, x1, y0, y1):
    return (x1 - x0) ** 2 + (y1 - y0) ** 2


def msd(x, y, frac):
    N = int(len(x) * frac)
    msd = []
    for lag in range(1, N):
        msd.append(
            np.mean(
                [
                    SquareDist(x[j], x[j + lag], y[j], y[j + lag])
                    for j in range(len(x) - lag)
                ]
            )
        )
    return np.array(msd)


def Get_params(numparams, dt, random_D, multiple_dt, 
               Nrange: list = [5, 600], Brange: list = [0.1, 2], 
               Rrange: list = [2, 25], 
               subalpharange: list = [0, 0.7],
               superalpharange: list = [1.3, 2], 
               Qrange: list = [1, 16], 
               Drandomrange: list = [10e-3, 1],
               Dfixed: int = 0.1):
    # bounds from table 1 Kowalek et al 2020
    Nmin, Nmax = Nrange
    Bmin, Bmax = Brange
    Rmin, Rmax = Rrange
    sub_alphamin, sub_alphamax = subalpharange
    super_alphamin, super_alphamax = superalpharange 
    Qmin, Qmax = Qrange

    # Gen parameters
    wiggle = np.random.uniform(0, 0.01, size=numparams)
    r_stucks = np.random.uniform(0, 0.01, size=numparams)
    Q = np.random.uniform(Qmin, Qmax, size=numparams)
    B = np.random.uniform(Bmin, Bmax, size=numparams) 
    Q1, Q2 = Q, Q

    if multiple_dt:
        raise ValueError("Multiple dt not implemented")   
    else:
        dt = dt

    NsND = np.random.randint(Nmin, Nmax, size=numparams)
    NsAD = np.random.randint(Nmin, Nmax, size=numparams)
    NsCD = np.random.randint(Nmin, Nmax, size=numparams)
    NsDM = np.random.randint(Nmin, Nmax, size=numparams)
    NstD = np.random.randint(Nmin, Nmax, size=numparams)
    TDM = NsDM * dt

    if random_D:
        Dmin, Dmax = Drandomrange
        D = loguniform.rvs(Dmin, Dmax, size=numparams)
    else:
        Dval = Dfixed
        D = np.random.uniform(Dval, Dval, size=numparams)

    # setting fixed confinement size instead of track length dependent
    angles = np.random.randint(0,180, size=(numparams,3))
    ellipse_dim = np.random.uniform(Bmin, Bmax, size=(numparams,3)) 
    r_c = np.sqrt(D * NsCD * dt / B)  # solving for r_c in eq. 8 Kowalek

    R = np.random.uniform(Rmin, Rmax, size=numparams)
    v = np.sqrt(R * 4 * D / TDM)  # solving for v in eq. 7 Kowalek

    subalpha = np.random.uniform(sub_alphamin, sub_alphamax, size=numparams)
    superalpha = np.random.uniform(super_alphamin, super_alphamax, size=numparams)

    # Compute sigma for ND, AD, CD and StD (stuck) from eq. 12 Kowalek
    sigmaND = np.sqrt(D * dt) / Q1
    sigmaAD = np.sqrt(D * dt) / Q1
    sigmaCD = np.sqrt(D * dt) / Q1
    sigmaStD = np.sqrt(D * dt) / Q1

    # Compute sigma for DM from eq. 12 Kowalek
    sigmaDM = np.sqrt(D * dt + v ** 2 * dt ** 2) / Q2

    return [NsND,
            NsAD,
            NsCD,
            NsDM,
            NstD,
            D * np.ones(numparams),
            dt * np.ones(numparams),
            D,
            r_c,
            ellipse_dim,
            angles,
            v,
            wiggle,
            r_stucks,
            subalpha,
            superalpha,
            sigmaND,
            sigmaAD,
            sigmaCD,
            sigmaDM,
            sigmaStD
            ]
    


def Gen_normal_diff(Ds, dt, sigma1s, Ns, dim=2, initial_state=[], withlocerr=True, min_len=5):
    traces = []
    if dim == 2:
        input_initial_state = initial_state
        initial_state = [0,0] if len(initial_state)==0 else initial_state
        x0, y0 = initial_state[0], initial_state[1]
        for D, n, sig in zip(Ds, Ns, sigma1s):
            if len(input_initial_state)>0 and n==min_len:
                n += 1
            xsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n)
            ysteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n)
            x, y = (
                np.concatenate([[x0], np.cumsum(xsteps)+x0]),
                np.concatenate([[y0], np.cumsum(ysteps)+y0]),
            )
            if withlocerr:
                x_noisy, y_noisy = (
                    x + np.random.normal(0, sig, size=x.shape),
                    y + np.random.normal(0, sig, size=y.shape),
                )
                traces.append(np.array([x_noisy, y_noisy]).T)
            else:
                traces.append(np.array([x, y]).T)
    if dim == 3:
        input_initial_state = initial_state
        initial_state = [0,0,0] if len(initial_state)==0 else initial_state
        x0, y0, z0 = initial_state[0], initial_state[1], initial_state[2]
        for D, n, sig in zip(Ds, Ns, sigma1s):
            if len(input_initial_state)>0 and n==min_len:
                n += 1
            xsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n)
            ysteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n)
            zsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n)
            x, y, z = (
                np.concatenate([[x0], np.cumsum(xsteps)+x0]),
                np.concatenate([[y0], np.cumsum(ysteps)+y0]),
                np.concatenate([[z0], np.cumsum(zsteps)+z0])
            )
            if withlocerr:
                x_noisy, y_noisy, z_noisy = (
                    x + np.random.normal(0, sig, size=x.shape),
                    y + np.random.normal(0, sig, size=y.shape),
                    z + np.random.normal(0, sig, size=z.shape)
                )
                traces.append(np.array([x_noisy, y_noisy, z_noisy]).T)
            else:
                traces.append(np.array([x, y, z]).T)

    if len(input_initial_state)>0:
        traces = [t[1:] for t in traces]

    return traces


def Gen_directed_diff(Ds, dt, vs, sigmaDM, Ns, dim=2, beta_set=None, initial_state=[], withlocerr=True, min_len=5):
    traces = []
    if dim == 2:
        input_initial_state = initial_state
        initial_state = [0,0] if len(initial_state)==0 else initial_state
        x0, y0 = initial_state[0], initial_state[1]

        for D, v, n, sig in zip(Ds, vs, Ns, sigmaDM):
            if len(input_initial_state)>0 and n==min_len:
                n += 1
            if beta_set is None:
                beta = np.random.uniform(0, 2 * np.pi)
            else:
                beta = beta_set
            dx, dy = v * dt * np.cos(beta), v * dt * np.sin(beta)

            xsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n) + dx
            ysteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n) + dy

            x, y = (
                np.concatenate([[x0], np.cumsum(xsteps)+x0]),
                np.concatenate([[y0], np.cumsum(ysteps)+y0]),
            )

            if withlocerr:
                x_noisy, y_noisy = (
                    x + np.random.normal(0, sig, size=x.shape),
                    y + np.random.normal(0, sig, size=y.shape),
                )
                traces.append(np.array([x_noisy, y_noisy]).T)
            else:
                traces.append(np.array([x, y]).T)
    if dim == 3:
        input_initial_state = initial_state
        initial_state = [0,0,0] if len(initial_state)==0 else initial_state
        x0, y0, z0 = initial_state[0], initial_state[1], initial_state[2]

        for D, v, n, sig in zip(Ds, vs, Ns, sigmaDM):
            if len(input_initial_state)>0 and n==min_len:
                n += 1
            if beta_set is not None: 
              theta, phi = beta_set
            else:
              theta_set, phi_set = None, None

            if theta_set is None:
                theta = np.random.uniform(0, np.pi)
            if phi_set is None:
                phi = np.random.uniform(0, 2 * np.pi)

            dx, dy, dz = v * dt * np.sin(phi)*np.cos(theta), v * dt * np.sin(phi)*np.sin(theta), v * dt * np.cos(phi)

            xsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n) + dx
            ysteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n) + dy
            zsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=n) + dz

            x, y, z = (
                np.concatenate([[x0], np.cumsum(xsteps)+x0]),
                np.concatenate([[y0], np.cumsum(ysteps)+y0]),
                np.concatenate([[z0], np.cumsum(zsteps)+z0])
            )

            if withlocerr:
                x_noisy, y_noisy, z_noisy = (
                    x + np.random.normal(0, sig, size=x.shape),
                    y + np.random.normal(0, sig, size=y.shape),
                    z + np.random.normal(0, sig, size=z.shape)
                )
                traces.append(np.array([x_noisy, y_noisy, z_noisy]).T)
            else:
                traces.append(np.array([x, y, z]).T)
    if len(input_initial_state)>0:
        traces = [t[1:] for t in traces]
    return traces


def _Take_subdiff_step(x0, y0, z0, D, dt, r_c, initial_state, dim=2, nsubsteps=100):
    dt_prim = dt / nsubsteps
    if dim==2:
        for i in range(nsubsteps):
            x1, y1 = (
                x0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
                y0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
            )
            if np.sqrt((x1-initial_state[0]) ** 2 + (y1-initial_state[1]) ** 2) < r_c:
                x0, y0 = x1, y1
        return x0, y0
    if dim==3:
        for i in range(nsubsteps):
            x1, y1, z1 = (
                x0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
                y0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
                z0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
            )
            if np.sqrt((x1-initial_state[0]) ** 2 + (y1-initial_state[1]) ** 2 + (z1-initial_state[2]) ** 2) < r_c:
                x0, y0, z0 = x1, y1, z1
        return x0, y0, z0
        

def Gen_confined_diff(Ds, dt, r_cs, sigmaCD, Ns, dim=2, initial_state=[], withlocerr=True, min_len=5):
    input_initial_state = initial_state
    def get_trace(x, dim, initial_state, withlocerr, min_len):
        if dim==2:
            D, dt, r_c, sig, n = x
            if len(initial_state)>0 and n==min_len:
                n += 1
            initial_state = [0,0] if len(initial_state)==0 else initial_state
            x0, y0 = initial_state[0], initial_state[1]
            xs, ys = [], []
            for i in range(n + 1):
                if (x0==0 and y0==0) or i>0:
                    xs.append(x0)
                    ys.append(y0)
                z0 = np.zeros_like(x0)
                x0, y0 = _Take_subdiff_step(x0, y0, z0, D, dt, r_c, initial_state)
            x, y = np.array(xs), np.array(ys)
            if withlocerr:
                x_noisy, y_noisy = (
                    x + np.random.normal(0, sig, size=x.shape),
                    y + np.random.normal(0, sig, size=y.shape),
                )
            else:
                x_noisy, y_noisy = x, y
            
            return np.array([x_noisy, y_noisy]).T
        if dim==3:
            D, dt, r_c, sig, n = x
            if len(initial_state)>0 and n==min_len:
                n += 1
                
            initial_state = [0,0,0] if len(initial_state)==0 else initial_state
            x0, y0, z0 = initial_state[0], initial_state[1], initial_state[2]
            xs, ys, zs = [], [], []

            for i in range(n + 1):
                if (x0==0 and y0==0 and z0==0) or i>0:
                    xs.append(x0)
                    ys.append(y0)
                    zs.append(z0)
                x0, y0, z0 = _Take_subdiff_step(x0, y0, z0, D, dt, r_c, initial_state, dim=dim)
            x, y, z = np.array(xs), np.array(ys), np.array(zs)
            if withlocerr:
                x_noisy, y_noisy, z_noisy = (
                    x + np.random.normal(0, sig, size=x.shape),
                    y + np.random.normal(0, sig, size=y.shape),
                    z + np.random.normal(0, sig, size=z.shape)
                )
            else:
                x_noisy, y_noisy, z_noisy = x, y, z
            
            return np.array([x_noisy, y_noisy, z_noisy]).T

    args = [(D, dt, r, sig, N) for D, r, sig, N in zip(Ds, r_cs, sigmaCD, Ns)]

    traces = []
    for i in range(len(Ns)):
        traces.append(get_trace(args[i], dim, initial_state, withlocerr, min_len))
    
    if len(input_initial_state)>0:
        traces = [t[1:] for t in traces]
    return traces


def New_Take_subdiff_step(x0, y0, z0, D, dt, center_ellipse, angles, ellipse_dim, dim=2, nsubsteps=100):
    dt_prim = dt / nsubsteps
    if dim==2:
        for i in range(nsubsteps):
            x1, y1, z1 = (
                x0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
                y0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
                np.zeros_like(x0)
            )
            if in_ellipse(x1,y1,z1, center_ellipse, angles, ellipse_dim):
                x0, y0 = x1, y1
        return x0, y0
    if dim==3:
        for i in range(nsubsteps):
            x1, y1, z1 = (
                x0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
                y0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
                z0 + np.random.normal(0, np.sqrt(2 * D * dt_prim)),
            )

            if in_ellipse(x1, y1, z1, center_ellipse, angles, ellipse_dim):
                x0, y0, z0 = x1, y1, z1
        return x0, y0, z0
        

def Gen_new_confined_diff(Ds, dt, sigmaCD, Ns, dim=2, initial_state=[], withlocerr=True, min_len=5, angles=[0,0], ellipse_dims=[0,0]):
    input_initial_state = initial_state
    def get_trace(x, dim, initial_state, withlocerr, min_len):
        if dim==2:
            D, dt, sig, n, angle, ellipse_dim = x
            if len(initial_state)>0 and n==min_len:
                n += 1

            initial_state = [0,0] if len(initial_state)==0 else initial_state
            x0, y0 = initial_state[0], initial_state[1]
            xs, ys = [], []
            for i in range(n + 1):
                if (x0==0 and y0==0) or i>0:
                    xs.append(x0)
                    ys.append(y0)
                z0 = np.zeros_like(x0)
                x0, y0 = New_Take_subdiff_step(x0, y0, z0, D, dt, initial_state, angle, ellipse_dim)
            x, y = np.array(xs), np.array(ys)
            if withlocerr:
                x_noisy, y_noisy = (
                    x + np.random.normal(0, sig, size=x.shape),
                    y + np.random.normal(0, sig, size=y.shape),
                )
            else:
                x_noisy, y_noisy = x, y
            
            return np.array([x_noisy, y_noisy]).T
        if dim==3:
            D, dt, sig, n, angle, ellipse_dim = x
            if len(initial_state)>0 and n==min_len:
                n += 1
                
            initial_state = [0,0,0] if len(initial_state)==0 else initial_state
            x0, y0, z0 = initial_state[0], initial_state[1], initial_state[2]
            xs, ys, zs = [], [], []
            for i in range(n + 1):
                if (x0==0 and y0==0 and z0==0) or i>0:
                    xs.append(x0)
                    ys.append(y0)
                    zs.append(z0)

                x0, y0, z0 = New_Take_subdiff_step(x0, y0, z0, D, dt, initial_state, angle, ellipse_dim, dim=dim)
            x, y, z = np.array(xs), np.array(ys), np.array(zs)
            if withlocerr:
                x_noisy, y_noisy, z_noisy = (
                    x + np.random.normal(0, sig, size=x.shape),
                    y + np.random.normal(0, sig, size=y.shape),
                    z + np.random.normal(0, sig, size=z.shape)
                )
            else:
                x_noisy, y_noisy, z_noisy = x, y, z
            
            return np.array([x_noisy, y_noisy, z_noisy]).T

    args = [(D, dt, sig, N, angle, ellipse_dim) for D, sig, N, angle, ellipse_dim in zip(Ds, sigmaCD, Ns, angles, ellipse_dims)]

    traces = []
    for i in range(len(Ns)):
        traces.append(get_trace(args[i], dim, initial_state, withlocerr, min_len))
    
    if len(input_initial_state)>0:
        traces = [t[1:] for t in traces]
    return traces



def Gen_anomalous_diff(Ds, dt, alphs, sigmaAD, Ns, dim=2, initial_state=[], withlocerr=True, min_len=5):
    Hs = alphs / 2
    traces = []
    if dim == 2:
        input_initial_state = initial_state
        initial_state = [0,0] if len(initial_state)==0 else initial_state
        x0, y0 = initial_state[0], initial_state[1]

        for D, n, sig, H in zip(Ds, Ns, sigmaAD, Hs):
            if len(input_initial_state)>0 and n==min_len:
                n += 1
            if max(alphs)<=1:
                while H>0.45:
                    H = np.random.uniform()
            if max(alphs)>=1:
                while H<0.55:
                    H = np.random.uniform()
            n = int(n)
            stepx, stepy = (
                np.sqrt(2 * D * dt) * fgn(n=n, hurst=H, length=n, method="daviesharte"),
                np.sqrt(2 * D * dt) * fgn(n=n, hurst=H, length=n, method="daviesharte"),
            )
            x, y = (
                np.concatenate([[x0], np.cumsum(stepx)+x0]),
                np.concatenate([[y0], np.cumsum(stepy)+y0]),
            )
            x_noisy, y_noisy = (
                x + np.random.normal(0, sig, size=x.shape),
                y + np.random.normal(0, sig, size=y.shape),
            )
            if withlocerr:
                traces.append(np.array([x_noisy, y_noisy]).T)
            else:
                traces.append(np.array([x, y]).T)
    
    if dim == 3:
        input_initial_state = initial_state
        initial_state = [0,0,0] if len(initial_state)==0 else initial_state
        x0, y0, z0 = initial_state[0], initial_state[1], initial_state[2]

        for D, n, sig, H in zip(Ds, Ns, sigmaAD, Hs):
            if len(input_initial_state)>0 and n==min_len:
                n += 1
            if max(alphs)<=1:
                while H>0.45:
                    H = np.random.uniform()
            if max(alphs)>=1:
                while H<0.55:
                    H = np.random.uniform()
            n = int(n)
            stepx, stepy, stepz = (
                np.sqrt(2 * D * dt) * fgn(n=n, hurst=H, length=n, method="daviesharte"),
                np.sqrt(2 * D * dt) * fgn(n=n, hurst=H, length=n, method="daviesharte"),
                np.sqrt(2 * D * dt) * fgn(n=n, hurst=H, length=n, method="daviesharte")
            )
            x, y, z = (
                np.concatenate([[x0], np.cumsum(stepx)+x0]),
                np.concatenate([[y0], np.cumsum(stepy)+y0]),
                np.concatenate([[z0], np.cumsum(stepz)+z0])
            )
            x_noisy, y_noisy, z_noisy = (
                x + np.random.normal(0, sig, size=x.shape),
                y + np.random.normal(0, sig, size=y.shape),
                z + np.random.normal(0, sig, size=z.shape)
            )
            if withlocerr:
                traces.append(np.array([x_noisy, y_noisy, z_noisy]).T)
            else:
                traces.append(np.array([x, y, z]).T)

    if len(input_initial_state)>0:
        traces = [t[1:] for t in traces]
    return traces


def _Take_stuck_step(dt, x0, y0, z0, wiggle, r_stuck, initial_state, dim=2, nsubsteps=100):
    dt_prim = dt / nsubsteps
    if dim==2:
        for i in range(nsubsteps):
            x1, y1 = (
                x0 + np.random.normal(0, wiggle),
                y0 + np.random.normal(0, wiggle)
            )
            if np.sqrt((x1-initial_state[0]) ** 2 + (y1-initial_state[1]) ** 2) < r_stuck:
                x0, y0 = x1, y1
        return x0, y0
    if dim==3:
        for i in range(nsubsteps):
            x1, y1, z1 = (
                x0 + np.random.normal(0, wiggle),
                y0 + np.random.normal(0, wiggle),
                z0 + np.random.normal(0, wiggle),
            )
            if np.sqrt((x1-initial_state[0]) ** 2 + (y1-initial_state[1]) ** 2 + (z1-initial_state[2]) ** 2) < r_stuck:
                x0, y0, z0 = x1, y1, z1
        return x0, y0, z0

def Gen_stuck_diff(wiggle, dt, r_stuck, sigmastD, Ns, dim=2, initial_state=[], withlocerr=True):
    def get_trace(x, dt, dim, initial_state, withlocerr):
        if dim==2:
            initial_state = [0,0] if len(initial_state)==0 else initial_state
            x0, y0 = initial_state[0], initial_state[1]
            r_stuck, wiggle, sig, n = x
            xs, ys = [], []

            for i in range(n + 1):
                if (x0==0 and y0==0) or i>0:
                    xs.append(x0)
                    ys.append(y0)
                z0 = np.zeros_like(x0)
                x0, y0 = _Take_stuck_step(dt, x0, y0, z0, wiggle, r_stuck, initial_state)
            x, y = np.array(xs), np.array(ys)
            if withlocerr:
                x_noisy, y_noisy = (
                    x + np.random.normal(0, sig, size=x.shape),
                    y + np.random.normal(0, sig, size=y.shape),
                )
            else:
                x_noisy, y_noisy = x, y
            
            return np.array([x_noisy, y_noisy]).T

        if dim==3:
            initial_state = [0,0,0] if len(initial_state)==0 else initial_state
            x0, y0, z0 = initial_state[0], initial_state[1], initial_state[2]
            r_stuck, wiggle, sig, n = x
            xs, ys, zs = [], [], []

            for i in range(n + 1):
                if (x0==0 and y0==0 and z0==0) or i>0:
                    xs.append(x0)
                    ys.append(y0)
                    zs.append(z0)
                x0, y0, z0 = _Take_stuck_step(dt, x0, y0, z0, wiggle, r_stuck, initial_state, dim=dim)
            x, y, z = np.array(xs), np.array(ys), np.array(zs)
            if withlocerr:
                x_noisy, y_noisy, z_noisy = (
                    x + np.random.normal(0, sig, size=x.shape),
                    y + np.random.normal(0, sig, size=y.shape),
                    z + np.random.normal(0, sig, size=z.shape)
                )
            else:
                x_noisy, y_noisy, z_noisy = x, y, z
            
            return np.array([x_noisy, y_noisy, z_noisy]).T

    args = [(w, r, sig, N) for w, r, sig, N in zip(wiggle, r_stuck, sigmastD, Ns)]

    traces = []
    for i in range(len(Ns)):
        traces.append(get_trace(args[i], dt, dim, initial_state, withlocerr))
    
    if len(initial_state)>0:
        traces = [t[1:] for t in traces]
    return traces
    

def subtrace_lengths(N, n_changepoints, min_len):
        """
        Create list of subtrace lengths.
        min_len: minimum length
        n_changepoints: number of subtraces
        N: overall length
        """
        pieces = []
        for idx in range(n_changepoints-1):
            # Draw number from minimum to max subtracted current total
            # and subtracted (n_changepoints-idx-1)*min_len to ensure
            # all subtraces can be minimum length        
            r = random.randint(min_len, N-sum(pieces)-(n_changepoints-idx-1)*min_len)
            pieces.append(r)
        pieces.append(int(N-sum(pieces)))
        return pieces

def Gen_changing_diff(n_traces, max_changepoints, min_parent_len, 
                      total_parents_len, dt, n_classes=4, dim=2,
                      random_D=False, multiple_dt=False,
                      Nrange: list = [5, 600], Brange: list = [0.005, 0.250], 
                      Rrange: list = [2, 25], 
                      subalpharange: list = [0, 0.7],
                      superalpharange: list = [1.3, 2], 
                      Qrange: list = [1, 16], 
                      Drandomrange: list = [10e-3, 1],
                      Dfixed: int = 0.1,
                      DMtype: str = 'active'):
    """
    input:
        n_traces: number of traces to create
        n_changepoints: number of changepoints in trace
        min_parent_len: min length of subtrace
    """
    params_matrix = Get_params(n_traces, dt, random_D, multiple_dt,
                               Nrange = Nrange, Brange = Brange, 
                               Rrange = Rrange, 
                               subalpharange = subalpharange,
                               superalpharange = superalpharange, 
                               Qrange = Qrange, 
                               Drandomrange = Drandomrange,
                               Dfixed = Dfixed)
    N, *_ = [params_matrix[i] for i in range(5)]
    Ds, r_c, ellipse_dims, angles, vs, wiggle, r_stuck, subalphas, superalphas, sigmaND, sigmaAD, sigmaCD, sigmaDM, sigmaStD = params_matrix[7:]
    trace_list = []
    label_list = []
    for i in range(n_traces):
        diff_types = []
        n_changepoints = np.random.choice(list(range(2, max_changepoints+1)))
        while len(diff_types)<=1:
            diff_types = random.choices(list(range(n_classes)), k=n_changepoints)
            diff_types = [diff_types[i] for i in range(0, len(diff_types)) if diff_types[i]!=diff_types[i-1]]
        while N[i]<min_parent_len*len(diff_types):
            N[i] = random.randint(min_parent_len*len(diff_types), total_parents_len)
        pieces = subtrace_lengths(N[i], len(diff_types), min_parent_len)
        trace = np.empty((0,dim))
        label = []
        for j, diff in enumerate(diff_types):
            n = pieces[j]
            initial_state = [] if len(trace)==0 else trace[-1]
            if diff == 0:
                subtrace = Gen_normal_diff([Ds[i]], dt, [sigmaND[i]], [n], initial_state=initial_state, dim=dim, min_len=min_parent_len, withlocerr=False)[0]
                trace = np.concatenate([trace, subtrace])
                label += [0]*len(subtrace)
            elif diff == 1:
                if DMtype=='active':
                    subtrace = Gen_directed_diff([Ds[i]], dt, [vs[i]], [sigmaDM[i]], [n], initial_state=initial_state, dim=dim, min_len=min_parent_len, withlocerr=False)[0]
                elif DMtype=='super':
                    subtrace = Gen_anomalous_diff([Ds[i]], dt, np.array([superalphas[i]]), [sigmaDM[i]], [n], initial_state=initial_state, dim=dim, min_len=min_parent_len, withlocerr=False)[0]
                trace = np.concatenate([trace, subtrace])
                label += [1]*len(subtrace)
            elif diff == 2:
                # Gen_new_confined_diff([Ds[i]], [r_c[i]], dt, [sigmaCD[i]], [n], initial_state=initial_state, dim=dim, min_len=min_parent_len)[0]
                subtrace = Gen_new_confined_diff([Ds[i]], dt, [sigmaCD[i]], [n], initial_state=initial_state, dim=dim, min_len=min_parent_len, ellipse_dims=[ellipse_dims[i]], angles=[angles[i]], withlocerr=False)[0]
                trace = np.concatenate([trace, subtrace])
                label += [2]*len(subtrace)
            elif diff == 3:
                subtrace = Gen_anomalous_diff([Ds[i]], dt, np.array([subalphas[i]]), [sigmaAD[i]], [n], initial_state=initial_state, dim=dim, min_len=min_parent_len, withlocerr=False)[0]
                trace = np.concatenate([trace, subtrace])
                label += [3]*len(subtrace)
            else:
                subtrace = Gen_stuck_diff([wiggle[i]], dt, [r_stuck[i]], [sigmaStD[i]], [n], dim=dim, initial_state=initial_state, withlocerr=False)[0]
                trace = np.concatenate([trace, subtrace])
                label += [3]*len(subtrace)

        trace_list.append(trace)
        label_list.append(label)
        
    return trace_list, label_list


def in_ellipse(x,y,z, center_ellipse, angles, ellipse_dim):
    """
    Finds the normalised distance of the point from the cell centre, 
    where a distance of 1 would be on the ellipse, less than 1 is inside, 
    and more than 1 is outside
    See: https://en.wikipedia.org/wiki/Rotation_of_axes
    """

    if len(center_ellipse)==2:
        angle = angles[0]
        ellipse_width, ellipse_height = ellipse_dim[0], ellipse_dim[1]

        # Find the point's x and y coordinates relative to the ellipse centre
        xc = x - center_ellipse[0]
        yc = y - center_ellipse[1]

        # Transform those using the angle to be the coordinates along the major and minor axes
        cos_angle = np.cos(np.radians(180.-angle))
        sin_angle = np.sin(np.radians(180.-angle))
        xct = xc * cos_angle - yc * sin_angle
        yct = xc * sin_angle + yc * cos_angle 

        rad_cc = (xct**2/(ellipse_width/2.)**2) + (yct**2/(ellipse_height/2.)**2)

    
    if len(center_ellipse)>2:
        ellipse_width, ellipse_height, ellipse_depth = ellipse_dim[0], ellipse_dim[1], ellipse_dim[2]
        yaw = angles[0]
        pitch = angles[1]
        roll = angles[2]
        yaw_rad = np.radians(180.-yaw)
        pitch_rad = np.radians(180.-pitch)
        roll_rad = np.radians(180.-roll)

        xc = x - center_ellipse[0]
        yc = y - center_ellipse[1]
        zc = z - center_ellipse[2]

        xct = xc*np.cos(yaw_rad)*np.cos(pitch_rad)+\
              yc*(np.cos(yaw_rad)*np.sin(pitch_rad)*np.sin(roll_rad)-np.sin(yaw_rad)*np.cos(roll_rad))+\
              zc*(np.cos(yaw_rad)*np.sin(pitch_rad)*np.cos(roll_rad)+np.sin(yaw_rad)*np.sin(roll_rad))


        yct = xc*np.sin(yaw_rad)*np.cos(pitch_rad)+\
              yc*(np.sin(yaw_rad)*np.sin(pitch_rad)*np.sin(roll_rad)+np.cos(yaw_rad)*np.cos(roll_rad))+\
              zc*(np.sin(yaw_rad)*np.sin(pitch_rad)*np.cos(roll_rad)-np.cos(yaw_rad)*np.sin(roll_rad))

              
        zct = -xc*np.sin(pitch_rad)+\
               yc*np.cos(pitch_rad)*np.sin(roll_rad)+\
               zc*np.cos(pitch_rad)*np.cos(roll_rad)

        rad_cc = (xct**2/(ellipse_width/2.)**2) + (yct**2/(ellipse_height/2.)**2) + (zct**2/(ellipse_depth/2.)**2) 
    return np.array(rad_cc <= 1.)