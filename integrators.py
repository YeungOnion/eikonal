r"""
This will be a module that will support the eikonal routines

In it will contain integration schemes for hankel transforms
for compact and long range functions as well as some other tools
that may or may not prove useful
"""


import numpy as np
from scipy.special import j0 as j0, j1 as j1, jv as jv, k0 as k0
from scipy.special import gamma as gamma
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.special
import scipy.ndimage
import hankel
import matplotlib.pyplot as plt
from scipy.optimize import fsolve as fsolve


def magni_FHT(f, N, alp=None, R=1, Q=1, S=None):
    r"""
    Returns FHT{f(x); from 0 to 1}=F(y) and domain points (log spaced)
    Inputs:
    - f, callable input signal
    - N, number of points
    - alp, resolution parameter
    - R, length providing "nearly" compact support
    - Q, wavenumber providing "                "
    - S, Fresnel Number = [aperture]^2 / (L * lam)
    """

    def phi0_correction(alpha):
        expalpha = np.exp(alpha)
        t1 = (2+expalpha)*expalpha
        t2 = (1+expalpha)**-2
        t3 = (1-expalpha)*(1+expalpha)
        return t1*t2*t3
    def roundN(N):
        lgN = np.log2(N)
        return 2**np.ceil(lgN)

    # re/set parameters
    N = roundN(N)
    if alp is None:
        alp = 1e-2
    else:
        alp_eqn = lambda z: z + np.log(1-np.exp(-z))/(N-1)
        alp = fsolve(alp_eqn, N**-.765534)
        # print(f"first ord:{N**-.765534}")
        # print(f"sol:{alp}")

    # set convenient values
    x0 = (np.exp(alp) + 1) * np.exp(-N*alp) / 2
    n = np.arange(N+1)
    two_n = np.arange(2*N)
    if S is None:
        S = Q*R # unitless scale 


    # make arrays in logspace for func evals
    x = x0*np.exp(alp*n) # N+1 elems
    y = x0*np.exp(alp*n)[:-1] # N elems
    ksi = np.exp(alp*(n-N)) # N+1 elems
    ksi[0] = 0

    # evaluate for convolution
    print('\tcalling func', flush=True)
    f = f(R*x) # N+1 elems
    phi = (f[:-1] - f[1:]) * ksi[1:] # N elems
    phi[0] *= phi0_correction(alp)

    # setup for xcorrelation
    print('\tevalling bessels', flush=True)
    # def j1_exp(z):
    #     sin_term = 3/4/np.sqrt(2*np.pi)/z**1.5
    #     cos_term =-2/np.sqrt(2*np.pi)/z**0.5
    #     return np.sin(z+np.pi/4)*sin_term + np.cos(z+np.pi/4)*cos_term
    # xy = S*x0*np.exp(alp*(two_n + 1 - N))
    # j = np.piecewise(xy, xy<1e4, (j1, j1_exp))
    j = scipy.special.j1(
            S*x0*np.exp(alp*(two_n + 1 - N))
            )

    # convolve and invert coordinate change 
    print('\tconvolving', flush=True)
    F = 1/y * np.correlate(j, np.conj(phi))[:-1]
    print('\toutputting', flush=True)
    return (Q*y, R/Q*F), (R*x, f)


def magni_FHT_nu1(f, N, alp=None, R=1, Q=1, S=None):
    r"""
    Returns FHT_nu=1{f(x); from 0 to 1}=F(y) and domain points (log spaced)
    Inputs:
    - f, callable input signal
    - N, number of points
    - alp, resolution parameter
    - R, length providing "nearly" compact support
    - Q, wavenumber providing "                "
    - S, Fresnel Number = [aperture]^2 / (L * lam)
    """


    def diff_exp(g, y):
        a = np.diff(np.log(y)).mean()
        dg = np.empty(g.shape, dtype=g.dtype)
        dy = np.empty(g.shape)

        dg[1:-1] = g[2:]-g[:-2]
        dg[0] = -3/2*g[0] + 2*g[1] - 1/2*g[2]
        dg[-1] = 3/2*g[-1] - 2*g[-2] + 1/2*g[-3]

        dy[1:-1] = 2*a*y[1:-1]
        dy[0] = 2*a*y[0]
        dy[-1] = 2*a*y[-1]
        dgdy = dg/dy
        return dgdy

    def f_0(b):
        b_linear = np.linspace(b[0], b[-1], 2**16)
        fval = f(b_linear)
        spl_r = spline(b_linear, fval.real, k=3)
        spl_i = spline(b_linear, fval.imag, k=3)

        dspl_r = spl_r.derivative()
        dspl_i = spl_i.derivative()
        dspl = dspl_r(b) + 1j*dspl_i(b)

        fval = f(b)

        return dspl + fval/b

    (q, F), (b, f_eval) = magni_FHT(f_0, N, alp, R, Q, S)
    F1 = F/q
    return (q, F1), (b, f_eval)


def test_magni():
    def exact(q): return k0(q)
    fig, axs = plt.subplots(2,1, sharex=True)
    for order, color in zip(range(8,12), ('r-','g-','c-','m-')):
        (q, F), (b, f) = magni_FHT(lambda b: 1/(1+b**2),
                N=2**order, alp=1e-2, R=100, Q=100)
        axs[0].plot(q, F, color, markersize=.5, label=f'N={2**order}')
        axs[1].semilogy(q, np.abs(F - exact(q)), color, label=f'N={2**order}')

    axs[0].plot(q, exact(q), 'b', label='exact', zorder=-1)
    axs[0].set_xlim((0, 10))
    axs[0].legend()
    axs[1].legend()
    plt.savefig('data/amplitudes/magni_conv.png', dpi=800)
    plt.show()


if __name__ == '__main__':
    test_magni()
