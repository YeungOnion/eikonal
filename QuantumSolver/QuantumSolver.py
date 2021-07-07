"""
Vim modeline not working, so
run this for editing convenience
imap <C-s> schro
imap <C-d> dirac
"""

"""
This is the doc for the QuantumSolver.py file
--
This contains functions and methods designed to solve the schrodinger and dirac equations both for bound and unbound states

Last edited: April 28 2020
Recovered file: April 12 2020
Created file: Jan 16 2020
"""

import numpy as np
import scipy
import scipy.optimize
import scipy.integrate
import functools

"""
For RK4 implementation and basic scipy style output e.g
sol = rk4solve(system, tVector, ic)
t.y = sol.t, sol.y
"""

class _ODEsolution:
    """
    Helper for rk4 solution output
    """
    def __init__(self, t, y):
        self.t = t
        self.y = y

def rk4solve(f, t, ic):
    """
    integration solver for rk4 routine
    """
    def rk4step(f, t, y, dt):
        """
        RK4 one step integrator, returns an object defined by the ode f(t, D_n*y_n)
        inputs:
        f - lambda of form f_val = f(t, y) with t scalar and y n-dim list/vector/array
        outputs:
        step - lambda func that requires (t, y, dt) that outputs n-dim list/vector/array
        """
        dy1 = dt * f( t       , y         )
        dy2 = dt * f( t + dt/2, y + dy1/2 )
        dy3 = dt * f( t + dt/2, y + dy2/2 )
        dy4 = dt * f( t + dt  , y + dy3   )
        return (dy1 + 2*dy2 + 2*dy3 + dy4)/6
    
    dt = t[1]-t[0]
    Nt = t.shape[0]
    Nd = ic.shape[0]
    y = np.full((Nt, Nd), np.nan)
    y[0] = ic
    
    for i, ti in enumerate(t):
        if i+1 < len(t):
            dt = t[i+1] - t[i]
            y[i+1] = y[i] + rk4step(f, ti, y[i], dt)
    return _ODEsolution(t, y.T)


"""
The equations defining the quantum systems radial component
"""

def ODE_Schro(V, ell, E):
    """
    Callable used to wrap the system into the ODE
    --
    Ex:
      SqrWell = ODE_Schro(V=lambda r: -V0/2*( (r<R) + (r<=R) ), ell=0, E=1)
      r, U = rk4solve(SqrWell, [rmin, rmax], ic, dr)
    """
    def _ODE_Schro(r, U):
        """
        The raw ode defining the system, designed for use in runge-kutta implementation
        """
        psi, dpsi = U
        ddpsi = ( ell*(ell+1)/r**2 + 2*(V(r)-E) ) * psi
        return np.array([dpsi, ddpsi])
    return _ODE_Schro

def ODE_Dirac(V, kap, E):
    """
    Callable used to wrap the system into the ODE
    --
    Ex:
      V = lambda r: -V0/2*( (r<R) + (r<=R) )
      S = lambda r: -V0 * (np.tanh(r/R) - 1)
      system = ODE_Dirac(V=(V, S), ell=0, E=1)
      r, U = rk4solve(system, [rmin, rmax], ic, dr)
    """
    V, S = V
    def _ODE_Dirac(r, U):
        """
        The raw ode defining the system, designed for use in runge-kutta implementation
        psi = 1 / i f(r) |+kap m; +/-> \
              r \   g(r) |-kap m; +/-> /
        """
        f, g = U
        df = (-kap/r) * f - ( (E-V(r)) + (1+S(r)) ) * g
        dg = ( kap/r) * g + ( (E-V(r)) - (1+S(r)) ) * f
        # df = (-kap/r) * f - ( (V(r)-E) - (1+S(r)) ) * g
        # dg = ( kap/r) * g - ( (V(r)-E) + (1+S(r)) ) * f
        return np.array([df, dg])
    return _ODE_Dirac

def ODE(V, ellORkap, E, model):
    assert (model=='schro' or model=='dirac'), "needs to be 'schro' or 'dirac'"
    if model=='schro':
        return ODE_Schro(V, ellORkap, E)
    if model=='dirac':
        return ODE_Dirac(V, ellORkap, E)


"""
Creates ICs (near 0 and outward)
"""

def MakeIC_Schro(V, ell, E, rmin):
    return np.array([rmin, (ell+1)]) * rmin**ell

def MakeIC_Dirac(V, kap, E, rmin):
    """
    Makes IC given the problem defined by V, kappa, E, and rmin
    Inputs:
    V - 2tuple of functions - the two potentials in Dirac Eqn
    kap - [half]int - S dot L eigenvalue
    E - real scalar - energy parameter s.t. E = k**2/2 * (mc^2)
    rmin - real scalar - starting domain value to integrate ODE from
    """
    V, S = V
    j = np.fabs(kap)-.5
    l = j + np.sign(kap)/2
    icLower = (l+1+kap)  /  ( (1+S(rmin)) + (E-V(rmin)) )
    return np.array((rmin, icLower))*1e-2

def MakeIC(V, ellORkap, E, rmin, model):
    assert (model=='schro' or model=='dirac'), "needs to be 'schro' or 'dirac'"
    if model=='schro':
        return MakeIC_Schro(V, ellORkap, E, rmin)
    if model=='dirac':
        return MakeIC_Dirac(V, ellORkap, E, rmin)

"""
Creates BCs (at r=R s.t. f(r, Ui) = f(Ui) for first order system)
"""
    
def MakeBC_Schro(V, ell, E, rmin):
    k = np.sqrt(-2*E)
    return np.array([1, -k])*rmin

def MakeBC_Dirac(V, kap, E, rmin):
    bcLower = np.sqrt((1-E)/(1+E))
    return np.array([1, bcLower])*1e-2

def MakeBC(V, ellORkap, E, rmin, model):
    assert (model=='schro' or model=='dirac'), "needs to be 'schro' or 'dirac'"
    if model=='schro':
        return MakeBC_Schro(V, ellORkap, E, rmin)
    if model=='dirac':
        return MakeBC_Dirac(V, ellORkap, E, rmin)

"""
Solvers for interior
"""

# def SolveInterior_Schro(V, ell, E, rbnds, dr):
#     """
#     Given the set of parameters, sets up and integrates the radial schrodinger eq outward.
#     Used to find bound and unbound solutions
#     Inputs:
#     V - lambda r - defines the potential for the system, asymptotic to zero
#     ell - positive int - the orbital angular momentum
#     E - real scalar - energy parameter s.t. E=k**2/2 * (mc^2)
#     rbnds - 2tuple - bounds on domain of solution (rmin!=0, [rmid/rmax])
#     dr - scalar - resolution of domain
#     """
#     # domain stuff
#     rmin, rmax = rbnds
#     N = int(np.ceil(rmax/dr))
#     r, dr = np.linspace(0, rmax, N+1, retstep=True)
#     r[0] = rmin
    
#     # ode stuff
#     system = ODE_Schro(V, ell, E)
#     ic = MakeIC_Schro(V, ell, E, rmin)
    
#     # solve and output
#     sol = rk4solve(system, r, ic)
#     return sol

# def SolveInterior_Dirac(V, kap, E, rbnds, dr):
#     """
#     Given the set of parameters, sets up and integrates the radial dirac eq outward.
#     Used to find bound and unbound solutions
#     Inputs:
#     V - 2tuple (lambda r, lambda r) - defines the potential for the system, asymptotic to zero
#     kap - int/halfint - the orbital angular momentum S dot L eigenvalue
#     E - real scalar - energy parameter
#     rbnds - 2tuple - bounds on domain of solution (rmin!=0, [rmid/rmax])
#     dr - scalar - resolution of domain
#     """
#     # domain stuff
#     rmin, rmax = rbnds
#     N = int(np.ceil(rmax/dr))
#     r, dr = np.linspace(0, rmax, N+1, retstep=True)
#     r[0] = rmin
    
#     # ode stuff
#     system = ODE_Dirac(V, kap, E)
#     ic = MakeIC_Dirac(V, kap, E, rmin)
    
#     # solve and output
#     sol = rk4solve(system, r, ic)
#     return sol

def SolveInterior(V, ellORkap, E, rbnds, dr, model):
    """
    Given the set of parameters, sets up and integrates the speicifed system outward.
    Used to find bound and unbound solutions
    Inputs:
    V - lambda r [schro] / (lambda r, lambda r) [dirac] - defines the potential for the system, zero at r=Inf
    ellORkap - int/halfint - the orbital angular momentum eigenvalue
    E - real scalar - energy parameter
    rbnds - 2tuple - bounds on domain of solution (rmin!=0, [rmid/rmax])
    dr - scalar - resolution of domain
    """
    # domain stuff
    rmin, rmax = rbnds
    N = int(np.ceil(rmax/dr))
    r, dr = np.linspace(0, rmax, N+1, retstep=True)
    r[0] = rmin
    
    # ode stuff
    system = ODE(V, ellORkap, E, model)
    ic = MakeIC(V, ellORkap, E, rmin, model)
    
    # solve and output
    sol = rk4solve(system, r, ic)
    return sol

"""
Solvers for exterior
"""
    
# def SolveExterior_Schro(V, ell, E, rbnds, dr):
#     """
#     Given the set of parameters, sets up and integrates the radial schrodinger eq inward.
#     Used to find bound solutions.
#     Inputs:
#     V - lambda r - defines the potential for the system, asymptotic to zero
#     ell - positive int - the orbital angular momentum
#     E - real scalar - energy parameter
#     rbnds - 2tuple - bounds on domain of solution (rmid, rmax)
#     dr - scalar - resolution of domain
#     """
#     # domain stuff
#     rmin, rmax = rbnds
#     N = int(np.ceil(rmax/dr))
#     r, dr = np.linspace(rmax, rmin, N+1, retstep=True)
    
#     # ode stuff
#     system = ODE_Schro(V, ell, E)
#     bc = MakeBC_Schro(V, ell, E, dr)
    
#     # solve and output
#     sol = rk4solve(system, r, bc)
#     return sol

# def SolveExterior_Dirac(V, kap, E, rbnds, dr):
#     """
#     Given the set of parameters, sets up and integrates the radial dirac eq inward.
#     Used to find bound solutions
#     Inputs:
#     V - 2tuple (lambda r, lambda r) - defines the potential for the system, asymptotic to zero
#     kap - int/halfint - the orbital angular momentum S dot L eigenvalue
#     E - real scalar - energy parameter
#     rbnds - 2tuple - bounds on domain of solution (rmid, rmax)
#     dr - scalar - resolution of domain
#     """
#     # domain stuff
#     rmin, rmax = rbnds
#     N = int(np.ceil(rmax/dr))
#     r, dr = np.linspace(rmax, rmin, N+1, retstep=True)
    
#     # ode stuff
#     system = ODE_Dirac(V, kap, E)
#     bc = MakeBC_Dirac(V, kap, E, rmin)
    
#     # solve and output
#     sol = rk4solve(system, r, bc)
#     return sol

def SolveExterior(V, ellORkap, E, rbnds, dr, model):
    """
    Given the set of parameters, sets up and integrates the speicifed system outward.
    Used to find bound solutions
    Inputs:
    V - lambda r [schro] / (lambda r, lambda r) [dirac] - defines the potential for the system, zero at r=Inf
    ellORkap - int/halfint - the orbital angular momentum eigenvalue
    E - real scalar - energy parameter
    rbnds - 2tuple - bounds on domain of solution (rmid, rmax)
    dr - scalar - resolution of domain
    """
    # domain stuff
    rmin, rmax = rbnds
    N = int(np.ceil(rmax/dr))
    r, dr = np.linspace(rmax, rmin, N+1, retstep=True)
    
    # ode stuff
    system = ODE(V, ellORkap, E, model)
    ic = MakeBC(V, ellORkap, E, rmin, model)
    
    # solve and output
    sol = rk4solve(system, r, ic)
    return sol

"""
Tools for bound state solver. Defined by finding a zero of some function at boundary bw in/exterior
"""

# def BCMatch_Schro(V, ell, rbnds, dr):
#     """
#     The boundary matching condition for the smoothness of the system
#     """
#     def fzero(E):
#         Int = SolveInterior_Schro(V, ell, E, rbnds[0:2], dr)
#         Ext = SolveExterior_Schro(V, ell, E, rbnds[1:3], dr)
#         return np.linalg.det(np.array( [Int.y[:, -1], Ext.y[:, -1]] ))
#     return np.vectorize(fzero)


# def BCMatch_Dirac(V, kap, rbnds, dr):
#     """
#     The boundary matching condition for the pairwise continuity of the system
#     """
    
#     def fzero(E):
#         Int = SolveInterior_Dirac(V, kap, E, rbnds[0:2], dr)
#         Ext = SolveExterior_Dirac(V, kap, E, rbnds[1:3], dr)
#         return np.linalg.det(np.array( [Int.y[:, -1], Ext.y[:, -1]] ))
#     return np.vectorize(fzero)


def BCMatch(V, ellORkap, rbnds, dr, model):
    def fzero(E):
        Int = SolveInterior(V, ellORkap, E, rbnds[0:2], dr, model)
        Ext = SolveExterior(V, ellORkap, E, rbnds[1:3], dr, model)
        return np.linalg.det(np.array( [Int.y[:, -1], Ext.y[:, -1]] ))
    return np.vectorize(fzero)
    

def MakeSoln(V, ell, Eguess, rbnds, dr, model):
    psiInt = SolveInterior(V, ell, Eguess, rbnds[0:2], dr, model)
    psiExt = SolveExterior(V, ell, Eguess, rbnds[1:3], dr, model)
    
    psiExt.y = psiExt.y * psiInt.y[0, -1] / psiExt.y[0, -1]
    psifunc = np.hstack( (psiInt.y, np.fliplr(psiExt.y)) )
    r = np.concatenate( (psiInt.t, np.flip(psiExt.t)) )
    
    I = Norm(r, psifunc, model)
    psifunc = psifunc/I
    
    return _ODEsolution(r, psifunc)


def Norm_Schro(r, psi):
    I2 = scipy.integrate.trapz(psi[0, 1:]*psi[0, 1:], r[1:])
    return np.sqrt(I2)


def Norm_Dirac(r, psi):
    I2 = scipy.integrate.trapz(psi[0, 1:]*psi[0, 1:] + psi[1, 1:]*psi[1, 1:], r[1:])
    return np.sqrt(I2)
    

def Norm(r, psi, model):
    assert (model=='schro' or model=='dirac'), "needs to be 'schro' or 'dirac'"
    if model=='schro':
        return Norm_Schro(r, psi)
    if model=='dirac':
        return Norm_Dirac(r, psi)    
    
# def FindEigen_Schro(V, ell, Ebnds, rbnds, dr):
#     """
#     This is the method used to find the eigenvalues and functions for the bound states to schrodinger 
#     Inputs:
#     V - lambda r - defines the potential for the system, asymptotic to zero
#     ell - positive int - the orbital angular momentum
#     Ebnds - 2tuple of real scalar - energy parameter bounds
#     rbnds - 3tuple of real scalar - bounds on domain of solution (rmin, rmid, rmax)
#     dr - real scalar - resolution of domain
#     """
#     if len(rbnds)!= 3:
#         print('That''s a no can do. We need 3 points on the domain. Roger.')
#         return None
    
#     # for rootfinding the soln
#     eigenE = scipy.optimize.brentq(BCMatch_Schro(V, ell, rbnds, dr), Ebnds[0], Ebnds[1])
    
#     eigenInt = SolveInterior_Schro(V, ell, eigenE, rbnds[0:2], dr)
#     eigenExt = SolveExterior_Schro(V, ell, eigenE, rbnds[1:3], dr)
    
#     eigenExt.y = eigenExt.y * eigenInt.y[0, -1] / eigenExt.y[0, -1]
#     eigenfunc = np.hstack( (eigenInt.y, np.fliplr(eigenExt.y)) )
#     r = np.concatenate( (eigenInt.t, np.flip(eigenExt.t)) )

#     return eigenE, _ODEsolution(r, eigenfunc)
    

# def FindEigen_Dirac(V, kap, Ebnds, rbnds, dr):
#     """
#     This is the method used to find the eigenvalues and functions for the bound states to schrodinger 
#     Inputs:
#     V - lambda r - defines the potential for the system, asymptotic to zero
#     kap - int/halfint - the orbital angular momentum
#     Ebnds - 2tuple of real scalar - energy parameter bounds
#     rbnds - 3tuple of real scalar - bounds on domain of solution (rmin, rmid, rmax)
#     dr - real scalar - resolution of domain
#     """
#     if len(rbnds)!= 3:
#         print('That''s a no can do. We need 3 points on the domain. Roger.')
#         return None
    
#     # for rootfinding the soln
#     eigenE = scipy.optimize.brentq(BCMatch_Dirac(V, kap, rbnds, dr), Ebnds[0], Ebnds[1])
    
#     eigenInt = SolveInterior_Dirac(V, kap, eigenE, rbnds[0:2], dr)
#     eigenExt = SolveExterior_Dirac(V, kap, eigenE, rbnds[1:3], dr)
    
#     eigenExt.y = eigenExt.y * eigenInt.y[0, -1] / eigenExt.y[0, -1]
#     eigenfunc = np.hstack( (eigenInt.y, np.fliplr(eigenExt.y)) )
#     r = np.concatenate( (eigenInt.t, np.flip(eigenExt.t)) )

#     return eigenE, _ODEsolution(r, eigenfunc)


def FindEigen(V, kap, Ebnds, rbnds, dr, model):
    """
    This is the method used to find the eigenvalues and functions for the bound states to schrodinger 
    Inputs:
    V - lambda r - defines the potential for the system, asymptotic to zero
    kap - int/halfint - the orbital angular momentum
    Ebnds - 2tuple of real scalar - energy parameter bounds
    rbnds - 3tuple of real scalar - bounds on domain of solution (rmin, rmid, rmax)
    dr - real scalar - resolution of domain
    """
    if len(rbnds)!= 3:
        print('That''s a no can do. We need 3 points on the domain. Roger.')
        return None
    
    assert (model=='schro' or model=='dirac'), "needs to be 'schro' or 'dirac'"
    # for rootfinding the soln
    eigenE = scipy.optimize.brentq(BCMatch(V, kap, rbnds, dr, model), Ebnds[0], Ebnds[1])
    
    eigenInt = SolveInterior(V, kap, eigenE, rbnds[0:2], dr, model)
    eigenExt = SolveExterior(V, kap, eigenE, rbnds[1:3], dr, model)
    
    eigenExt.y = eigenExt.y * eigenInt.y[0, -1] / eigenExt.y[0, -1]
    eigenfunc = np.hstack( (eigenInt.y, np.fliplr(eigenExt.y)) )
    r = np.concatenate( (eigenInt.t, np.flip(eigenExt.t)) )

    return eigenE, _ODEsolution(r, eigenfunc)


"""
This section is for unbound states, same interior BC, so it's easy
"""

def SolveUnbound(V, ellORkap, E, rbnds, dr, model):
    """
    Given the set of parameters, sets up and solves the unbounded solution
    Used to find bound and unbound solutions
    Inputs:
    V - lambda r [schro] / (lambda r, lambda r) [dirac] - defines the potential for the system, zero at r=Inf
    ellORkap - int/halfint - the orbital angular momentum eigenvalue
    E - real scalar - energy parameter
    rbnds - 2tuple - bounds on domain of solution (rmin!=0, [rmid/rmax])
    dr - scalar - resolution of domain
    """

    assert (model=='schro' or model=='dirac'), "needs to be 'schro' or 'dirac'"
    return SolveInterior(V, ellORkap, E, rbnds, dr, model)


def FindWaveNumberFromEnergy(E, model):
    assert (model=='schro' or model=='dirac'), "needs to be 'schro' or 'dirac'"
    if model=='schro':
        k = np.sqrt(2*E)
    else:
        k = np.sqrt(E**2 - 1)
    return k


def MakeUnboundDomain(k, rbnds, dr, model):
    """
    Given wavenumber and domain bounds, creates numpy vector of floats for
    radius coordinate, `r`, with n=20 points on each wave, for at least m=10
    full wavelengths
    k - positive scalar float - asymptotic wavenumber
    rbnds - 2 element numpy vector, positive - bounds of domain
    """
    m, n = 0, 20
    L = 2*np.pi/k
    rmax = max((rbnds[-1], m*L))
    dr = min((L/n, dr))
    return (rbnds[0], rmax), dr


def MakeUnitAmplitude(k, sol):
    t, y = sol.t, sol.y
    L = 2*np.pi/k
    idx = np.argmin(np.abs(t-L))
    yEnd = y[0,-idx:]
    amp = np.max(yEnd)
    sol.y /= amp
    return sol


def FindPhaseShift(k, sol, ellORkap, model):
    """
    Solves matrix problem fitting the last n=0 points of the solution
    to the form psi(r;k) = A jl(kr) + B nl(kr) where delta = atan2(A,B)
    """
    # determine model
    assert (model=='schro' or model=='dirac'), "needs to be 'schro' or 'dirac'"

    n = 20 # at time of writing this is one period
    # determine effective ell (upper component is fine)
    if model=='schro':
        ellUpper = int(ellORkap)
    if model=='dirac':
        j = np.fabs(ellORkap)-.5
        ellUpper = int(j + np.sign(ellORkap)/2)

    # setup for naming, jl/nl and r, x = A(r)\b(r)
    jl = lambda z: k*z*scipy.special.spherical_jn(ellUpper, k*z)
    nl = lambda z: k*z*scipy.special.spherical_yn(ellUpper, k*z)
    r = np.array(sol.t[-n:])
    b = np.empty(n)
    A = np.empty((n, 2))

    # populate A,b
    b = sol.y[0,-n:]
    A[:n,0] = jl(r)
    A[:n,1] = nl(r)

    # solve the matrix eqn for tan(delta) based on last two points in soln
    x = np.linalg.lstsq(A, b, rcond=None)[0]

    return np.arctan2(x[1],x[0]), x[0]*jl(sol.t) + x[1]*nl(sol.t), jl(sol.t)

