#!/usr/bin/env python
# coding: utf-8

# In[1]:


# should be using python >=3.7
import sys
print(*sys.version_info, sep='.')


# In[2]:


# setup for plotting
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

# load numeric libs
import numpy as np
import scipy.integrate
import scipy
import QuantumSolver as quantum
TWOPI = 2*np.pi

# plotting utilities
def CenterYAx(ax=None):
    if ax is None:
        ax = plt.gca()
    ylims = ax.get_ylim()
    ylim = max(np.fabs(ylims))
    ax.set_ylim((-ylim, ylim))

def PlotSolns(sol, color, labelStr=None, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(sol.t, sol.y[0], c=color, label=labelStr)
#     ax.plot(sol.t, sol.y[1], '--', c=color)
    CenterYAx(plt.gca())


# # Units
# $$
# \hbar c = 197.326\,{\rm{MeV\,fm}} = mc^2 = 939\,\rm{MeV} = 1
# $$
# 
# Gives that
# $[E]=mc^2=939\,\rm{MeV}$, $[L]=\hbar c/mc^2=0.2101\rm{fm}$
# 
# And these eqns
# 
# ## Schrodinger radial
# 
# With solutions of the form,
# $$
# \newcommand{\bra}[1]{\langle{#1}|}
# \newcommand{\ket}[1]{|{#1}\rangle}
# \langle \vec{r}\ket{\psi} = \frac{U_{El}(r)}{r} Y_{lm}(\theta,\phi)
# $$
# 
# with [binding] energy eigenvalues and vectors determined from this system,
# $$
# \newcommand{\dd}[1][]{\rm{d}#1}
# \newcommand{\dv}[2][]{\frac{\dd{#1}}{\dd{#2}}}
# \biggl[-\frac{1}{2}\bigl(\dv{r}\bigr)^2 + \frac{\ell(\ell+1)}{2r^2} + \bigl(\tilde{V}(r)-B\bigr)\biggr]U(r) = 0
# $$
# 
# With IC and bound state BC,
# $U(r\rightarrow0)=Ar^{\ell+1}$, and $U(r\rightarrow\infty)=B\exp\bigl(-\sqrt{2|B|}\,r\bigr)$
# 
# ## Dirac radial
# With solutions of the form,
# $$
# \langle \vec{r}\ket{\psi} = \frac{1}{r}\,\begin{pmatrix} i f(r) \,\Omega_{+\kappa m}\ket{\pm} \\ \;g(r) \,\Omega_{-\kappa m}\ket{\pm}\end{pmatrix}
# $$
# 
# with energy eigenvalues and vectors determined from this system,
# $$
# \begin{align}
# \biggl(\dv{r}+\frac{\kappa}{r}\biggr)\,f &= -\bigl( E^*(r) + M^*(r)\bigr)\,g \\
# \biggl(\dv{r}-\frac{\kappa}{r}\biggr)\,g &= +\bigl( E^*(r) - M^*(r)\bigr)\,f 
# \end{align}\\
# M^*(r) = 1+S(r)\\ E^*(r) = E-V(r)
# $$
# 
# With IC and bound state BC,
# $\langle r\ket{\psi}(r=\delta r\rightarrow0)=Ar^\ell_+\begin{pmatrix}r \\ -\frac{(1+\ell_++\kappa)}{M^*(\delta r)+E^*(\delta r)}\end{pmatrix}$, and $U(r\rightarrow\infty)=B\exp\bigl(-\sqrt{1-E^2}\,r\bigr)\begin{pmatrix}1\\\sqrt\frac{1-E}{1+E}\end{pmatrix}$
# 
# Relating the energies with natural units
# $|B| = |1-E|$

# In[9]:


V0, a0, R = 0.05235, 3.33, 19.529
def V(r):
    return -V0 / (1+np.exp((r-R)/a0))
def S(r):
    return 0*r

scipy.integrate.quadrature(V, 0, 20*R)[0]/np.pi


# In[4]:


# wrap the problem for easy changing/comparison

class QMprob:
    def __init__(self, model, potential, ellORkap):
        assert (model=='schro' or model=='dirac'), "needs to be 'schro' or 'dirac'"
        self.model = model
        self.potential = potential
        self.ellORkap = ellORkap
        self.Emin = 0 if model =='schro' else 1
        
diracProb = QMprob('dirac', (V,S), -1)
schroProb = QMprob('schro', V, 0)


# In[5]:


# extract from problem object
prob = diracProb
potential = prob.potential
ellORkap = prob.ellORkap
model = prob.model
Emin = prob.Emin

dr = 1e-3
rbnds = dr/10, 10*R
E = 2 + Emin


k = quantum.FindWaveNumberFromEnergy(E, model)
rbnds, dr = quantum.MakeUnboundDomain(k, rbnds, dr, model)
sol = quantum.SolveUnbound(potential, ellORkap, E, rbnds, dr, model)
sol = quantum.MakeUnitAmplitude(k, sol)

r = sol.t
psi = sol.y[0,:]

plt.figure(1)
plt.plot(k*r/TWOPI, psi)


delta, fitSoln, refSoln = quantum.FindPhaseShift(k, sol, ellORkap, model)
plt.plot(k*r/TWOPI, fitSoln, '--')
plt.plot(k*r/TWOPI, refSoln, ':')
print(delta)

plt.legend(('sol','fit','ref'), loc=1)


# In[6]:


nd = 21
dE = np.logspace(-3,1, nd)
delta = np.empty(dE.shape)

for i in range(nd):
    Ei = dE[i] + Emin*np.sign(dE[i])
    k = quantum.FindWaveNumberFromEnergy(Ei, model)
    rbnds, dr = quantum.MakeUnboundDomain(k, rbnds, dr, model)
    sol = quantum.SolveUnbound(potential, ellORkap, Ei, rbnds, dr, model)
    sol = quantum.MakeUnitAmplitude(k, sol)
    delta[i], _, _= quantum.FindPhaseShift(k, sol, ellORkap, model)
    print(f"{i}: {Ei}")
    
plt.figure(3)
plt.plot(np.abs(dE), delta/np.pi)
plt.ylabel('$\delta/\pi$')
plt.xlabel('$||E|-M|$')
plt.xscale('log')


# In[6]:


nd = 21
dE = np.logspace(-3,1, nd)
delta = np.empty(dE.shape)

for i in range(nd):
    Ei = dE[i] + Emin*np.sign(dE[i])
    k = quantum.FindWaveNumberFromEnergy(Ei, model)
    rbnds, dr = quantum.MakeUnboundDomain(k, rbnds, dr, model)
    sol = quantum.SolveUnbound(potential, ellORkap, Ei, rbnds, dr, model)
    sol = quantum.MakeUnitAmplitude(k, sol)
    delta[i], _, _= quantum.FindPhaseShift(k, sol, ellORkap, model)
    print(f"{i}: {Ei}")
    
plt.figure(3)
plt.plot(np.abs(dE), delta/np.pi)
plt.ylabel('$\delta/\pi$')
plt.xlabel('$||E|-M|$')
plt.xscale('log')


# In[12]:


nd = 21
dE = -np.logspace(-4,1, nd)
delta = np.empty(dE.shape)

for i in range(nd):
    Ei = dE[i] + Emin*np.sign(dE[i])
    k = quantum.FindWaveNumberFromEnergy(Ei, model)
    rbnds = quantum.MakeUnboundDomain(k, rbnds, dr, model)
    sol = quantum.SolveUnbound(potential, ellORkap, Ei, rbnds, dr, model)
    sol = quantum.MakeUnitAmplitude(k, sol)
    delta[i], _, _= quantum.FindPhaseShift(k, sol, ellORkap, model)
    print(f"{i}: {Ei}")

plt.plot(np.abs(dE), delta/np.pi, 'r')
plt.xscale('log')


# In[13]:


plt.grid()

