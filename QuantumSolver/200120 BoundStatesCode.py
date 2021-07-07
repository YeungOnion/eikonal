#!/usr/bin/env python
# coding: utf-8

# In[1]:


# should be using python >=3.7
import sys
print(sys.version_info[0:2])


# In[2]:


# setup for good codes
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

import numpy as np
import scipy.integrate
# import RK4util
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
    ax.plot(sol.t, sol.y[1], '--', c=color)
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
# \biggl(\dv{r}+\frac{\kappa}{r}\biggr)\,f &= -\bigl( M^*(r) + E^*(r)\bigr)\,g \\
# \biggl(\dv{r}-\frac{\kappa}{r}\biggr)\,g &= +\bigl( M^*(r) - E^*(r)\bigr)\,f 
# \end{align}\\
# M^*(r) = 1+S(r)\\ E^*(r) = E-V(r)
# $$
# 
# With IC and bound state BC,
# $\langle r\ket{\psi}(r=\delta r\rightarrow0)=Ar^\ell_+\begin{pmatrix}r \\ -\frac{(1+\ell_++\kappa)}{M^*(\delta r)+E^*(\delta r)}\end{pmatrix}$, and $U(r\rightarrow\infty)=B\exp\bigl(-\sqrt{1-E^2}\,r\bigr)\begin{pmatrix}1\\\sqrt\frac{1-E}{1+E}\end{pmatrix}$
# 
# Relating the energies with natural units
# $|B| = |1-E|$

# # Given Problem
# 
# Energy like Woods-Saxon, no mass like potential.
# $$
# V(r) = -V_0\biggl(1+\exp\bigl(\frac{r-R}{a}\bigr)\biggr)^{-1}; \quad S(r)=0
# $$
# 
# - $A = 40$
# - $a_0 = 0.7\rm{fm} \rightarrow 3.33$
# - $R = (1.2\rm{fm})\,A^{1/3} = 4.1039{\rm{fm}} \rightarrow 19.529$
# - $V_0 = 50\rm{MeV} \rightarrow 0.05325$

# In[3]:


V0, a0, R = 0.05235, 3.33, 19.529
def V(r):
    return -V0 / (1+np.exp((r-R)/a0))
#     return -V0 * np.sinh(R/a0) / (np.cosh(r/a0) + np.cosh(R/a0))
def S(r):
    return 0*r


# # Solve the Schrodinger problem for bound states

# In[4]:


model = 'schro'
ell = 0
B = -.72*V0
# B = -.24*V0
dr = 5e-2
rbnds = dr/10, 20, 120

# solve and plot for some energy
plt.figure(0)
sol = quantum.SolveInterior(V, ell, B, rbnds[0:2], dr, model)
PlotSolns(sol, 'xkcd:blue', 'Interior', plt.gca())
plt.ylabel('$\psi_{int}$')

sol = quantum.SolveExterior(V, ell, B, rbnds[1:3], dr, model)
PlotSolns(sol, 'xkcd:red', 'Exterior', plt.gca().twinx())
plt.ylabel('$\psi_{ext}$')

sol = quantum.MakeSoln(V, ell, B, rbnds, dr, model)
plt.figure(1)
PlotSolns(sol, 'xkcd:green')
plt.ylabel('$\psi$')

# plot BCmatchCond as k varies
plt.figure(2)
Blist = np.linspace(-.99, -.01, 20)*V0
BCcond = quantum.BCMatch(V, ell, rbnds, dr, model)
BCvals = BCcond(Blist)
plt.plot( np.abs(Blist)/V0, BCvals , '.-')
plt.xlabel('$|B|/V_0$')
plt.grid()
plt.ylim((-10,10))


# In[6]:


Bbnds = {0: ((-.73,-.70), (-.25,-.2)), 1: ((-.33,.39),(-.007,-.0085))} # in units V0
# Bexact ={0: ()}
Bexact = {0: (-0.03796169771240042, -0.012781727878502123), 1:()}


NeedToSearchEigen = (len(Bexact[ell]) < len(Bbnds[ell]))
c = ['xkcd:red','xkcd:blue','xkcd:green']

plt.figure(3)
# solve and plot
if NeedToSearchEigen is True:
    for i,Bguess in enumerate(Bbnds[ell]):
        Bguess = V0*np.array(Bguess)
        B, sol = quantum.FindEigen(V, ell, Bguess, rbnds, dr, model)
        PlotSolns(sol, c[i], '$B=%.3fV_0=%.3E$'%(B/V0, B))
        print(B)

else:
    for i,B in enumerate(Bexact[ell]):
        sol = quantum.MakeSoln(V, ell, B, rbnds, dr, model)
        PlotSolns(sol, c[i], '$B=%.3fV_0=%.3E$'%(B/V0, B))

plt.axvline(R,c='k',linestyle='--')
plt.legend()


# # Solve Bound state Dirac Problem
# 

# In[19]:


model = 'dirac'
kap = 1
E = .987
dr = 5e-2
rbnds = dr/10, 20, 120

# solve and plot for some energy
plt.figure(-1)
sol = quantum.SolveInterior((V,S), kap, E, rbnds[0:2], dr, model)
plt.plot(sol.t, sol.y[0], 'b')
plt.ylabel('$\psi_{int}$')
CenterYAx()

sol = quantum.SolveExterior((V,S), kap, E, rbnds[1:3], dr, model)
plt.gca().twinx().plot(sol.t, sol.y[0], 'r')
plt.ylabel('$\psi_{ext}$')
CenterYAx()

# plot BCmatchCond as k varies
plt.figure(-2)
Elist = np.linspace(-.99,-.01, 40)*V0 + 1
BCcond = quantum.BCMatch((V,S), kap, rbnds, dr, model)
BCvals = BCcond(Elist)
plt.subplot(1,2,1)
plt.plot( (1-Elist)/V0, BCvals , '.-')
plt.xlabel('$|B|/V_0$')
plt.grid()
plt.ylim((-10,10))

plt.subplot(1,2,2)
plt.plot( Elist/V0, BCvals , '.-')
plt.xlabel('$E/V_0$')
plt.grid()
plt.ylim((-10,10))


# In[18]:


18.858*V0


# In[26]:


Ebnds = {-1: ((18.36,18.38), (18.84,18.87)), 1: ((-.33,.39),(-.007,-.0085))} # in units V0
Eexact = {-1: (0.9620280850376638, 0.9869199155620734) }

NeedToSearchEigen = (len(Eexact[kap]) < len(Ebnds[kap]))

plt.figure(-3)
# solve and plot
if NeedToSearchEigen is True:
    for Eguess in Ebnds[kap]:
        Eguess = V0*np.array(Eguess)
        E, sol = quantum.FindEigen((V,S), kap, Eguess, rbnds, dr, model)
        B = np.abs(1-E)
        plt.plot(sol.t, sol.y[0], label='$B=%.3fV_0=%.3f$'%(B/V0, B))
        plt.plot(sol.t, sol.y[1], '--', label='$E=%.3fV_0=%.3f$'%(B/V0, B))
        plt.axvline(R,c='k',linestyle='--')
        print(E, end=', ')

else:
    for E in Eexact[kap]:
        sol = quantum.MakeSoln((V,S), kap, E, rbnds, dr, model)
        plt.plot(sol.t, sol.y.T, label='$E=%.3fV_0=%.3f$'%(E/V0, E))
        plt.axvline(R,c='k',linestyle='--')

plt.legend()

