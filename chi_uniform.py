"""
This will serve as a place to store these functions.
Could be used as direct closed form versions of
functions or could return callables that are based
on numerical values (spline or other interpolants)
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from scipy.special import gamma as gamma
import re
import sys

hc_ = 197.32698  # MeV fm
e2_ = 1 / 137.035999
proton_mass_ = 938.272088 / hc_
GF_ = 1.026826e-5 / (proton_mass_)**2
QW_n_ = -0.9878
QW_p_ = 0.0721

# probe is electron
m_ = 0.51099895/hc_
Etest_ = 2200./hc_  # MeV/hc

# Pb, N=126, Z=82
Z_ = 82
N_ = 126

# # Ca, N=28,  Z=20
# Z_ = 20
# N_ = 28


def eta(E):
    kap = np.sqrt(E**2 - m_**2)
    return E / kap * Z * e2_


# a useful thing to have around
def circ(ksi, R=1):
    return np.sqrt(R**2 - ksi**2)


def chi_0(b, E, R, Z, N):
    kap = np.sqrt(E**2 - m_**2)
    return 2j * eta(E) * np.log(kap * b)


# pack into t1 and t2 like in jorges work
# use np piecwise for vectorized evaluations
def chi_c1(b, E, R, Z, N):
    kap = np.sqrt(E**2 - m_**2)

    def t1(b):
        b = b / R

        def leq(b):
            a1 = 4 / 3 * (1 - b**2 / 4) * circ(b)
            a2 = np.log((1 + circ(b)) / b)
            return -2 * Z * e2_ * (a1 - a2)

        def geq(b):
            return 0 * b

        return np.piecewise(b, b < 1, [leq, geq])

    return 1j * E / kap * t1(b)


def chi_c2(b, E, R, Z, N):
    kap = np.sqrt(E**2 - m_**2)

    def t2(b):
        b = b / R

        def leq(b):
            a1 = circ(b) * (2 * b**4 - 14 * b**2 + 27) / 15
            a2 = 1 / b * np.arctan(b / circ(b))
            return 2 * (Z * e2_)**2 / R * (a1 + a2)

        def geq(b):
            return (Z * e2_)**2 * np.pi / (b * R)

        return np.piecewise(b, b < 1, [leq, geq])

    return -1j / 2 / kap * t2(b)


def chi_c(b, E, R, Z, N):
    return chi_c1(b, E, R) + chi_c2(b, E, R, Z, N)


def chi_so(b, E, R, Z, N):
    def tso(b):
        b = b / R

        def leq(b):
            return 2 / (b * R)**2 * (Z * e2_) * (1 - circ(b)**3)

        def geq(b):
            return 2 / (b * R)**2 * (Z * e2_)

        return np.piecewise(b, b < 1, [leq, geq])

    return 1j * b / 2 / (E + m_) * tso(b)


def chi_a(b, E, R, Z, N):
    QW = N*QW_n_ + Z*QW_p_
    coef = 1j / 2 / np.sqrt(2) * GF_ * QW

    def ta(b):
        b = b / R

        def leq(b):
            return 3 / (2 * np.pi * R**2) * circ(b)

        def geq(b):
            return 0 * b

        return np.piecewise(b, b < 1, [leq, geq])

    return coef * ta(b)


def f0_func(q, E, R_ch, R_wk, use_ratio=True):
    kap = np.sqrt(E**2 - m_**2)
    exp_pt1 = (q / 2 / kap)**(2j * eta(E))
    if use_ratio:
        exp_pt2 = gamma(1 - 1j * eta(E)) / gamma(1 + 1j * eta(E))
    else:
        sig = np.angle(scipy.special.gamma(1 + 1j * eta(E)))
        exp_pt2 = np.exp(-2j * sig)
    return 2 * eta(E) * (kap / q**2) * exp_pt1 * exp_pt2


def F0_func(b, E, R_ch, R_wk):
    kap = np.sqrt(E**2 - m_**2)
    phase_factor = np.exp(-chi_0(b, E, R_ch), Z, N)
    coef = 1j * kap
    return coef * phase_factor * (1 - np.exp(-chi_c(b, E, R_ch)), Z, N)


F0_func._name = r'F0'
F0_func._nu = 0


def F1_func(b, E, R_ch, R_wk):
    kap = np.sqrt(E**2 - m_**2)
    phase_factor = np.exp(-chi_0(b, E, R_ch), Z, N)
    coef = 1j * kap
    return coef * phase_factor * (1 - np.exp(-chi_c1(b, E, R_ch)), Z, N)


F1_func._name = r'F1'
F1_func._nu = 0


def F2_func(b, E, R_ch, R_wk):
    kap = np.sqrt(E**2 - m_**2)
    phase_factor = np.exp(-chi_0(b, E, R_ch) - chi_c1(b, E, R_ch), Z, N)
    coef = 1j * kap
    return coef * phase_factor * (1 - np.exp(-chi_c2(b, E, R_ch)), Z, N)


F2_func._name = r'F2'
F2_func._nu = 0


def Fn_func(b, E, R_ch, R_wk):
    kap = np.sqrt(E**2 - m_**2)
    phase_factor = np.exp(-chi_0(b, E, R_ch) - chi_c(b, E, R_ch), Z, N)
    coef = -kap
    return coef * phase_factor * chi_so(b, E, R_ch, Z, N)


Fn_func._name = r'Fn'
Fn_func._nu = 1


def Fk_func(b, E, R_ch, R_wk):
    kap = np.sqrt(E**2 - m_**2)
    phase_factor = np.exp(-(chi_0(b, E, R_ch) + chi_c(b, E, R_ch)), Z, N)
    coef = 1j * kap
    return coef * phase_factor * chi_a(b, E, R_wk, Z, N)


Fk_func._name = r'Fk'
Fk_func._nu = 0

# below is all for testing purposes


def eval_chi(Etest, b, item_num, R_ch, R_wk, Z, N, postfix=''):

    X0  = chi_0  (b, Etest, R_ch, Z, N)
    X1  = chi_c1 (b, Etest, R_ch, Z, N)
    X2  = chi_c2 (b, Etest, R_ch, Z, N)
    Xso = chi_so (b, Etest, R_ch, Z, N)
    Xa  = chi_a  (b, Etest, R_wk, Z, N)

    dat = np.stack(
        (b, np.abs(X0), np.abs(X1), np.abs(X2), np.abs(Xso), np.abs(Xa)))
    head_string = ("    b".ljust(25) + "    X0".ljust(25) +
                   "    X1".ljust(25) + "    X2".ljust(25) +
                   "    Xso".ljust(25) + "    Xa".ljust(25))

    fname = 'data/profile_funcs/chi_OY'
    np.savetxt(fname+postfix+'.txt', dat.T, header=head_string)

    labels = re.split(' +', head_string)
    plt.plot(b, dat[item_num], label=labels[item_num])
    plt.legend()


def eval_F(Etest, b, item_num, R_ch, R_wk, keep_jacobian=True, postfix=''):

    if keep_jacobian:
        F1_int = b * scipy.special.jn(0, b) * F1_func(b, Etest, R_ch, R_wk)
        F2_int = b * scipy.special.jn(0, b) * F2_func(b, Etest, R_ch, R_wk)
        Fn_int = b * scipy.special.jn(1, b) * Fn_func(b, Etest, R_ch, R_wk)
        Fk_int = b * scipy.special.jn(0, b) * Fk_func(b, Etest, R_ch, R_wk)
    else:
        F1_int = F1_func(b, Etest, R_ch, R_wk)
        F2_int = F2_func(b, Etest, R_ch, R_wk)
        Fn_int = Fn_func(b, Etest, R_ch, R_wk)
        Fk_int = Fk_func(b, Etest, R_ch, R_wk)

    F0_int = F1_int + F2_int

    head_string = ("    b".ljust(25) + "    F0_r".ljust(25) +
                   "    F0_i".ljust(25) + "    Fn_r".ljust(25) +
                   "    Fn_i".ljust(25) + "    Fk_r".ljust(25) +
                   "    Fk_i".ljust(25))
    dat = np.stack((b, F0_int.real, F0_int.imag, Fn_int.real, Fn_int.imag,
                    Fk_int.real, Fk_int.imag))

    fname = 'data/profile_funcs/Fintegrand_OY'
    np.savetxt(fname+postfix+'.txt', dat.T, header=head_string)

    labels = re.split(' +', head_string)
    plt.plot(b, dat[item_num], label=labels[item_num])
    plt.legend()


if __name__ == "__main__":
    # compute, export, and show chi functions
    item_num = 2
    fname_postfix = ''

    if len(sys.argv) > 4:
        print("use a max of 3 parameters after the script name")
    if len(sys.argv) > 3:
        fname_postfix = sys.argv[4]
    if len(sys.argv) > 2:
        R_ch_ = sys.argv[1]
        R_wk_ = sys.argv[2]
    else:
        R_ch_ = 3.477*np.sqrt(5./3)
        R_wk_ = 3.777*np.sqrt(5./3)

    b = np.linspace(0, 10, 101)[1:]
    # b = np.logspace(-3,1,101)
    eval_chi(Etest_, b, item_num, R_ch=R_ch_, R_wk=R_wk_, postfix=fname_postfix)
    plt.clf()

    # compute, export, and show F integrands
    eval_F(Etest_, b, item_num, R_ch=R_ch_, R_wk=R_wk_, keep_jacobian=True, postfix=fname_postfix)
    plt.yscale('symlog')
    plt.clf()
