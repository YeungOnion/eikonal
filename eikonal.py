r"""
helper functions needed to run code for eikonal
Requires module for profile functions, `chi`
 - chi_0
 - chi_c
 - chi_so
 - chi_a
"""

from scipy.integrate import simps as simps
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.special import jv as jv
import numpy as np
import hankel
from integrators import magni_FHT, magni_FHT_nu1

###
# TODO: these can be changed as needed
import chi_uniform as chi_funcs
###


def test_f0(q, R_ch, R_wk, DEBUG=False):
    f0 = chi_funcs.f0_func(q, chi_funcs.Etest_, R_ch, R_wk)
    if DEBUG:
        print('f0 complete')
    return f0.real, f0.imag


def test_F0_short(q, R_ch, R_wk, DEBUG=False):
    def F(b, qi):
        func = np.vectorize(chi_funcs.F1_func)
        jacobian = b * jv(chi_funcs.F0_func._nu, qi * b)
        return func(b, chi_funcs.Etest_, R_ch, R_wk) * jacobian

    db, bmax = 5e-2, R_ch
    b = np.arange(0, bmax + 2 * db, db)
    b[0] = 1e-6
    B, Q = np.meshgrid(b, q, sparse=True)
    F = simps(F(B, Q).real, B) + 1j * simps(F(B, Q).imag, B)
    # dat = np.array((q,F.real, F.imag))

    if DEBUG:
        print('F1 complete')
    return F.real, F.imag


def test_F0(q_in, R_ch, R_wk, DEBUG=False):
    def F(b):
        return chi_funcs.F0_func(b, chi_funcs.Etest_, R_ch, R_wk)

    (q, F), (b, f) = magni_FHT(F, N=2**16, alp=1e-2, R=500, Q=500)
    spl_r = spline(q, F.real)
    spl_i = spline(q, F.imag)
    if DEBUG:
        print('F0 complete')
    return spl_r(q_in), spl_i(q_in)


def test_F0_long(q_in, R_ch, R_wk, DEBUG=False):
    def F(b):
        return chi_funcs.F2_func(b, chi_funcs.Etest_, R_ch, R_wk)

    (q, F), (b, f) = magni_FHT(F, N=2**16, alp=1e-2, R=1000, Q=1000)
    spl_r = spline(q, F.real)
    spl_i = spline(q, F.imag)
    if DEBUG:
        print('F2 complete')
    return spl_r(q_in), spl_i(q_in)


def test_F0_long_ogata(q, R_ch, R_wk, DEBUG=False):
    def F(b):
        return np.vectorize(chi_funcs.F2_func)(b, chi_funcs.Etest_, R_ch, R_wk)

    ht = hankel.HankelTransform(nu=chi_funcs.F2_func._nu, N=120, h=0.03)
    F = ht.transform(F, q, ret_err=False)

    if DEBUG:
        print('F2 complete')
    return F.real, F.imag


def test_Fn(q_in, R_ch, R_wk, DEBUG=False):
    def F(b):
        return chi_funcs.Fn_func(b, chi_funcs.Etest_, R_ch, R_wk)

    (q, F), (b, f) = magni_FHT_nu1(F, N=2**16, alp=1e-2, R=500, Q=500)
    spl_r = spline(q, F.real)
    spl_i = spline(q, F.imag)
    if DEBUG:
        print('Fn complete')
    return spl_r(q_in), spl_i(q_in)


def test_Fn_ogata(q, R_ch, R_wk, DEBUG=False):
    # db, bmax = 1e-2, chi_funcs.R_*1.1
    # b = np.arange(0, bmax, db)
    # b[0] = 1e-3

    def F(b):
        ff = np.vectorize(chi_funcs.Fn_func)
        return ff(b, chi_funcs.Etest_, R_ch, R_wk)

    ht = hankel.HankelTransform(nu=1, N=120, h=0.03)
    F = ht.transform(F, q, ret_err=False)

    if DEBUG:
        print('Fn complete')
    return F.real, F.imag


def test_Fkap(q, R_ch, R_wk, DEBUG=False):
    db, bmax = 1e-2, R_wk * 1.1
    b = np.arange(0, bmax, db)
    b[0] = 1e-3

    def F(b, qi):
        ff = np.vectorize(chi_funcs.Fkap_func)
        jacobian = b * jv(chi_funcs.Fkap_func._nu, qi * b)
        return ff(b, chi_funcs.Etest_, R_ch, R_wk) * jacobian

    B, Q = np.meshgrid(b, q, sparse=True)
    F = simps(F(B, Q).real, B) + 1j * simps(F(B, Q).imag, B)

    print('Fk complete')
    return F.real, F.imag


def test_all(q, R_ch, R_wk, use_magni=True, DEBUG=False, fname='PVES'):
    dat = np.array(q)
    dat = np.vstack((dat, test_f0(q, R_ch, R_wk, DEBUG)))
    dat = np.vstack((dat, test_F0_short(q, R_ch, R_wk, DEBUG)))

    if use_magni:
        F2 = test_F0_long(q, R_ch, R_wk, DEBUG)
        dat = np.vstack((dat, F2))
        dat = np.vstack((dat, test_Fn(q, R_ch, R_wk, DEBUG)))
        fname += '_magni'

    else:
        F2 = test_F0_long_ogata(q, R_ch, R_wk, DEBUG)
        dat = np.vstack((dat, F2))
        dat = np.vstack((dat, test_Fn_ogata(q, R_ch, R_wk, DEBUG)))
        fname += '_ogata'

    dat = np.vstack((dat, test_Fkap(q, R_ch, R_wk, DEBUG)))
    headstring = ('q f0_r f0_i F1_r F1_i F2_r F2_i ' 'Fn_r Fn_i Fk_r Fk_i')


    np.savetxt(fname + '.out', dat.T, header=headstring)
    return fname


def eikonal_asymmetries(F0, Fn, Fk):
    x_sec = np.abs(F0)**2 + np.abs(Fn)**2 + np.abs(Fk)**2
    An = 2 * (F0 * np.conj(Fn)).real / x_sec
    Ak = 2 * (F0 * np.conj(Fk)).real / x_sec
    return x_sec, An, Ak


if __name__ == '__main__':
    theta = np.linspace(0,30,481)[1:]
    Etest, R_ch = 1060./chi_funcs.hc_, 5.501 # Pb, N=126, Z=82
    # Etest, R_ch = 2200./chi_funcs.hc_, 3.477 # Ca, N=28,  Z=20

    kap = np.sqrt(Etest**2 - chi_funcs.m_**2)
    q = 2 * kap * np.sin(theta * np.pi / 180 / 2)

    R_ch_list = R_ch * np.r_[1,1,1,1] * np.sqrt(5/3)
    R_wk_list = (R_ch + np.r_[0:.3:4j])*np.sqrt(5/3)

    for i, (R_ch, R_wk) in enumerate(zip(R_ch_list, R_wk_list)):
        parameter_string = f"R_ch={R_ch}, R_wk={R_wk}"
        fname = 'data/amplitudes/radius' + str(i) + '_OY'
        print(parameter_string)

        fname = test_all(q, R_ch, R_wk, use_magni=False, DEBUG=True, fname=fname)

        dat = np.loadtxt(fname + '.out').T
        q = dat[0]
        F0 = np.sum(dat[[1, 3, 5]], axis=0) + 1j * np.sum(dat[[2, 4, 6]], axis=0)
        Fn = dat[7] + 1j * dat[8]
        Fk = dat[9] + 1j * dat[10]
        x_sec, An, Ak = eikonal_asymmetries(F0, Fn, Fk)

        dat = np.stack((q, x_sec, An, Ak, theta)).real.astype('float128')
        fname = fname.replace('amplitudes', 'asymmetries')
        np.savetxt(fname + '.out', dat.T, header='q xsec An Ak #' + parameter_string)
