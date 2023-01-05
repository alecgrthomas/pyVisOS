"""
functions for nonlinear analysis of 1d flattop beam-drive wakes

Contains
---
kpz_gt_1_2
get_gamma-m_gt_1_2
get_gamma_max
get_z_deltan_0
get_n_behind_driver
get_n_in_driver
n_in_driver_gt_1_2
"""

import numpy as np
from scipy import optimize

import scipy.optimize as opt
optimize = opt
from scipy import special as sp

def kpz_gt_1_2(phi,alpha):
    consta = np.sqrt(2*alpha)
    constb = np.sqrt(2*alpha - 1)
    kappa_sq = 1 / consta**2
    kappap_sq = constb**2 / consta**2
    
    int_nc_sq = kappap_sq * sp.ellipkinc(phi,kappa_sq)
    int_nc_sq -= sp.ellipeinc(phi, kappa_sq)
    int_nc_sq += np.tan(phi) * np.sqrt(1 - kappa_sq*np.sin(phi)**2)
    int_nc_sq_kpsq = int_nc_sq
    int_g_an = consta * int_nc_sq_kpsq
    return int_g_an
def get_gamma_m_gt_1_2(nd0, kp0Ld, n_p):
    alpha = nd0 / n_p
    kp = np.sqrt(n_p)
    consta = np.sqrt(2*alpha)
    constb = np.sqrt(2*alpha - 1)
    kappa_sq = 1 / consta**2
    kappap_sq = constb**2 / consta**2
    
    solns = opt.root_scalar(lambda phi: kpz_gt_1_2(phi,alpha)*2 + kp * kp0Ld*constb**2, bracket=[-np.pi/2,0])

    phib = solns.root
    xrb = 1 + np.tan(phib)**2
    betab = (1 - xrb**2) / (1 + xrb**2)
    sgnEb = 1.0
    abs_dx_dtau_b = np.sqrt(2*(1-alpha) - 1./xrb + (2*alpha - 1)*xrb)
    Eb = sgnEb * kp*abs_dx_dtau_b
    gb = (1 + xrb**2) / 2 / xrb
    pb = gb * betab
    gm = 1 - alpha + alpha * xrb
    return gm,gb,pb,Eb,xrb,phib

def get_gamma_max(nd0,kp0Ld,n_p):
    """
    Find wake amplitude behind beam driver
    
    Parameters
    ---
    nd0 : driver density
    kp0Ld : float, driver length (k_p0 Ld)
    n_p : plasma density
    
    Returns
    ---
    gm_R : float, maximum gamma attained in wave behind dr
    gb : float, fluid gamma at the end of the driver
    xrb : float, Rosenzweig's x = potential at the end of the driver
    phib : float, if nd0/kp0Ld != 0.5 then return phib, the angle
        needed to get in-driver quantities.
        If nd/kp0Ld == 0.5, return 0
    """

    ald = nd0 / n_p
    kp = np.sqrt(n_p)
    gm = 0.
    gb = 0.
    phib = 0.
    sgnEb = 1.0
    if (ald < 0.5):
        ksq = 2 * ald
        solns = opt.root_scalar(lambda phi: (1-ksq) * kp*kp0Ld - 2 * (sp.ellipe(ksq) - sp.ellipeinc(phi, ksq)), bracket = [-20,20])
        phib = solns.root

        xrb = 1 + ksq /(1-ksq) * np.cos(phib)**2
        sgnEb = np.sign(np.sin(2*phib))
    elif ald == 0.5:
        solns = opt.root_scalar(lambda x: np.sqrt(x)*np.sqrt(x-1) + np.arcsinh(np.sqrt(x-1)) - kp*kp0Ld, bracket=[1,100])
        xrb = solns.root
    elif ald > 0.5:
        gm_al,gb_al,betab_al,Eb,xrb,phib = get_gamma_m_gt_1_2(nd0, kp0Ld, n_p)

    else:
        raise TypeError('ald must be nonnegative float')
    abs_dx_dtau = np.sqrt(2*(1-ald) - 1./xrb + (2*ald - 1) * xrb)
    Eb = sgnEb * kp*abs_dx_dtau
    betab = (1 - xrb**2) / (1 + xrb**2)
    gb = (1 + xrb**2) / 2 / xrb
    pb = betab * gb
    gm = 1 - ald + ald * xrb
    return gm,gb,pb,Eb,xrb,phib

def get_z_deltan_0(nd0,kp0Ld,n_p):
    """
    get the position in the wake(relative to front of beam driver) where $\Delta n=0$
    
    Parameters
    ---
    nd0
    kp0Ld
    n_p
    
    Returns
    ---
    lambda_wave : float, nonlinear plasma wavelength at density n_p, 
            amplitude determined from provided parameters
    z_deltan_0 : float, position in wake where delta n = 0
    
    Examples
    ---
    >>> get_z_deltan_0(.1,.1*2*np.pi,1.)
    (6.284685944931384, -6.287685609010141, -1.947910629819441, -3.5816471992845216, -5.091753434324511)
    """

    kp = np.sqrt(n_p)
    gm,gb,betab,sgnEb,xrb,_ = get_gamma_max(nd0,kp0Ld,n_p)

    kappa_sq = (gm - 1) / (gm + 1)
    kappap = np.sqrt(2 / (gm + 1))
    phi1 = -np.pi + np.arcsin(np.sqrt((gm - gb) / (gm - 1)))
    z1 = 2 / kappap * sp.ellipeinc(phi1,kappa_sq)
    z1 -= kappap * sp.ellipkinc(phi1,kappa_sq)
    z1 -= 2*np.sqrt(kappa_sq) / kappap * np.sin(phi1)
    z1 /= kp

    phi_dn_min = -np.pi
    phi_dn_0 = -3*np.pi / 2
    pni_dn_max = -2*np.pi
    phis = -np.pi/2 * np.array([2., 3., 4.])
    zs_dn = 1/kp * 2 / kappap * sp.ellipeinc(phis,kappa_sq)
    zs_dn -= 1/kp * kappap * sp.ellipkinc(phis,kappa_sq)
    zs_dn -= 1/kp * 2*np.sqrt(kappa_sq) / kappap * np.sin(phis)
    
    k_wave = kp * np.pi / 2 * kappap / (2 * sp.ellipe(kappa_sq) - kappap * sp.ellipk(kappa_sq))
    lambda_wave = 2*np.pi / k_wave


    return lambda_wave, zs_dn[-1],zs_dn[0]-z1-kp0Ld, zs_dn[1]-z1-kp0Ld, zs_dn[2]-z1-kp0Ld #z2-z1-kp0Ld#, z2, z1


def get_wake_behind_driver(np_np0, nd0_np0,kp0Ld0, wave_beta = 1.0,n_cycles = 3):

    kp = np.sqrt(np_np0)
    gm,gb,pb,qEb,xrb,_ = get_gamma_max(nd0_np0,kp0Ld0,np_np0)
    qEb /= kp
    
    phi_wake = np.arctan2(qEb * np.sqrt(1+gb), pb*np.sqrt(2))
    phis = np.linspace(phi_wake,phi_wake + n_cycles * 2*np.pi)

    kappa_sq = (gm-1.) / (gm + 1.)
    kappa = np.sqrt(kappa_sq)
    kappa_p = np.sqrt(2 / (gm + 1.))

    #should wave_beta be moved to d_tau?
    tau_b = wave_beta * 1/kp*(2 / kappa_p * sp.ellipeinc(phi_wake, kappa_sq) - kappa_p * sp.ellipkinc(phi_wake, kappa_sq))
    d_tau_b = -1/kp * 2 * kappa / kappa_p * np.sin(phi_wake)
    tau_b= tau_b + d_tau_b

    tau0list = wave_beta * 1/kp*(2 / kappa_p * sp.ellipeinc(phis, kappa_sq) - kappa_p * sp.ellipkinc(phis, kappa_sq))
    d_tau = -1/kp * 2 * kappa / kappa_p * np.sin(phis)
    tau_behind_driver = tau0list + d_tau - tau_b+kp0Ld0

    gammas =  gm - (gm-1) * np.sin(phis)**2
    ps = np.sqrt(gammas**2 - 1) * np.sign(np.cos(phis))
    betas = ps / gammas
    n_behind_driver = np_np0 / (1-betas)
    qe_behind_driver = -kp*np.sin(phis)*np.sqrt(2*(gm - 1))

    return tau_behind_driver, n_behind_driver, qe_behind_driver

def get_n_in_driver(nd0,kp0Ld,n_p):
    """
    get density profile within driver
    
    
    
    Returns
    ---
    z_in_driver
    n_in_driver
    
    Notes
    ---
    See Rosenzweig, Nonlinear dynamics in pwfa?
    """
    ald = nd0 / n_p
    kp = np.sqrt(n_p)
    gm,gb,betab,sgnEb,xrb,phib = get_gamma_max(nd0,kp0Ld,n_p)

    z_in_driver = np.array([])
    xr = np.array([])
    if ald < 0.5:
        phi = np.linspace(phib,np.pi/2)
        z_in_driver = 2 / (1 - 2*ald)/np.sqrt(n_p) * (sp.ellipe(2*ald) - sp.ellipeinc(phi,2*ald))
        xr = 1 + 2*ald / (1 - 2*ald) * np.cos(phi)**2
        sgn_sin2phi = np.sign(np.sin(2*phi))
    elif ald == 0.5:
        xr = np.linspace(1, xrb)
        z_in_driver = 1./np.sqrt(n_p) * (np.sqrt(xr)*np.sqrt(xr-1) + np.arcsinh(np.sqrt(xr-1)) )
    elif ald > 0.5:
        z_in_driver, n_in_driver, e_in_driver = n_in_driver_gt_1_2(nd0, kp0Ld, n_p)
        z_in_driver *= -1
#         phi = -np.linspace(phib, np.pi/2)
#         xr = 1 + np.tan(phi)**2
# #         z_in_driver = 
#         asq = 2*ald
#         bsq = 2*ald - 1
#         kappa_sq = 1. / asq
#         kappap_sq = bsq / asq
#         z_in_driver = kappap_sq*sp.ellipkinc(phi,kappa_sq)
#         z_in_driver -= sp.ellipeinc(phi,kappa_sq)
#         z_in_driver += np.tan(phi)*np.sqrt(1-1/asq*np.sin(phi)**2)
#         z_in_driver /= bsq / kp
    beta = (1 - xr**2) / (1 + xr**2)
    if ald <= 0.5:
        n_in_driver = n_p / (1 - beta)
        e_in_driver = kp * np.sqrt(2*(1-ald) - 1./xr + (2*ald - 1)*xr)
    if ald < 0.5:
        e_in_driver *= sgn_sin2phi
    # e = 2(1−nb)−1/x+ (2nb−1)x
    return z_in_driver, n_in_driver, e_in_driver


def n_in_driver_gt_1_2(nd0, kp0Ld, n_p):
    alpha = nd0 / n_p
    kp = np.sqrt(n_p)
    consta = np.sqrt(2*alpha)
    constb = np.sqrt(2*alpha - 1)
    kappa_sq = 1 / consta**2
    kappap_sq = constb**2 / consta**2

    gm,gb,betab,sgnEb,xrb,phib = get_gamma_m_gt_1_2(nd0,kp0Ld,n_p)
    phi_list = np.linspace(0,phib)
#     int_nc_sq = kappap_sq * sp.ellipkinc(phi_list,kappa_sq)
#     int_nc_sq -= sp.ellipeinc(phi_list, kappa_sq)
#     int_nc_sq += np.tan(phi_list) * np.sqrt(1 - kappa_sq*np.sin(phi_list)**2)
#     int_nc_sq_kpsq = int_nc_sq
#     int_g_an = consta * int_nc_sq_kpsq
    int_g_an = kpz(phi_list, alpha)


    z_in_driver = 2*int_g_an / constb**2 / kp
    xr = 1 + np.tan(phi_list)**2
    beta = (1 - xr**2) / (1 + xr**2)
    n_in_driver = n_p / (1 - beta)
    e_in_driver = kp * np.sqrt(2*(1-alpha) - 1./xr + (2*alpha - 1)*xr)
    return  z_in_driver , n_in_driver, e_in_driver

def kpz(phi,alpha):
    consta = np.sqrt(2*alpha)
    constb = np.sqrt(2*alpha - 1)
    kappa_sq = 1 / consta**2
    kappap_sq = constb**2 / consta**2

    int_nc_sq = kappap_sq * sp.ellipkinc(phi,kappa_sq)
    int_nc_sq -= sp.ellipeinc(phi, kappa_sq)
    int_nc_sq += np.tan(phi) * np.sqrt(1 - kappa_sq*np.sin(phi)**2)
    int_nc_sq_kpsq = int_nc_sq
    int_g_an = consta * int_nc_sq_kpsq
    return int_g_an

