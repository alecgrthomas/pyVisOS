"""useful functions for plotting FBPIC simulation data

Contains
---
plot_initial_final_spectra : function
plot_spectrum_v_time : function
pa2d_lineout_movie : function
pa_movie_2d : function
run_from_ipython : function
"""
import numpy as np
import os
from matplotlib import pyplot as plt
import h5py as h5
from scipy import constants as cnst
from scipy.integrate import odeint
from scipy import special as sp
from scipy import integrate as nt
from scipy.interpolate import interp1d
import time # time
from tqdm import tqdm
import nonlinear_functions as nlf

import osh5io
import osh5def
import osh5vis
import osh5utils
import osh5dir

from matplotlib.animation import FFMpegWriter
import matplotlib.animation as manimation

def get_tau_dn0(nd0_np0, kp0Ld0, n_p):
    
    v_gamma = np.vectorize(nlf.get_gamma_max)
    wave_beta = 1.0
    kp = np.sqrt(n_p)
    gm,gb,pb,Eb,xrb,_ = v_gamma(nd0_np0,kp0Ld0,n_p)
    q = -1
    qEb_kp = Eb / kp
    betab = pb / gb
    nb = n_p / (1-betab)
    phi_wake = np.arctan2(qEb_kp * np.sqrt(1+gb), pb*np.sqrt(2))
    phi_wake_unwrap = np.unwrap(phi_wake * 2) / 2
    phi_wake_pi = phi_wake_unwrap / np.pi
    next_int = np.ceil(phi_wake_pi)
    next_odd_int = next_int + np.mod(next_int + 1,2) 
    phi_dn0 = np.pi * (next_odd_int + 0.5)

    kappa_sq = (gm-1.) / (gm + 1.)
    kappa = np.sqrt(kappa_sq)
    kappa_p = np.sqrt(2 / (gm + 1.))

    #should wave_beta be moved to d_tau?
    tau_b = wave_beta * 1/kp*(2 / kappa_p * sp.ellipeinc(phi_wake_unwrap, kappa_sq) - kappa_p * sp.ellipkinc(phi_wake_unwrap, kappa_sq))
    d_tau_b = -1/kp * 2 * kappa / kappa_p * np.sin(phi_wake_unwrap)
    tau_b= tau_b + d_tau_b

    tau_dn0 = wave_beta * 1/kp*(2 / kappa_p * sp.ellipeinc(phi_dn0, kappa_sq) - kappa_p * sp.ellipkinc(phi_dn0, kappa_sq))
    d_tau_dn0 = -1/kp * 2 * kappa / kappa_p * np.sin(phi_dn0)
    tau_dn0 = tau_dn0 + d_tau_dn0 - tau_b+kp0Ld0
    return tau_dn0


def pa_coupled_nonlinear(y,t, nd0_np0, kp0Ld0):
        n_p, nc = y
        n_p_interp = np.linspace(0.01,1.1)
        tau_dn0_list = get_tau_dn0(nd0_np0, kp0Ld0, n_p_interp)
        dn = n_p_interp[1]-n_p_interp[0]
        dtaudn0_dn = (tau_dn0_list[2:] - tau_dn0_list[:-2]) / 2 / dn
        dtau_dn0_dn_fun = interp1d(n_p_interp[1:-1], dtaudn0_dn)
        dxi_dn = -dtau_dn0_dn_fun(n_p)
        dnp_dt = -n_p/2/nc / dxi_dn
        
        gm,_,_,_,_,_ = nlf.get_gamma_max(nd0_np0, kp0Ld0, n_p)
        dnc_dt = np.sqrt(2)*n_p**1.5 * np.sqrt(gm - 1)
        return [dnp_dt, dnc_dt]
    

    
def get_spectrum_v_time(base_directory,field,species,n_iterations):
    """ plot total spectra from FBPIC time series data
    
    Parameters
    ---
    sim_dir : default './', where `diags/hdf5/` directory is
    coordinate : default 'x', component of E to get spectrum of
    this is to stop having to rerun to generate new picture each time!
    """
    t1 = time.time()
    print('generating spectrum v time data')
    #ts_add = LpaDiagnostics(sim_dir + 'diags/hdf5/')
    
    #ex, meta_ex = ts_add.get_field(field = 'E', coord = coordinate, klim = None, iteration = 0, slice_across = 'r')
    (field_name,fldmap) = osh5dir.getfieldname(base_directory,field,species,'')
    ex = osh5dir.getfile(field_name,0)
    
    tmin = ex.run_attrs.get('TIME')[0]
    
    n_ft = int(ex.size/2)
    
    Lz = ex.axes[0][-1]-ex.axes[0][0]
    klist = 2*np.pi / Lz * np.arange(0,ex.size)

    klim_ind = int(ex.size/2)
    lambdas = np.zeros_like(klist)
    lambdas[1:] = 2*np.pi / klist[1:]
    #n_iterations = ts_add.iterations.size
    total_spectra = np.zeros([ex.size, n_iterations])
    k_max = np.zeros(n_iterations)
    meanks = np.zeros(n_iterations)
    #for ii, iter_num in tqdm(enumerate(ts_add.iterations)):
    for ii in tqdm(range(n_iterations)):
        #exi, meta_exi = ts_add.get_field(field = 'E', coord = coordinate, iteration = iter_num, slice_across = 'r')
        exi = osh5dir.getfile(field_name,ii)
        exi_ft = abs(np.fft.fft(exi)) / np.sqrt(exi.size)
        total_spectra[:,ii] = exi_ft**2
        max_ind = np.argmax(exi_ft[:klim_ind])
        k_max[ii] = klist[max_ind]
        meanks[ii] = np.dot(total_spectra[:n_ft,ii],klist[:n_ft]) / np.sum(total_spectra[:n_ft,ii])
        
    #tmin_fs = ts_add.tmin * 1e15
    #tmax_fs = ts_add.tmax * 1e15
    tmax = exi.run_attrs.get('TIME')[0]

    time_out = np.linspace(tmin,tmax,meanks.size)
    
    return time_out,klist,total_spectra,k_max,meanks

def get_theory_curve(t_fs,nc0_np0,om0):

    # AGRT - now add theoretical curve
    #nc0_np0 = 100
    #om0 = 2.36e15
    kp0Ld0 = 1.0 #0.475
    nd0_np0 = 0.8
    tf = t_fs[-1]
    tsol = np.linspace(0, tf, 10000) #2*int(tf)+1)

    sol = odeint(pa_coupled_nonlinear, [1,nc0_np0], tsol, args=(nd0_np0, kp0Ld0))
    
    t_out = tsol*np.sqrt(nc0_np0)/om0*1e15
    
    om_predict = np.sqrt(sol[:,1]/sol[0,1])
    

    
    # ----------- end spectrum over time plot -------
    # ----------- lineout movie -------------
    
    return t_out,om_predict

def get_time_history(base_directory,field,species,n_iterations):
    """ plot total spectra from FBPIC time series data
    
    Parameters
    ---
    sim_dir : default './', where `diags/hdf5/` directory is
    coordinate : default 'x', component of E to get spectrum of
    this is to stop having to rerun to generate new picture each time!
    """
    t1 = time.time()
    print('generating spectrum v time data')
    #ts_add = LpaDiagnostics(sim_dir + 'diags/hdf5/')
    
    #ex, meta_ex = ts_add.get_field(field = 'E', coord = coordinate, klim = None, iteration = 0, slice_across = 'r')
    (field_name,fldmap) = osh5dir.getfieldname(base_directory,field,species,'')
    ex = osh5dir.getfile(field_name,0)
    
    tmin = ex.run_attrs.get('TIME')[0]
    
        
    Lz = ex.axes[0][-1]-ex.axes[0][0]

    data = np.zeros([ex.size, n_iterations])
    
    for ii in tqdm(range(n_iterations)):
        exi = osh5dir.getfile(field_name,ii)
        data[:,ii] = exi
                
    tmax = exi.run_attrs.get('TIME')[0]

    time_out = np.linspace(tmin,tmax,n_iterations)
    
    return time_out,data

