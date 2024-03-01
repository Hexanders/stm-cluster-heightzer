from time import time
from numpy import arange, array, pi
from scipy import optimize
from IPython.display import Image
import siunits as siu
from scipy import constants as const
from typing import Callable
import numpy as np 
def atoms_in_trancated_sphere_hm(tau:float, h:float, rho:float, atomic_mass:float):
    '''
    Calculate quantity of atoms in volume of trunkated sphere like in Ingo Barke Diss
    
    Parameter:
        tau: faktor between widht a and height h : tua = a/h
        h: height of cluster
        rho: density of elemnt from witch are clusters made of
        ato,ic_mass: atoic mass of the emelent
    '''
    return (pi*h**3. * (0.5*tau -1./3.)) * rho / atomic_mass

def V_trancated_sphere_hm(tau:float, h:float):
    '''
    Calculate volume of trunkated sphere like in Ingo Barke Diss
    
    Parameter:
        tau: faktor between widht a and height h : tua = a/h
        h: height of cluster
    '''
    return pi*h**3. * (0.5*tau -1./3.)

def tau_sphere(V:float, h:float):
    """
    Tau from V_trancated_sphere_hm just for optimisation purpes
    """
    return (V/(pi*h**3.)+1./3.)*2.

def V_simple_sphere(h:float ):
    """
    Volume of a sphere with the height h = 2R
    """
    return 4./3.*pi*(h/2)**3.

def atoms_in_simple_sphere(tau:float, h:float, rho:float, atomic_mass:float):
    return 4./3.*pi*(h/2)**3. * rho / atomic_mass


def V_ellipsoid_sym(tau:float, h:float):
    '''
    Calculate volume of Elepsoid. V = 4/3 * pi *  a *  b * c
    assumption : a = b = tau*c, with c = 1/2 *h 
    -> V = 4/3 *pi * 1/2* tau *h 
    Parameter:
        tau: factor see above
        h: height of cluster
    '''
    return 1/6 * pi * tau **2. * h**3.

def atoms_in_ellipsoid_sym(tau:float, h:float, rho:float, atomic_mass:float):
    '''
    Calculate quantity of atoms in volume of Elepsoid. V = 4/3 * pi *  a *  b * c
    assumption : a = b = tau*c, with c = 1/2 *h 
    -> V = 4/3 *pi * 1/2* tau *h 
    Parameter:
        tau: factor see above
        h: height of cluster
        rho: density of element from witch are clusters made of
        atomic_mass: atoic mass of the emelent
    '''
    return (1/6 * pi * tau **2. * h**3.) * rho / atomic_mass

def V_sum(function:float, tau:float, heights_array:float):
    """
    Summ over all Volumuse for heigts array with specific function:
        V_ellipsoid_sym(tau,h)
        V_trancated_sphere_hm
    """
    return array([function(tau,i) for i in heights_array]).sum() 

def atoms_sum(function:float, tau:float, heights_array:float, rho:float, atomic_mass:float):
    """
    Summ over all Volumuse for heigts array with specific function:
        V_ellipsoid_sym(tau,h)
        V_trancated_sphere_hm
    """
    return array([function(tau,i, rho, atomic_mass) for i in heights_array]).sum() 

def V_ellipsoid_asym(tau:float, gamma:float, h:float):
    '''
    Calculate volume of Elepsoid. V = 4/3 * pi *  a *  b * c
    assumption : a = gamma* b; b = tau*c, with c = 1/2 *h 
    -> V = 4/3 *pi * gamma * tau* 1/2 *h 
    Parameter:
        tau, gamma : factors see above
        h: height of cluster
    '''
    return 2/3 * pi * tau *gamma * h


def inter_volume(function: Callable, tau_range: list = (0.0,10,0.1), list_of_clusters: list =None):
    """
    Calculate volume for Sperical or elipsoidal model (function) within an intervall (tau_range) 
    function;: V_ellipsoid_sym / V_trancated_sphere_hm
    """
    # start_time = time()
    volume_list_hm = {}
    for a in arange(tau_range[0],tau_range[1],tau_range[2]):
        coord_hights_volum = []
        for i in list_of_clusters:
            coord_hights_volum.append(function(a,i[3]))
        volume_a = array(coord_hights_volum).sum()    
        volume_list_hm[a] = volume_a 
    # print("--- %s seconds ---" % (time() - start_time))
    return volume_list_hm

def inter_atoms_in_volume(function: Callable, tau_range:list = (0.0,10,0.1), list_of_clusters:list =None, rho:float = None, atomic_mass:float = None):
    """
    Calculate atoms in volume for Sperical or elipsoidal model (function) within an intervall (tau_range) 
    function: atoms_in_ellipsoid_sym / atoms_in_trancated_sphere_hm
    """
    # start_time = time()
    atoms_number_list_hm = {}
    for a in arange(tau_range[0],tau_range[1],tau_range[2]):
        coord_hights_volum = []
        for i in list_of_clusters:
            V = function(a,i, rho, atomic_mass)
            coord_hights_volum.append(V)
        atoms_in_interval = array(coord_hights_volum).sum()    
        atoms_number_list_hm[a] = atoms_in_interval 
    # print("--- %s seconds ---" % (time() - start_time))
    return atoms_number_list_hm


def finde_V_root(tau:float, func:Callable,  heigts_list:list, aim_volum:float):
    """
    Helpf function for finding roots for Volumen estimation.
    
    Example:
        optimize.root(finde_V_root,0.01, args=(V_trancated_sphere_hm,hightsRT[:,3],ML_30_RT)).x[0] for getting x value of function V_trancated_sphere_hm
    """
    return aim_volum - V_sum(func, tau, heigts_list)

def finde_atoms_in_V_root(tau:float, func:Callable,  heights_array:list, rho:float, atomic_mass:float, aim_atoms_in_V:float):
    """
    Helper function for finding roots for atoms in volum estimation.
    
    Example:
        optimize.root(finde_V_root,0.01, args=(V_trancated_sphere_hm,hightsRT[:,3],ML_30_RT)).x[0] for getting x value of function V_trancated_sphere_hm
    """
    return aim_atoms_in_V - atoms_sum(func, tau, heights_array, rho, atomic_mass)


def calc_tau_with_error(clheights: list = [],
                        area : float = None,
                        ML : float = None,
                        ML_delta: float = None, 
                        atoms_per_sqM :float= None,
                        atom_mass: float = None,
                        densety: float = None,
                        heits_erro_persent = 0,
                       atoms_in_geometry: Callable = atoms_in_ellipsoid_sym):
    
   
    """
    Calculate tau (τ) with error estimation.

    This function calculates tau (τ) along with its error using an optimization algorithm. 
    Tau represents the parameter that is optimized to match the observed data with the 
    model's predictions.
    For example how many atoms in each  truncatet spere ore in oblate elepsiod as model for every cluster on surface

    Args:
        clheights (list): List of cluster heights.
        area (float): Area of the surface (default is None).
        ML (float): Monolayer coverage (default is None).
        ML_delta (float): Error in monolayer coverage (default is None).
        atoms_per_sqM (float): Number of atoms per square meter (default is None).
        atom_mass (float): Mass of each atom (default is None).
        density (float): Density of the material (default is None).
        atoms_in_geometry (Callable): Function to calculate the number of atoms in the given geometry 
                                      (default is atoms_in_ellipsoid_sym).

    Returns:
        dict: A dictionary containing the calculated tau (τ) and its error (tau_err), 
              along with the optimization result (result_otimization_for_tau).

    Note:
        This function uses an optimization algorithm to find the value of tau (τ) that 
        minimizes the difference between the observed cluster heights and the model's predictions.
        It estimates the error in tau (τ) based on the variation in the optimization result.
    """
    
    atoms_in_persentage_ML = (area * ML)*atoms_per_sqM
    
    tau_ellips = optimize.fsolve(finde_atoms_in_V_root,1., ## calculate tau
                                args=(atoms_in_geometry,
                                      clheights,
                                      densety,
                                      atom_mass,
                                      atoms_in_persentage_ML),
                                  full_output=True)
    
    del_tau  = np.sum(tau_ellips[1]['fvec'] ** 2)
    
    del_tau_ellips_plus = tau_ellips[0][0] - optimize.fsolve(finde_atoms_in_V_root,1., 
                                                     args=(atoms_in_geometry,
                                                          clheights-clheights*heits_erro_persent,
                                                          densety,
                                                          atom_mass,
                                                          atoms_in_persentage_ML + atoms_in_persentage_ML*ML_delta),
                                                     full_output=True)[0][0]
    
    del_tau_ellips_minus = -tau_ellips[0][0] + optimize.fsolve(finde_atoms_in_V_root,1., 
                                                     args=(atoms_in_geometry,
                                                          clheights+clheights*heits_erro_persent,
                                                          densety,
                                                          atom_mass,
                                                          atoms_in_persentage_ML - atoms_in_persentage_ML*ML_delta),
                                                     full_output=True)[0][0]
    del_tau_h = (del_tau_ellips_minus + del_tau_ellips_plus)/2### mittelwert aus dem mximalen und minimalen wert der Fehler
    tau_ellips_err = np.sqrt( del_tau**2. + del_tau_h**2.) 
    return {'tau':tau_ellips[0][0], 'tau_err':tau_ellips_err,'tau_plus:':del_tau_ellips_minus, 'tau_minus':del_tau_ellips_plus, 'result_otimization_for_tau': tau_ellips}
