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


def V_ellipsoid_sym(tau:float, h:float, deltau:float = 0.0, delh:float = 0.0):
    '''
    Calculate volume of Elepsoid. V = 4/3 * pi *  a *  b * c
    assumption : a = b = tau*c, with c = 1/2 *h 
    -> V = 4/3 *pi * 1/2* tau *h 
    Parameter:
        tau: factor see above
        h: height of cluster
    '''
    Volume = 1/6 * pi * tau **2. * h**3.
    Volume_err = np.sqrt((1/3. * pi * tau * h**3.)**2. * deltau**2. + (1./2 * pi * tau **2. * h**2.)**2. * delh**2.)
    return (Volume, Volume_err)

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
def capacity_ellipsoid(a,h):
    """
    Calculate the capacity of an ellipsoid with axis lengths a, b, and c.

    This function calculates the electrical capacity of an ellipsoid, where:
    - a is the ellipsoid axis parallel to the surface,
    - h is the height of the cluster (and also the ellipsoid axis perpendicular to the surface).

    Args:
        a (float): Length of the ellipsoid axis parallel to the surface.
        h (float): Height of the cluster (and also the ellipsoid axis perpendicular to the surface).

    Returns:
        float: The capacity of the ellipsoid.

    Note:
        The formula used for calculating the capacity of the ellipsoid is derived from electrostatics.
        It assumes that 'a' is greater than 'h' and that 'a' and 'h' are equal if the ellipsoid is a sphere.
    """
    return 4.*const.pi*const.epsilon_0* ( np.sqrt( (a/2.)**2. - (h/2.)**2. ) ) /  ( (const.pi/2.) - np.arctan( ( h/2. )/np.sqrt( (a/2.)**2. -(h/2.)**2.) )  )
    
def delta_capacity_ellipsoid(a, del_a, h , del_h):
    """
    Calculate the error in the capacity of an ellipsoid with error propagation.

    This function calculates the error in the electrical capacity of an ellipsoid
    by considering the errors in the input parameters a and h, based on error propagation.

    Args:
        a (float): Length of the ellipsoid axis parallel to the surface.
        del_a (float): Error in the length of the ellipsoid axis parallel to the surface.
        h (float): Height of the cluster (and also the ellipsoid axis perpendicular to the surface).
        del_h (float): Error in the height of the cluster.

    Returns:
        float: The deviation in the capacity of the ellipsoid.

    Note:
        This function uses error propagation techniques to estimate the deviation in the capacity
        of the ellipsoid based on the errors in the input parameters a and h.
    """
    epsilon_0 = const.physical_constants['vacuum electric permittivity'][0]
    del_epsilon_0 = const.physical_constants['vacuum electric permittivity'][2]
    kappa = np.sqrt(a**2.-h**2.)

    ###  partial derivation of h
    part_one = epsilon_0 * ( 4.*const.pi*h/( kappa* (2.* np.arctan(h/kappa) - const.pi) ) +
    
                             8.*const.pi /( 2.* np.arctan(h/kappa) - const.pi)
                           )
    ###  partial derivation of a splittet in sub parts 1 and 2 for clarity
    part_two1 = 4.*const.pi*epsilon_0*(-2.*h*kappa - 2.*a**2. *np.arctan(h/kappa) +const.pi*a**2.)
    
    part_two2 = a * kappa *(const.pi -2.*np.arctan(h/kappa))**2. 
    
    part_two = part_two1 / part_two2
    
    ###  partial derivation of epsilon_0
    part_tree =  2.*const.pi*kappa / ( const.pi/2. - np.arctan(h/kappa) )
    
    return np.sqrt( part_one**2. * del_h**2.  + part_two**2. * del_a**2. + part_tree**2.*del_epsilon_0**2.)

def capacity_sphere(h:float, delh:float = 0.0):
    """
    Calculate the capacity of a sphere with a given radius.

    This function computes the electrical capacity of a sphere with a radius 'h/2',
    where 'h' represents the diameter of the sphere.

    Args:
        h (float): Diameter of the sphere.

    Returns:
        float: The capacity of the sphere.

    Note:
        The formula used for calculating the capacity of the sphere is derived from electrostatics.
        It assumes that the sphere is composed of a uniform material and has no charge distribution.
    """
    c = 4.*const.pi*const.epsilon_0 * ( h/2. )
    c_err = 4.*const.pi*const.epsilon_0 * ( 1/2. )*delh
    return (c,c_err)

def delta_capacity_sphere(delh):
    """
    Calculate the error in the capacity of a sphere with error propagation.

    This function computes the deviation in the electrical capacity of a sphere with
    error propagation based on the error in the diameter 'delh'.

    Args:
        delh (float): Error in the diameter of the sphere.

    Returns:
        float: The deviation in the capacity of the sphere.

    Note:
        This function assumes that the sphere is composed of a uniform material and has no charge distribution.
    """
    return 4.*const.pi*const.epsilon_0 * ( delh/2. )
    
def surface_capacitance_shpere(r,h, sum_end = 5000):
    """
    based on https://solar.physics.montana.edu/dana/ph519/sph_cap.pdf
    r sphere radius
    h distance between sphere center and surface, so basicaly h = d+r, wiht d distance from shere to surface 
    """
    """
    Calculate the "surface" capacitance of a sphere (meaning capacitance of the sphere induced by the surface).

    This function computes the surface capacitance of a sphere using the formula provided
    in the reference: https://solar.physics.montana.edu/dana/ph519/sph_cap.pdf

    Args:
        r (float): Radius of the sphere.
        h (float): Distance between the sphere center and the surface.
                   It's essentially the sum of the radius and the distance from the sphere to the surface.
        sum_end (int): Number of terms in the summation for calculating the function F_of_psi. 
                       Higher values provide more accurate results at the cost of increased computation time.
                       Default is 10000.

    Returns:
        float: The surface capacitance of the sphere.

    Note:
        This function assumes that the sphere is composed of a uniform material and has no charge distribution.
        The surface capacitance is calculated based on the formula provided in the referenced document.
    """
    epsilon_0 = const.physical_constants['vacuum electric permittivity'][0]
    del_epsilon_0 = const.physical_constants['vacuum electric permittivity'][2]
    def F_of_psi(psi,sum_end = sum_end):
        n = 0
        summu_jammy = 0.
        while n <= sum_end:
            summu_jammy += (1/np.sinh((n+1)*psi))
            n += 1
        return np.sinh(psi)*summu_jammy
    return 4*np.pi*epsilon_0*r*F_of_psi(np.arccosh(h/r))

def surface_capacitance_shpere_with_errors(r, h, delr=None, delh = None, sum_end = 5000):
    """
    based on https://solar.physics.montana.edu/dana/ph519/sph_cap.pdf
    r sphere radius
    h distance between sphere surface and surface of the substrate, so basicaly d = h - r, with d distance from sphere center to surface 
    """
    """
    Calculate the "surface" capacitance of a sphere (meaning capacitance of the sphere induced by the surface).

    This function computes the surface capacitance of a sphere using the formula provided
    in the reference: https://solar.physics.montana.edu/dana/ph519/sph_cap.pdf

    Args:
        r (float): Radius of the sphere.
        h (float): Distance between the sphere center and the surface.
                   It's essentially the sum of the radius and the distance from the sphere to the surface.
        sum_end (int): Number of terms in the summation for calculating the function F_of_psi. 
                       Higher values provide more accurate results at the cost of increased computation time.
                       Default is 10000.

    Returns:
        float: The surface capacitance of the sphere.

    Note:
        This function assumes that the sphere is composed of a uniform material and has no charge distribution.
        The surface capacitance is calculated based on the formula provided in the referenced document.
    """
    epsilon_0 = const.physical_constants['vacuum electric permittivity'][0]
    del_epsilon_0 = const.physical_constants['vacuum electric permittivity'][2]
    kapa = 4.*np.pi*epsilon_0
    d = h + r
    def F_of_psi(psi,sum_end = sum_end):
        n = 0
        summu_jammy = 0.
        while n <= sum_end:
            summu_jammy += (1/np.sinh((n+1)*psi))
            n += 1
        return summu_jammy
    
    # def del_R_of_sum(psi,d=d,r=r,sum_end = sum_end):
    #     n = 0
    #     summu_jammy = 0.
    #     while n <= sum_end:
    #         summu_jammy += (d*(n+1) * ( 1/ np.tanh( (n+1) * psi ) ) ) / ( r**2 * np.sqrt( (d/r) -1. ) * np.sqrt( (d+r)/r  ) * np.sinh( (n+1) * psi ) )
    #         n += 1
    #     return summu_jammy
    
    # def del_D_of_sum(psi,d=d,r=r,sum_end = sum_end):
    #     n = 0
    #     summu_jammy = 0.
    #     while n <= sum_end:
    #         summu_jammy += ( (n+1)*(1/np.tanh((n+1)*psi) ) ) / ( r*np.sqrt((d/r)-1)*np.sqrt((d+r)/r) * np.sinh((n+1)*psi))
    #         n += 1
    #     return summu_jammy

    def del_of_sum(psi,sum_end = sum_end):
        n = 0
        summu_jammy = 0.
        while n <= sum_end:
            summu_jammy += (n+1) / ( np.tanh((n+1)*psi) * np.sinh((n+1)*psi) )
            n += 1
        return summu_jammy
    psi = np.arccosh(d/r)
    
    if delh or delr:
        del_of_sum_result =  del_of_sum(psi) / (np.sqrt((d/r)-1)*np.sqrt((d+r)/r))
        if delr == None:
            delr = .0
        else:
            first_a = np.sinh(psi) * F_of_psi(psi)
            first_b = r * F_of_psi(np.arccosh(d/r)) * (d**2. / (r**2. * np.sqrt( (d-r)/(d+r) ) *(d+r) ) )
            first_c = r * np.sinh(psi) * (d / r**2.) * del_of_sum_result
        if delh == None:
            delh = .0
        else:
            second_a = r * ( d /( r * np.sqrt( (d-r)/(d+r) ) * (d+r) ) ) * F_of_psi(psi)  
            second_b = r *  np.sinh(psi) * (del_of_sum_result/r)
            
        first = kapa * (first_a + first_b + first_c)
        second = kapa * (second_a + second_b)
        errors = np.sqrt(first**2.*delr**2. + second **2.*delh**2. )
    else:
        errors = 0.0
    return (4*np.pi*epsilon_0*r*np.sinh(psi)*F_of_psi(psi) , errors) 

def radius_of_spher_volume(v:float, v_err:float = 0.0):
    """
    Returns radius of the spher with the given volume v
    """
    radius = (v*3./(4.*np.pi))**(1./3.)
    radius_err = 1/(6.**(2./3)*(np.pi)**(1/3)* (v)**(2./3))*v_err
    return (radius, radius_err)

def C_ell_overC_spher(tau:float, tau_err:float = 0.0):
    x = np.sqrt(tau**2.-1.)
    error = np.sqrt((x/ (tau**(5./3) *(np.pi/2. - np.arctan(1/x)))) **2. * tau_err**2.)
    return ( (x * tau**(-2./3.)) / (np.pi/2.-np.arctan(1/x)), error ) 
