# cython: boundscheck=False, wraparound=False,
# cython: cdivision_warnins=True
# cython: initializedcheck=True

import numpy as np
import pandas as pd
from cypolyscat cimport wig, wig3j0, setup_wigner

cimport cython 
from libc.math cimport sqrt, exp
# from libc.complex import cexp
from scipy.linalg.cython_blas cimport zgbmv
from scipy.linalg.cython_lapack cimport zgbsv


cdef extern from "complex.h" nogil:
    double complex cexp(double complex)
    double cabs(double complex)

# Parameters and constants

cdef double kB = 3.1668114E-6 #(Eh/K)

# These are the values for CF3I
cdef:
    double A   = 5750E6   * 1.51983E-16 # MHz to a. u.
    double B   = 1523E6   * 1.51983E-16
    double B   = 1515E6   * 1.51983E-16
    double DJ  = 0.164E3  * 1.51983E-16
    double DJK = 0.9925E3 * 1.51983E-16

#==============================================================================

cdef construct_Hrot(int dim, double[:] Jvec, int K,
                    double[:] Jout):
    """Write the field-free, diagonal Hamiltonian into the vector Jout."""
    cdef:
        int i
        double kk = <double> K, j
    
    kk = kk * kk
    kk = (A - B) * kk
    
    for i in range(dim):
        j = Jvec[i]
        j = j * j + j
        j = j * B
        Jout[i] = j + kk

    return

cdef eng_distort(int dim, double[:] Jvec, int K,
                     double[:] Jout):
    """Correct Jout for centrifugal distortions"""
    cdef:
        int i
        double kk = <double> K, j, j2

    kk = kk * kk
    
    for i in range(dim):
        j = Jvec[i]
        j = j * j + j

        j2 = j * j
        j2 = DJ * j2

        j = DJK * j * kk

        Jout[i] = Jout[i] - j - j2

    return

#===============================================================================

cpdef dipexcit_matele(int Jmax, int K, int M, int q):
    """Return the matrix of matrix elements for a dipole excitation with field component q.
       q=0 corresponds to a parallel transition, q=+/-1 to a perpendicular one."""
    cdef:
        int i, j, k, l
        int cleb_i, wig_i, i_start, i_stop
        int i_min = max(abs(K), abs(M))
        double j1, j2
        double[:] clf = np.zeros(200)
        double[:] Jvec
        double[:,:] wigout
        double phas
        
    dim = Jmax + 1 - i_min
    wigKM = np.zeros([2, dim, dim])
    wigout = np.zeros([dim, dim])
    Jvec = np.arange(dim, dtype=np.float_) + i_min
               
    phas = (-1) ** abs(K + M)
    
    for j in range(dim):
        i_start = wig(1, <int> Jvec[j], -K,  0, clf)
        cleb_i = <int> max(i_min - i_start,  0)
        wig_i  = <int> max(i_start - i_min,  0)
        i_stop = <int> max(dim - wig_i, 0)
        wigKM[0,j,wig_i:wig_i+i_stop] = clf[cleb_i:cleb_i+i_stop]
    
        i_start = wig(1, <int> Jvec[j], -M, -q, clf)
        cleb_i = <int> max(i_min - i_start,  0)
        wig_i  = <int> max(i_start - i_min,  0)
        i_stop = <int> max(dim - wig_i, 0)
        wigKM[1,j,wig_i:wig_i+i_stop] = clf[cleb_i:cleb_i+i_stop]

        for k in range(dim):
            j1 = Jvec[j]
            j2 = Jvec[k]
            
            j1 = 2 * j1 + 1
            j2 = 2 * j2 + 1    
            
            j1 = sqrt(j1 * j2)
            wigout[j,k] = wigKM[0,j,k] * wigKM[1,j,k] * j1 * phas
            
    return np.asarray(wigout)

#==============================================================================

def propagate_psi(int Jmax,
                 #double[:] time,
                 #double[:] field,
                 #linear=True,
                  coeffs=[]):
    """Do the time integration over a laser pulse amplitude vector called field for all quantum numbers."""
    cdef:
        int ki, mi, k, m, i, j, l
#       int j21 = Jmax + 1
        int J1 = Jmax + 1
        int i_min, dim
        int[:] mk = np.linspace(0, Jmax, J1, dtype=np.int32)
        
        double ii
        
        int[:] ipiv = np.arange(J1, dtype=np.int32)
#       double complex[::1,:] H_rot
#       double complex[::1,:] H_ind
        
#       double complex[::1,:] c_cur
#       double complex[::1,:] c_new
 
#       double complex[::1,:] Hstat
#       double complex[::1,:] H_cur
#       double complex[::1,:] H_new

#       double[:,:,::1] wig
        double[:] Jvec
        double[:] EJK
        
#       double[:,:] Javg
        
    res = list()
    eng = list()
    jav = list()
    
#   for ki in range(J1):
#       k = mk[ki]
#       
#       if linear:
#           k = 0
    k = 0

    for mi in range(J1):
        m = mk[mi]
        
        i_min = max(abs(k), abs(m))
        dim = J1 - i_min
    
#       ipiv = np.arange(dim, dtype=np.int32)
#       H_rot = np.zeros([7, dim], dtype=np.complex_, order='F')
#       H_ind = np.zeros([7, dim], dtype=np.complex_, order='F')

#       c_cur = np.zeros([dim, dim], dtype=np.complex_, order='F')
#       c_new = np.zeros([dim, dim], dtype=np.complex_, order='F')

#       Hstat = np.zeros([7, dim], dtype=np.complex_, order='F')
#       H_cur = np.zeros([7, dim], dtype=np.complex_, order='F')
#       H_new = np.zeros([7, dim], dtype=np.complex_, order='F')

#       wig  = np.zeros([2, dim, dim], dtype=np.float_, order='C')
        Jvec = np.zeros(dim)
        EJK  = np.zeros(dim)
        
#       Javg = np.zeros([time.shape[0], dim])
    
        for i in range(dim):
            ii = <double> i
            Jvec[i] = <double> i_min + ii
            c_cur[i,i] = 1
            
        construct_Hrot(dim, Jvec, k, EJK)
        H_ind = dipexcit_matele(dim, Jvec, k, m, wig, H_ind)

#       for i in range(dim):
#           Hstat[2,i] = EJK[i]

#       if any(coeffs):
#           c_cur = coeffs[k,m].unstack().values.T
            
#       propagateCN(dim, H_cur, H_new, Hstat, H_ind,
#                   c_cur, c_new, time, field,
#                   ipiv, Jvec, Javg)
        
        iv = np.asarray(Jvec)
        inx = pd.MultiIndex.from_product([[k], [m], iv, iv], names=['K', 'M', 'Jinit', 'J'])
        
        sr = pd.Series(np.asarray(H_ind).ravel(), index=inx)
        res.append(sr)
        
        eng_distort(dim, Jvec, k, EJK)
        
        inx = pd.MultiIndex.from_product([[k], [m], iv], names=['K', 'M', 'J'])
        eng.append(pd.Series(np.asarray(EJK), inx))
        
#       inx = pd.MultiIndex.from_product([[k], [m], np.asarray(time), iv], names=['K', 'M', 'time', 'J2'])
#       rav = pd.Series(np.asarray(Javg).ravel(), index=inx)
#       jav.append(rav)

            
 #      if linear:
 #          break
        
    return pd.concat(eng), pd.concat(res)#, pd.concat(jav)
