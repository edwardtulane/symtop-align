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
#   double B   = 1523E6   * 1.51983E-16
    double B   = 1515E6   * 1.51983E-16
    double DJ  = 0.164E3  * 1.51983E-16
    double DJK = 0.9925E3 * 1.51983E-16

    double al_para = 61.
    double al_perp = 45.
    double del_al = al_para - al_perp
    
#     DJ = 7.95E3  * 1.51983E-16
#     DJK = 9.94E4 * 1.51983E-16

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


cpdef cos2_matele(int Jmax, int K, int M):
    """Return the matrix of cos2 matrix elements between the rotational basis set functions"""
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
        i_start = wig(2, <int> Jvec[j], -K, 0, clf)
        cleb_i = <int> max(i_min - i_start, 0)
        wig_i  = <int> max(i_start - i_min, 0)
        i_stop = <int> max(dim - wig_i, 0)
        wigKM[0,j,wig_i:wig_i+i_stop] = clf[cleb_i:cleb_i+i_stop]
    
        i_start = wig(2, <int> Jvec[j], -M, 0, clf)
        cleb_i = <int> max(i_min - i_start, 0)
        wig_i  = <int> max(i_start - i_min, 0)
        i_stop = <int> max(dim - wig_i, 0)
        wigKM[1,j,wig_i:wig_i+i_stop] = clf[cleb_i:cleb_i+i_stop]

        for k in range(dim):
            j1 = Jvec[j]
            j2 = Jvec[k]
            
            j1 = 2 * j1 + 1
            j2 = 2 * j2 + 1    
            
            j1 = sqrt(j1 * j2)
            wigout[j,k] = ((2/3.) * wigKM[0,j,k] * wigKM[1,j,k] * j1 * phas)
            
        i_start = wig(0, <int> Jvec[j], -K, 0, clf)
        cleb_i = <int> max(i_min - i_start, 0)
        wig_i  = <int> max(i_start - i_min, 0)
        i_stop = <int> max(dim - wig_i, 0)
        wigKM[0,j,wig_i:wig_i+i_stop] = clf[cleb_i:cleb_i+i_stop]
    
        i_start = wig(0, <int> Jvec[j], -M, 0, clf)
        cleb_i = <int> max(i_min - i_start, 0)
        wig_i  = <int> max(i_start - i_min, 0)
        i_stop = <int> max(dim - wig_i, 0)
        wigKM[1,j,wig_i:wig_i+i_stop] = clf[cleb_i:cleb_i+i_stop]
        
        j1 = Jvec[j]
            
        j1 = 2 * j1 + 1

        wigout[j,j] = wigout[j,j] + ((1/3.) * wigKM[0,j,j] * wigKM[1,j,j] * j1 * phas)

    return np.asarray(wigout)


cdef construct_Hind(int dim, double[:] Jvec, int K, int M,
                    double[:,:,:] wigKM,
                    double complex[:,:] H_out):
    """Write the field-induced Hamiltonian into H_out. An array for the Wigner coefficient matrix wigKM
    needs to be provided."""
    cdef:
        int i, j, k, l
        int cleb_i, wig_i, i_start, i_stop
        int i_min = <int> Jvec[0]
        double j1, j2
        double[:] clf = np.zeros(200)
        double phas
        
    phas = (-1) ** abs(K + M)
    
    for j in range(dim):
        i_start = wig(2, <int> Jvec[j], K, 0, clf)
        cleb_i = <int> max(i_min - i_start, 0)
        wig_i  = <int> max(i_start - i_min, 0)
        i_stop = <int> max(dim - wig_i, 0)
        wigKM[0,j,wig_i:wig_i+i_stop] = clf[cleb_i:cleb_i+i_stop]
    
        i_start = wig(2, <int> Jvec[j], M, 0, clf)
        cleb_i = <int> max(i_min - i_start, 0)
        wig_i  = <int> max(i_start - i_min, 0)
        i_stop = <int> max(dim - wig_i, 0)
        wigKM[1,j,wig_i:wig_i+i_stop] = clf[cleb_i:cleb_i+i_stop]


    cdef:
        int diff, ioff, joff, rng, diag, ix
        
    for diag in range(5):
        diff = 2 - diag
        ioff = max(diff, 0)
        joff = max(-diff,0)
        rng = dim - abs(diff)
        
        for ix in range(rng):
            k = ioff + ix
            l = joff + ix
            
            j1 = Jvec[k]
            j2 = Jvec[l]
            
            j1 = 2 * j1 + 1
            j2 = 2 * j2 + 1    
            
            j1 = sqrt(j1 * j2)
#             phas = 1
            H_out[diag,k] = <double complex> (j1 * phas * wigKM[0,k,l] * wigKM[1,k,l])
    return


#==============================================================================

def propagate_psi(int Jmax,
                  double[:] time,
                  double[:] field,
                  linear=True,
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
        double complex[::1,:] H_rot
        double complex[::1,:] H_ind
        
        double complex[::1,:] c_cur
        double complex[::1,:] c_new
 
        double complex[::1,:] Hstat
        double complex[::1,:] H_cur
        double complex[::1,:] H_new

        double[:,:,::1] wig
        double[:] Jvec
        double[:] EJK
        
        double[:,:] Javg
        
    res = list()
    eng = list()
    jav = list()
    
    for ki in range(J1):
        k = mk[ki]
        
        if linear:
            k = 0
            
        for mi in range(J1):
            m = mk[mi]
            
            i_min = max(abs(k), abs(m))
            dim = J1 - i_min
        
            ipiv = np.arange(dim, dtype=np.int32)
            H_rot = np.zeros([7, dim], dtype=np.complex_, order='F')
            H_ind = np.zeros([7, dim], dtype=np.complex_, order='F')

            c_cur = np.zeros([dim, dim], dtype=np.complex_, order='F')
            c_new = np.zeros([dim, dim], dtype=np.complex_, order='F')

            Hstat = np.zeros([7, dim], dtype=np.complex_, order='F')
            H_cur = np.zeros([7, dim], dtype=np.complex_, order='F')
            H_new = np.zeros([7, dim], dtype=np.complex_, order='F')

            wig  = np.zeros([2, dim, dim], dtype=np.float_, order='C')
            Jvec = np.zeros(dim)
            EJK  = np.zeros(dim)
            
            Javg = np.zeros([time.shape[0], dim])
        
            for i in range(dim):
                ii = <double> i
                Jvec[i] = <double> i_min + ii
                c_cur[i,i] = 1
                
            construct_Hrot(dim, Jvec, k, EJK)
            construct_Hind(dim, Jvec, k, m, wig, H_ind)

            for i in range(dim):
                Hstat[2,i] = EJK[i]

            if any(coeffs):
                c_cur = coeffs[k,m].unstack().values.T
                
            propagateCN(dim, H_cur, H_new, Hstat, H_ind,
                        c_cur, c_new, time, field,
                        ipiv, Jvec, Javg)
            
            iv = np.asarray(Jvec)
            inx = pd.MultiIndex.from_product([[k], [m], iv, iv], names=['K', 'M', 'Jinit', 'J'])
            
            sr = pd.Series(np.asarray(c_cur.T).ravel(), index=inx)
            res.append(sr)
            
            eng_distort(dim, Jvec, k, EJK)
            
            inx = pd.MultiIndex.from_product([[k], [m], iv], names=['K', 'M', 'J'])
            eng.append(pd.Series(np.asarray(EJK), inx))
            
            inx = pd.MultiIndex.from_product([[k], [m], np.asarray(time), iv], names=['K', 'M', 'time', 'J2'])
            rav = pd.Series(np.asarray(Javg).ravel(), index=inx)
            jav.append(rav)

            
        if linear:
            break
        
    return pd.concat(eng), pd.concat(res), pd.concat(jav)

def propagate_fieldfree(Jmax, int K, int M,
                        double[:] eng, 
                        double complex[:,:] coeffs, 
                        double time):

    cdef:
        int j, l
        int cdim
        
        double complex c, e, t
        double complex [:,:] cff_prop = np.zeros_like(coeffs, dtype=np.complex_)
        
#     i_min = max(abs(K), abs(M))
    
    cdim = coeffs.shape[0]
    t = <double complex> time
    
    for j in range(cdim):
        for l in range(cdim):
            e = <double complex> eng[l]
            c = coeffs[j,l]
            cff_prop[j,l] = <double complex> c * cexp(-1j * t * e)
            
    return np.array(cff_prop)

#=========================================================================================================

@cython.wraparound(False)
@cython.boundscheck(False)
def propagateCN(int dim, double complex[::1,:] H_new, double complex[::1,:] H_old,
                double complex[::1,:] Hstat, double complex[::1,:] H_ind,
                double complex[::1,:] c_old, double complex[::1,:] c_new,
                double[:] time, double[:] field,
                int[:] ipiv, double[:] Jvec, double[:,:] Javg):
    """Do the time integration for a set of K and M quantum numbers. The final result is contained in c_new."""
    cdef:
        int i,j, t
        int m = 5, lda = 7
        int one = 1, two = 2
        int length = time.shape[0] - 1
        int info
        double jfac
        double complex[:] norm = np.zeros(dim, dtype=np.complex_)
        double complex f
        double complex fac = 1., nul = 0
        double complex[:] old_slc, upd_slc
        double complex delT = time[1] - time[0]
        
    f = field[0]
    f = (-2/3.)*(f*f*del_al) / 4. # eps_fac
    
    for j in range(dim):
        for i in range(5):
            H_old[i,j] = -0.5j * delT * (Hstat[i,j] + f * H_ind[i,j])
        H_old[2,j] = H_old[2,j] + 1
        
    for t in range(length):
        f = field[t + 1]
        f = (-2/3.)*(f*f*del_al) / 4. # eps_fac
        
        for j in range(dim):
            for i in range(5):
                H_new[i+2,j] = 0.5j * delT * (Hstat[i,j] + f * H_ind[i,j])
            H_new[4,j] = H_new[4,j] + 1
            
#         print np.asarray(H:)

        for j in range(dim):
            old_slc = c_old[:,j]
            upd_slc = c_new[:,j]
#                 trans, m,   n,   kl,     ku, alph,               lda      x        inx, beta,    y,        incy)    
            zgbmv('N', &dim, &dim, &two, &two, &fac, &H_old[0,0], &lda, &old_slc[0], &one, &nul, &upd_slc[0], &one)
    
        for j in range(dim):
            for i in range(5):
                H_old[i,j] = H_new[i+2,j].conjugate()
                
#               N,    kl,   ku, nrhs,       AB, ldab,   ipiv,      b,          ldb, info
        zgbsv(&dim, &two, &two, &dim, &H_new[0,0], &lda, &ipiv[0], &c_new[0,0], &dim, &info)
    
        for j in range(dim):
            jfac = Jvec[j] * Jvec[j] + Jvec[j]
            for i in range(dim):
                c_old[i,j] = c_new[i,j] # / norm[j]

#                 jfac = jfac * 
                Javg[t+1, i] = Javg[t+1, i] + jfac * <double> (c_old[i,j] * c_old[i,j].conjugate())#abs(c_old[i,j]) ** 2
                
#=========================================================================================================
#=========================================================================================================

def boltzmann_dist(int Jmax, double T, linear=True):
    """Return the Boltzmann distribution for a given temperature T up to Jmax
       (Maximum probability density at sqrt(kB * T / 2B) - 0.5 [a. u.]). """
    cdef:
        int ki, mi, k, m, i
        int j21 = 2 * Jmax + 1
        int J1 = Jmax + 1
        int i_min, dim
        int[:] mk = np.linspace(-Jmax, Jmax, j21, dtype=np.int32)
        
        double ii, e
        double kbT = 1 / (kB * T)
        
        double[:] Jvec# = np.zeros(J1)
        double[:] EJK# = np.zeros(J1)
        double[:] degs
        
    res = list()
    
    for ki in range(j21):
        k = mk[ki]
        
        if linear:
            k = 0
        
        i_min = abs(k)
        dim = J1 - i_min

        Jvec = np.zeros(dim)
        EJK = np.zeros(dim)        
        degs = np.zeros(dim)
        
        for i in range(dim):
            ii = <double> i
            Jvec[i] = <double> i_min + ii
            degs[i] = Jvec[i] * 2 + 1

        construct_Hrot(dim, Jvec, k, EJK)     
        
        for i in range(dim):
            e = EJK[i]
            EJK[i] = exp(-1 * e * kbT) * degs[i]

        iv = np.asarray(Jvec)
        inx = pd.MultiIndex.from_product([[k], iv], names=['K', 'J'])
        sr = pd.DataFrame({'occ': np.asarray(EJK), 'deg': np.asarray(degs)}, index=inx)
        res.append(sr)        


        if linear:
            break
        
    return pd.concat(res)

#=========================================================================================================
#=========================================================================================================
def generate_wigner_KM(int K, int M,
                       int Jmax,
                       int thetaN):
    
    cdef:
        int J1 = Jmax + 1
        int i_min, dim
        double ii
        
        double[:] Jvec
        double[:,:] ds
    
    i_min = max(abs(K), abs(M))
    dim = J1 - i_min
                  
    Jvec = np.zeros(dim)    
    th = np.linspace(0, np.pi, thetaN)    
    ds = np.zeros([dim, thetaN])

    setup_wigner(Jmax, K, M, dim, thetaN, ds)

    for i in range(dim):
        ii = <double> i
        Jvec[i] = <double> i_min + ii

    iv = np.asarray(Jvec)
    inx = pd.MultiIndex.from_product([[K], [M], iv], names=['K', 'M', 'J'])
    ds_df = pd.DataFrame(np.asarray(ds), index=inx, columns=th) 
    
    return ds_df
    
def generate_wigner_func(int Jmax,
                         int thetaN,
                         linear=True):
    """As they are notoriously slow to evaluate, precompute the rotational eigenfunctions |JKM> and store them 
       in a multi-indexed DataFrame."""
    cdef:
        int ki, mi, k, m, i, # j, l
        int j21 = 2 * Jmax + 1
        int J1 = Jmax + 1
        int i_min, dim
        int[:] mk = np.linspace(-Jmax, Jmax, j21, dtype=np.int32)
        double ii
        
        double[:] Jvec
        double[:,:] ds, dp
    
    ds_col = list()
#     dp_col = list()
    
    
    th = np.linspace(0, np.pi, thetaN)
    
    for ki in range(j21):
        k = mk[ki]
        
        if linear:
            k = 0
            
        for mi in range(j21):
            m = mk[mi]
            
            i_min = max(abs(k), abs(m))
            dim = J1 - i_min
                  
            Jvec = np.zeros(dim)    
            ds = np.zeros([dim, thetaN])
#             dp = ds.copy()
 
            setup_wigner(Jmax, k, m, dim, thetaN, ds)
    
            for i in range(dim):
                ii = <double> i
                Jvec[i] = <double> i_min + ii

            iv = np.asarray(Jvec)
            inx = pd.MultiIndex.from_product([[k], [m], iv], names=['K', 'M', 'J'])
            ds_df = pd.DataFrame(np.asarray(ds), index=inx, columns=th)
#             dp_df = pd.DataFrame(np.asarray(dp), index=inx, columns=th)
            
            ds_col.append(ds_df)
#             dp_col.append(dp_df)
            
            print 'K=%4i --- M=%4i' % (k, m)
            
        if linear:
            break
            
    return pd.concat(ds_col)#, pd.concat(dp_col)

#=========================================================================================================

@cython.wraparound(False)
@cython.boundscheck(False)
def evolve_cos2(int Jmax, int K, int M,
                double[:] eng, 
                double complex[:,:] coeffs, 
                double[:] time,
                double[:] boltz):

    cdef:
        int i, j, l, m
        int tdim, imin, cdim
        
        double cele, wele
        double complex c, e, t
        double[:] c2 = time.copy()
        double [:,:] reduced_prod
        double[:,:] w
        double complex[:,:,:] outer_prod
        
    tdim = time.shape[0]
#     i_min = max(abs(K), abs(M))
    cdim = coeffs.shape[0]
    
    outer_prod = np.zeros([tdim, cdim, cdim], dtype=np.complex_)
    reduced_prod = np.zeros([tdim, cdim])
    
    
    for i in range(tdim):
        for j in range(cdim):
            for l in range(cdim):
                t = <double complex> time[i]
                e = <double complex> eng[l]
                c = coeffs[j,l]
                outer_prod[i,j,l] = <double complex> c * cexp(-1j * t * e)
            
    w = cos2_matele(Jmax, K, M)
    
    for i in range(tdim):
        for j in range(cdim):
            for l in range(cdim):
                for m in range(cdim):
                    cele = <double> (outer_prod[i,j,l] * outer_prod[i,j,m].conjugate())
                    wele = w[l,m] * cele
                    reduced_prod[i,j] = reduced_prod[i,j] + wele
                    
    for i in range(tdim):
        c2[i] = 0
        for j in range(cdim):
            c2[i] = c2[i] + reduced_prod[i,j] * boltz[j]
            
        
    return np.array(c2)

from cython.parallel import prange

@cython.wraparound(False)
@cython.boundscheck(False)
def evolve_angdist(int Jmax, int K, int M,
                   double[:] eng, 
                   double complex[:,:] coeffs, 
                   double[:] time,
                   double[:,:] ds,
                   double[:] boltz):

    cdef:
        int i, j, l, m, th
        int tdim, imin, cdim
        
        double prod
        double complex c, e, t, d
        double complex c2, dele
        double[:,:] angdist
        double complex[:,:,:] reduced_prod
        double complex[:,:,:] outer_prod
        
    tdim = time.shape[0]
    thdim = ds.shape[1]
#     i_min = max(abs(K), abs(M))
    cdim = coeffs.shape[0]
    
    outer_prod = np.zeros([tdim, cdim, cdim], dtype=np.complex_)
    reduced_prod = np.zeros([tdim, cdim, thdim], dtype=np.complex_)
    angdist = np.zeros([tdim, thdim])
    
    
#     for i in prange(tdim, nogil=True, num_threads=6):
    for i in range(tdim):
#         for th in range(thdim):
            for j in range(cdim):
                for l in range(cdim):
                    t = <double complex> time[i]
                    e = <double complex> eng[l]
                    c = coeffs[j,l]
#                     d = <double complex> ds[l, th]
                    outer_prod[i,j,l] = <double complex> c * cexp(-1j * t * e)
#     for i in prange(tdim, nogil=True, num_threads=2):
    for i in range(tdim):
        for j in range(cdim):
            for l in range(cdim):
#               for m in range(cdim):
                    c2 = outer_prod[i,j,l]
                    for th in range(thdim):
                        dele = <double complex> ds[l,th]
#                       dele = dele * ds[m,th]
                        dele = dele * c2
                        reduced_prod[i,j,th] = reduced_prod[i,j,th] + dele # <double> ds[l,th] * ds[m,th] * c2

    for i in range(tdim):
#     for i in prange(tdim, nogil=True, num_threads=6):

        for j in range(cdim):
              for th in range(thdim):      
                prod = <double> cabs(reduced_prod[i,j,th])
                prod = prod * prod
                angdist[i,th] +=  prod * boltz[j]
            
        
    return np.array(angdist)

def propagate_psi_plusminus(int Jmax,
                  double[:] time,
                  double[:] field,
                  linear=True,
                  coeffs=[]):
    """Do the time integration over a laser pulse amplitude vector called field for all quantum numbers."""
    cdef:
        int ki, mi, k, m, i, j, l
        int j21 = 2 * Jmax + 1
        int J1 = Jmax + 1
        int i_min, dim
        int[:] mk = np.linspace(-Jmax, Jmax, j21, dtype=np.int32)
        
        double ii
        
        int[:] ipiv = np.arange(J1, dtype=np.int32)
        double complex[::1,:] H_rot
        double complex[::1,:] H_ind
        
        double complex[::1,:] c_cur
        double complex[::1,:] c_new
 
        double complex[::1,:] Hstat
        double complex[::1,:] H_cur
        double complex[::1,:] H_new

        double[:,:,::1] wig
        double[:] Jvec
        double[:] EJK
        
        double[:,:] Javg
        
    res = list()
    eng = list()
    jav = list()
    
    for ki in range(j21):
        k = mk[ki]
        
        if linear:
            k = 0
            
        for mi in range(j21):
            m = mk[mi]
            
            i_min = max(abs(k), abs(m))
            dim = J1 - i_min
        
            ipiv = np.arange(dim, dtype=np.int32)
            H_rot = np.zeros([7, dim], dtype=np.complex_, order='F')
            H_ind = np.zeros([7, dim], dtype=np.complex_, order='F')

            c_cur = np.zeros([dim, dim], dtype=np.complex_, order='F')
            c_new = np.zeros([dim, dim], dtype=np.complex_, order='F')

            Hstat = np.zeros([7, dim], dtype=np.complex_, order='F')
            H_cur = np.zeros([7, dim], dtype=np.complex_, order='F')
            H_new = np.zeros([7, dim], dtype=np.complex_, order='F')

            wig  = np.zeros([2, dim, dim], dtype=np.float_, order='C')
            Jvec = np.zeros(dim)
            EJK  = np.zeros(dim)
            
            Javg = np.zeros([time.shape[0], dim])
        
            for i in range(dim):
                ii = <double> i
                Jvec[i] = <double> i_min + ii
                c_cur[i,i] = 1
                
            construct_Hrot(dim, Jvec, k, EJK)
            construct_Hind(dim, Jvec, k, m, wig, H_ind)

            for i in range(dim):
                Hstat[2,i] = EJK[i]

            if any(coeffs):
                c_cur = coeffs[k,m].unstack().values.T
                
            propagateCN(dim, H_cur, H_new, Hstat, H_ind,
                        c_cur, c_new, time, field,
                        ipiv, Jvec, Javg)
            
            iv = np.asarray(Jvec)
            inx = pd.MultiIndex.from_product([[k], [m], iv, iv], names=['K', 'M', 'Jinit', 'J'])
            
            sr = pd.Series(np.asarray(c_cur.T).ravel(), index=inx)
            res.append(sr)
            
            eng_distort(dim, Jvec, k, EJK)
            
            inx = pd.MultiIndex.from_product([[k], [m], iv], names=['K', 'M', 'J'])
            eng.append(pd.Series(np.asarray(EJK), inx))
            
            inx = pd.MultiIndex.from_product([[k], [m], np.asarray(time), iv], names=['K', 'M', 'time', 'J2'])
            rav = pd.Series(np.asarray(Javg).ravel(), index=inx)
            jav.append(rav)

            
        if linear:
            break
        
    return pd.concat(eng), pd.concat(res), pd.concat(jav)

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
            wigout[j,k] = wigKM[0,j,k] * wigKM[1,j,k] * j1 * phas)
            
    return np.asarray(wigout)
