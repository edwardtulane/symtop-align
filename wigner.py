"""Generate Rotational Eigenfunctions up to Jmax on a theta grid with thetaN points.
Store them in a hdf5 file in the calc directory. The precalculated functions can be retrieved
from the storage via multi-indexing (The indexing is (K, M, J)!).

ds is the Wigner small d matrix element, dp its conjugate function.
NOTE: I removed the conjugate, as it is redundant. Just keep track of the phase."""

import os
import numpy as np
import pandas as pd

from ipyparallel import Client, interactive, require

if __name__ == '__main__':

    Jmax = 50

    thetaN = 257

    linear = False

    if linear:
        ext = 'linear'
        balance = np.array([.28, .40, .49, .58, .70])
    else:
        ext = 'symtop'
        balance = np.array([.284, .431, .568, .716])

    cl = Client()
    view = cl[:]

    view['thN'] = thetaN
    view['Jmax'] = Jmax

    @require('os')
    @interactive
    def wig_funs():
        os.chdir('/home/brausse/program/align-symtop/')
        from symtop import generate_wigner_KM

        for (ki, mi) in zip(k, m):
                ds.append(generate_wigner_KM(ki, mi, Jmax, thN))


    j21 = np.arange(-Jmax, Jmax+1)

    k, m = np.meshgrid(j21, j21)
    k = k.ravel()
    m = m.ravel()

    if linear:
        m = j21
        k = np.zeros_like(j21)

    part = m.shape[0] * balance

    for i, dv in enumerate(cl):
        dv['k'] = np.split(k, part.astype(int))[i]
        dv['m'] = np.split(m, part.astype(int))[i]

    view['ds'] = list()

    ar = view.apply(wig_funs)
    ar.wait_interactive()

    path = os.path.expanduser('~/calc/align/wignerD-J%i-th%i-%s.h5' % (Jmax, thetaN, ext))

    with pd.HDFStore(path) as st:
        st['ds'] = pd.concat(view.gather('ds'))
