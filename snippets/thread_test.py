import numpy as np

from pyscf.pbc import gto,scf


if __name__ == "__main__":
    atom = "He 0 0 0; He 0.75 0 0"
    a = np.eye(3) * 3
    basis = "gth-dzvp"
    pseudo = "gth-pade"

    cell = gto.Cell(atom=atom, a=a, basis=basis, pseudo=pseudo)
    cell.mesh = [21,21,21]
    cell.build()
    cell.verbose = 6

    kmesh = [2,2,1]
    kpts = cell.make_kpts(kmesh)

    mf = scf.KRHF(cell, kpts)
    mf.kernel()