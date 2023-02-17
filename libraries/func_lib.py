import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import linalg
from pyscf import gto, scf, mcscf, fci, cc, mp,ao2mo
from scipy.linalg import norm, eig, qz, block_diag, eigh, orth, fractional_matrix_power, expm, svd
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment, minimize, root,newton
from opt_einsum import contract
from scipy.io import loadmat, savemat

import warnings
import sys
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'lines.linewidth': 3})
np.set_printoptions(linewidth=300,precision=6,suppress=True)

def get_U_matrix(x,molecule,basis,reference_determinant,reference_overlap):
    U_matrices=[]
    C_matrices=[]
    for xval in x:
        mol = gto.Mole()
        if isinstance(xval,tuple) or isinstance(xval,np.ndarray) or isinstance(xval,list):
        	mol.atom=molecule(*xval)
        else:
        	mol.atom=molecule(xval)
        mol.basis = basis
        mol.unit="bohr"
        mol.build()
        hf=scf.RHF(mol)
        hf.kernel()
        C=hf.mo_coeff
        S=mol.intor("int1e_ovlp")
        C_new=localize_procrustes_ovlp(mol,hf.mo_coeff,hf.mo_occ,reference_determinant,S,reference_overlap)

        U_rot=np.real(scipy.linalg.fractional_matrix_power(S,0.5))@C_new
        U_matrices.append(U_rot)
        C_matrices.append(C_new)
    return U_matrices, C_matrices


def make_mol(molecule,x,basis="6-31G",charge=0):
	"""Helper function to create Mole object at given geometry in given basis"""
	mol=gto.Mole()
	if isinstance(x,list):
		mol.atom=molecule(*x)
	else:
		mol.atom=molecule(x)
	mol.basis = basis
	mol.unit= "Bohr"
	mol.charge=charge
	mol.build()
	return mol

def get_reference_determinant(molecule_func,refx,basis,charge=0,return_overlap=False):
    mol = gto.Mole()
    mol.unit = "bohr"
    mol.charge = charge
    mol.cart = False
    mol.build(atom=molecule_func(*refx), basis=basis)
    hf = scf.RHF(mol)
    hf.kernel()
    S=mol.intor("int1e_ovlp")
    if return_overlap:
        return np.asarray(hf.mo_coeff), S
    return np.asarray(hf.mo_coeff)
def CCSD_energy_curve(molecule_func,xvals,basis):
	"""Returns CCSD energy along a geometry"""
	E=[]
	for x in xvals:
		mol = gto.Mole()
		mol.unit = "bohr"
		mol.build(atom=molecule_func(*x), basis=basis)
		hf = scf.RHF(mol)
		hf.kernel()
		mycc = cc.CCSD(hf)
		mycc.kernel()
		if mycc.converged:
			E.append(mycc.e_tot)
		else:
			E.append(np.nan)
	return E

def orthogonal_procrustes_ovlp(mo_new,reference_mo,overlap_new,overlap_reference):
    overlap_new_sqrtm=np.real(scipy.linalg.sqrtm(overlap_new))
    overlap_reference_sqrtm=np.real(scipy.linalg.sqrtm(overlap_reference))
    A=mo_new
    B=reference_mo.copy()

    M=A.T@overlap_new_sqrtm@overlap_reference_sqrtm@B # This does not....
    #M=A.T@B #This works just the way I expected and intent to!!
    U,s,Vt=scipy.linalg.svd(M)
    return U@Vt, 0

def localize_procrustes_ovlp(mol,mo_coeff,mo_occ,ref_mo_coeff,overlap_new,overlap_reference,mix_states=False,active_orbitals=None,nelec=None, return_R=False,weights=None):
    """Performs the orthgogonal procrustes on the occupied and the unoccupied molecular orbitals.
    ref_mo_coeff is the mo_coefs of the reference state.
    If "mix_states" is True, then mixing of occupied and unoccupied MO's is allowed.
    """
    if active_orbitals is None:
        active_orbitals=np.arange(len(mo_coeff))
    if nelec is None:
        nelec=int(np.sum(mo_occ))
    active_orbitals_occ=active_orbitals[:nelec//2]
    active_orbitals_unocc=active_orbitals[nelec//2:]
    mo_coeff_new=mo_coeff.copy()
    if mix_states==False:
        mo=mo_coeff[:,active_orbitals_occ]
        premo=ref_mo_coeff[:,active_orbitals_occ]
        R1,scale=orthogonal_procrustes_ovlp(mo,premo,overlap_new,overlap_reference)
        mo=mo@R1
        mo_unocc=mo_coeff[:,active_orbitals_unocc]
        premo=ref_mo_coeff[:,active_orbitals_unocc]
        R2,scale=orthogonal_procrustes_ovlp(mo_unocc,premo,overlap_new,overlap_reference)
        mo_unocc=mo_unocc@R2
        mo_coeff_new[:,active_orbitals_occ]=np.array(mo)
        mo_coeff_new[:,active_orbitals_unocc]=np.array(mo_unocc)
        R=block_diag(R1,R2)
    elif mix_states==True:
        mo=mo_coeff[:,active_orbitals]
        premo=ref_mo_coeff[:,active_orbitals]
        R,scale=orthogonal_procrustes(mo,premo)
        mo=mo@R

        mo_coeff_new[:,active_orbitals]=np.array(mo)

    if return_R:
        return mo_coeff_new,R
    else:
        return mo_coeff_new

def canonical_orthonormalization(T,S,threshold=1e-8):
    """Solves the generalized eigenvector problem.
    Input:
    T: The symmetric matrix
    S: The overlap matrix
    threshold: eigenvalue cutoff
    Returns: The lowest eigenvalue & eigenvector
    """
    ###The purpose of this procedure here is to remove all very small eigenvalues of the overlap matrix for stability
    s, U=np.linalg.eigh(S) #Diagonalize S (overlap matrix, Hermitian by definition)
    U=np.fliplr(U)
    s=s[::-1] #Order from largest to lowest; S is an overlap matrix, hence we (ideally) will only have positive values
    s=s[s>threshold] #Keep only largest eigenvalues
    spowerminushalf=s**(-0.5) #Take s
    snew=np.zeros((len(U),len(spowerminushalf)))
    sold=np.diag(spowerminushalf)
    snew[:len(s),:]=sold
    s=snew


    ###Canonical orthogonalization
    X=U@s
    Tstrek=X.T@T@X
    epsilon, Cstrek = np.linalg.eigh(Tstrek)
    idx = epsilon.argsort()[::1] #Order by size (non-absolute)
    epsilon = epsilon[idx]
    Cstrek = Cstrek[:,idx]
    C=X@Cstrek
    lowest_eigenvalue=epsilon[0]
    lowest_eigenvector=C[:,0]
    return lowest_eigenvalue,lowest_eigenvector
