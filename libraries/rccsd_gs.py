from qs_ref import *
from quantum_systems.custom_system import construct_pyscf_system_rhf
import basis_set_exchange as bse
from rccsd import RCCSD
import coupled_cluster
import rhs_t
from coupled_cluster.rccsd import energies as rhs_e
from opt_einsum import contract
import numpy as np
from scipy.optimize import minimize, root,newton
import time
np.set_printoptions(linewidth=300,precision=8,suppress=True)
class sucess():
    """Mini helper class that resembles scipy's OptimizeResult."""
    def __init__(self,x,success,nfev):
        self.x=x
        self.success=success
        self.nfev=nfev

def basischange_clusterOperator(U,t1,t2):
    "Use U as change of basis operator to go from basis t1 to a new t1-tilde, same for t2"
    n_virt=t1.shape[0]
    n_occ=t1.shape[1]
    U_occ=U[:n_occ,:n_occ] #Occupied orbital rotation
    U_virt=U[n_occ:,n_occ:] #Virtual orbital rotation
    new_t1 = contract('ij,ai,ab->bj', U_occ, t1, U_virt)
    new_t2 = contract("ik,jl,abij,ac,bd->cdkl",U_occ,U_occ,t2,U_virt,U_virt)
    return new_t1,new_t2

def orthonormalize_ts(t1s: list,t2s: list):
    """
    Given lists of t1 and t2 amplitudes, orthogonalizes the vectors t_1\osumt_2 using SVD
    Input:
    t1s,t2s (lists): Lists of same lengths for t1 and t2 amplitudes

    returns:
    t1s,t2s (lists): Orthogonalized lists with the same span
    coefs (matrix): The linear combinations to express original elements in terms of new elements
    """
    t_tot=[]
    a,i=t1s[0].shape
    for j in range(len(t1s)):
        t_tot.append(np.concatenate((t1s[j],t2s[j]),axis=None)) #   T1 \osum T2
    t_tot=np.array(t_tot)
    t_tot_old=t_tot.copy()
    t_tot=t_tot.T
    U,s,Vt=svd(t_tot,full_matrices=False)
    t_tot=(U@Vt).T #This is the unitary matrix closest to t_tot
    t1_new=[]
    t2_new=[]
    coefs=np.zeros((len(t1s),len(t1s))) #Coefficients to transform between the old and the new representations
    for j in range(len(t1s)):
        for k in range(len(t1s)):
            coefs[j,k]=t_tot_old[j,:]@t_tot[k,:]

    for j in range(len(t1s)):
        new_t1=np.reshape(t_tot[j,:a*i],(a,i))
        new_t2=np.reshape(t_tot[j,a*i:],(a,a,i,i))
        t1_new.append(new_t1)
        t2_new.append(new_t2)
    return t1_new,t2_new,coefs
def orthonormalize_ts_lowdin(t1s: list,t2s: list):
    """
    Given lists of t1 and t2 amplitudes, orthogonalizes the vectors t_1\osumt_2 using SVD
    Input:
    t1s,t2s (lists): Lists of same lengths for t1 and t2 amplitudes

    returns:
    t1s,t2s (lists): Orthogonalized lists with the same span
    coefs (matrix): The linear combinations to express original elements in terms of new elements
    """
    t_tot=[]
    a,i=t1s[0].shape
    for j in range(len(t1s)):
        t_tot.append(np.concatenate((t1s[j],t2s[j]),axis=None)) #   T1 \osum T2
    t_tot=np.array(t_tot)
    t_tot_old=t_tot.copy()
    t_tot=t_tot
    overlap=t_tot@t_tot.T
    print(overlap)
    overlap_12=scipy.linalg.sqrtm(overlap)
    t_tot=np.linalg.inv(overlap_12)@t_tot
    t1_new=[]
    t2_new=[]
    coefs=np.zeros((len(t1s),len(t1s))) #Coefficients to transform between the old and the new representations
    for j in range(len(t1s)):
        for k in range(len(t1s)):
            coefs[j,k]=t_tot_old[j,:]@t_tot[k,:]

    for j in range(len(t1s)):
        new_t1=np.reshape(t_tot[j,:a*i],(a,i))
        new_t2=np.reshape(t_tot[j,a*i:],(a,a,i,i))
        t1_new.append(new_t1)
        t2_new.append(new_t2)
    return t1_new,t2_new,coefs

def setUpsamples(sample_x,molecule_func,basis,rhf_mo_ref,rhf_S_ref,mix_states=False,type="procrustes",weights=None):
    """
    Sets up lambda and t-amplitudes for a set of geometries.

    Input:
    sample_x (list): The geometries at which the molecule is constructed.
    molecule_func (function->String): A function which returns a string
        corresponding to a molecular geometry as function of parameters from sample_x.
    basis (string): The basis set.
    rhf_mo_ref (matrix or list of matrices):
        if matrix: Use Procrustes algorithm to use "best" HF reference, then calculate amplitudes based on that
        if list: Use coefficient matrices in the list directly to calculate amplitudes from.
    mix_states (bool): if general procrustes orbitals should be used.
    weights: Outdated parameter

    Returns:
    t1s,t2s,l1s,l2s: CCSD ampltidue and lambda equations
    sample_energies: CCSD Energies for the desired samples
    """
    t1s=[]
    t2s=[]
    l1s=[]
    l2s=[]
    sample_energies=[]
    for k,x in enumerate(sample_x):
        ref_state=rhf_mo_ref
        system = construct_pyscf_system_rhf_ref(
            molecule=molecule_func(*x),
            basis=basis,
            add_spin=False,
            anti_symmetrize=False,
            reference_state=ref_state,
            reference_overlap=rhf_S_ref,
            mix_states=mix_states,
            weights=None
        )
        rccsd = RCCSD(system, verbose=False)
        ground_state_tolerance = 1e-8
        rccsd.compute_ground_state(
            t_kwargs=dict(tol=ground_state_tolerance),
            l_kwargs=dict(tol=ground_state_tolerance),

        )
        t, l = rccsd.get_amplitudes()
        t1s.append(t[0])
        t2s.append(t[1])
        l1s.append(l[0])
        l2s.append(l[1])
        sample_energies.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
    return t1s,t2s,l1s,l2s,sample_energies

class EVCSolver():
    """
    Class to solve EVC equations. Contains AMP-CCEVC and some helper functions functions. Uses restricted determinants and restricted CC theory.

    Methods:
    __init__: Initialize
    solve_CCSD: Return CCSD energies at self.all_x
    solve_AMP_CCSD: solves AMP-CCEVC equations and returns AMP-CCEVC energies for given sample amplitudes
    """
    def __init__(self,all_x,molecule_func,basis,reference_determinant,t1s,t2s,l1s,l2s,reference_overlap=None,givenC=False,sample_x=None,mix_states=False,natorb_truncation=None):
        self.all_x=all_x
        self.molecule_func=molecule_func
        self.sample_x=sample_x
        self.basis=basis
        self.reference_determinants=reference_determinant #The reference determiant or a list of different reference determinants.
        self.reference_overlap=reference_overlap
        for i in range(len(t1s)):
            t1s[i]=np.array(t1s[i])
            t2s[i]=np.array(t2s[i])
            l1s[i]=np.array(l1s[i])
            l2s[i]=np.array(l2s[i])
        self.t1s=t1s
        self.t2s=t2s
        self.l1s=l1s
        self.l2s=l2s
        self.mix_states=mix_states
        self.natorb_truncation=natorb_truncation
        self.num_params=np.prod(t1s[0].shape)+np.prod(t2s[0].shape)
        self.coefs=None
        self.givenC=givenC
    def solve_CCSD_noProcrustes(self,xtol=1e-8):
        """
        Solves the CCSD equations.

        Returns:
        E_CCSD (list): CCSD Energies at all_x.
        """
        self.num_iter=[]
        E_CCSD=[]
        E_HF=[]
        for k,x_alpha in enumerate(self.all_x):
            ref_state=self.reference_determinants
            system = construct_pyscf_system_rhf( #Construct a canonical-orbital HF state.
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=False,
                anti_symmetrize=False,
            )
            try:
                rccsd = RCCSD(system, verbose=False)
                ground_state_tolerance = xtol
                rccsd.compute_ground_state(t_kwargs=dict(tol=ground_state_tolerance))
                HF_energy=system.compute_reference_energy().real
                E_HF.append(HF_energy)
                E_CCSD.append(HF_energy+rccsd.compute_energy().real)
                print("Number iterations: %d"%rccsd.num_iterations)
                self.num_iter.append(rccsd.num_iterations)
            except:
                E_CCSD.append(np.nan)
                E_HF.append(np.nan)
        return E_HF,E_CCSD
    def solve_CCSD_previousgeometry(self,procrustes_orbitals,xtol=1e-8):
        """Use the previous geometry as a start guess for CCSD calculations. for the first geometry, no previous geometry is used. (MP2 guess)"""
        E_CCSD=[]
        self.num_iter=[]
        for k,x_alpha in enumerate(self.all_x):
            ref_state=self.reference_determinants
            system,C_canonical = construct_pyscf_system_rhf_ref( #With these parameters, canonical orbitals are used!
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=False,
                anti_symmetrize=False,
                reference_state=None,
                givenC=None,
                mix_states=self.mix_states,
                truncation=self.natorb_truncation,
                return_C=True
            )

            if k==0:
                rccsd = RCCSD(system, include_singles=True)
                molecule=pyscf.M(atom=self.molecule_func(*x_alpha),basis=self.basis)
                S=molecule.intor("int1e_ovlp")
            else:
                molecule=pyscf.M(atom=self.molecule_func(*x_alpha),basis=self.basis)
                S=molecule.intor("int1e_ovlp")
                C_new,Uinv=localize_procrustes_ovlp(None,C_canonical,None,C_prev,S,S_prev,nelec=sum(molecule.nelec), return_R=True) #Unitary to go from canonical orbitals to Procrustes
                U=np.conj(Uinv.T) #The unitary operation to go from Procrustes orbitals to canonical orbitals!
                t1_new,t2_new=basischange_clusterOperator(U,start_guess_amplitudes[0],start_guess_amplitudes[1])
                start_guess_amplitudes=[t1_new,t2_new]

                rccsd = RCCSD(system, include_singles=True,start_guess=start_guess_amplitudes)

            ground_state_tolerance = xtol
            rccsd.compute_ground_state(t_kwargs=dict(tol=ground_state_tolerance))
            t, l = rccsd.get_amplitudes()
            start_guess_amplitudes=[t[0],t[1]]
            E_CCSD.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
            print("Number iterations: %d"%rccsd.num_iterations)
            self.num_iter.append(rccsd.num_iterations)
            C_prev=C_canonical
            S_prev=S
        return E_CCSD
    def solve_CCSD_startguess(self,start_guess_t1_list,start_guess_t2_list,procrustes_orbitals, basis_change_from_Procrustes=True,xtol=1e-8):
        """
        Solves the CCSD equations. A start guess for the t amplitudes needs to be provided.
        Optional: If basis change is true,
        a transform of the t1 and the t2 to the canonical orbitals will be performed, assuming that the t1 and t2 amplitudes refer
        to Procrustes orbitals.

        Returns:
        E_CCSD (list): CCSD Energies at all_x.
        """
        E_CCSD=[]
        self.num_iter=[]
        for k,x_alpha in enumerate(self.all_x):
            ref_state=self.reference_determinants
            reference_overlap=self.reference_overlap
            system,C_canonical = construct_pyscf_system_rhf_ref( #With these parameters, canonical orbitals are used!
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=False,
                anti_symmetrize=False,
                reference_state=None,
                givenC=None,
                mix_states=self.mix_states,
                truncation=self.natorb_truncation,
                return_C=True
            )
            print(x_alpha)
            if basis_change_from_Procrustes:
                U=np.linalg.inv(procrustes_orbitals[k])@C_canonical
                t1_new,t2_new=basischange_clusterOperator(U,start_guess_t1_list[k],start_guess_t2_list[k])
                start_guess_amplitudes=[t1_new,t2_new]
            else:
                start_guess_amplitudes=[start_guess_t1_list[k],start_guess_t2_list[k]]
            rccsd = RCCSD(system, include_singles=True,start_guess=start_guess_amplitudes)
            ground_state_tolerance = xtol
            rccsd.compute_ground_state(t_kwargs=dict(tol=ground_state_tolerance))
            E_CCSD.append(system.compute_reference_energy().real+rccsd.compute_energy().real)
            print("Number iterations: %d"%rccsd.num_iterations)
            self.num_iter.append(rccsd.num_iterations)
        return E_CCSD
    def calculate_CCSD_energies_from_guess(self,start_guess_t1_list,start_guess_t2_list,procrustes_orbitals,basis_change_from_Procrustes=True,xtol=1e-8):
        """
        Returns CCSD energies given the t1 and t2 amplitudes in Procrustes basis.
        If basis_change_from_Procrustes is true and procrustes orbitals are given, energies
        are calculated with respect to Procrustes orbitals.
        """
        E_CCSD=[]
        for k,x_alpha in enumerate(self.all_x):
            ref_state=self.reference_determinants
            reference_overlap=self.reference_overlap
            system,C_canonical = construct_pyscf_system_rhf_ref( #With these parameters, canonical orbitals are used!
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=False,
                anti_symmetrize=False,
                reference_state=None,
                givenC=None,
                mix_states=self.mix_states,
                truncation=self.natorb_truncation,
                return_C=True
            )
            if basis_change_from_Procrustes:
                U=np.linalg.inv(procrustes_orbitals[k])@C_canonical
                t1_new,t2_new=basischange_clusterOperator(U,start_guess_t1_list[k],start_guess_t2_list[k])
                start_guess_amplitudes=[t1_new,t2_new]
            else:
                start_guess_amplitudes=[start_guess_t1_list[k],start_guess_t2_list[k]]
            rccsd = RCCSD(system, include_singles=True,start_guess=start_guess_amplitudes)
            reference_energy=system.compute_reference_energy().real
            CCSD_correction=rccsd.compute_energy().real
            E_CCSD.append(reference_energy+CCSD_correction)
        return E_CCSD
    def solve_AMP_CCSD(self,occs=1,virts=0.5,xtol=1e-5,maxfev=60, start_guess_list=None):
        """
        Solves the AMP_CCSD equations.

        Input:
        occs (float): The percentage (if below 1) or number (if above 1) of occupied orbitals to include in amplitude calculations
        virts (float): The percentage (if below 1) or number (if above 1) of virtual orbitals to include in amplitude calculations
        xtol (float): Convergence criterion for maximum amplitude error
        maxfev (int): Maximal number of Newton's method iterations
        start_guess (list): List of lists with starting parameters [[c_1,\dots,c_L]_1, [c_1,\dots,c_L]_2, \dots]
        Returns:
        energy (list): AMP-CCEVC Energies at all_x.
        """
        energy=[]
        t1_copy=self.t1s #Reset after a run of AMP_CCEVC such that WF-CCEVC can be run afterwards
        t2_copy=self.t2s
        t1s_orth,t2s_orth,coefs=orthonormalize_ts(self.t1s,self.t2s) # Use orthonormal t1 and t2 amplitudes
        if self.coefs is None:
            self.coefs=coefs
        self.t1s=t1s_orth #Use the orthogonalized forms of t1s and t2s
        self.t2s=t2s_orth
        t2=np.zeros(self.t2s[0].shape)
        for i in range(len(self.t1s)):
            t2+=self.t2s[i]
        t1_v_ordering=contract=np.einsum("abij->a",t2**2)
        t1_o_ordering=contract=np.einsum("abij->i",t2**2)
        important_o=np.argsort(t1_o_ordering)[::-1]
        important_v=np.argsort(t1_v_ordering)[::-1]
        chosen_t=np.zeros(self.t2s[0].shape)
        if occs is None:
            occs_local=self.t2s[0].shape[2]
        elif occs<1.1:
            occs_local=int(self.t2s[0].shape[0]*occs)
        else:
            occs_local=occs
        if virts is None:
            virts_local=self.t2s[0].shape[0]//2
        elif virts<1.1:
            virts_local=int(self.t2s[0].shape[0]*virts)
        else:
            virts_local=virts
        self.used_o=np.sort(important_o[:occs_local])
        self.used_v=np.sort(important_v[:virts_local])
        chosen_t[np.ix_(self.used_v,self.used_v,self.used_o,self.used_o)]=1
        self.picks=chosen_t.reshape(self.t2s[0].shape)
        self.picks=(self.picks*(-1)+1).astype(bool)

        self.nos=important_o[:occs_local]
        self.nvs=important_v[:virts_local]
        self.num_iterations=[]
        self.times=[]
        self.projection_errors=[]
        projection_errors=[]
        self.t1s_final=[] #List of the t1 solutions for each x
        self.t2s_final=[] #list of the t2 solutions for each x
        for k,x_alpha in enumerate(self.all_x):
            ref_state=self.reference_determinants
            system = construct_pyscf_system_rhf_ref(
                molecule=self.molecule_func(*x_alpha),
                basis=self.basis,
                add_spin=False,
                anti_symmetrize=False,
                reference_state=ref_state,
                reference_overlap=self.reference_overlap,
                mix_states=self.mix_states,
                truncation=self.natorb_truncation
            )
            f = system.construct_fock_matrix(system.h, system.u)
            ESCF=system.compute_reference_energy().real
            self._system_jacobian(system)
            closest_sample_x=np.argmin(np.linalg.norm(np.array(self.sample_x)-x_alpha,axis=1))
            try:
                start_guess=start_guess_list[k]
            except TypeError: #Not a list
                start_guess=self.coefs[:,closest_sample_x]

            except IndexError: #List to short
                start_guess=self.coefs[:,closest_sample_x]
            self.times_temp=[]

            sol=self._own_root_diis(start_guess,args=[system],options={"xtol":xtol,"maxfev":maxfev})
            final=sol.x
            self.num_iterations.append(sol.nfev)
            self.times.append(np.sum(np.array(self.times_temp)))
            t1=np.zeros(self.t1s[0].shape)
            t2=np.zeros(self.t2s[0].shape)
            for i in range(len(self.t1s)):
                t1+=final[i]*self.t1s[i] #Starting guess
                t2+=final[i]*self.t2s[i] #Starting guess
            self.t1s_final.append(t1)
            self.t2s_final.append(t2)
            t1_error = rhs_t.compute_t_1_amplitudes(f, system.u, t1, t2, system.o, system.v, np)
            t2_error = rhs_t.compute_t_2_amplitudes(f, system.u, t1, t2, system.o, system.v, np)
            max_proj_error=np.max((np.max(abs(t1_error)),np.max(abs(t2_error)) ))
            self.projection_errors.append(max_proj_error)
            newEn=rhs_e.compute_rccsd_ground_state_energy(f, system.u, t1, t2, system.o, system.v, np)+ESCF
            if sol.success==False:
                energy.append(np.nan)
            else:
                energy.append(newEn)
        self.t1s=t1_copy
        self.t2s=t2_copy
        return energy

    def _system_jacobian(self,system):
        """
        Construct quasi-newton jacobian for a given geometry.
        """
        f = system.construct_fock_matrix(system.h, system.u)
        no=system.n
        nv=system.l-system.n
        t1_Jac=np.zeros((nv,no))
        t2_Jac=np.zeros((nv,nv,no,no))
        for a in range(nv):
            for i in range(no):
                t1_Jac[a,i]=f[a+no,a+no]-f[i,i]
        for a in range(nv):
            for b in range(nv):
                for i in range(no):
                    for j in range(no):
                        t2_Jac[a,b,i,j]=f[a+no,a+no]-f[i,i]+f[b+no,b+no]-f[j,j] #This is really crappy, as the diagonal approximation does not hold. Though it works!!
        self.t1_Jac=t1_Jac
        self.t2_Jac=t2_Jac
    def _error_function(self,params,system):
        """
        Projection error given a set of pareters [c_1,...,c_L]

        Input:
        params (list): The parameters [c_1,...,c_L] for which to caluclate the error
        system: The molecule with information

        Returns:
            List: Projection errors
        """
        t1=np.zeros(self.t1s[0].shape)
        t2=np.zeros(self.t2s[0].shape)

        for i in range(len(self.t1s)):
            t1+=params[i]*self.t1s[i] #Starting guess
            t2+=params[i]*self.t2s[i] #Starting guess

        f = system.construct_fock_matrix(system.h, system.u)
        start=time.time()
        t1_error = rhs_t.compute_t_1_amplitudes_REDUCED_new(f, system.u, t1, t2, system.o, system.v, np,self.picks,self.nos,self.nvs)
        t2_error = rhs_t.compute_t_2_amplitudes_REDUCED_new(f, system.u, t1, t2, system.o, system.v, np,self.picks,self.nos,self.nvs)
        end=time.time()
        self.times_temp.append(end-start)
        ts=[np.concatenate((self.t1s[i],self.t2s[i]),axis=None) for i in range(len(self.t1s))]
        t_error=np.concatenate((t1_error,t2_error),axis=None)
        projection_errors=np.zeros(len(self.t1s))
        t1_error_flattened=t1_error.flatten()
        t2_error_flattened=t2_error.flatten()
        for i in range(len(projection_errors)):
            projection_errors[i]+=t1_error_flattened@self.t1s[i].flatten()
            projection_errors[i]+=t2_error_flattened@self.t2s[i].flatten()
        return projection_errors
    def _jacobian_function(self,params,system):
        """
        quasi-newton jacobian for a given geometry as expressed in terms of params [c_1,\dots,c_L]
        """
        t1=np.zeros(self.t1s[0].shape)
        t2=np.zeros(self.t2s[0].shape)
        Ts=[]
        for i in range(len(self.t1s)):
            t1+=params[i]*self.t1s[i] #Starting guess
            t2+=params[i]*self.t2s[i] #Starting guess
            #Ts.append(t2s[i][picks].flatten())
            Ts.append(self.t2s[i][np.ix_(self.used_v,self.used_v,self.used_o,self.used_o)].flatten())

        jacobian_matrix=np.zeros((len(params),len(params)))
        for i in range(len(params)):
            for j in range(i,len(params)):
                jacobian_matrix[j,i]+=contract("k,k,k->",self.t1s[i][np.ix_(self.used_v,self.used_o)].flatten(),self.t1s[j][np.ix_(self.used_v,self.used_o)].flatten(),self.t1_Jac[np.ix_(self.used_v,self.used_o)].flatten())
                jacobian_matrix[j,i]+=contract("k,k,k->",Ts[i],Ts[j],self.t2_Jac[np.ix_(self.used_v,self.used_v,self.used_o,self.used_o)].flatten())
                jacobian_matrix[i,j]=jacobian_matrix[j,i]
        return jacobian_matrix
    def _own_root_diis(self,start_guess,args,options={"xtol":1e-3,"maxfev":25},diis_start=2,diis_dim=5):
        """Root finding algorithm based on quasi-newton and DIIS.

        Input:
        start_guess (list): Start guess for coefficients
        args [list]: A list containing the system
        options (dict): Tolerance (xtol) and number of maximum evaluations
        diis_start: How many iterations to do before starting diis_start
        diis_dim: Dimensionality of DIIS subspace

        Returns:
        Sucess object with final solution, number of iterations, and wether convergence was reached
        """
        guess=start_guess
        xtol=options["xtol"]
        maxfev=options["maxfev"]
        iter=0
        errors=[]
        amplitudes=[]
        updates=[]
        error=10*xtol #Placeholder to enter while loop

        #Calculate diis_start iterations without DIIS
        while np.max(np.abs(error))>xtol and iter<diis_start:
            jacobian=self._jacobian_function(guess,*args)
            error=self._error_function(guess,*args)
            update=-np.linalg.inv(jacobian)@error
            guess=guess+update
            errors.append(error)
            updates.append(update)
            amplitudes.append(guess)
            iter+=1

        #DIIS algorithm
        while np.max(np.abs(error))>xtol and iter<maxfev:

            #Calculate DIIS B-matrix from Jacobian guess update
            B_matrix=np.zeros((len(updates)+1,len(updates)+1))
            for i,ei in enumerate(updates):
                for j, ej in enumerate(updates):
                    B_matrix[i,j]=np.dot(ei,ej)
            for i in range(len(updates)+1):
                B_matrix[i,-1]=B_matrix[-1,i]=-1
            B_matrix[-1,-1]=0
            sol=np.zeros(len(updates)+1)
            sol[-1]=-1
            input=np.zeros(len(updates[0]))
            try:
                weights=np.linalg.solve(B_matrix,sol)
                for i,w in enumerate(weights[:-1]):
                    input+=w*amplitudes[i] #Calculate new approximate ampltiude guess vector
                    #errsum+=w*updates[i]
            except np.linalg.LinAlgError: #If DIIS matrix is singular, use most recent quasi-newton step
                input=guess
            #errsum=np.zeros(len(updates[0]))


            jacobian=self._jacobian_function(input,*args)
            error=self._error_function(input,*args)
            update=-np.linalg.inv(jacobian)@error #Calculate update vector
            guess=input+update
            errors.append(error)
            updates.append(update)
            amplitudes.append(guess)
            if len(errors)>=diis_dim: #Reduce DIIS space to dimensionality threshold
                errors.pop(0)
                amplitudes.pop(0)
                updates.pop(0)
            iter+=1
        success=iter<maxfev
        #print("Num iter: %d"%iter)
        return sucess(guess,success,iter)
