"""
Get approximate AMP-CCEVC amplitudes for HF molecule in small basis
"""
import sys
sys.path.append("../../libraries")
from rccsd_gs import *
from machinelearning import *
from func_lib import *
from numba import jit
import pickle

basis="cc-pVTZ"
charge = 0
molecule=lambda x:  "H 0 0 0; F 0 0 %f"%x;molecule_name="HF"
refx=[1.75]
print(molecule(*refx))

"""This procedure guarantees that exactly the same coefficient matrices at all geometries are produced, for reproducibility"""
try:
    reference_determinant=np.loadtxt("inputs/HF_ref_det_%s.txt"%basis)
    reference_overlap=np.loadtxt("inputs/HF_ref_S_%s.txt"%basis)
except FileNotFoundError:
    reference_determinant,reference_overlap=get_reference_determinant(molecule,refx,basis,charge,True)
    np.savetxt("inputs/HF_ref_det_%s.txt"%basis,reference_determinant)
    np.savetxt("inputs/HF_ref_S_%s.txt"%basis,reference_overlap)
sample_geom1=np.linspace(1.5,4,7)
geom_alphas1=np.linspace(1.4,4.1,81)
#geom_alphas1=sample_geom1
geom_alphas=[[x] for x in geom_alphas1]

sample_geom=[[x] for x in sample_geom1]
sample_geom1=np.array(sample_geom).flatten()

#This step is relatively expensive and could easily be moved elsewhere, but this program is a proof of concept, not the fastest possible implementation.
sample_U,sample_Procrustes=get_U_matrix(sample_geom1,molecule,basis,reference_determinant,reference_overlap)
target_U,target_Procrustes=get_U_matrix(geom_alphas1,molecule,basis,reference_determinant,reference_overlap)

t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,reference_overlap,mix_states=False,type="procrustes")

evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,reference_overlap,sample_x=sample_geom,mix_states=False)

"""
Set up machine learning for t amplitudes
"""

t1s_orth,t2s_orth,t_coefs=orthonormalize_ts_lowdin(evcsolver.t1s,evcsolver.t2s)
t1_machinelearn=[]
t2_machinelearn=[]
kernel=RBF_kernel_unitary_matrices #Use extended RBF kernel
stds=np.zeros(len(geom_alphas))
predictions=[]
ml_params=[]
for i in range(len(sample_geom1)):
    mean,std,parameters=get_model(sample_U,t_coefs[i]-np.mean(t_coefs[i]),kernel,target_U)
    predictions.append(mean+np.mean(t_coefs[i]))
    stds+=(std)
ml_params.append(parameters)
means=np.array(predictions)
for i in range(len(geom_alphas1)):
    t1_temp=np.zeros_like(t1s[0])
    t2_temp=np.zeros_like(t2s[0])
    for j in range(len(t_coefs)):
        t1_temp+=means[j][i]*t1s_orth[j]
        t2_temp+=means[j][i]*t2s_orth[j]
    t1_machinelearn.append(t1_temp)
    t2_machinelearn.append(t2_temp)

print("Initial")
xtol=1e-8 #Convergence tolerance
E_ML_U=evcsolver.calculate_CCSD_energies_from_guess(t1_machinelearn,t2_machinelearn,target_Procrustes,xtol=xtol) #But we are not getting them... BITCH ASS

evcsolver.solve_CCSD_startguess(t1_machinelearn,t2_machinelearn,target_Procrustes,xtol=xtol)
niter_machinelearn=evcsolver.num_iter
outdata={}
outdata["basis"]=basis
outdata["molecule_name"]=molecule_name
outdata["sample_geometries"]=sample_geom1
outdata["test_geometries"]=geom_alphas1
outdata["sample_energies"]=sample_energies
outdata["ML"]=niter_machinelearn

outdata["coefficients"]=t_coefs
outdata["sample_U"]=sample_U
outdata["target_U"]=target_U
outdata["CC_sample_amplitudes_procrustes"]=[t1s_orth,t2s_orth]
outdata["CC_sample_amplitudes"]=[t1s,t2s,l1s,l2s]
outdata["reference_determinant"]=reference_determinant
outdata["Machine_learning_parameters"]=ml_params
outdata["energies_ML"]=E_ML_U
outdata["sample_t1"]=t1s_orth
outdata["sample_t2"]=t2s_orth
outdata["t_transform"]=t_coefs

file="energy_data/HF_machinelearning_%s_%d.bin"%(basis,len(sample_geom1))
import pickle
with open(file,"wb") as f:
    pickle.dump(outdata,f)
