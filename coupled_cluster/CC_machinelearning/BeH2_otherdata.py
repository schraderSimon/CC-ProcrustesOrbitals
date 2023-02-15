import sys
sys.path.append("../../libraries")
from rccsd_gs import *
from func_lib import *
from numba import jit
from machinelearning import *
from matrix_operations import *
from helper_functions import *
from mpl_toolkits.axes_grid1 import ImageGrid

molecule_name="BeH2"
basis = 'cc-pVTZ'
basis_set = bse.get_basis(basis, fmt='nwchem')
charge = 0
def molecule(x,y):
    return """Be 0 0 0; H -%f 0 0; H %f 0 0"""%(x,y)
refx=(2,2)
reference_determinant,reference_overlap=get_reference_determinant(molecule,refx,basis,return_overlap=True)
sample_geom_new=[]
x=np.linspace(2.1,5.9,5)
for i in range(len(x)):
    for j in range(len(x)):
        sample_geom_new.append([x[i],x[j]])
sample_geom=np.array(sample_geom_new)
span=np.linspace(2,6,10)
geom_alphas=[]
for x in span:
    for y in span:
        geom_alphas.append((x,y))
x, y = np.meshgrid(span,span)
target_U,target_Procrustes=get_U_matrix(geom_alphas1,molecule,basis,reference_determinant,reference_overlap)
t1s,t2s,l1s,l2s,sample_energies=setUpsamples(sample_geom,molecule,basis,reference_determinant,reference_overlap,mix_states=False,type="procrustes")

evcsolver=EVCSolver(geom_alphas,molecule,basis,reference_determinant,t1s,t2s,l1s,l2s,reference_overlap,sample_x=sample_geom,mix_states=False)
xtol=1e-6 #Convergence tolerance

E_CCSD=evcsolver.solve_CCSD_previousgeometry(target_Procrustes,xtol=xtol)
niter_prevGeom=evcsolver.num_iter

E_CCSD,E_HF=evcsolver.solve_CCSD_noProcrustes(xtol=xtol)
niter_CCSD=evcsolver.num_iter

E_AMP_red_10=evcsolver.solve_AMP_CCSD(occs=1,virts=0.1,xtol=xtol)
t1s_reduced=evcsolver.t1s_final
t2s_reduced=evcsolver.t2s_final
evcsolver.solve_CCSD_startguess(t1s_reduced,t2s_reduced,target_Procrustes,xtol=xtol)
niter_AMP_startguess10=evcsolver.num_iter

E_AMP_red_20=evcsolver.solve_AMP_CCSD(occs=1,virts=0.2,xtol=xtol)
t1s_reduced=evcsolver.t1s_final
t2s_reduced=evcsolver.t2s_final
evcsolver.solve_CCSD_startguess(t1s_reduced,t2s_reduced,target_Procrustes,xtol=xtol)
niter_AMP_startguess20=evcsolver.num_iter
outdata={}
outdata["basis"]=basis
outdata["molecule_name"]=molecule_name
outdata["sample_geometries"]=sample_geom1
outdata["test_geometries"]=geom_alphas1
outdata["sample_energies"]=sample_energies
outdata["MP2"]=niter_CCSD
outdata["EVC_10"]=niter_AMP_startguess10
outdata["EVC_20"]=niter_AMP_startguess20
outdata["prevGeom"]=niter_prevGeom
outdata["energies_CCSD"]=E_CCSD
outdata["energies_AMP_20"]=E_AMP_red_20
outdata["energies_AMP_10"]=E_AMP_red_10
outdata["energies_HF"]=E_HF


file="energy_data/BeH2_AMPCCEVC_%s_%d.bin"%(basis,len(sample_geom))
import pickle
with open(file,"wb") as f:
    pickle.dump(outdata,f)
sys.exit(1)
