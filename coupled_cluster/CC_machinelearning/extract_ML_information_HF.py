import sys
sys.path.append("../../libraries")
from rccsd_gs import *
from func_lib import *
basis = 'cc-pVTZ'
import pickle
from matplotlib.ticker import MaxNLocator
num_points=7
basis=basis
file="energy_data/HF_machinelearning_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_GP=pickle.load(f)
niter_GP=data_GP["ML"]
energies=np.array(data_GP["energies_ML"])
target_geometries=np.array(data_GP["test_geometries"])
sample_geometries=np.array(data_GP["sample_geometries"])
GP_parameters=data_GP["Machine_learning_parameters"]
orthogonal_sample_t1=data_GP["sample_t1"]
orthogonal_sample_t2=data_GP["sample_t2"]
orthogonalization_information=data_GP["t_transform"]


#The Original cluster operator at sample geometry i can then be obtained as
t1_original=np.zeros_like(orthogonal_sample_t1[0])
t2_original=np.zeros_like(orthogonal_sample_t2[0])
i=2 #Can be between (0,len(sample_geometries)-1)
for j in range(len(orthogonalization_information)):
    t1_original+=orthogonalization_information[j][i]*orthogonal_sample_t1[j]
    t2_original+=orthogonalization_information[j][i]*orthogonal_sample_t2[j]

#The machine learning parameters are contained in GP_parameters can be accessed as follows:
for i in range(len(sample_geometries)):
    print("This concerns the component for the %d'th orthogonal sample parameter"%i)
    print("sigma_f^2: %f"%GP_parameters[i][0])
    print("l^2: %f \n"%GP_parameters[i][1])
