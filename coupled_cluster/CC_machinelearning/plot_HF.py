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
file="energy_data/HF_machinelearning_bestGeometries_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_GP_auto=pickle.load(f)
file="energy_data/HF_AMPCCEVC_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_AMP=pickle.load(f)

CCSD_energies=np.array(data_AMP["energies_CCSD"])
HF_energies=np.array(data_AMP["energies_HF"])

if sum(HF_energies-CCSD_energies)<0: #This is because I made a small bug writing to file
    temp=HF_energies.copy()
    HF_energies=CCSD_energies.copy()
    CCSD_energies=temp.copy()

niter_tr_sum_startguess10_7=data_AMP["EVC_10"]
niter_tr_sum_startguess20_7=data_AMP["EVC_20"]
niter_prevGeom=data_AMP["prevGeom"]
niter_GP_7=data_GP["ML"]
niter_GP_auto_7=data_GP_auto["ML"]
niter_MP2=data_AMP["MP2"]
tr_sum_20_energies_7=data_AMP["energies_AMP_20"]
tr_sum_10_energies_7=data_AMP["energies_AMP_10"]
GP_energies_7=np.array(data_GP["energies_ML"])
target_geometries=np.array(data_GP["test_geometries"])
sample_geometries_GP_7=np.array(data_GP["sample_geometries"])
GP_auto_energies_7=np.array(data_GP_auto["energies_ML"])
sample_geometries_GP_auto_7=np.array(data_GP_auto["sample_geometries"])

num_points=10
file="energy_data/HF_machinelearning_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_GP=pickle.load(f)
file="energy_data/HF_machinelearning_bestGeometries_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_GP_auto=pickle.load(f)
file="energy_data/HF_AMPCCEVC_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_AMP=pickle.load(f)
niter_tr_sum_startguess10_10=data_AMP["EVC_10"]
niter_tr_sum_startguess20_10=data_AMP["EVC_20"]
niter_prevGeom=data_AMP["prevGeom"]
niter_GP_10=data_GP["ML"]
niter_GP_auto_10=data_GP_auto["ML"]
niter_MP2=data_AMP["MP2"]
tr_sum_20_energies_10=data_AMP["energies_AMP_20"]
tr_sum_10_energies_10=data_AMP["energies_AMP_10"]
GP_energies_10=np.array(data_GP["energies_ML"])
#target_geometries=np.array(data_GP["test_geometries"])
sample_geometries_GP_10=np.array(data_GP["sample_geometries"])

GP_auto_energies_10=np.array(data_GP_auto["energies_ML"])
sample_geometries_GP_auto_10=np.array(data_GP_auto["sample_geometries"])


fig,ax=plt.subplots(2,1,sharey=False,sharex=True,figsize=(10,10))

for x in (sample_geometries_GP_7[:-1]):
    ax[0].axvline(x,linestyle="--",color="blue",alpha=0.3)
ax[0].axvline(sample_geometries_GP_7[-1],linestyle="--",color="blue",alpha=0.3,label="Sample geom.")
for x in (sample_geometries_GP_auto_7[:-1]):
    ax[0].axvline(x,linestyle="--",color="red",alpha=0.3)
ax[0].axvline(sample_geometries_GP_auto_7[-1],linestyle="--",color="red",alpha=0.3, label="Sample geom. (auto)")
ax[0].plot(target_geometries,1000*(CCSD_energies-GP_energies_7),label="GP",color="blue")
ax[0].plot(target_geometries,1000*(CCSD_energies-GP_auto_energies_7),color="red",label="GP (auto)")
ax[0].plot(target_geometries,1000*(CCSD_energies-tr_sum_10_energies_7),label=r"tr. sum 10%",color="green")
ax[0].plot(target_geometries,1000*(CCSD_energies-tr_sum_20_energies_7),label=r"tr. sum 20%",color="darkorange")

ax[0].set_ylabel(r"$\Delta E$ (mHartree)")
for x in (sample_geometries_GP_10):
    ax[1].axvline(x,linestyle="--",color="blue",alpha=0.3)
for x in (sample_geometries_GP_auto_10):
    ax[1].axvline(x,linestyle="--",color="red",alpha=0.3)

ax[1].plot(target_geometries,1000*(CCSD_energies-GP_energies_10),label="GP",color="blue")
ax[1].plot(target_geometries,1000*(CCSD_energies-tr_sum_10_energies_10),label=r"tr. sum 10%",color="green")
ax[1].plot(target_geometries,1000*(CCSD_energies-tr_sum_20_energies_10),label=r"tr. sum 20%",color="darkorange")
ax[1].plot(target_geometries,1000*(CCSD_energies-GP_auto_energies_10),label="GP (auto)",color="red")

ax[1].set_ylabel(r"$\Delta E$ (mHartree)")
ax[1].set_xlabel("Intermolecular distance (Bohr)")
ax[0].legend(prop={'size': 16.5},columnspacing=0.0,handletextpad=0.0,labelspacing=0)
ax[1].set_title("10 sample geometries")
ax[0].set_title("7 sample geometries")
plt.suptitle("Deviation from CCSD energy (HF)")

plt.tight_layout()
plt.savefig("plots/HF_energy.pdf")
plt.show()

fig,ax=plt.subplots(2,1,sharey=False,sharex=True,figsize=(10,10))

for x in (sample_geometries_GP_7[:-1]):
    ax[0].axvline(x,linestyle="--",color="blue",alpha=0.3)
ax[0].axvline(sample_geometries_GP_7[-1],linestyle="--",color="blue",alpha=0.3,label="Sample geom.")
for x in (sample_geometries_GP_auto_7[:-1]):
    ax[0].axvline(x,linestyle="--",color="red",alpha=0.3)
ax[0].axvline(sample_geometries_GP_auto_7[-1],linestyle="--",color="red",alpha=0.3, label="Sample geom. (auto)")

ax[0].plot(target_geometries,niter_GP_7,label="GP",color="blue")
ax[0].plot(target_geometries,niter_GP_auto_7,label="GP (auto)",color="red")

ax[0].plot(target_geometries,niter_tr_sum_startguess10_7,"--",label=r"tr. sum 10%",color="green")
ax[0].plot(target_geometries,niter_tr_sum_startguess20_7,"--",label=r"tr. sum 20%",color="darkorange")
ax[0].plot(target_geometries,niter_MP2,label="MP2",color="purple")
ax[0].plot(target_geometries,niter_prevGeom,label="prevGeom",color="brown")

ax[0].set_title("7 sample geometries")
ax[0].set_ylabel(r"Number of iterations")
for x in (sample_geometries_GP_10):
    ax[1].axvline(x,linestyle="--",color="blue",alpha=0.3)
for x in (sample_geometries_GP_auto_10):
    ax[1].axvline(x,linestyle="--",color="red",alpha=0.3)

ax[1].plot(target_geometries,niter_GP_10,label="GP",color="blue")
ax[1].plot(target_geometries,niter_GP_auto_10,label="GP (auto)",color="red")

ax[1].plot(target_geometries,niter_tr_sum_startguess10_10,"--",label=r"tr. sum 10%",color="green")
ax[1].plot(target_geometries,niter_tr_sum_startguess20_10,"--",label=r"tr. sum 20%",color="darkorange")
ax[1].plot(target_geometries,niter_MP2,label="MP2",color="purple")
ax[1].plot(target_geometries,niter_prevGeom,label="prevGeom",color="brown")

ax[1].set_ylim([2.9,17])

ax[0].set_ylim([6,20])
ax[1].set_ylabel(r"Number of iterations")
ax[1].set_xlabel("Intermolecular distance (Bohr)")
ax[1].set_title("10 sample geometries")

ax[0].legend(loc="upper left",prop={'size': 16.5},columnspacing=0.0,handletextpad=0.0,labelspacing=0)
plt.suptitle("Number of iterations (HF)")
for i in range(len(ax)):
    ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig("plots/HF_niter.pdf")

plt.show()

correlation_energy=-(CCSD_energies-HF_energies)
p_error_GP_10=np.mean(abs(CCSD_energies-GP_energies_10)/correlation_energy)
p_error_tr_sum_10_10=np.mean(abs(CCSD_energies-tr_sum_10_energies_10)/correlation_energy)
p_error_tr_sum_20_10=np.mean(abs(CCSD_energies-tr_sum_20_energies_10)/correlation_energy)
p_error_GP_auto_10=np.mean(abs(CCSD_energies-GP_auto_energies_10)/correlation_energy)

p_error_GP_7=np.mean(abs(CCSD_energies-GP_energies_7)/correlation_energy)
p_error_tr_sum_10_7=np.mean(abs(CCSD_energies-tr_sum_10_energies_7)/correlation_energy)
p_error_tr_sum_20_7=np.mean(abs(CCSD_energies-tr_sum_20_energies_7)/correlation_energy)
p_error_GP_auto_7=np.mean(abs(CCSD_energies-GP_auto_energies_7)/correlation_energy)
print("Errors: GP 10: %f"%(1*(1-p_error_GP_10)))
print("Errors: GP auto 10: %f"%(1*(1-p_error_GP_auto_10)))
print("Errors: tr. sum 10, 10: %f"%(1*(1-p_error_tr_sum_10_10)))
print("Errors: tr. sum 20, 10: %f"%(1*(1-p_error_tr_sum_20_10)))

print("Errors: GP 7: %f"%(1*(1-p_error_GP_7)))
print("Errors: GP auto 7: %f"%(1*(1-p_error_GP_auto_7)))
print("Errors: tr. sum 10, 7: %f"%(1*(1-p_error_tr_sum_10_7)))
print("Errors: tr. sum 20, 7: %f"%(1*(1-p_error_tr_sum_20_7)))
