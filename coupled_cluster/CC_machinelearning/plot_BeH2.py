import sys
sys.path.append("../../libraries")
from rccsd_gs import *
from func_lib import *
basis = 'cc-pVTZ'
import pickle

import matplotlib as mpl
mpl.rcParams['font.size'] = 25

num_points=25
basis=basis
file="energy_data/BeH2_machinelearning_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_ML=pickle.load(f)
file="energy_data/BeH2_machinelearning_bestGeometries_%s_%d.bin"%(basis,num_points)
with open(file,"rb") as f:
    data_GP_top=pickle.load(f)
file="energy_data/BeH2_AMPCCEVC_%s_%d.bin"%(basis,num_points)

import pickle
with open(file,"rb") as f:
    energy_dict=pickle.load(f)
GP_sample_geometries=np.array(data_ML["sample_geometries"])
GP_top_sample_geometries=np.array(data_GP_top["sample_geometries"])
CCSD_energies=CCSD=np.array(energy_dict["energies_CCSD"]).reshape((10,10))
HF_energies=np.array(energy_dict["energies_HF"]).reshape((10,10))

E_ML=np.array(data_ML["energies_ML"]).reshape((10,10))
E_GP_auto=np.array(data_GP_top["energies_ML"]).reshape((10,10))
E_tr_sum_20=np.array(energy_dict["energies_AMP_20"]).reshape((10,10))
E_tr_sum_10=np.array(energy_dict["energies_AMP_10"]).reshape((10,10))

niter_tr_sum_10=np.array(energy_dict["EVC_10"]).reshape((10,10))
niter_tr_sum_20=np.array(energy_dict["EVC_20"]).reshape((10,10))
niter_prevGeom=np.array(energy_dict["prevGeom"]).reshape((10,10))
niter_ML=np.array(data_ML["ML"]).reshape((10,10))
niter_GP_auto=np.array(data_GP_top["ML"]).reshape((10,10))
niter_MP2=np.array(energy_dict["MP2"]).reshape((10,10))

x=y=np.linspace(2,6,10)
test_geom=energy_dict["test_geometries"]
E_MLerr=abs(E_ML-CCSD)*1000
E_GP_auto_err=abs(E_GP_auto-CCSD)*1000
E_20_err=abs(E_tr_sum_20-CCSD)*1000
E_10_err=abs(E_tr_sum_10-CCSD)*1000

cmap="viridis"
z_min=0
alpha=0.9
z_max=np.amax( np.concatenate( (E_MLerr.ravel(),E_GP_auto_err.ravel())))
#z_max=np.amax(E_MLerr.ravel())
fig,grid=plt.subplots(2,2,sharey=True,sharex=True,figsize=(15,10))
im0=grid[0,0].pcolormesh(x, y, E_MLerr, cmap=cmap,shading='auto',vmin=z_min,vmax=np.amax(E_MLerr),alpha=alpha)
grid[0,0].set_title("GP")
grid[0,0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[0,0].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[0,0].scatter(GP_sample_geometries[:,0],GP_sample_geometries[:,1],color="magenta",marker="*")

im2=grid[1,0].pcolormesh(x, y, E_10_err, cmap=cmap,shading='auto',vmin=z_min,vmax=np.amax(E_10_err),alpha=alpha)
grid[1,0].set_title("truncated sum (10%)")
grid[1,0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[1,0].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[1,0].scatter(GP_sample_geometries[:,0],GP_sample_geometries[:,1],color="magenta",marker="*")

im3=grid[1,1].pcolormesh(x, y, E_20_err, cmap=cmap,shading='auto',vmin=z_min,vmax=np.amax(E_20_err),alpha=alpha)
grid[1,1].set_title("truncated sum (20%)")
grid[1,1].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[1,1].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[1,1].scatter(GP_sample_geometries[:,0],GP_sample_geometries[:,1],color="magenta",marker="*")








im1=grid[0,1].pcolormesh(x, y, E_GP_auto_err, cmap=cmap,shading='auto',vmin=z_min,vmax=np.amax(E_GP_auto_err),alpha=alpha)

grid[0,1].scatter(GP_top_sample_geometries[:,0],GP_top_sample_geometries[:,1],color="magenta",marker="*")

grid[0,1].set_title("GP (auto)")
grid[0,1].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[0,1].set_xlabel(r"distance $H^2$-Be (Bohr)")
plt.suptitle("Absolute deviation from CCSD energy")
colorbar=fig.colorbar(im1,label='Error (mHartree)')
colorbar=fig.colorbar(im0,label='Error (mHartree)')
colorbar=fig.colorbar(im2,label='Error (mHartree)')
colorbar=fig.colorbar(im3,label='Error (mHartree)')

plt.tight_layout()
plt.savefig("plots/BeH2_energies.pdf")
plt.show()

fig,grid=plt.subplots(2,2,sharey=True,sharex=True,figsize=(15,10))
im0=grid[0,0].pcolormesh(x, y, niter_ML, cmap=cmap,shading='auto',vmin=np.amin(niter_ML),vmax=np.amax(niter_ML),alpha=alpha)
grid[0,0].set_title("GP")
grid[0,0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[0,0].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[0,0].scatter(GP_sample_geometries[:,0],GP_sample_geometries[:,1],color="magenta",marker="*")

im2=grid[1,0].pcolormesh(x, y, niter_tr_sum_10, cmap=cmap,shading='auto',vmin=np.amin(niter_tr_sum_10),vmax=np.amax(niter_tr_sum_10),alpha=alpha)
grid[1,0].set_title("truncated sum (10%)")
grid[1,0].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[1,0].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[1,0].scatter(GP_sample_geometries[:,0],GP_sample_geometries[:,1],color="magenta",marker="*")

im3=grid[1,1].pcolormesh(x, y, niter_tr_sum_20, cmap=cmap,shading='auto',vmin=np.amin(niter_tr_sum_20),vmax=np.amax(niter_tr_sum_20),alpha=alpha)
grid[1,1].set_title("truncated sum (20%)")
grid[1,1].set_xlabel(r"distance $H^2$-Be (Bohr)")
grid[1,1].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[1,1].scatter(GP_sample_geometries[:,0],GP_sample_geometries[:,1],color="magenta",marker="*")


im1=grid[0,1].pcolormesh(x, y, niter_GP_auto, cmap=cmap,shading='auto',vmin=np.amin(niter_GP_auto),vmax=np.amax(niter_GP_auto),alpha=alpha)

grid[0,1].scatter(GP_top_sample_geometries[:,0],GP_top_sample_geometries[:,1],color="magenta",marker="*")

grid[0,1].set_title("GP (auto)")
grid[0,1].set_ylabel(r"distance $H^1$-Be (Bohr)")
grid[0,1].set_xlabel(r"distance $H^2$-Be (Bohr)")
plt.suptitle("Absolute deviation from CCSD energy")
colorbar=fig.colorbar(im1,label=r'$\Delta$ num. iter.')
colorbar=fig.colorbar(im0,label=r'$\Delta$ num. iter.')
colorbar=fig.colorbar(im2,label=r'$\Delta$ num. iter.')
colorbar=fig.colorbar(im3,label=r'$\Delta$ num. iter.')

plt.tight_layout()
plt.savefig("plots/BeH2_niter.pdf")
niter_GP_auto=np.array(niter_GP_auto,dtype=float)
niter_ML=np.array(niter_ML,dtype=float)
niter_tr_sum_10=np.array(niter_tr_sum_10,dtype=float)
niter_tr_sum_20=np.array(niter_tr_sum_20,dtype=float)
# We have to remove 9 points that are exactly the sample points, otherwise we are counting "too well".
first_9=list(zip(*np.where(niter_GP_auto <= 2)))[:9]
for elem in first_9:
    i, j= elem
    niter_GP_auto[i,j]=np.nan
print("Average number of iterations")
print("MP2: %f"%np.nanmean(niter_MP2))
print("ML auto: %f"%np.nanmean(niter_GP_auto))
print("ML: %f"%np.nanmean(niter_ML))
print("AMP 20: %f"%np.nanmean(niter_tr_sum_20))
print("AMP 10: %f"%np.nanmean(niter_tr_sum_10))
plt.show()
correlation_energy=-(CCSD_energies-HF_energies)
p_error_GP_10=np.mean(abs(CCSD_energies-E_ML)/correlation_energy)
p_error_tr_sum_10_10=np.mean(abs(CCSD_energies-E_tr_sum_10)/correlation_energy)
p_error_tr_sum_20_10=np.mean(abs(CCSD_energies-E_tr_sum_20)/correlation_energy)
p_error_GP_adv_10=np.mean(abs(CCSD_energies-E_GP_auto)/correlation_energy)

print("Errors: GP: %f"%(1*(1-p_error_GP_10)))
print("Errors: GP adv: %f"%(1*(1-p_error_GP_adv_10)))
print("Errors: tr. sum 10: %f"%(1*(1-p_error_tr_sum_10_10)))
print("Errors: tr. sum 20: %f"%(1*(1-p_error_tr_sum_20_10)))
