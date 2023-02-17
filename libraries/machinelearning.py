import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import sys
from scipy.linalg import block_diag
from scipy.optimize import minimize, minimize_scalar

def RBF_kernel_unitary_matrices(list_U1,list_U2,kernel_params=[1,1]):
    #if kernel_params[1]<-0.5:
    #    kernel_params[1]=-0.5
    l_1=np.exp(kernel_params[1])
    sigma_1=np.exp(kernel_params[0])
    noise=1e-10#np.exp(kernel_params[4])
    norm=np.zeros((len(list_U1),len(list_U2)))
    n=min([len(list_U1),len(list_U2)])
    for i in range(len(list_U1)):
        for j in range(len(list_U2)):
            U1=list_U1[i]
            U2=list_U2[j]
            norm[i,j]=np.linalg.norm(U1-U2)
    kernel_mat= sigma_1*np.exp(-0.5*norm**2/l_1)
    for i in range(n):
        kernel_mat[i,i]+=noise
    return kernel_mat
def GP(X1, y1, X2, kernel_func,kernel_params):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """
    # Kernel of the observations
    Σ11 = kernel_func(X1, X1,kernel_params)
    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2,kernel_params)
    # Solve
    solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T
    # Compute posterior mean
    μ2 = solved @ y1
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2,kernel_params)
    Σ2 = Σ22 - (solved @ Σ12)
    return μ2, Σ2  # mean, covariance
def log_likelihood(kernel_params,data_X,y,kernel):
    cov_matrix=kernel(data_X,data_X,kernel_params) #The covariance matrix
    cov_matrix=cov_matrix
    det=np.linalg.det(cov_matrix)
    log_det=np.log(det)
    inv_times_data=np.linalg.solve(cov_matrix,y)
    return -(-0.5*y.T@inv_times_data-0.5*log_det)
def find_best_model(U_list,y,kernel,start_params):
    y_new=y
    sol=minimize(log_likelihood,x0=start_params,args=(U_list,y,kernel),bounds=[(None,None),(0.25,None)])
    #Uses exponential terms, and e^0.25~1.3 as stated in the paper
    best_sigma=sol.x
    return best_sigma
def get_model(U_list,y,kernel,U_list_target,start_params=[-2,0]):
    best_sigma=find_best_model(U_list,y,kernel,start_params)
    new, Σ2 = GP(U_list, y, U_list_target, kernel,best_sigma)
    return new, np.diag(Σ2),np.exp(best_sigma)
