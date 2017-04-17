#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import autograd.numpy as np
from autograd import value_and_grad
import matplotlib.pyplot as plt
from pyDOE import lhs
from sklearn.cluster import KMeans
from Utilities import kernel, Normalize, fetch_minibatch, stochastic_update_Adam
import time

np.random.seed(12345)

def predict(X_star):
    Z = ModelInfo["Z"]
    m = ModelInfo["m"]
    S = ModelInfo["S"]
    hyp = ModelInfo["hyp"]
    K_u_inv = ModelInfo["K_u_inv"]
    
    N_star = X_star.shape[0]
    partitions_size = 10000
    (number_of_partitions, remainder_partition) = divmod(N_star, partitions_size)
    
    mean_star = np.zeros((N_star,1));
    var_star = np.zeros((N_star,1));
    
    for partition in range(0,number_of_partitions):
        print("Predicting partition: %d" % (partition))
        idx_1 = partition*partitions_size
        idx_2 = (partition+1)*partitions_size
        
        # Compute mu
        psi = kernel(Z, X_star[idx_1:idx_2,:], hyp[:-1])    
        K_u_inv_m = np.matmul(K_u_inv,m)   
        mu = np.matmul(psi.T,K_u_inv_m)
        
        mean_star[idx_1:idx_2,0:1] = mu;        
    
        # Compute cov  
        Alpha = np.matmul(K_u_inv,psi)
        cov = kernel(X_star[idx_1:idx_2,:], X_star[idx_1:idx_2,:], hyp[:-1]) - \
                np.matmul(psi.T, np.matmul(K_u_inv,psi)) + np.matmul(Alpha.T, np.matmul(S,Alpha))
        var = np.abs(np.diag(cov)) + np.exp(hyp[-1])
        
        var_star[idx_1:idx_2,0] = var

    print("Predicting the last partition")
    idx_1 = number_of_partitions*partitions_size
    idx_2 = number_of_partitions*partitions_size + remainder_partition
    
    # Compute mu
    psi = kernel(Z, X_star[idx_1:idx_2,:], hyp[:-1])    
    K_u_inv_m = np.matmul(K_u_inv,m)   
    mu = np.matmul(psi.T,K_u_inv_m)
    
    mean_star[idx_1:idx_2,0:1] = mu;        

    # Compute cov  
    Alpha = np.matmul(K_u_inv,psi)
    cov = kernel(X_star[idx_1:idx_2,:], X_star[idx_1:idx_2,:], hyp[:-1]) - \
            np.matmul(psi.T, np.matmul(K_u_inv,psi)) + np.matmul(Alpha.T, np.matmul(S,Alpha))
    var = np.abs(np.diag(cov)) + np.exp(hyp[-1])
    
    var_star[idx_1:idx_2,0] = var
    
    
    return mean_star, var_star


def likelihood_UB(hyp):
    X = ModelInfo["X_batch"]
    y = ModelInfo["y_batch"]
    Z = ModelInfo["Z"]
    m = ModelInfo["m"]
    S = ModelInfo["S"]
    jitter_cov = ModelInfo["jitter_cov"]
       
    N = X.shape[0]
    M = Z.shape[0]
    
    logsigma_n = hyp[-1]
    sigma_n = np.exp(logsigma_n)
    
    # Compute K_u_inv
    K_u = kernel(Z, Z, hyp[:-1])    
    K_u_inv = np.linalg.solve(K_u + np.eye(M)*jitter_cov, np.eye(M))
#    L = np.linalg.cholesky(K_u  + np.eye(M)*jitter_cov)    
#    K_u_inv = np.linalg.solve(np.transpose(L), np.linalg.solve(L,np.eye(M)))
    
    ModelInfo.update({"K_u_inv": K_u_inv})
      
    # Compute mu
    psi = kernel(Z, X, hyp[:-1])    
    K_u_inv_m = np.matmul(K_u_inv,m)   
    MU = np.matmul(psi.T,K_u_inv_m)
    
    # Compute cov
    Alpha = np.matmul(K_u_inv,psi)
    COV = kernel(X, X, hyp[:-1]) - np.matmul(psi.T, np.matmul(K_u_inv,psi)) + \
            np.matmul(Alpha.T, np.matmul(S,Alpha))
    
    # Compute NLML        
    Beta = y - MU
    NLML_1 = np.matmul(Beta.T, Beta)/(2.0*sigma_n*N)
    
    NLML_2 = np.trace(COV)/(2.0*sigma_n*N)
    NLML_3 = logsigma_n/2.0 + np.log(2.0*np.pi)/2.0
    NLML = NLML_1 + NLML_2 + NLML_3
    
    return NLML[0,0]


def update_m_S():
    X = ModelInfo["X_batch"]
    y = ModelInfo["y_batch"]
    Z = ModelInfo["Z"]
    m = ModelInfo["m"]
    S = ModelInfo["S"]
    hyp = ModelInfo["hyp"]
    jitter = ModelInfo["jitter"]
    K_u_inv = ModelInfo["K_u_inv"]
       
    N = X.shape[0]
    sigma_n = np.exp(hyp[-1])
    
    # Compute mu(X)
    psi = kernel(Z, X, hyp[:-1])    
    K_u_inv_m = np.matmul(K_u_inv,m)   
    MU = np.matmul(psi.T,K_u_inv_m)
    
    # Compute cov(X,X)
    Alpha = np.matmul(K_u_inv,psi)
    COV = kernel(X, X, hyp[:-1]) - np.matmul(psi.T, np.matmul(K_u_inv,psi)) + \
            np.matmul(Alpha.T, np.matmul(S,Alpha))
    
    COV_inv = np.linalg.solve(COV  + np.eye(N)*sigma_n + np.eye(N)*jitter, np.eye(N))
#    L = np.linalg.cholesky(COV  + np.eye(N)*sigma_n + np.eye(N)*jitter) 
#    COV_inv = np.linalg.solve(np.transpose(L), np.linalg.solve(L,np.eye(N)))
    
    # Compute cov(Z, X)
    cov_ZX = np.matmul(S,Alpha)
    
                 
    # Update m and S
    alpha = np.matmul(COV_inv, cov_ZX.T)
    m = m + np.matmul(cov_ZX, np.matmul(COV_inv, y-MU))    
    S = S - np.matmul(cov_ZX, alpha)
       
    return m, S
    

def init_params():
    X = ModelInfo["X"]
    M = ModelInfo["M"]
    (N,D) = X.shape;
    idx = np.random.permutation(N)
    N_subset = min(N, 10000)
    kmeans = KMeans(n_clusters=M, random_state=0).fit(X[idx[0:N_subset],:])
    Z = kmeans.cluster_centers_

    hyp = np.log(np.ones(D+1))
    logsigma_n = np.array([-4.0])
    hyp = np.concatenate([hyp, logsigma_n])

    m = np.zeros((M,1))
    S = kernel(Z,Z,hyp[:-1])

    ModelInfo.update({"hyp": hyp})
    ModelInfo.update({"Z": Z});
    ModelInfo.update({"m": m});
    ModelInfo.update({"S": S});

def train():
    init_params();
    
    max_iter = ModelInfo["max_iter"]
    N_batch = ModelInfo["N_batch"]
    hyp = ModelInfo["hyp"]
    monitor_likelihood = ModelInfo["monitor_likelihood"]
    
    # Adam optimizer parameters
    mt_hyp = np.zeros(hyp.shape)
    vt_hyp = np.zeros(hyp.shape)
    lrate = ModelInfo["lrate"]
    
    print("Total number of parameters: %d" % (hyp.shape[0]))
    
    # Gradients from autograd
    UB = value_and_grad(likelihood_UB)
    
    start = time.time()
    for i in range(1,max_iter+1):
        # Fetch minibatch
        X_batch, y_batch = fetch_minibatch(X,y,N_batch) 
        ModelInfo.update({"X_batch": X_batch})
        ModelInfo.update({"y_batch": y_batch})
        
        # Compute likelihood_UB and gradients 
        hyp = ModelInfo["hyp"]
        NLML, D_NLML = UB(hyp)    
        
        # Update hyper-parameters
        hyp, mt_hyp, vt_hyp = stochastic_update_Adam(hyp, D_NLML, mt_hyp, vt_hyp, lrate, i)
        
        # Update m and S
        m, S = update_m_S()
        
        ModelInfo.update({"hyp": hyp})
        ModelInfo.update({"m": m})
        ModelInfo.update({"S": S})
        
        
        if i % monitor_likelihood == 0:
            end = time.time()
            print("Iteration: %d, likelihood_UB: %e, elapsed time: %.2f seconds" % (i, NLML, end-start))
            start = time.time()

###############################################################################
###############################################################################

if __name__ == "__main__":
    
    # Setup
    N = 6000
    D = 1
    lb = 0.0*np.ones((1,D))
    ub = 1.0*np.ones((1,D))    
    noise = 0.1

    # Configuration
    ModelInfo = {"N_batch": 1}
    ModelInfo.update({"M": 10})
    ModelInfo.update({"lrate": 1e-3})
    ModelInfo.update({"max_iter": 6000})
    ModelInfo.update({"monitor_likelihood": 10})
    ModelInfo.update({"jitter": 1e-8})
    ModelInfo.update({"jitter_cov": 1e-8})
    ModelInfo.update({"Normalize_input_data": 1})
    ModelInfo.update({"Normalize_output_data": 1})
    
    # Generate traning data
    def f(x):
        return x*np.sin(4*np.pi*x)    
    X = lb + (ub-lb)*lhs(D, N)
    y = f(X) + noise*np.random.randn(N,1)
    
    # Generate test data
    N_star = 400
    X_star = lb + (ub-lb)*np.linspace(0,1,N_star)[:,None]
    y_star = f(X_star)
    
    # Normalize Input Data
    if ModelInfo["Normalize_input_data"] == 1:
        X_m = np.mean(X, axis = 0)
        X_s = np.std(X, axis = 0)
        X = Normalize(X, X_m, X_s)
        
        X_star = Normalize(X_star, X_m, X_s)

    # Normalize Output Data
    if ModelInfo["Normalize_output_data"] == 1:
        y_m = np.mean(y, axis = 0)
        y_s = np.std(y, axis = 0)   
        y = Normalize(y, y_m, y_s)
        
        y_star = Normalize(y_star, y_m, y_s)

    ModelInfo.update({"X": X})
    ModelInfo.update({"y": y})
    
    # Training
    train()
    
    # Prediction
    mean_star, var_star = predict(X_star)
    
    # Plot Results
    Z = ModelInfo["Z"]
    m = ModelInfo["m"]
    plt.figure(1)
    plt.plot(X,y,'b+',alpha=0.1)
    plt.plot(Z,m, 'ro', alpha=1)
    plt.plot(X_star, y_star, 'b-')
    plt.plot(X_star, mean_star, 'r--')
    lower = mean_star - 2.0*np.sqrt(var_star)
    upper = mean_star + 2.0*np.sqrt(var_star)
    plt.fill_between(X_star.flatten(), lower.flatten(), upper.flatten(), facecolor='orange', alpha=0.5)
