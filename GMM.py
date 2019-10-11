#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import main
import kmeans
from scipy.stats import multivariate_normal


def e_step(X,pi,mu,sigma,n_clusters):
    N = X.shape[0] 
    gamma = np.zeros((N, n_clusters))
    const_c = np.zeros(n_clusters)    
    for c in range(n_clusters):
        gamma[:,c] = pi[c] * multivariate_normal.pdf(X, mu[c,:], sigma[c])
    # normalize across columns to make a valid probability
    gamma_norm = np.sum(gamma, axis=1)[:,np.newaxis]
    gamma /= gamma_norm
    
    return gamma


def m_step(X,gamma,pi,mu,sigma,n_clusters):
        N = X.shape[0]
        dim = X.shape[1]
        pi = np.mean(gamma, axis = 0)
        mu = np.dot(gamma.T, X) / np.sum(gamma, axis = 0)[:,np.newaxis]

        for c in range(n_clusters):
            x = X - mu[c, :]            
            gamma_diag = np.diag(gamma[:,c])
            x_mu = np.matrix(x)
            gamma_diag = np.matrix(gamma_diag)
            sigma_c = x.T * gamma_diag * x
            sigma[c,:,:]=(sigma_c) / np.sum(gamma, axis = 0)[:,np.newaxis][c]

        return pi, mu, sigma


def GMM_clustering(dataset,n_clusters,max_itr=1000):
    kmean_res, kmean_centroids = kmeans.k_means(dataset,n_clusters)
    data = dataset.to_numpy()
    centroids = kmean_centroids.to_numpy()
    datadim = centroids.shape[1]
    clusters = np.unique(kmean_res)
    
    #initialize parameters
    initial_means = centroids
    initial_cov = np.zeros((n_clusters,datadim,datadim))
    initial_pi = np.zeros((n_clusters))
    
    ct = 0
    for cluster in clusters:
        ids = np.where(kmean_res == cluster)
        initial_pi[ct] = len(ids[0])/len(kmean_res)
        de_mean = dataset.iloc[ids] - initial_means[ct,:]
        Nk = len(ids[0])
        initial_cov[ct,:,:] = np.dot(initial_pi[ct] * de_mean.T, de_mean) / Nk
        ct += 1
    
    pi = initial_pi
    mu = initial_means
    sigma = initial_cov
        
    itr = 0
    while(True):
        gamma = e_step(data,pi,mu,sigma,5)
        new_pi, new_mu, new_sigma = m_step(data,gamma,pi,mu,sigma,5)
        if itr > max_itr or ((new_pi-pi).all() and (new_mu-mu).all() and (new_sigma-sigma).all()):
            break
        else:
            pi = new_pi
            mu = new_mu
            sigma = new_sigma
            itr += 1
            
    labels = np.zeros((data.shape[0], n_clusters))
    for c in range(n_clusters):
        labels[:,c] = pi[c] * multivariate_normal.pdf(data,mu[c,:],sigma[c])
    labels = labels.argmax(1)
    labels = clusters[labels]
    
    return labels, mu