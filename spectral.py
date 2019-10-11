#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

def laplacianFully(df, k):
    num_rows = df.shape[0]
    W = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        a = df.iloc[i, :]
        for j in range(i+1, num_rows):
            b = df.iloc[j, :]
            W[i][j] = np.linalg.norm(a-b)
            W[j][i] = W[i][j]

    # use gaussian kernel for similarity function

    # calculate a matrix of variance measure for each individual dist in W (local scaling)
    # get the d(si, sk) -- single row at row k of sorted W
    # (kth smallest dist in all rows/objects for each attribute)
    ksub = np.sort(W, axis=0)
    ksub = ksub[k]=
    # calculate sigma squared using matrix multiplication
    ksub = ksub[np.newaxis].T
    sigma_square = np.multiply(ksub, ksub.T)

    A = np.exp(-1 * W * W / sigma_square)

    # calculate degree matrix
    D = np.zeros((num_rows, num_rows))
    rowsums = np.sum(A, axis=1).tolist()
    for q in range(num_rows):
        D[q][q] = rowsums[q]

    return D, A

def laplacianEpsilon(df, epsilon):
    num_rows = df.shape[0]
    A = np.zeros((num_rows, num_rows))
    D = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        a = df.iloc[i, :]
        for j in range(i + 1, num_rows):
            b = df.iloc[j, :]
            # use euclidean function for similarity measure
            euclidean = np.linalg.norm(a - b)
            if euclidean >= epsilon:
                A[i][j] = euclidean
                A[j][i] = A[i][j]

        D[i][i] = np.sum(A[i])
    return D, A


# get matrix of first k eigenvectors as columns
def eigen(L, k):
    values, vectors = np.linalg.eig(L)
    idx = np.argsort(values)
    values = values[idx]
    vectors = vectors[:,idx]
    V = vectors[:, 0:k]
    return values, V

def sklearnKmeans(V, k):
    kmeans = KMeans(n_clusters=k, init='random', random_state= 0 ).fit(V)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return clusters, centroids

# the following is for comparison b/w sklearn and our self implemented spectral clustering
def sklearnSpectral(dataset, k):
    sklearn = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=0).fit(dataset)
    clusters = sklearn.labels_
    centroids = "null"
    return clusters, centroids

def spectral_cluster(dataset, k):

    D, A = laplacianFully(dataset, k)
    L = np.subtract(D, A)
    W, V = eigen(L, k)
    clusters, centroids = sklearnKmeans(V, k)

    return clusters, centroids
