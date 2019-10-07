#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
from sklearn.cluster import KMeans, SpectralClustering


def gaussianKernel(a, b):
    sigma = 1
    euclidean = np.linalg.norm(a-b)
    res = math.exp(-0.5 * (euclidean ** 2) / (sigma ** 2 ) )
    return res


def laplacianFully(df):
    num_rows = df.shape[0]
    A = np.zeros((num_rows, num_rows))
    D = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        a = df.iloc[i, :]
        for j in range(i+1, num_rows):
            b = df.iloc[j, :]
            # use gaussian function for similarity measure
            A[i][j] = gaussianKernel(a, b)
            A[j][i] = A[i][j]
        D[i][i] = np.sum(A[i])
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
            #gaussian = gaussianKernel(a, b)
            if euclidean >= epsilon:
                A[i][j] = euclidean
                A[j][i] = A[i][j]

        D[i][i] = np.sum(A[i])
    return D, A


# get matrix of first k eigenvectors as columns
def eigen(L, k):
    W, vectors = np.linalg.eig(L)

    V = vectors[:, 0:k]

    return W, V

def sklearnKmeans(V, k):
    kmeans = KMeans(n_clusters=k, init='random', random_state= 0 ).fit(V)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_
    print(clusters)
    print(centroids)
    return clusters, centroids

# the following is for comparison b/w sklearn and our self implemented spectral clustering
def sklearnSpectral(dataset, k):
    sklearn = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=0).fit(dataset)
    clusters = sklearn.labels_
    centroids = "null"
    return clusters, centroids

def spectral_cluster(dataset, k):

    # D, A = laplacianFully(dataset)
    D, A = laplacianEpsilon(dataset, 1.5)
    L = np.subtract(D, A)
    # Lnorm = np.linalg.inv(D) * L
    print(A)
    print(D)
    print(L)

    W, V = eigen(L, k)
    clusters, centroids = sklearnKmeans(V, k)

    # clusters, centroids = sklearnSpectral(dataset, k)
    return clusters, centroids
