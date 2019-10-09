#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def init(df):
    # return a matrix of distance between all pairs of points
    n_rows = df.shape[0]
    mat = np.full((n_rows, n_rows), np.inf)
    for i in range(n_rows):
        p_i = df.iloc[i,:]
        for j in range(0, n_rows):
            p_j = df.iloc[j,:]
            mat[i][j] = euclidean(p_i, p_j)
    return mat


def merge(clusters, pair):
    # merge the two closest clusters
    clusters[pair[0]] += clusters[pair[1]]
    clusters.pop(pair[1])
    return clusters


def update(arr, pair, clusters, p_mat, linkage):
    # recalcute distance matrix
    n_rows = len(clusters)
    mat = np.full((n_rows, n_rows),np.inf)
    for i in range(n_rows):
        for j in range(n_rows):
            if linkage == "MIN":
                mat[i][j] = single(clusters[i], clusters[j], p_mat)
            elif linkage == "MAX":
                mat[i][j] = complete(clusters[i], clusters[j], p_mat)                
    return mat

def single(c_i, c_j, p_mat):
    # return distance of the closest pairs of points from two clusters
    min_dist = float('inf')
    for i in c_i:
        for j in c_j:
            if p_mat[i][j] < min_dist:
                min_dist = p_mat[i][j]
    return min_dist

def complete(c_i, c_j, p_mat):
    # return distance of the farthest pairs of points from two clusters    
    max_dist = -1
    for i in c_i:
        for j in c_j:
            if p_mat[i][j] > max_dist:
                max_dist = p_mat[i][j]
    return max_dist

def euclidean(a, b):
    # return euclidean distance of two points
    return np.linalg.norm(a-b)

def hac(dataset, k, linkage):
    # hierarchical agglomerative clustering with two different inter-cluster distances
    # linkage: max(complete-linkage) or min(single-linkage)
    n_rows = dataset.shape[0]
    clusters = []
    for n in range(n_rows):
        clusters.append([n])
    p_mat = init(dataset)
    d_mat = p_mat
    # repeat until k number of clusters remain
    while(len(clusters) > k):
        # find the two closest clusters
        min_dist = float('inf')
        pair = (-1, -1)
        for i in range(len(d_mat)):
            for j in range(len(d_mat)):
                dist = d_mat[i][j]
                if i!= j and dist < min_dist:
                    min_dist = dist
                    pair = (i, j)
                    # print(pair, min_dist)
        # merge the two closest clusters
        clusters = merge(clusters, pair)
        # update distance matrix
        d_mat = update(d_mat, pair, clusters,p_mat,linkage)     
    # format output
    cluster_final = np.zeros(n_rows, dtype=int)
    for i in range(len(clusters)):
        c = clusters[i]
        for p in c:
            cluster_final[p] = i
    # print(cluster_final)
    return cluster_final