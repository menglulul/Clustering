#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def init(df, k):
    # return randomly selected k points with its range
    range = df.shape[0]
    ix = np.random.choice(range, size=k, replace=False)
    centroids = df.iloc[ix,:].reset_index(drop=True)
    return centroids

def euclidean(a, b):
    # return euclidean distance of two points
    return np.linalg.norm(a-b)

def assign(df, centroids):
    # calculate distances to centroids and assign to the closest centroid
    # return the clustering result in ndarray
    num_rows = df.shape[0]
    closest = np.zeros(num_rows)
    for i in range(num_rows):
        d = df.iloc[i,:]
        min_dist = float('inf')
        for centroid in range(centroids.shape[0]):
            c = centroids.iloc[centroid,:]
            dist = euclidean(d, c)
            if dist < min_dist:
                min_dist = dist
                closest[i] = centroid
    return closest


def update(df, clusters):
    # calculate new centroid as the mean of all points that belongs to the cluster
    # return updated centroids in dataframe
    return df.groupby(clusters).mean()


def k_means(dataset, k):
    print('dataset:')
    print(dataset)
    centroids = init(dataset, k)
    clusters = []
    print('initial centroids:')
    print(centroids)
    # assign points to clusters and update centroids
    # repeat until centroids no longer change
    while(True):
        clusters = assign(dataset, centroids)
        new_c = update(dataset, clusters)
        if new_c.equals(centroids):
            break
        else:
            centroids = new_c
            print(new_c)
    # format output
    clusters += 1
    clusters = clusters.astype(int)
    return clusters, centroids