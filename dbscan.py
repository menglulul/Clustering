#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

def DBSCAN(data, eps, minpts):
    data = data.to_numpy()
    cluster = 0
    visited = np.zeros(len(data))
    labels = np.zeros(len(data))
    
    for i in range(len(data)):
        if visited[i]==0:
            visited[i]=1
            neighborPts = regionQuery(data, i, eps)
            if len(neighborPts)<minpts:
                labels[i]=-1
            else:
                #expandCluster
                cluster += 1
                labels[i]=cluster
                j=0
                while j<len(neighborPts):
                    index = neighborPts[j]
                    if visited[index]==0:
                        visited[index]=1
                        new_neighbor = regionQuery(data, index, eps)
                        if len(new_neighbor)>=minpts:
                            neighborPts = neighborPts+new_neighbor
                    if labels[index]==0:
                        labels[index]=cluster
                    j+=1
            
                
    return labels


def regionQuery(data, center, eps):
    neighborPts = []
    for i in range(len(data)):
            dist = np.linalg.norm(data[center]-data[i])
            if(dist<eps):
                neighborPts.append(i)
    return neighborPts

def distance(a,b):
    res = 0
    for i in range(len(a)):
        res += (a[i]-b[i])*(a[i]-b[i])
    res = math.sqrt(res)
    return res