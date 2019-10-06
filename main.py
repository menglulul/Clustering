#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import kmeans
import visualization as vs
import pca as mypca

def read(file_path):
    data = pd.read_csv(file_path,sep=r'\t', header=None, engine='python')
    ds =  data.iloc[:,2:]
    gt = data.iloc[:,1]
    return ds, gt

# calculate rand index and jaccard coefficient
def validate(p, c):
    m11 = m00 = m10 = m01 = 0
    for i in range(len(p)):
        for j in range(len(p)):
            if p[i] == p[j] and c[i] == c[j]:
                m11 += 1
            if p[i] != p[j] and c[i] == c[j]:
                m10 += 1
            if p[i] == p[j] and c[i] != c[j]:
                m01 += 1
            if p[i] != p[j] and c[i] != c[j]:
                m00 += 1
    rand_index = (m11 + m00) / (m11 + m10 + m01 + m00)
    jaccard = m11 / (m11 + m10 + m01)
    return rand_index, jaccard

# apply pca and draw scatter plot
def visualize(dataset, labels, title):
    pca_res = mypca.pca(dataset.to_numpy())
    vs.visualization(pca_res, labels, title, 'PC')

def main():
    dataset, ground_truth = read("cho.txt")
    # dataset, ground_truth = read("iyer.txt")

    # kmeans_res = kmeans.k_means(dataset, 3)
    
    # visualize(dataset, ground_truth, 'groundtruth')
    # visualize(dataset, kmeans_res, 'kmeans')
    
    # ix, jaccard = validate(ground_truth, kmeans_res)
    # print('rand_index: ', ix)
    # print('jaccard: ', jaccard)

if __name__ == "__main__":
    main()
