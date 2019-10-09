#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import kmeans, hierarchical
import visualization as vs
import pca as mypca

# from sklearn.cluster import KMeans
# from sklearn.metrics import jaccard_score
# from sklearn.cluster import AgglomerativeClustering

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

    visualize(dataset, ground_truth, 'groundtruth')

    # kmeans clustering
    # kmeans_res, kmeans_centroids = kmeans.k_means(dataset, 5)
    # print('kmeans_centroids: ', kmeans_centroids)
    # visualize(dataset, kmeans_res, 'kmeans')
    
    # kmeans_ix, kmeans_jaccard = validate(ground_truth, kmeans_res)
    # print('kmeans_rand_index: ', kmeans_ix)
    # print('kmeans_jaccard: ', kmeans_jaccard)
    
    # compare kmeans to sklearn implementation
    # sky_kmeans = KMeans(n_clusters=5)
    # sky_kmeans_res = sky_kmeans.fit(dataset).predict(dataset)
    # sky_kmeans_centroids = sky_kmeans.cluster_centers_
    # print('sky_kmeans_centroids: ', sky_kmeans_centroids)
    # sky_kmeans_labels = sky_kmeans.labels_
    # visualize(dataset, sky_kmeans_labels, 'sky_kmeans')
    
    # sky_kmeans_ix, sky_kmeans_jaccard = validate(ground_truth, sky_kmeans_res)
    # print('sky_kmeans_rand_index: ', sky_kmeans_ix)
    # print('sky_kmeans_jaccard: ', sky_kmeans_jaccard)

    # hierarchical agglomerative clustering
    # clustering = AgglomerativeClustering(linkage='complete', n_clusters=5)
    # clustering.fit(dataset)
    # visualize(dataset, clustering.labels_, 'sk_hac_complete')    
    hac_res = hierarchical.hac(dataset, 5, 'MIN')
    visualize(dataset, hac_res, 'hac_single')
    hac_res = hierarchical.hac(dataset, 5, 'MAX')
    visualize(dataset, hac_res, 'hac_conplete')

if __name__ == "__main__":
    main()
