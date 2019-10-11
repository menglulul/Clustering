#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import jaccard_similarity_score

import kmeans
import hierarchical
import spectral
import GMM
import dbscan as my_dbscan
import visualization as vs
import pca as mypca


def read(file_path):
    data = pd.read_csv(file_path,sep=r'\t', header=None, engine='python')
    ds =  data.iloc[:,2:]
    gt = data.iloc[:,1]
    return ds, gt

def RandIndex(truth, pred):
    sklearnRandScore = adjusted_rand_score(truth, pred)
    return sklearnRandScore

# apply pca and draw scatter plot
def visualize(dataset, labels, title):
    pca_res = mypca.pca(dataset.to_numpy())
    vs.visualization(pca_res, labels, title, 'PC')

def main():

    dataset, ground_truth = read("cho.txt")
    visualize(dataset, ground_truth, 'groundtruth')

    # kmeans clustering
    for i in range(4, 5):
        kmeans_res, kmeans_centroids = kmeans.k_means(dataset, i)
        kmeans_ix = RandIndex(ground_truth, kmeans_res)
        print('kmeans: k = {}, RandIndex = {}'.format(i, kmeans_ix))
        visualize(dataset, kmeans_res, 'kmeans')

    # hierarchical agglomerative(single link)
    for i in range(238, 239):
        hac_min_res = hierarchical.hac(dataset, i, 'MIN')
        hac_min_ix = RandIndex(ground_truth, hac_min_res)
        print('hac_min: n_clusters = {}, RandIndex = {}'.format(i, hac_min_ix))
        visualize(dataset, hac_min_res, 'hac_min')
    
    # hierarchical agglomerative(complete link)
    for i in range(5, 6):
        hac_max_res = hierarchical.hac(dataset, i, 'MAX')
        hac_max_ix = RandIndex(ground_truth, hac_max_res)
        print('hac_max: n_clusters = {}, RandIndex = {}'.format(i, hac_max_ix))
        visualize(dataset, hac_max_res, 'hac_max')

    # spectral clustering
    for i in range(12, 13):
        spec_res, spec_centroids = spectral.spectral_cluster(dataset, i)
        spec_ix = RandIndex(ground_truth, spec_res)
        print('spectral: k = {}, RandIndex = {}'.format(i, spec_ix))
        visualize(dataset, spec_res, 'spectral')

    # GMM clustering
    for i in range(5, 6):
        GMM_res, GMM_centroids = GMM.GMM_clustering(dataset, i)
        GMM_ix = RandIndex(ground_truth, GMM_res)
        print('GMM: k = {}, RandIndex = {}'.format(i, GMM_ix))
        visualize(dataset, GMM_res, 'GMM')

    
# def dbscan_clustering():
#     dataset, ground_truth = read("cho.txt")
#     visualize(dataset, ground_truth, 'groundtruth')
    
#     file = open('dbscan_cho.csv', mode='w')
#     writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     best_rand_index = 0
#     for eps in np.arange(1,1.3,0.05):
#         for min_samples in range(2,16):
#             labels = my_dbscan.DBSCAN(dataset, eps, min_samples)
#             rand_index, jaccard = sklearnIndex(ground_truth, labels)
#             if rand_index>best_rand_index:
#                 best_rand_index = rand_index
#                 visualize(dataset, labels, 'dbscan')
#                 print('eps',eps)
#                 print('min_samples',min_samples)
#                 print('rand_index',rand_index)
#                 print('jaccard',jaccard)
#             writer.writerow([eps,min_samples,rand_index,jaccard])
    
#     dataset, ground_truth = read("iyer.txt")
#     visualize(dataset, ground_truth, 'groundtruth')
    
#     file = open('dbscan_iyer.csv', mode='w')
#     writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     best_rand_index = 0
#     for eps in np.arange(0.8,1.2,0.05):
#         for min_samples in range(2,12):
#             labels = my_dbscan.DBSCAN(dataset, eps, min_samples)
#             rand_index, jaccard = sklearnIndex(ground_truth, labels)
#             if rand_index>best_rand_index:
#                 best_rand_index = rand_index
#                 visualize(dataset, labels, 'dbscan')
#                 print('eps',eps)
#                 print('min_samples',min_samples)
#                 print('rand_index',rand_index)
#                 print('jaccard',jaccard)
#             writer.writerow([eps,min_samples,rand_index,jaccard])
            
#     #sklearn DBSCAN
#     db = DBSCAN(eps=1, min_samples=5).fit(dataset)
#     labels = db.labels_
#     visualize(dataset, labels, 'dbscan')
    
#     #my DBSCAN
#     #parameters: dataset, eps, minpts
#     my_db = my_dbscan.DBSCAN(dataset, 1, 5)
#     visualize(dataset, my_db, 'my_dbscan')
    
if __name__ == "__main__":
    main()
