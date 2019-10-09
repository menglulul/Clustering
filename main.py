#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import kmeans
import spectral
import visualization as vs
import pca as mypca
import GMM

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import jaccard_similarity_score

from sklearn.cluster import DBSCAN
import dbscan as my_dbscan

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

def sklearnIndex(truth, pred):
    sklearnRandScore = adjusted_rand_score(truth, pred)
    sklearnJaccardScore = jaccard_similarity_score(truth, pred)
    return sklearnRandScore, sklearnJaccardScore

# apply pca and draw scatter plot
def visualize(dataset, labels, title):
    pca_res = mypca.pca(dataset.to_numpy())
    vs.visualization(pca_res, labels, title, 'PC')

def main():

    # dataset, ground_truth = read("cho.txt")
    dataset, ground_truth = read("iyer.txt")

    # to-do try out different k to report best parameter setting
    k = 11

    visualize(dataset, ground_truth, 'groundtruth')

    # kmeans clustering
    kmeans_res, kmeans_centroids = kmeans.k_means(dataset, k)
    print(sklearnIndex(ground_truth, kmeans_res))
    print('kmeans_centroids: ', kmeans_centroids)
    visualize(dataset, kmeans_res, 'kmeans')

    kmeans_ix, kmeans_jaccard = validate(ground_truth, kmeans_res)
    print('kmeans_rand_index: ', kmeans_ix)
    print('kmeans_jaccard: ', kmeans_jaccard)
    
    # compare kmeans to sklearn implementation
    # sky_kmeans = KMeans(n_clusters=5)
    # sky_kmeans_res = sky_kmeans.fit(dataset).predict(dataset)
    # randScore, jaccardScore = sklearnIndex(ground_truth, sky_kmeans_res)
    # print(randScore)
    # print(jaccardScore)
    # sky_kmeans_centroids = sky_kmeans.cluster_centers_
    # print('sky_kmeans_centroids: ', sky_kmeans_centroids)
    # sky_kmeans_labels = sky_kmeans.labels_
    # visualize(dataset, sky_kmeans_labels, 'sky_kmeans')
    #
    # sky_kmeans_ix, sky_kmeans_jaccard = validate(ground_truth, sky_kmeans_res)
    # print('sky_kmeans_rand_index: ', sky_kmeans_ix)
    # print('sky_kmeans_jaccard: ', sky_kmeans_jaccard)

    # spectral clustering
    spec_res, spec_centroids = spectral.spectral_cluster(dataset, k)
    print('spectral_centroids: ', spec_centroids)
    visualize(dataset, spec_res, 'spectral')

    sklearnRandScore, sklearnJaccardScore = sklearnIndex(ground_truth, spec_res)
    print('spectral rand index: ', sklearnRandScore)
    print('spectral jaccard index', sklearnJaccardScore)

    # GMM clustering
    GMM_res, GMM_centroids = GMM.GMM_clustering(dataset,k)
    print('GMM_centroids: ', GMM_centroids)
    visualize(dataset, GMM_res, 'GMM')

    GMM_ix, GMM_jaccard = validate(ground_truth, GMM_res)
    print('GMM_rand_index: ', GMM_ix)
    print('GMM_jaccard: ', GMM_jaccard)
    
    #DBSCAN clustering
    
    #sklearn DBSCAN
    db = DBSCAN(eps=1, min_samples=5).fit(dataset)
    labels = db.labels_
    visualize(dataset, labels, 'dbscan')
    
    #my DBSCAN
    #parameters: dataset, eps, minpts
    my_db = my_dbscan.DBSCAN(dataset, 1, 5)
    visualize(dataset, my_db, 'my_dbscan')

    
if __name__ == "__main__":
    main()
