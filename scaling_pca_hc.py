#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 08:20:13 2019

@author: test
"""

import numpy as np
import random
import os
import sys
import matplotlib.pyplot as  plt
import pandas as pd
from pandas import read_table, read_csv
import seaborn as sns




os.chdir("../datasets/")
# read the csv table using Pandas
pima = read_csv("diabetes.csv", sep=",")

# pre-process as in lecture 1
# see # Basic data cleaning and preparation
# imputation of zero values with mean values for non-zero
pima.loc[pima.BMI==0,"BMI"] = pima[(pima.BMI!=0)]["BMI"].mean()
pima.loc[pima.BloodPressure==0,"BloodPressure"] = pima[(pima.BloodPressure!=0)]["BloodPressure"].mean()
pima.loc[pima.Glucose==0,"Glucose"] = pima[(pima.Glucose!=0)]["Glucose"].mean()
pima.loc[pima.Insulin==0,"Insulin"] = pima[(pima.Insulin!=0)]["Insulin"].mean()
pima.loc[pima.SkinThickness==0,"SkinThickness"] = pima[(pima.SkinThickness!=0)]["SkinThickness"].mean()


###############################################################################
# SCALING
###############################################################################
bmi = pima["BMI"]
# Standard scaler
bmi_std = ((bmi - bmi.mean())/ bmi.std())
# Plot a kernel density estimate and rug plot
sns.displot(bmi_std, hist=False, rug=True, color="r")
# MIN/MAX
bmi_minmax = ((bmi - bmi.min())/ (bmi.max() - bmi.min()))
sns.displot(bmi_minmax, hist=False, rug=True, color="r")



input_attribs = pima.columns[0:8]
output = pima.columns[8]

# Now using Scikit-learn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Min-Max scaling; leave out the output attribute
mms = MinMaxScaler()
pima_mms = mms.fit_transform(pima.loc[:,input_attribs])


type(pima_mms)
# standard scaler; leave out the output attribute
stdsc = StandardScaler()
pima_stdsc = stdsc.fit_transform(pima.loc[:,input_attribs])

# calculate the distance matrix based on euclidean distance
from scipy.spatial.distance import pdist, squareform
# pdist: Pairwise distances between observations in n-dimensional space. Returns a condensed distance matrix Y.
# squareform creates a symmetrical matrix of the pairwise distances from condensed distance matrices.
row_dist = pd.DataFrame(squareform(pdist(pima_mms, metric="euclidean")))


###############################################################################
# PCA implemented via NumPy SVD function
###############################################################################
# PCA assumes that the dataset is centered around the origin
pima_inputs = pima.loc[:,input_attribs]
pima_outputs = pima.loc[:,output]

pima_centered = pima_inputs - pima_inputs.mean(axis = 0)
U, s, V = np.linalg.svd(pima_centered)
pc1 = V.T[:,0]
pc2 = V.T[:,1]
# project to d dimensions
W2 = V.T[:,:2]
pima_2d = pima_centered.dot(W2)
pima_2d.columns = ["PC1", "PC2"]
sns.relplot(x="PC1", y="PC2", data = pima_2d);

###############################################################################
# PCA via Scikit-Learn
###############################################################################
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
pima_2d = pca.fit_transform(pima_inputs)
# Explained Variance Ratio
pca.explained_variance_ratio_ * 100
sum_pc1_pc2_var = sum(pca.explained_variance_ratio_ * 100)
print ("First 2 PCs explain: %.3f of variance" % sum_pc1_pc2_var)

# Choosing the right number of dimensions
# rerun withou specifying number of dimensions
pca = PCA()
pca.fit(pima_inputs)
cumsum = np.cumsum(pca.explained_variance_ratio_)
idx = range(len(cumsum))
df = pd.DataFrame({"pc":idx,"var":cumsum})
sns.relplot(x="pc", y="var", data = df, kind ="line");

# recover data after reduction
#pca = PCA(n_components = 154)
#X_reduced = pca.fit_transform(X_train)
#X_recovered = pca.inverse_transform(X_reduced)


# plot decision regions of a classifier
# lets perform dimensionality reduction 
# following by training a classifier (logistic regression)
# use the function from Sebastian Raschka to plot the decision regions see code
# in the end of file

from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

#PCA with 2 PCS
pca = PCA(n_components = 2)
pima_2d = pca.fit_transform(pima_inputs)
# fit a LR model on the output data
lr = LogisticRegression()
lr.fit(pima_2d, pima_outputs)

plot_decision_regions(pima_2d, pima_outputs, classifier = lr)
plt.xlabel("PC1")
plt.xlabel("PC2")
plt.legend(loc="lower left")
plt.show()

###############################################################################
# HC using Scipy
###############################################################################
# apply the complete linkage agglomeration
from scipy.cluster.hierarchy import linkage

# use the complete input sample matrix
row_clusters = linkage(pima_mms, method="complete", metric="euclidean")

# have a closer look at the clustering results in format of Pandas DataFrame
# the 1st and 2nd col denote the most dissimilar element in the cluster
# 3rd col denotes the distance between elements in 1 and 2
# 4rd col denotes number of elements in cluster
cols_labels = ["row label 1","row label 2","distance","# items in clust"]
idx_labels = ["cluster %d" % (i+1) for i in range(row_clusters.shape[0])]
pd.DataFrame(row_clusters, columns = cols_labels, index = idx_labels)

# visualize the results in the form of a dendrogram
from scipy.cluster.hierarchy import dendrogram
row_dendro = dendrogram(row_clusters, labels = pima.index.values)
plt.plot(figsize=(18, 13))
plt.ylabel("Euclidean distace")
plt.show()

###############################################################################
# HC using SCIKIT_LEARN
###############################################################################
from sklearn.cluster import AgglomerativeClustering
# AgglomerativeClustering : recursively merges the pair of clusters that minimally 
# increases a given linkage distance.
# returns cluster labels
# n_clusters: The number of clusters to find.
# affinity: Metric used to compute the linkage.
# linkage: {"ward", "complete", "average", "single"}
agc = AgglomerativeClustering(n_clusters = 2, affinity = "euclidean")
hc_labels = agc.fit_predict(pima_mms)
print("Cluster labels: %s" % hc_labels)

# clustering of SAMPLES X ATTRIBUTES
sns.clustermap(pima_mms, metric="euclidean", standard_scale=1, method="complete", cmap="Blues")
# clustering of SAMPLES X SAMPLES
sns.clustermap(row_dist, metric="euclidean", method="complete", cmap="Blues")

# Lets add coloured leaves
# Prepare a vector of color mapped to the 'OUTCOME' column
my_palette = dict(zip(pima.Outcome.unique(), ["orange","green"]))
row_colors = pima.Outcome.map(my_palette)
#Dendrogram with heatmap and coloured leaves
sns.clustermap(row_dist, metric="euclidean", method="complete", cmap="Blues", row_colors=row_colors)
plt.savefig("test.png", size=(20,20))

###############################################################################
# Clustering with K-Means
###############################################################################
from sklearn.cluster import KMeans
km = KMeans(n_clusters = 2, init="random", n_init = 10, max_iter = 300, tol = 1e-04, random_state = 0)
outcome_km = km.fit_predict(pima_mms)
# compare the obtained clusters with the real outcome values
km_acc = sum(pima.Outcome == outcome_km) / len(outcome_km) * 100
print ("K-Means Accuracy: %.3f" % km_acc)

# we have a 8-dimensional dataset; we can visualize 2 of the dimensions w.r.t to the cluster solution
sns.relplot(x = "BMI", y = "Glucose", hue = outcome_km , data = pima, palette=["m", "g"], style=outcome_km);


# Selecting the right value of K
# distribution of within-cluster SSE (Sum of Squared Errors) also called distortion
distortions = []
for i in range(1,11):
    km = KMeans (n_clusters = i, init = "k-means++", n_init = 10, max_iter = 300, random_state = 0)
    km.fit(pima_mms)
    distortions.append(km.inertia_)
# plot the values
plt.plot(range(1,11), distortions, marker = "o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.show()



###############################################################################
# https://github.com/rasbt/mlxtend/issues/97
# Decision region plot for more than 4 classes  by Sebastian Raschka
###############################################################################
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.1):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))+1])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl)
