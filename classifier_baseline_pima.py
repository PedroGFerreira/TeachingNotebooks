#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:51:08 2019

@author: Pedro G. Ferreira
"""

# Required software
# Python3.*, numpy, pandas, scikit-learn, SciPy, Matplotlib (and other required and depedent packages)
# Recommended ways to install Python Packages: Conda or Pip

# Pip is the official Python package installer. You can use pip to install packages from the Python Package Index and other indexes.
# "Conda is an open source package management system and environment management system that runs on Windows, macOS and Linux. Conda quickly installs, runs and updates packages and their dependencies." Includes installation for Python packages

# install packge numpy
#$ pip install numpy

# install a specific package version
#$ pip install numpy=1.15

#update and upgrade a package
#$ pip install --upgrade numpy

# unistall package
#$ pip uninstall numpy

# with conda installation is similar to pip
#$ conda install numpy

# Follow Pandas installation
#https://pandas.pydata.org/pandas-docs/stable/install.html

# Install matplotlib
#https://matplotlib.org/3.1.1/users/installing.html

# Follow Scikit-Learn installation
#https://scikit-learn.org/stable/install.html

# Follow Jupyter notebook installation
#https://jupyter.readthedocs.io/en/latest/install.html

# Recommended Editors: Spyder or ATOM


import numpy as np
import random
import os
import sys
import matplotlib.pyplot as  plt
import pandas as pd
from pandas import read_table, read_csv
import seaborn as sns



# Import the PIMA INDIAN DIABETES DATASET
# Retrieved from Kaggle: https://www.kaggle.com/uciml/pima-indians-diabetes-database
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
# The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.
# Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.
# Task: Can you build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?

os.chdir("../datasets/")
# read the csv table using Pandas
pima = read_csv("diabetes.csv", sep=",")
type(pima)
# get the the first 5 rows of the table
pima.head()
# name of the columns
pima.columns
# quick description of the data
pima.info()
# for the outcome verify the number of classes
pima["Outcome"].value_counts()
# Summary of the numerical attributes
pima.describe()
# look at a specific attribute
pima.describe()["BMI"]
pima.describe()["Age"]

# histogram for each numerical attribute
#%matplotlib inline         # for jupyter notebooks
pima.hist(bins=50, figsize=(20,15))
plt.show()

# Basic data cleaning and preparation
# Some features seem to be well distributed; others have values that call our attention
pima["BMI"].hist(bins = 50)
# how many cases have zero value
sum(pima["BMI"] == 0)
sum(pima.BMI == 0)
# Select the cases where the BMI values are zero
pima[(pima.BMI==0)]
# lets make an imputation of these values based on the mean value
# of the BMI entries that are different from zero
pima[(pima.BMI!=0)]["BMI"].mean()
# now select the entries in the table by first selecting the rows with 
# criteria BMI == 0 and then the column BMI
pima.loc[pima.BMI==0,"BMI"] = pima[(pima.BMI!=0)]["BMI"].mean()
# apply the same procedure for the other variables
pima.loc[pima.BloodPressure==0,"BloodPressure"] = pima[(pima.BloodPressure!=0)]["BloodPressure"].mean()
pima.loc[pima.Glucose==0,"Glucose"] = pima[(pima.Glucose!=0)]["Glucose"].mean()
pima.loc[pima.Insulin==0,"Insulin"] = pima[(pima.Insulin!=0)]["Insulin"].mean()
pima.loc[pima.SkinThickness==0,"SkinThickness"] = pima[(pima.SkinThickness!=0)]["SkinThickness"].mean()


# pima.iloc[pos] does selection by position
# pima.iloc[0,:] first row
# pima.iloc[0,0]
# pima.loc[idx] does selection by index

# Select all cases where Glucose is higher than 100
pima.loc[pima.Glucose>100,:]
pima.loc[pima.Glucose>100,:].shape

# Select outcomes for cases where Glucose is higher than 100 and BMI less than average
pima.loc[(pima.Glucose>100) & (pima.BMI < pima.BMI.mean()),"Outcome"].value_counts()
# select cases and make scatter plot
pima.loc[(pima.Glucose>100) & (pima.BMI < pima.BMI.mean()),["Age","Insulin"]].plot.scatter(x = "Age", y="Insulin")


# Look for correlations between the variables
pima.corr()
# sort correlation values by BMI
pima.corr()["BMI"].sort_values(ascending=False)
pima.corr()["Outcome"].sort_values(ascending=False)

#visualize correlations using heatmap
fig, ax = plt.subplots()
heatmap = plt.pcolor(pima.corr())
cbar = plt.colorbar(heatmap)
# We want to show all labels
ax.set_xticks(np.arange(len(pima.columns)))
ax.set_yticks(np.arange(len(pima.columns)))
ax.set_xticklabels(pima.columns)
ax.set_yticklabels(pima.columns)
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.show()

# Plot selected numerical attributed against every other
# use pandas scatter_matrix function
from pandas.plotting import scatter_matrix
attributes = ["BMI", "Glucose","Age","Insulin"]
scatter_matrix(pima[attributes], figsize=(12,8))

# plot the Outcome w.r.t Glucose
pima.plot(kind ="scatter", x = "Outcome", y = "Glucose", alpha=0.1)

# Include Seaborn for some graphics
sns.set(style = "whitegrid")

# scatter plot: joint distribution of two variables
sns.relplot(x = "BMI", y = "Glucose", data = pima);

# Create count plot with region on the y-axis
sns.countplot(data = pima, x = "Outcome")

# add information of the outcome on the previous scatter plot
sns.relplot(x = "BMI", y = "Glucose", hue="Outcome" , data = pima);

# add semantic variable changes the size of each point according to the number of pregnancies
sns.relplot(x = "BMI", y = "Glucose", hue = "Outcome" , data = pima, size = "Pregnancies");

# boxplots of BMI grouped by outcome 
sns.boxplot(y="BMI",  x = "Outcome", palette=["m", "g"], data = pima)

# check the mean values;
print(" BMI for 0 ", pima.loc[pima.Outcome == 0,"BMI"].mean(), " BMI for 1: ", pima.loc[pima.Outcome == 1,"BMI"].mean())

# create a violiplot instead of boxplot
sns.violinplot(data=pima[["Glucose","Insulin"]], inner="points", size=(14,8))

# jointplots are also useful for better understanding point density
sns.jointplot(x = "Glucose", y = "Insulin", data = pima, color="b")
sns.jointplot(x = "Glucose", y = "Insulin", data = pima, color="b", kind = "reg")

#################################################################################
# BUILD A SIMPLE PREDICTIVE MODEL
from sklearn.model_selection import train_test_split
#################################################################################
# create a train/test dataset
SEED = 123
test_size = 0.3

X = pima.iloc[:,0:7]
Y = pima.iloc[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state = SEED)

# test the proportion of each Outcome class in the datasets
y_train.value_counts() / len(y_train) * 100
y_test.value_counts() / len(y_test) * 100

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state = SEED, stratify=Y)

# 1) You got the data and explored it
# 2) Clean the data (basic pre-processing)
# 3) create a train and test dataset
# Now select and train a ML model
# Two models: Instance-based: KNN and Model-based Logistic Regression

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import  KNeighborsClassifier

# create and train a LR model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# prediction probability for each class
log_reg.predict_proba(X_train)
# prediction of the training set
log_reg_preds_train = log_reg.predict(X_train)
# calculate the percentage of cases correctly classified
log_reg_acc_train = sum(log_reg_preds_train == y_train)/ len(y_train)

# prediction of the test set
log_reg_preds_test = log_reg.predict(X_test)
# calculate the percentage of cases correctly classified
log_reg_acc_test = sum(log_reg_preds_test == y_test)/ len(y_test)


# create and train a KNN model
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
# prediction of the training set
knn_preds_train = knn.predict(X_train)
# calculate the percentage of cases correctly classified
knn_acc_train = sum(knn_preds_train == y_train)/ len(y_train)
# prediction of the test set
knn_preds_test = knn.predict(X_test)
# calculate the percentage of cases correctly classified
knn_acc_test = sum(knn_preds_test == y_test)/ len(y_test)


print ("Logistic Regression Train Accuracy: %.3f" % log_reg_acc_train)
print ("Logistic Regression Test Accuracy: %.3f" % log_reg_acc_test)
print ("KNN Train Accuracy: %.3f" % knn_acc_train)
print ("KNN Test Accuracy: %.3f" % knn_acc_test)


# Better evaluation using Cross-validation
from sklearn.model_selection import cross_val_score
log_reg_scores = cross_val_score(log_reg, X, Y, scoring="accuracy", cv = 10)
knn_scores = cross_val_score(knn, X, Y, scoring="accuracy", cv = 10)

print ("Logistic Regression Accuracy Avg: %.3f  Std: %.3f" % (log_reg_scores.mean()*100, log_reg_scores.std()*100))
print ("KNN Accuracy Avg: %.3f  Std: %.3f" % (knn_scores.mean()*100, knn_scores.std()*100))
