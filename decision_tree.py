#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:13:24 2022
@author: Pedro G. Ferreira
Target: Students of AI and Data Science
Description: Demonstration of building a simple predictive model based on decision trees. Predict the "cut" variable
- train/test split
- model training and test evaluatiom
- accuracy calculation
"""
# 1. load modules
import pandas as pd
import numpy
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn 
from sklearn.model_selection import train_test_split

# 2. Change to directory with the dataset and load the dataset diamonds2.csv
os.chdir('/yourdirectory/datasets/')
df = pd.read_csv("diamonds.csv", sep=",")


# 3. Create a new dataset with the following features Carat, Price, Cut (keep order of features)
dmd = df[["carat", "price", "cut"]]


#################################################################################
# BUILD A SIMPLE PREDICTIVE MODEL
#################################################################################
#3.5
X = dmd[["carat", "price"]]
Y = dmd["cut"]


# 3.6 Apply the train_test_split to create a train and a test dataset
# i) create a SEED value e.g. SEED = 123
# ii) use the parameter stratify as True. What is the meaning of this parameter?
# iii) Create a 70/30 division test_size=0.3
test_size = 0.3
SEED = 123
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state = SEED, stratify=Y)


# 3.7 calculate the distribution of the output label (target) in the test and train sets; use valu_counts() compare the 
# distribution
y_train.value_counts()
y_test.value_counts()


# 3.8 create a decision tree model and test its accurracy in the train and the test set
# import DecisionTreeClassifier from sklearn.tree
# with max_leaf_nodes = 3  and the criterion="gini" and random_state = 0 
from sklearn.tree import DecisionTreeClassifier
dt3_gini = DecisionTreeClassifier(max_leaf_nodes = 3, random_state=0, criterion="gini")


# 3.9 use the fit() function from the created DT
# This allows to train the model
dt3_gini.fit(X_train, y_train)


# 3.10 prediction on the training set
# use the predict() function; apply it to the X_train dataset
dt3_gini_preds_train = dt3_gini.predict(X_train)

# 3.11 calculate the percentage of cases correctly classified
# compare the resulting vector (predicted) with the vector of actual values (y_train)
# sum all the positive cases and divide by the total number of cases (length of y_train)
# This will yield the accurracy
dt3_gini_acc_train = sum(dt3_gini_preds_train == y_train)/ len(y_train)

# 3.12 repeat the previous step for the test dataset

# prediction of the test set
dt3_gini_preds_test = dt3_gini.predict(X_test)
# calculate the percentage of cases correctly classified
dt3_gini_acc_test = sum(dt3_gini_preds_test == y_test)/ len(y_test)


# 3.13 Plot the train and test accurracy
print ("Decision Tree Train Accuracy Gini: %.3f" % dt3_gini_acc_train)
print ("Decision Tree Test Accuracy Gini: %.3f" % dt3_gini_acc_test)






