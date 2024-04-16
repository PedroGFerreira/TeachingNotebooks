!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:18:18 2019

@author: Pedro G: Ferreira
"""

import numpy as np
import random
import os
import sys
import matplotlib.pyplot as  plt
import pandas as pd
from pandas import read_table, read_csv
import seaborn as sns


from sklearn.model_selection import train_test_split, cross_val_score


os.chdir("../datasets/")
# read the csv table using Pandas
default = read_csv("Default.txt", sep="\t")
# 3 input variables and 1 output variable: default

##############################################################################
# SECTION 1: COMPARING THE RESULT OF A CLASSIFIER WITH THE BASELINE ESTIMATOR
##############################################################################
# create a train/test dataset
SEED = 123
test_size = 0.3

from sklearn.preprocessing import LabelEncoder
default_encoded = default
encoder = LabelEncoder()
default_encoded["default"] = encoder.fit_transform(default["default"])
default_encoded["student"] = encoder.fit_transform(default["student"])

X = default_encoded.iloc[:,1:4]  # input data
Y = default_encoded.iloc[:,0]    # labels
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state = SEED)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state = SEED)

sgd_clf_cv_scores = cross_val_score(sgd_clf, X_train, y_train, scoring="accuracy", cv = 10, verbose = 0)
np.mean(sgd_clf_cv_scores)
# Out[64]: 0.9190616715544317

# DummyClassifier is a classifier that makes predictions using simple rules.
# This classifier is useful as a simple baseline to compare with other (real) classifiers. Do not use it for real problems.
# “stratified”: generates predictions by respecting the training set’s class distribution.
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy = "stratified")
dummy_cv_scores = cross_val_score(dummy, X_train, y_train, scoring="accuracy", cv = 10)
np.mean(dummy_cv_scores)
# Out[65]: 0.9664295233255578

##############################################################################
# SECTION 2: BUILD A CONFUSION MATRIX
##############################################################################
from sklearn.model_selection import cross_val_predict
# cross_val_predict performs k.fold cross validation but instead of returning
# scores returns the predictions made on each test fold
# you can get a prediction for each instance in the training set
y_train_predictions = cross_val_predict(sgd_clf, X_train, y_train, cv = 10)



from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_train_predictions)
# each row represents actual class / true label
# each column represents predicted class / label


from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_train, y_train_predictions)
recall = recall_score(y_train, y_train_predictions)
f1 = f1_score(y_train, y_train_predictions)
print ("Precision %.3f; Recall %.3f; F1 %.3f" % (precision, recall, f1))


##############################################################################
# SECTION 3: ROC CURVE
##############################################################################
from sklearn.metrics import roc_curve, roc_auc_score

# decision_function: Predict confidence scores for samples.
# the decision_function returns  the distance between the hyperplane defined
# by the classifier and the training instances
y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv = 10, method="decision_function")
fpr, tpr, thresholds = roc_curve(y_train, y_scores)

lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label="SGD")
#plt.plot((fpr-0.1), (tpr-0.1),label='other', color='navy', linestyle=':', linewidth=4) # add another classifier scores
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

roc_auc_score(y_train, y_scores)

##############################################################################
# SECTION 4 : MULTICLASS CLASSIFICATION ON IRIS DATASET
##############################################################################
from sklearn.datasets import load_iris
iris = load_iris()

# input data 
iris.data
# features names
iris.feature_names
# output/target variable
iris.target
# output/target variable names
list(iris.target_names)


# train a multi-class classifier with SGD on the iris datasetw
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_size, random_state = SEED)
sgd_clf = SGDClassifier(random_state = SEED)
# sklearn automatically detects multiclasses and runs OvA or OvO (for SVMs)
# it creates 3 binary classifiers
sgd_clf.fit(X_train, y_train)

# among the 105 training cases; select one random case to predict
rnd_case = 104
# get the predicted class
predicted_class = sgd_clf.predict(X_train[rnd_case,:].reshape(1, -1) )
# get the decision scores
predicted_scores = sgd_clf.decision_function(X_train[rnd_case,:].reshape(1, -1) )
# the predicted class is the one with highest decision score
np.argmax(predicted_scores)

# plot the errors across all classes
y_train_predictions = cross_val_predict(sgd_clf, X_train, y_train)
conf_mat = confusion_matrix(y_train, y_train_predictions)
conf_mat = pd.DataFrame(conf_mat)
conf_mat.index.name = 'Actual'
conf_mat.columns.name = 'Predicted'
sns.set(font_scale=1.4)#for label size
sns.heatmap(conf_mat, cmap="Blues", annot=True,annot_kws={"size": 16})# font size


##############################################################################
# SECTION 5 : CREATE A PRE-PROCESSING PIPELINE FOR THE PIMA DATASET
##############################################################################

pima = read_csv("diabetes.csv", sep=",")
# convert the zero values registering absence as NAN
pima.loc[pima.BMI==0,"BMI"] = np.nan
pima.loc[pima.BloodPressure==0,"BloodPressure"] = np.nan
pima.loc[pima.Glucose==0,"Glucose"] = np.nan
pima.loc[pima.Insulin==0,"Insulin"] = np.nan
pima.loc[pima.SkinThickness==0,"SkinThickness"] = np.nan


# number of NANs per row
for i in range(len(pima.index)) :
    print("Nan in row ", i , " : " ,  pima.iloc[i].isnull().sum())

# make imputation of NA values based on the median values
from sklearn.preprocessing import Imputer
imp = Imputer(strategy = "median")
pima_trans = imp.fit_transform(pima)
# number of NANs per row
for i in range(pima_trans.shape[0]):
    print("Nan in row ", i , " : " ,  np.isnan(pima_trans[i,:]).sum())


# define the Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

numeric_pipeline = Pipeline (
        [ ("imputer", SimpleImputer(strategy = "median")),
         ("std_scale", StandardScaler())
        ]
        )


# define the input variables and the input values matrix
input_attribs = pima.columns[0:8]
output = pima.columns[8]
pima_input = pima.loc[:,input_attribs]

# apply the pipeline to transform the data
pima_input_processed = numeric_pipeline.fit_transform(pima_input)
pima_input_processed = pd.DataFrame(pima_input_processed)
pima_input_processed.columns = input_attribs


##############################################################################
# SECTION 6 : Logistic Regression Classifier for Default dataset
##############################################################################
# see Section 1 to load the dataset
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

X = default_encoded.iloc[:,1:4]  # input data
Y = default_encoded.iloc[:,0]    # labels
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state = SEED)

sns.boxplot(y="balance",  x = "default", palette=["m", "g"], data = default)
# train the model based on the income input feature
log_reg.fit(X_train["balance"].values.reshape(-1, 1), y_train)
lr_scores = cross_val_score(log_reg, X_train, y_train, scoring="accuracy", cv = 10)


#plot estimated probabilities for clients with  income  0 to 7000
X_new = np.linspace(0, 2500, 100).reshape(-1,1)
y_prob = log_reg.predict_proba(X_new)
plt.plot(X_new, y_prob[:,1], "g-", label="Not Default")
plt.plot(X_new, y_prob[:,0], "b--", label="Default")
plt.xlabel('Balance')
plt.ylabel('Class Probability')

##############################################################################
# SECTION 7: Decision Tree for Iris Dataset
##############################################################################
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=test_size, random_state = SEED)

tree_clf = DecisionTreeClassifier(max_depth = 2)
tree_clf.fit(X_train, y_train)
tree_scores = cross_val_score(tree_clf, X_train, y_train, scoring="accuracy", cv = 10)

graph = export_graphviz(tree_clf, out_file="iris_tree.dot", feature_names=iris.feature_names, class_names = iris.target_names, rounded=True, filled = True)
# open the iris_tree.dot
# remove the \n
# paste in http://www.webgraphviz.com/

# test the tree with different maxdepth values

y_train_predictions = cross_val_predict(tree_clf, X_train, y_train)
conf_mat = confusion_matrix(y_train, y_train_predictions)
conf_mat = pd.DataFrame(conf_mat)
conf_mat.index.name = 'Actual'
conf_mat.columns.name = 'Predicted'
sns.set(font_scale=1.4)#for label size
sns.heatmap(conf_mat, cmap="Blues", annot=True,annot_kws={"size": 16})# font size


##############################################################################
# SECTION 8: Random Forests
##############################################################################
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=SEED)

rf_clf.fit(X_train, y_train)
rf_scores = cross_val_score(rf_clf, X_train, y_train, scoring="accuracy", cv = 10)

# View a list of the features and their importance scores
# Get numerical feature importances
importances = list(rf_clf.feature_importances_)
# List of tuples with feature and its importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(iris.feature_names, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
sns.barplot(x=importances, y = iris.feature_names,label="Feature Importance")

# HYPER PARAMETER GRID SEARCH
for n_estimators in [50, 100, 200, 500, 1000]:
    rf_clf = RandomForestClassifier(random_state=SEED, n_estimators = n_estimators)
    rf_scores = cross_val_score(rf_clf, X_train, y_train, scoring="accuracy", cv = 10)
    print("Num estimators: %.4f Accuracy: %.4f" %(n_estimators, rf_scores.mean()))
    
for max_depth in [2, 4, 6, 10, 50]:
    rf_clf = RandomForestClassifier(random_state=SEED, n_estimators = 100, max_depth = max_depth)
    rf_scores = cross_val_score(rf_clf, X_train, y_train, scoring="accuracy", cv = 10)
    print("max_depth: %d Accuracy: %.4f" %(max_depth, rf_scores.mean()))
    
for min_samples_leaf in [1, 2, 4, 6, 8, 10]:
    rf_clf = RandomForestClassifier(random_state=SEED, n_estimators = 100, min_samples_leaf = min_samples_leaf)
    rf_scores = cross_val_score(rf_clf, X_train, y_train, scoring="accuracy", cv = 10)
    print("min_samples_leaf: %d Accuracy: %.4f" %(min_samples_leaf, rf_scores.mean()))
   
##############################################################################
# SECTION 9: SVMs
##############################################################################
from sklearn.svm import SVC 
    
# Penalty parameter C of the error term.
for C_param in [0.01, 0.1, 1, 10, 20]:
    svc_clf = SVC(random_state=SEED, C = C_param, gamma = "auto")
    svc_scores = cross_val_score(svc_clf, X_train, y_train, scoring="accuracy", cv = 10)
    print("C %f Accuracy: %.4f" %(C_param, svc_scores.mean()))

# Specifies the kernel type to be used in the algorithm. 
for k in ["linear", "poly", "rbf", "sigmoid"]:
    svc_clf = SVC(random_state=SEED, kernel = k, gamma = "auto", C = 1.0)
    svc_scores = cross_val_score(svc_clf, X_train, y_train, scoring="accuracy", cv = 10)
    print("kernel %s Accuracy: %.4f" %(k, svc_scores.mean()))



