#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:13:24 2022
@author: Pedro G. Ferreira
Target: Students of AI and Data Science
Description: Demonstration of several data pre-processing steps, including:
- selection and filtering
- plotting data for exploratory analysis
- conversion and variable encoding
- scaling and standardization
"""
import pandas as pd
import numpy
import os
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Change to directory with the dataset and load the dataset diamonds2.csv
os.chdir('/yourdirectory/datasets/')
dmd = pd.read_csv("diamonds2.csv")

# 2. Change the name of the very first columne do idx and set this variable as the index of the dataset
# Tip:first get the columns names into a list, change the first element of the list to "idx" and set the index with set_index()

# get the columns name
cols = list(dmd.columns)
#change the first col name and set as index
cols[0] = "idx"
dmd.columns = cols
dmd.set_index("idx", inplace=True)


# 3. Check how many and which features have non-null values
# use the info() function or sum all the isnull() values per column 
dmd.info()
dmd.cut.isnull().sum()


# 4. Create a subset of your original dataset with only the attributes: carat, cut, depth, price
# Tip. Create a list with the selected features. 
# df[['width','length','species']] --> Select multiple columns with specific names.
dmd[["carat", "cut","depth", "price"]]
dmd.loc[:,["carat", "cut","depth", "price"]]

# 5. Use the function pairplot() in seaborn to plot all features in pairwise manner.
# use the hue parameter to set the colors according to the value of cut
# set the parameter vars as the list of variables that you want to plot except the cut feature
# ["carat", "depth", "price"]
sns.pairplot(dmd.loc[:,["carat", "cut","depth", "price"]], vars=["carat", "depth", "price"], hue="cut")

# 6. check if there are negative values
sum(dmd.price < 0)
sum(dmd.x < 0)
sum(dmd.y < 0)
sum(dmd.z < 0)
sum(dmd.depth < 0)
sum(dmd.carat < 0)
sum(dmd.table < 0)


# 7. set the variables that are negative to null values
#  use the float("NaN")
# make the selection of cases with negative values using .loc
# e.g. df.loc[df['a'] > 10, ['a','c']]
# Select rows meeting logical condition, and only the specific columns 
dmd.loc[dmd.price < 0, "price"] = float("NaN")
dmd.loc[dmd.x < 0, "x"] = float("NaN")
dmd.loc[dmd.y < 0, "y"] = float("NaN")
dmd.loc[dmd.z < 0, "z"] = float("NaN")

# 8. set the null values to the mean of the features excluding the null values
# i) test all entries in the feature that are false, 
# use the pandas isna() function to test e.g. test === pd.isna(df.feat) == False
# ii) select the cases where the above test holds using .loc[], e.g. df.loc[test, "feat"]
# iii) apply the mean function to the resulting column
dmd.loc[pd.isna(dmd.x) == False,"x"].mean()
# iv) set this value where the test above is different to False
dmd.loc[pd.isna(dmd.x) != False,"x"] = dmd.loc[pd.isna(dmd.x) == False,"x"].mean()
dmd.loc[pd.isna(dmd.y) != False,"y"] = dmd.loc[pd.isna(dmd.y) == False,"y"].mean()
dmd.loc[pd.isna(dmd.z) != False,"z"] = dmd.loc[pd.isna(dmd.z) == False,"z"].mean()
dmd.loc[pd.isna(dmd.price) != False,"price"] = dmd.loc[pd.isna(dmd.price) == False,"price"].mean()


# 9. for the missing values in cut use the most frequent values
# from the cases with non missing values
# i) use the function value_counts() to find the most frequent value
# ii) set the feature cut with na value different than false to the most frequent value
dmd.loc[pd.isna(dmd.cut) != False,"cut"] = "Ideal"


# 10. redo the figure in step 5 and save to file fig.pdf
# tip use the savefig function from matplotlib.pyplot
sns.pairplot(dmd.loc[:,["carat", "cut","depth", "price"]], vars=["carat", "depth", "price"], hue="cut")
plt.savefig('fig.pdf')


# 11. Check if the feature clarity needs processing
# convert all the valus to upper case. Use the function upper() applied to strings, e.g. x.upper()
# create a list of all the values in clarity converted to upper case
# use list comprehension e.g. [x.upper() for x in dmd.clarity]
# set the column clarity with this new value
dmd.clarity = [x.upper() for x in dmd.clarity]
# recheck that all values are in upper case (use the value_counts() applied to the column)


# 12. encode the variable cut as a numeric value; create a new feature
# called cut_num as the numeric encoding of cut
# use the function LabelEncoder from sklearn.preprocessing
# check the sklearn cheat sheet for an example
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
dmd["cut_num"] = enc.fit_transform(dmd.cut)


# 13. encode the variable cut_num as a one hot encondig value; set a new feature called cut_ohe
# use the function OneHotEncoder from sklearn.preprocessing
# first use the fit function on the OneHotEncoder object
# use reshape(-1, 1) in the selected column, e.g. df.cut_num.reshape(-1, 1)
# apply the transform function on the OneHotEncoder object
# use the toarray() to obtain the converted matrix
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
dmd_cut_reshaped = dmd.cut_num.to_numpy().reshape(-1, 1)
ohe.fit(dmd_cut_reshaped)
cut_ohe = ohe.transform(dmd_cut_reshaped).toarray()


# 14. Calculate a new feature called price_zscore as the standard Z-score of the feature price
dmd["price_zscore"] = (dmd.price - dmd.price.mean())/dmd.price.std()

# 15. Calculate a new feature called price_min_max as the min max scaling of the feature price
# check its range visualizing as an histogram
dmd["price_min_max"] = (dmd.price - dmd.price.min())/(dmd.price.max() - dmd.price.min())
dmd["price_min_max"].hist()

