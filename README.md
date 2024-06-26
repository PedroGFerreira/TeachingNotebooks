Notebooks and scripts used for teaching Machine Learning and Data Science in different courses at BSc, MSc and PhD. 

# Scripts4
- **diamond_preprocess.py**: perform some data pre-processing steps, including normalization of the data, imputation, feature selection and creation and data visualization for exploration

- **decision_tree.py**: create a simple decision tree model for a classifier to predict the "cut" of the diamond: "Fair", "Good", "Very Good", "Ideal", "Premium".

- **Auto.ipynb**: notebook to explore the dataset, visualize the data and build simple predictive models to predict origin of the car: european, USA, or Asian.

- **classifier_baseline_pima.**: script to explore the dataset, visualize the data and build two simple predictive models. Note: the PIMA Diabetes dataset is used here. This dataset contains sensitive health data. Although this is a public dataset, its use and analysis should be done taking into account the sensitive nature of this data, keeping in mind that it refers to real people.

- **scaling_pca_hc.py**: script to perform unsupervised learning. Principal Component Analysis with 2 implementations. Clustering: k-means and Hierarchical Clustering. Impact of normalization and scaling of data is tested.

- **classifier_comparison.py**: script comparing the results of a classifier with the baseline estimator; building a confusion matrix; roc curve; multiclass classification; creating a pre-processing pipeline; logistic regression, decision tree, svm, random forests classifiers.



# Data

Note that these datasets are not not my property and have not been generated by me.
These are open-source datasets that are provided here for convenience of reproducibility while running the datasets.

**Please, acknowledge properly the source and authors of this data!**


## Auto
Source: This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University. The dataset was used in the 1983 American Statistical Association Exposition. The original dataset has 397 observations, of which 5 have missing values for the variable "horsepower". These rows are removed here. The original dataset is avaliable as a CSV file in the docs directory, as well as at https://www.statlearning.com.
References: James, G., Witten, D., Hastie, T., and Tibshirani, R. (2013) An Introduction to Statistical Learning with applications in R, https://www.statlearning.com, Springer-Verlag, New York

## Cars93
Details: These cars represent a random sample for 1993 models that were in both Consumer Reports and PACE Buying Guide. Only vehicles of type small, midsize, and large were include.
Further description can be found in Lock (1993). Use the URL http://jse.amstat.org/v1n1/datasets.lock.html.
Source: Lock, R. H. (1993) 1993 New Car Data. Journal of Statistics Education 1(1).

## College
Details: This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University. The dataset was used in the ASA Statistical Graphics Section's 1995 Data Analysis Exposition.
Source: James, G., Witten, D., Hastie, T., and Tibshirani, R. (2013) An Introduction to Statistical Learning with applications in R, https://www.statlearning.com, Springer-Verlag, New York

## Default
Details: This research aimed at the case of customers' default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods.
Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Introductory Paper: The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. By I. Yeh, Che-hui Lien. 2009 Published in Expert systems with applications

## Diamonds
Details: Analyze diamonds by their cut, color, clarity, price, and other attributes. ~54K cases and 10 attributes
Source: https://www.kaggle.com/datasets/shivam2503/diamonds


## Hitters
Details:This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University. This is part of the data that was used in the 1988 ASA Graphics Section Poster Session. The salary data were originally from Sports Illustrated, April 20, 1987. The 1986 and career statistics were obtained from The 1987 Baseball Encyclopedia Update published by Collier Books, Macmillan Publishing Company, New York.
Source:James, G., Witten, D., Hastie, T., and Tibshirani, R. (2013) An Introduction to Statistical Learning with applications in R, https://www.statlearning.com, Springer-Verlag, New York

