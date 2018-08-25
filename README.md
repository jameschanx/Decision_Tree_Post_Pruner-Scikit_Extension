﻿# Decision Tree Post Pruner (Sci-kit Learn Extension)
James Chan 2017

### Overview
At the time this was written (9/12/17) Sci-kit Learn's DecisionTreeClassifier does not support error-reduction post pruning, which is an effective way to reduce overfitting and potentially improve testing accuracy. The decision tree in the examples below uses Gini as split criterion with a leaf size of 1.  

### Pruning - Before vs. After
The learning curves below compare the out-of-sample accuracy before and after post-pruning.  Each training size interval takes the average accuracy between 32 trials with random shuffling.  

<img src="https://raw.githubusercontent.com/jameschanx/Decision_Tree_Post_Pruner-Scikit_Extension/master/abalone.png" align="left" style="height: 300px"/>

#### Abalone Dataset
Estimate the age of abalone based on features such as size, gender, and weight

Source: https://archive.ics.uci.edu/ml/datasets/abalone
#### Remark:
The accuracy after post-pruning is about 1.0 to 1.5% higher than a tree that hasn't been pruned.
###### &nbsp;
## 

<img src="https://raw.githubusercontent.com/jameschanx/Decision_Tree_Post_Pruner-Scikit_Extension/master/wine.png" align="left" style="height: 300px"/>

#### Wine Dataset
Estimate the quality of red wine on a scale of 1-10 assigned by people.  Input features include alcohol content, malic acid content, and color intensity.

Source: https://archive.ics.uci.edu/ml/datasets/wine
#### Remark:
The accuracy after post-pruning is about 0.0 to 2.0% higher than a tree that hasn't been pruned.
###### &nbsp;
## 

<img src="https://raw.githubusercontent.com/jameschanx/Decision_Tree_Post_Pruner-Scikit_Extension/master/diabetes.png" align="left" style="height: 300px"/>

#### Pima Indian Diabetes Dataset
Predict whether subject has diabetes based on predictor such as blood pressure, BMI, and age

Source: https://www.kaggle.com/uciml/pima-indians-diabetes-database
#### Remark:
The accuracy after post-pruning is about 1.0 to 2.0% than a tree that hasn't been pruned.
###### &nbsp;
## 

<img src="https://raw.githubusercontent.com/jameschanx/Decision_Tree_Post_Pruner-Scikit_Extension/master/iris.png" align="left" style="height: 300px"/>

#### Iris Dataset
Predict the type of Iris base on features such as sepal length, petal length, and petal width.

Source: https://archive.ics.uci.edu/ml/datasets/iris
#### Remark:
There is no visible improvement to the accuracy after post-pruning.

###### &nbsp;
###### &nbsp; 
### Conclusion
Post-pruning has the desired property that it is likely to reduce both the variance and the bias in a decision tree learner.  It should be considered whenever a decision tree algorithm is used.