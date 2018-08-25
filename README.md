# Decision Tree Post Pruner (Sci-kit Learn Extension)
James Chan 2017

### Overview
At the time this was written (9/12/17) Sci-kit Learn's DecisionTreeClassifier does not support error-reduction post pruning, which is an effective way to reduce overfitting and potentially improve testing accuracy. My code takes advantage of the existing decision tree data structure once the tree has been created.

### Post Pruning Algorithm
The decision tree in the examples below uses Gini as split criterion with a leaf size of 1.  The post pruning algorithm used is taken from Machine Learning (1997) by Tom Mitchell.  
<img src="https://raw.githubusercontent.com/jameschanx/Decision_Tree_Post_Pruner-Scikit_Extension/master/before_after_prune.png"/>
<center> Figure 1. Before Post-Pruning vs After Post-Pruning</center>


The algorithm is very simple, basically while improvement in training accuracy is 
observed, continue to remove leaves.
## 

### Assess Post Pruner
#### Train Test Split
The learning curves below compare the out-of-sample accuracy before and after post-pruning.  Each training size interval takes the average accuracy between 32 trials with random shuffling. 

<div style="text-align:right"><img src="https://raw.githubusercontent.com/jameschanx/Decision_Tree_Post_Pruner-Scikit_Extension/master/train_test_split.png" style="height: 400px" /></div>

<center> Figure 2. Train Test Split Scheme</center>

To assess accurately the effect of pruing, we need to take two separate test sets.  One will be tested on the non-pruned tree, and the other will be tested on the post-pruned tree. As mentioned previously, we shuffle our data and average the results of multiple trials in order to get an unbiased assessment of post-pruning results 

#### Pruning Results Visualized

<img src="https://raw.githubusercontent.com/jameschanx/Decision_Tree_Post_Pruner-Scikit_Extension/master/abalone.png" align="left"/>

#### Abalone Dataset
Estimate the age of abalone based on features such as size, gender, and weight

Source: https://archive.ics.uci.edu/ml/datasets/abalone
#### Remark:
The accuracy after post-pruning is about 1.0 to 1.5% higher than a tree that hasn't been pruned.
###### &nbsp;
## 

<img src="https://raw.githubusercontent.com/jameschanx/Decision_Tree_Post_Pruner-Scikit_Extension/master/wine.png" align="left"/>

#### Wine Dataset
Estimate the quality of red wine on a scale of 1-10 assigned by people.  Input features include alcohol content, malic acid content, and color intensity.

Source: https://archive.ics.uci.edu/ml/datasets/wine
#### Remark:
The accuracy after post-pruning is about 0.0 to 2.0% higher than a tree that hasn't been pruned.
###### &nbsp;
## 

<img src="https://raw.githubusercontent.com/jameschanx/Decision_Tree_Post_Pruner-Scikit_Extension/master/diabetes.png" align="left"/>

#### Pima Indian Diabetes Dataset
Predict whether subject has diabetes based on predictor such as blood pressure, BMI, and age

Source: https://www.kaggle.com/uciml/pima-indians-diabetes-database
#### Remark:
The accuracy after post-pruning is about 1.0 to 2.0% than a tree that hasn't been pruned.
###### &nbsp;
## 

<img src="https://raw.githubusercontent.com/jameschanx/Decision_Tree_Post_Pruner-Scikit_Extension/master/iris.png" align="left"/>

#### Iris Dataset
Predict the type of Iris base on features such as sepal length, petal length, and petal width.

Source: https://archive.ics.uci.edu/ml/datasets/iris
#### Remark:
There is no visible improvement to the accuracy after post-pruning.

###### &nbsp;
###### &nbsp; 
### Conclusion
Post-pruning has the desired property that it very often reduces both the variance and the bias in a decision tree learner.  It should be considered whenever a decision tree algorithm is used.