"""
@author: James
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from copy import deepcopy
#from dt_analysis import DTC_validation_curve, DTC_learn_curve
#from dt_analysis import RFC_validation_curve, RFC_learn_curve

def error_reduction_pruning(clf, X_test, y_test):
    #by James Chan, jchan70@gatech.edu, 8/28/17
    clf2 = deepcopy(clf)
    current_score = clf2.score(X_test, y_test)
    temp_structure = clf2.tree_.__getstate__()
    temp_table = temp_structure['nodes']
    temp_values = temp_structure['values']
    
    pruning_occurred = True
    while(pruning_occurred):
        pruning_occurred = False
        for row, i in zip(temp_table, range(temp_values.shape[0])):
            if row[0] != -1:
                lc_index = row[0]
                rc_index = row[1]
                lc = temp_table[lc_index]
                rc = temp_table[rc_index]
                #see if both children are leaves
                if lc[0] == -1 and rc[0] == -1:
                    #keep a copy of the original values for later to restore if needed
                    original_values = temp_values[i].copy()
                    original_left_child = row[0]
                    original_right_child = row[1]
                    #collapse all samples into the category with highest sample count
                    num_samples = sum(temp_values[i,0])
                    highest_samples_index = np.argmax(temp_values[i,0])
                    temp_values[i,0] = temp_values[i,0] * 0
                    temp_values[i,0, highest_samples_index] = num_samples
                    #set things in row to make it a leaf
                    row[0] = -1
                    row[1] = -1
                    #check if score improved, if so, update score, if not, change it back.  
                    clf2.tree_.__setstate__(temp_structure)
                    new_score = clf2.score(X_test, y_test)
                    if new_score > current_score:
                        #print('pruned node {}, new score is {}, samples count {}'.format(i, new_score, num_samples))
                        pruning_occurred = True
                        current_score = new_score
                    else:
                        temp_values[i] = original_values.copy()
                        row[0] = original_left_child
                        row[1] = original_right_child
                        clf2.tree_.__setstate__(temp_structure)
    clf.tree_.__setstate__(temp_structure)
    return clf
    
def evaluate_pruner(X, y, title):
    test_scores = []
    pruned_scores = []
    
    training_sizes = [i for i in range(10,101)]
    for ts in training_sizes:
        print(ts)
        acc = []
        acc_pruned = []
        #average of n trials
        for i in range(32):
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = .2)
            X_train, X_prune, Y_train, Y_prune = train_test_split(X_train, Y_train, test_size = .25)
            X_batch = X_train[:int(ts/100 * X_train.shape[0])]
            Y_batch = Y_train[:int(ts/100 * Y_train.shape[0])]    
        
            dt = DTC()
            dt.fit(X_batch, Y_batch)
            testing_score = dt.score(X_test, Y_test)
            dt = error_reduction_pruning(dt, X_prune, Y_prune)
            post_pruning_score = dt.score(X_test, Y_test)
            acc.append(testing_score)
            acc_pruned.append(post_pruning_score)
        
        #average scores
        test_scores.append(np.mean(acc))
        pruned_scores.append(np.mean(acc_pruned))
    
    plt.figure()
    plt.plot(training_sizes, test_scores, label='Out-of-sample, Before Pruning')
    plt.plot(training_sizes, pruned_scores, label='Out-of-sample, After Pruning')
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlabel('Training Size [%]')
    
    
if __name__=="__main__":
    #load data
    X_abalone = pd.read_hdf('data/X_abalone.hdf')
    X_abalone_scaled = scale(X_abalone)
    Y_abalone = pd.read_hdf('data/y_abalone.hdf')
    
    X_wine = pd.read_hdf('data/X_wine.hdf')
    X_wine_scaled = scale(X_wine)
    Y_wine = pd.read_hdf('data/y_wine.hdf')
    
    X_diabetes = pd.read_hdf('data/X_diabetes.hdf')
    X_diabetes_scaled = scale(X_diabetes)
    Y_diabetes = pd.read_hdf('data/y_diabetes.hdf')
    
    X_titanic = pd.read_hdf('data/X_titanic.hdf')
    y_titanic = pd.read_hdf('data/y_titanic.hdf')
    
    X_iris = pd.read_hdf('data/X_iris.hdf')
    X_iris_scaled = scale(X_iris)
    y_iris = pd.read_hdf('data/y_iris.hdf')

    evaluate_pruner(X_abalone_scaled, Y_abalone, 'Abalone')    
    evaluate_pruner(X_wine_scaled, Y_wine, 'Wine')    
    evaluate_pruner(X_diabetes_scaled, Y_diabetes, 'Pima Indian Diabetes')    
#    evaluate_pruner(X_titanic, y_titanic)    
    evaluate_pruner(X_iris_scaled, y_iris, 'Iris')    
