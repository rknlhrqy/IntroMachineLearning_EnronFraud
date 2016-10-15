# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 21:10:54 2016

@author: kenren
"""

#%matplotlib inline

import sys
import pickle
import math
import time
import matplotlib.pyplot
sys.path.append("../tools/")

from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# Open the file and read the dataset
with open("../final_project/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Remote any data where salary is NaN.
data_dict_no_NaN = {name:data_dict[name] for name in data_dict if data_dict[name]["salary"] != "NaN"}

# Define the function to draw the scatterplot of the dataset
# based on the chosen features.
# Red dots in the scatter plot are POI's 
# Blue dots are not POI's
def draw_scatterplot(featurelist, datadict):
    data = featureFormat(datadict, featurelist)
    for point in data:
        if point[0] != 0:
            color = 'red'
        else:
            color = 'blue'
        matplotlib.pyplot.scatter( point[1], point[2], color = color)
    matplotlib.pyplot.xlabel( featurelist[1] )
    matplotlib.pyplot.ylabel( featurelist[2] )
    matplotlib.pyplot.show()
    
# Have a look at the distribution of the dataset
# based on salary and bonus.    
features_list = ['poi','salary', 'bonus']
draw_scatterplot(features_list, data_dict_no_NaN)

# Apparently there exist one outlier from the scatterplot above whose
# salary is more than 2.5e7. Check the document "enron61702insiderpay.pdf"
# and find that it is "Total". And it should be removed from the dataset.
# 
# Find this outlier from the dataset and remove it from the dataset.
def Remove_Max_Outlier(datadict, feature):
    v=list(k[feature] for k in datadict.values())
    k=list(datadict.keys())
    name = k[v.index(max(v))]
    
    del datadict[name]
    return datadict
    
data_dict_no_Outlier = Remove_Max_Outlier(data_dict_no_NaN, "total_payments")

# Draw the scatterplot for the dataset again.
draw_scatterplot(features_list, data_dict_no_Outlier)


# Define a function to set all NaN value of the defined feature as zero.
def NaN_to_Zero(featurelist, datadict):
    for key, value in datadict.items():
        for i in featurelist:
            if type(value[i]) is float and math.isnan(value[i]) or type(value[i]) is str and value[i] == "NaN" :
                datadict[key][i] = 0
    return datadict
    
# List all features.    
features_list = ['poi', 'bonus', 'deferral_payments', 'director_fees', 'exercised_stock_options', \
                 'expenses', 'loan_advances', 'long_term_incentive', 'other', 'restricted_stock', \
                 'salary', 'total_payments', 'total_stock_value']

# Set NaN value of the above features to be zero
data_dict_no_NaN2 = NaN_to_Zero(features_list, data_dict_no_Outlier)

# Apply the calculation of log(1+data) to all financial values in the dataset
# This is to get more "normal" distribution. 
for name in data_dict_no_NaN2:
    data_point = data_dict_no_NaN2[name]
    for i in features_list:
        if i != 'poi':
            data_point[i] = math.log(float(data_point[i]) + 1.0)

# Define the function to calculate the ratio of some messages 
# over all messages.
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    ### this function returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    fraction = 0.
    if poi_messages == "NaN" or all_messages == "NaN" or all_messages == 0:
        return fraction
    fraction = 1.0 * poi_messages / all_messages
    return fraction

# Go through the dataset. Calculate "fraction_from_poi" and 
# "fraction_to_poi". And then add both features into the dataset.

# "fraction_from_poi" is the ratio of the number of messages received
# from a POI by a person over the total number of messages received
# by this person.

# "fraction_to_poi" is the ratio of the number of messages sent to a POI
# from a person over the total number of messages sent by this person.
for name in data_dict_no_NaN2:

    data_point = data_dict_no_NaN2[name]

    #print
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    data_point["fraction_to_poi"] = fraction_to_poi
    
# Draw a scatterplot for the two new features.
features_list = ['poi', 'fraction_from_poi', 'fraction_to_poi']
draw_scatterplot(features_list, data_dict_no_NaN2)

# define "my_dataset", which is required for verification of the result.
my_dataset = data_dict_no_NaN2

# Define the function to calculate precision and recall.
def precision_recall(labels, predictions):
    ind_true_pos = [i for i in range(0,len(labels)) if (predictions[i]==1) & (labels[i]==1)]
    ind_false_pos = [i for i in range(0,len(labels)) if ((predictions[i]==1) & (labels[i]==0))]
    ind_false_neg = [i for i in range(0,len(labels)) if ((predictions[i]==0) & (labels[i]==1))]
#    ind_true_neg = [i for i in range(0,len(labels)) if ((predictions[i]==0) & (labels[i]==0))]
    precision = 0
    recall = 0
    
    ind_labels = [i for i in range(0,len(labels)) if labels[i]==1]
    
    if len(ind_labels) !=0:
        if float( len(ind_true_pos) + len(ind_false_pos))!=0:
            precision = float(len(ind_true_pos))/float( len(ind_true_pos) + len(ind_false_pos))
        if float( len(ind_true_pos) + len(ind_false_neg))!=0:
            recall = float(len(ind_true_pos))/float( len(ind_true_pos) + len(ind_false_neg))
        return precision, recall
    else:
        return -1,-1

# Define a custom scoring function. This scoring function will return
# the minimum  between precision and recall.
def custom_scorer(labels, predictions):
    precision, recall = precision_recall(labels, predictions)
    min_score = min(precision, recall)
    return min_score
score = make_scorer(custom_scorer, greater_is_better=True)

# Define the function to otuput the outcomes of the gridSearchCV().
def get_outcomes(gridCV, fh):

    '''Gets the print out of all the outcomes from the grid_search. It prints out the 
    best parameters found by the model and the outcomes of the test of the model on 
    the test set.'''
    
    print "Best parameters from the grid search: ", gridCV.best_params_
    fh.write("Best parameters from the grid search: " + str(gridCV.best_params_) + "\n")
    clf_gridCV = gridCV.best_estimator_
    print "\nBest Estimator Accuracy:", clf_gridCV.score(features_test, labels_test)
    fh.write("\nBest Estimator Accuracy: " + str(clf_gridCV.score(features_test, labels_test)) + "\n")
    clf_gridCV_pred = clf_gridCV.predict(features_test)
    fh.write("\n\nRecall Score: "+ str(recall_score(labels_test, clf_gridCV_pred)) + "\n")
    print "\n\nRecall Score:", recall_score(labels_test, clf_gridCV_pred)
    fh.write("\n\nRecall Score: " + str(recall_score(labels_test, clf_gridCV_pred)) + "\n")
    print "\n\nPrecision Score:", precision_score(labels_test, clf_gridCV_pred)


# Open the file to print the result.
f = open("poi_id_result.txt", 'w')

# Define the features which I choose.        
features_list = ['poi', 'exercised_stock_options', 'salary']
print features_list
f.write(str(features_list) + "\n")

# Change the format of the dataset.                 
data = featureFormat(my_dataset, features_list, remove_all_zeroes=False, sort_keys = True)

# Split the data into labels and features.
labels, features = targetFeatureSplit(data)

# split the data into the testing data (30%) and training data (70%)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# call Decision Tree classifier and its parameters.
dtc = DecisionTreeClassifier(random_state = 42, criterion = "entropy", splitter = "best")


t0 = time.time()
t0_value = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0))
print "Training starts at: ", t0_value
f.write("Training starts at: " + str(t0_value) + "\n")
dtc.fit(features_train, labels_train)
t1 = time.time()
t1_value = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1))
f.write("Training ends at: " + str(t1_value) + "\n")
print "Training ends at: ", t1_value

labels_pred = dtc.predict(features_test)

ascore = accuracy_score(labels_test, labels_pred)

f1score = f1_score(labels_test, labels_pred, average=None)

print "Accuracy Score: ", ascore

print "F1 Score: ", f1score

# Close file.
f.close()








