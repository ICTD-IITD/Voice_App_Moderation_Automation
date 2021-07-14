import argparse
import csv
import os
import pickle
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import time

#Plot Confusion Matrix for testing and Training Accuracies
def plotConfusionMatrix(confusionMatrix, classes, fname):

    plt.figure(figsize=(10, 7))
    
    ax = sn.heatmap(confusionMatrix, fmt="d", annot=True, cbar=False,
                cmap=sn.cubehelix_palette(15),
                xticklabels=classes, yticklabels=classes)
    # Move X-Axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    ax.set(xlabel="Predicted", ylabel="Actual")
    
    
    figure =fname + ".png"
    
    plt.title(fname  , y = 1.08 , loc = "center")
    plt.savefig(figure)
    plt.show()
    plt.close()
	
def train_model_SVM(train_Data, test_Data, model_name, scaler_name):
    x_train = train_Data[:,0:136]
    y_train = train_Data[:,136]
    x_test = test_Data[:,0:136]
    y_test = test_Data[:,136]
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    #pickle.dump(scaler, open('gender_scaler','wb'))
    pickle.dump(scaler, open(scaler_name, 'wb'))
    # svm = SVC(C=2, gamma = 0.03).fit(x_train, y_train)
    svm = SVC(C=1, gamma=0.01).fit(x_train, y_train)
    train_pred = svm.predict(x_train)

    # print("Accuracy on training set: ",svm.score(y_train, train_pred))
    print("Accuracy on training set: ",accuracy_score(y_train, train_pred))
    y_pred = svm.predict(x_test)
    # accuracy = svm.score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test set: ",(accuracy))
    
    #save model
    pickle.dump(svm, open(model_name, 'wb'))	
    
    return y_test, y_pred, accuracy


def k_fold_validation_train_model(input_file, model_name, scaler_name, k=10):
    rows = []
    
    with open(input_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row[1:])
   
    rows = np.array(rows).astype(float)
    print(rows.shape)
	
    kf = KFold(n_splits=k, shuffle = True)
    kf.get_n_splits(rows)
    accuracies = []
    test_true = np.array([])
    test_predicted = np.array([])
    i = 0

    for train_index, test_index in kf.split(rows):
        print("Chunk OF Data for iteration : " + str(i))
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_data = rows[train_index]
        test_data = rows[test_index]
        #model_name = "genderClassifier" + str(i);
        start_time = time.time()
        y_true, y_pred, accuracy = train_model_SVM(train_data, test_data, model_name, scaler_name)
        end_time = time.time()
        print("Time taken to train SVM gender classifier : {} seconds".format(end_time-start_time))
        accuracies.append(accuracy)
        test_true = np.append(test_true,y_true)
        test_predicted = np.append(test_predicted, y_pred)		
        i = i+1
        break    # do not break if you want to evaluate over all the K-Folds

    accuracies = np.array(accuracies)
    print("Accuracies : " , accuracies)
    print("Average Accuracy : {:.3f}".format(np.average(accuracies)))
    #print(test_true,test_predicted)
    print("F1_score", f1_score(test_true, test_predicted, average = None))
    classes = list(sorted(set(test_true)))
    results = confusion_matrix(test_true, test_predicted, labels = classes)
    plotConfusionMatrix(results, classes, "genderClassifier")	
    
    print("Training and validation finished")

def main(feats_file, model_name, scaler_name):
    k_fold_validation_train_model(feats_file, model_name, scaler_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats_file', type=str, help='File path for extracted features', required=True)
    parser.add_argument('--model_name', type=str, help='Name for the gender classifier model', required=True)
    parser.add_argument('--scaler_name', type=str, help='Name of the standard scaler used for feature normalization', required=True)
    args = parser.parse_args()
    main(args.feats_file, args.model_name, args.scaler_name)

#k_fold_validation_train_model("gender_classifier_features_data.csv")

