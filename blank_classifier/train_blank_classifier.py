import argparse
import csv
import os
import pickle
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import time


# Plot Confusion Matrix for testing and Training Accuracies
def plotConfusionMatrix(confusionMatrix, classes, fname):

    plt.figure(figsize=(10, 7))

    ax = sn.heatmap(confusionMatrix, fmt="d", annot=True, cbar=False,
                    cmap=sn.cubehelix_palette(15),
                    xticklabels=classes, yticklabels=classes)
    # Move X-Axis to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    ax.set(xlabel="Predicted", ylabel="Actual")

    figure = fname + ".png"

    plt.title(fname, y=1.08, loc="center")
    plt.show()
    plt.savefig(figure)
    plt.close()


def train_model_RF(train_Data, test_Data, model_name, scaler_name):
    x_train = train_Data[:, 0:136]
    y_train = train_Data[:, 136]
    x_test = test_Data[:, 0:136]
    y_test = test_Data[:, 136]
    scaler = StandardScaler()
    scaler.fit(x_train)
    # pickle.dump(scaler, open("scaler_acc_rej", 'wb'))
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    calibrated_forest = CalibratedClassifierCV(base_estimator=RandomForestClassifier(n_estimators=10,criterion="entropy"))
    pipe = Pipeline([('select', SelectKBest()),('model', calibrated_forest)])
    param_grid = {'select__k': [1, 2],'model__base_estimator__max_depth': [50, 75, 100, 150]}
    search = GridSearchCV(pipe, param_grid, cv=5)
    clf = search.fit(x_train, y_train)
    train_pred = clf.predict(x_train)
        
    print("Accuracy on training set: {:.3f}".format(accuracy_score(y_train, train_pred)))
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test set: {:.3f}".format(accuracy))

    pickle.dump(scaler, open(scaler_name, 'wb'))
    pickle.dump(clf, open(model_name, 'wb'))

    return y_test, y_pred, accuracy


def k_fold_validation_train_model(input_file, model_name, scaler_name, k=10):
    """
        Script to train the Random Forest model for accepting or rejecting an audio file
    """
    # Define the base directory path
    # base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    rows = []
    with open(input_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        # fields = csvreader.next()
        for row in csvreader:
            try:
                row = [float(i) for i in row[1:]]
            except:
                # Fill all blanks with 0
                for idx, r in enumerate(row):
                    if row[idx] == '':
                        row[idx] = '0'
                row = [float(i) for i in row[1:]]
                print("Error")
                # sys.exit()
            rows.append(row)
    rows = np.array(rows).astype(float)
    print(rows.shape)

    # np.random.shuffle(rows)
    # print(rows.shape)

    non_fin = []
    for i in range(0, rows.shape[0]):
        if(np.isfinite(rows[i,:]).all() != True):
            # print(np.isfinite(rows[i,:]).all())
            print(i+1)
            # print(rows[i,:])
            non_fin.append(i)
            
    rows = np.delete(rows, non_fin, axis=0)
    print(rows.shape)
    np.random.shuffle(rows)

    kf = KFold(n_splits=k, shuffle=True, random_state=int(time.time()))
    kf.get_n_splits(rows)
    accuracies = []
    test_true = np.array([])
    test_predicted = np.array([])
    i = 0

    for train_index, test_index in kf.split(rows):
        print("Chunk OF Data for iteration : " + str(i))
        train_data = rows[train_index]
        test_data = rows[test_index]
        # print(len(train_index),len(test_index))
        start_time = time.time()
        y_true, y_pred, accuracy = train_model_RF(train_data, test_data, model_name, scaler_name)
        end_time = time.time()
        print("Time taken to train RF blank classifier : {} seconds".format(end_time-start_time))
        accuracies.append(accuracy)
        test_true = np.append(test_true, y_true)
        test_predicted = np.append(test_predicted, y_pred)
        i = i + 1
        break    # do not break if you want to evaluate over all the K-Folds

    accuracies = np.array(accuracies)
    print("Accuracies : ", accuracies)
    print("Average Accuracy : {:.3f}".format(np.average(accuracies)))
    print("F1_score", f1_score(test_true, test_predicted, average=None))
    classes = list(sorted(set(test_true)))
    results = confusion_matrix(test_true, test_predicted, labels=classes)
    print("Confusion matrix: {}".format(results))
    
    plotConfusionMatrix(results, classes, "blank_classifier_cm")
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
