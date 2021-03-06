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

def make_prediction(feats_file, model_name, scaler_name):
    rows = []
    aud_names = []

    with open(feats_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            aud_names.append(row[0])
            rows.append(row[1:])
   
    rows = np.array(rows).astype(float)
    # print(rows.shape)
    print("Number of testing data points : {}\n".format(rows.shape[0]))

    x_test = rows[:,0:136]
    y_true = rows[:,136]
    with open(scaler_name, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_name, 'rb') as f:
        svm = pickle.load(f)

    x_test = scaler.transform(x_test)
    y_pred = svm.predict(x_test)

    # Convert labels into strings
    #y_pred = ['Female' if x==2 else 'Male' for x in y_pred]

    print("Confusion Matrix of Test set")
    print(confusion_matrix(y_true, y_pred))
    print("\nAccuracy on Test set: {}\n".format(accuracy_score(y_true, y_pred)))
    
    # res_df = pd.DataFrame({aud_col_name: aud_names, 'Predicted ML Gender': y_pred})
    # pred_df = pd.merge(fdf, res_df, how='inner', on=[aud_col_name])

def main(feats_file, model_name, scaler_name):
    pred_df = make_prediction(feats_file, model_name, scaler_name)
    # pred_df.to_excel(out_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--in_file", type=str, help="File containing audio information",
    #     default="latest_short_acc_rej.xlsx", required=True)
    # parser.add_argument("--aud_col_name", type=str, help="Name of the columns containing audio links",
    #     default='Recording audio link', required=True)
    parser.add_argument('--feats_file', type=str, help='File path for extracted features', required=True)
    parser.add_argument('--model_name', type=str, help='Name for the gender classifier model', required=True)
    parser.add_argument('--scaler_name', type=str, help='Path of the scaler model', required=True)
    # parser.add_argument('--out_file', type=str, help='File containing the output prediction', required=True)
    args = parser.parse_args()
    main(args.feats_file, args.model_name, args.scaler_name)

#k_fold_validation_train_model("gender_classifier_features_data.csv")
