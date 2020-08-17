import argparse
import os
import sys

import numpy as np
import pandas as pd

def main(feats_csv, train_csv, test_csv):
    filedf = pd.read_csv(feats_csv)

    # shuffle the dataframe a number of times
    for i in range(3):
        filedf = filedf.sample(frac=1).reset_index(drop=True)
    
    # Split the df into 80% train and 20% test splits
    msk = np.random.rand(len(filedf)) < 0.8
    train_df = filedf[msk]
    test_df = filedf[~msk]

    print("Length of train features : {}".format(len(train_df)))
    print("Length of test features : {}".format(len(test_df)))

    # Save the train and the test dataframes
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats_csv", type=str, help="The csv containing all the audio features",
        required=True, default="acc_rej_feats.csv")
    parser.add_argument("--train_csv", type=str, help="Name of out csv containing all the train features",
        required=True, default="train_acc_rej_feats.csv")
    parser.add_argument("--test_csv", type=str, help="Name of out csv containing all the test features",
        required=True, default="test_acc_rej_feats.csv")
    args = parser.parse_args()
    main(args.feats_csv, args.train_csv, args.test_csv
