'''
Script to compile the data from training and test folders into a single dataset but placing softer conditions.
This has been done because the test data unnecessarily contained more than 10% data.
The data should also be randomized before training and appropriate positive and negative samples must be taken.
Also this code is in python2 to be compatible with the production environment.
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd

def get_combined_df(filedf):
    df4 = filedf[(filedf['state'].fillna('') == 'REJ') & (filedf['tags'].fillna('').apply(lambda x: 'blank' in str(x).lower().strip()))]
    df1 = filedf[filedf['state'].fillna('') == 'PUB']
    df2 = filedf[(filedf['state'].fillna('') == 'ARC') & (filedf['rating'].fillna(-1).apply(lambda x: int(x) == 4 or int(x) == 5))]
    #df3 = filedf[(filedf['state'].fillna('') == 'REJ') & (filedf['tags'].fillna('').apply(lambda x: 'published' in str(x).lower().strip()))]
    #df4 = filedf[(filedf['state'].fillna('') == 'REJ') & (filedf['rating'].fillna(-1).apply(lambda x: int(x) in [0,1,2])) & (filedf['tags'].fillna('').apply(lambda x: 'blank' in str(x).lower().strip() or 'noisy' in str(x).lower().strip()))]
    combined_df = pd.concat([df1, df2, df4])
    return combined_df

def main(training_data_pth, test_data_pth, out_file):
    final_df = pd.DataFrame()
    # Go over the training data
    train_files = os.listdir(training_data_pth)
    file_names = ['bihar_item_data - all and xtra columns.xlsx',
                    'bihar_nalanda_Item_data - all and xtra columns.xlsx',
                    'jharkhand_item_data - all and xtra columns.xlsx',
                    'mp_item_data - all and xtra columns.xlsx',
                    'up_hindi belt_Item_data - all and xtra columns.xlsx']

    for train_file in train_files:
        if train_file in file_names:
            print("Entering Training file : {}".format(train_file))
            filedf = pd.read_excel(os.path.join(training_data_pth, train_file))
            combined_df = get_combined_df(filedf)
            final_df = pd.concat([final_df, combined_df])
    
    # Go over the test data
    test_files = os.listdir(test_data_pth)
    file_names = ['bihar_clubs.xlsx', 'bmgf-Nalanda_items.xlsx', 'jharkhand_clubs.xlsx',
                    'mp_instance.xlsx', 'up_instance_crea.xlsx']
    for test_file in test_files:
        if test_file in file_names:
            print("Entering test file : {}".format(test_file))
            filedf = pd.read_excel(os.path.join(test_data_pth, test_file))
            filedf = filedf.drop(columns=['creation_time'])
            filedf = filedf.rename(columns={'current_state': 'state', 'id': 'item_id'})
            combined_df = get_combined_df(filedf)
            final_df = pd.concat([final_df, combined_df])

    print("Generate the final Dataframe of shape : {}".format(final_df.shape))
    final_df.to_excel(out_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_pth", help="Path for the training data folder",
        default='/home/oniondev/Desktop/aman/gramvaani/ml_datasets/content-moderation/data/training_data',
        type=str, required=True)
    parser.add_argument("--test_data_pth", help="Path for the test data folder",
        default='/home/oniondev/Desktop/aman/gramvaani/ml_datasets/content-moderation/data/test_data',
        type=str, required=True)
    parser.add_argument("--out_file", help="Name of the output file",
        default='accept_reject_combined_data.xlsx',
        type=str, required=True)
    args = parser.parse_args()
    main(args.training_data_pth, args.test_data_pth, args.out_file)
