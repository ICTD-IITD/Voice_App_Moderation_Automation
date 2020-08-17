'''
Script to compile 100,000 items from the 417,041 items generated using compile_data.py
This script should be executed after accept_reject_combined_data.xlsx has been generated
from compile_data.py script
'''
import argparse
import os
import sys

import pandas as pd

def main(in_file_pth, out_file_name):
    filedf = pd.read_excel(in_file_pth)

    # Accepted items
    df1 = filedf[filedf['state'].fillna('') == 'PUB']
    df2 = filedf[(filedf['state'].fillna('') == 'ARC') & (filedf['rating'].fillna(-1).apply(lambda x: int(x) == 4 or int(x) == 5))]
    # df3 = filedf[(filedf['state'].fillna('') == 'REJ') & (filedf['tags'].fillna('').apply(lambda x: 'published' in str(x).lower().strip()))]
    
    # Get the latest accepted items for each type
    print("PUB items : {}".format(df1.shape[0]))
    print("ARC with 4,5 rating items : {}".format(df2.shape[0]))
    print("REJ with published tag : {}".format(df3.shape[0]))
    df1 = df1[df1.shape[0]-7000:]
    df2 = df2[df2.shape[0]-18000:]
    # df3 = df3[df3.shape[0]-21000:]
    acc_df = pd.concat([df1, df2])

    # Rejected items
    rej_df = filedf[(filedf['state'].fillna('') == 'REJ') & (filedf['rating'].fillna(-1).apply(lambda x: int(x) in [0,1,2])) & (filedf['tags'].fillna('').apply(lambda x: 'blank' in str(x).lower().strip() or 'noisy' in str(x).lower().strip()))]

    # Compile the latest data
    lat_acc_df = acc_df
    lat_rej_df = rej_df[rej_df.shape[0]-25000:]

    print("shape of latest accepted items : {}".format(lat_acc_df.shape))
    print("shape of latest rejected items : {}".format(lat_rej_df.shape))

    # Create the excel with latest data
    out_df = pd.concat([lat_acc_df, lat_rej_df])
    out_df.to_excel(out_file_name, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_pth", type=str, required=True,
        help="Path of the file for which stats are required",
        default="/home/oniondev/Desktop/aman/gramvaani/ml_datasets/content-moderation/scripts/new_preprocessing/accept_reject_combined_data_final.xlsx")
    parser.add_argument("--out_file_name", type=str, required=True,
        help="Name of the output file to store the latest data",
        default="latest_acc_rej_data.xlsx")
    args = parser.parse_args()
    main(args.input_file_pth, args.out_file_name)
