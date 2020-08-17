import argparse
import csv
import os
import sys

import pandas as pd
import numpy as np

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from tqdm import tqdm

def extract_feats(input_file, acc_flag):
    [Fs, x] = audioBasicIO.readAudioFile(input_file)    #This function returns a numpy array that stores the audio samples of a specified WAV of AIFF file ; Fs -> sampling rate (16kHz), x -> signal
    F,f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.05*Fs, 0.025*Fs) # F : (34, num_frames_in_audio), f_names : names of features
    l = []
    l.append(input_file)
    for j in range(34):
        l.append(np.percentile(F[j, :], 25))
        l.append(np.percentile(F[j, :], 50))
        l.append(np.percentile(F[j, :], 75))
        l.append(np.percentile(F[j, :], 95))
    l.append((len(F[0])+1)/40)            # Appending the length of the audio file (length = num_frames*frame_len) and frame_len = 25ms
    if acc_flag:
        l.append(2)         # label for accepted items : 2
    else:
        l.append(1)         # label for rejected items : 1
    return l

def get_feats(df, auds_dir, feats_dir, acc_flag):
    aud_links = df['audio_url'].fillna('').tolist()
    wav_links = os.listdir(auds_dir)
    all_feats = []
    stored_feats = os.listdir(feats_dir)
    print("Extracting features")
    for aud_link in tqdm(aud_links):
        aud_name = aud_link.split('/')[-1].replace('mp3', 'wav')
        if aud_name in wav_links:
            if aud_name.replace('wav', 'npy') in stored_feats:
                try:
                    feats = list(np.load(os.path.join(feats_dir, aud_name.replace('wav', 'npy'))))
                    feats.insert(0, os.path.join(auds_dir, aud_name))
                except:
                    feats = extract_feats(os.path.join(auds_dir, aud_name), acc_flag)
                    np_feats = np.asarray(feats[1:])
                    np.save(os.path.join(feats_dir, aud_name.replace('wav', 'npy')), np_feats)
            else:
                feats = extract_feats(os.path.join(auds_dir, aud_name), acc_flag)
                np_feats = np.asarray(feats[1:])
                np.save(os.path.join(feats_dir, aud_name.replace('wav', 'npy')), np_feats)
            all_feats.append(feats)
    print("Extracted feats")
    return all_feats

def main(in_file, auds_dir, out_file, feats_dir):
    filedf = pd.read_excel(in_file)

    # Accepted items
    df1 = filedf[filedf['state'].fillna('') == 'PUB']
    df2 = filedf[(filedf['state'].fillna('') == 'ARC') & (filedf['rating'].fillna(-1).apply(lambda x: int(x) == 4 or int(x) == 5))]
    # df3 = filedf[(filedf['state'].fillna('') == 'REJ') & (filedf['tags'].fillna('').apply(lambda x: 'published' in str(x).lower().strip()))]
    acc_df = pd.concat([df1, df2])

    # Rejected items
    rej_df = filedf[(filedf['state'].fillna('') == 'REJ') & (filedf['rating'].fillna(-1).apply(lambda x: int(x) in [0,1,2])) & (filedf['tags'].fillna('').apply(lambda x: 'blank' in str(x).lower().strip() or 'noisy' in str(x).lower().strip()))]

    acc_feats = get_feats(acc_df, auds_dir, feats_dir, acc_flag=True)
    rej_feats = get_feats(rej_df, auds_dir, feats_dir, acc_flag=False)
    
    # the combined acc and rej feats
    acc_feats.extend(rej_feats)         # acc feats now contains all the feats (acc+rej)

    # Saving the output features
    print("Saving the generated features")
    with open(out_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(acc_feats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, help="File containing audio information",
        default="latest_short_acc_rej.xlsx", required=True)
    parser.add_argument("--auds_dir", type=str, help="Directory containing audio wav files",
        default="audio_files_wav", required=True)
    parser.add_argument("--out_file", type=str, help="File containing the audio features",
        default="acc_rej_feats.csv", required=True)
    parser.add_argument("--feats_dir", type=str, help="Dir containing the audio features",
        default="feats_dir", required=True)
    args = parser.parse_args()
    main(args.in_file, args.auds_dir, args.out_file, args.feats_dir)
