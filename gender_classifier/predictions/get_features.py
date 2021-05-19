import argparse
import csv
import os
import sys
import time

import pandas as pd
import numpy as np

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from statistics import mean, stdev
from tqdm import tqdm

def extract_feats(aud_link, input_file, fem_flag):
    [Fs, x] = audioBasicIO.readAudioFile(input_file)    #This function returns a numpy array that stores the audio samples of a specified WAV of AIFF file ; Fs -> sampling rate (16kHz), x -> signal
    F,f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.05*Fs, 0.025*Fs) # F : (34, num_frames_in_audio), f_names : names of features
    l = []

    l.append(aud_link)
    for j in range(34):
        l.append(min(F[j]))
        l.append(max(F[j]))
        l.append(mean(F[j]))
        l.append(stdev(F[j]))

    if fem_flag:
        l.append(2)         # label for female contributions : 2
    else:
        l.append(1)         # label for male contributions : 1
    return l

def get_feats(df, aud_col_name, auds_dir, feats_dir, fem_flag):
    aud_links = df[aud_col_name].fillna('').tolist()
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
                    feats.insert(0, aud_link)
                except:
                    try:
                        feats = extract_feats(aud_link, os.path.join(auds_dir, aud_name), fem_flag)
                        np_feats = np.asarray(feats[1:])
                        np.save(os.path.join(feats_dir, aud_name.replace('wav', 'npy')), np_feats)
                    except Exception as e:
                        print("Error in audio : {} and error : {}".format(aud_name, str(e)))
            else:
                try:
                    feats = extract_feats(aud_link, os.path.join(auds_dir, aud_name), fem_flag)
                    np_feats = np.asarray(feats[1:])
                    np.save(os.path.join(feats_dir, aud_name.replace('wav', 'npy')), np_feats)
                except Exception as e:
                    print("Error in audio : {} and error : {}".format(aud_name, str(e)))
            all_feats.append(feats)
    print("Extracted feats")
    return all_feats

def main(in_file, aud_col_name, auds_dir, out_file, feats_dir):
    filedf = pd.read_excel(in_file)

    start_time = time.time()
    fem_feats = get_feats(filedf, aud_col_name, auds_dir, feats_dir, fem_flag=True)
    end_time = time.time()

    # Saving the output features
    print("Saving the generated features")
    with open(out_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(fem_feats)

    print("Time to extract features for female contributions: {:.2f} seconds".format(end_time-start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, help="File containing audio information",
        default="latest_short_acc_rej.xlsx", required=True)
    parser.add_argument("--aud_col_name", type=str, help="Name of the columns containing audio links",
        default='Recording audio link', required=True)
    parser.add_argument("--auds_dir", type=str, help="Directory containing audio wav files",
        default="audio_files_wav", required=True)
    parser.add_argument("--out_file", type=str, help="File containing the audio features",
        default="acc_rej_feats.csv", required=True)
    parser.add_argument("--feats_dir", type=str, help="Dir containing the audio features",
        default="feats_dir", required=True)
    args = parser.parse_args()
    main(args.in_file, args.aud_col_name, args.auds_dir, args.out_file, args.feats_dir)
