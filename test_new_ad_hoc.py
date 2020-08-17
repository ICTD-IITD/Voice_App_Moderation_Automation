"""
Script to test from the unseen test data in the ad-hoc mode for GV stack
"""

import argparse
import contextlib
import os
import pickle
import sys
import wave

import numpy as np
import pandas as pd
import soundfile as sf

from pydub import AudioSegment
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
from subprocess import Popen
from tqdm import tqdm

from get_features import get_feats

def get_clean_df(filedf, test_urls):
    filedf = filedf.fillna('')
    filedf = filedf[filedf['audio_url'].apply(lambda x: x.split('/')[-1].split('.mp3')[0] in test_urls)]
    filedf = filedf.drop_duplicates(subset=['audio_url'], keep='first')
    return filedf

def download_audio(audio_link, dest_pth_mp3, dest_pth_wav):
    present_audios = os.listdir(dest_pth_mp3)
    # for link in audio_links: # We can loop over the audios when we want to download all audios
    aud_name = audio_link.split('/')[-1].split('.mp3')[0]
    in_file = aud_name + '.mp3'
    out_file = aud_name + '.wav'
    if in_file not in present_audios:
        try:
            os.system('wget -P ' + dest_pth_mp3 + ' ' + audio_link + ' >/dev/null 2>&1') # Hides the console output, can also use subprocess
            # Convert the downloaded mp3 audio into wav format
            os.system('ffmpeg -i '+ os.path.join(dest_pth_mp3, in_file) + ' -acodec pcm_s16le -ac 1 -ar 16000 ' +
                os.path.join(dest_pth_wav, out_file)+' >/dev/null 2>&1')
            # Remove the mp3 file because we do not want its transcript
            # os.system('rm ' + os.path.join(dest_pth, in_file))
        except Exception as e:
            print("Download error for link :{} with error {}".format(audio_link, str(e)))

def get_features(aud_links, auds_dir, feats_dir):
    df = pd.DataFrame({'audio_url':aud_links})  # This is done for corresponding with the get_feats function
    # Get and save the audio features
    feats = get_feats(df, auds_dir, feats_dir, acc_flag=True)

    feats = [f[1:] for f in feats]  # Remove the audio name from the features

    return feats

def make_prediction(feats, model_name, scaler_name, use_length_feat):
    with open(model_name, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_name, 'rb') as f:
        scaler = pickle.load(f)
    
    test_df = pd.DataFrame.from_records(feats).fillna(0)

    if use_length_feat == "False":
        test_df = test_df.drop(test_df.columns[-2], axis=1)

    test_feats = test_df.iloc[:,:-1].values
    true_labels = test_df.iloc[:,-1].values

    pred_labels = model.predict(scaler.transform(test_feats))

    return pred_labels, true_labels

def main(complete_file_pth, model_ckpt_pth, scaler_model_ckpt_pth, aud_dest_pth_mp3, aud_dest_pth_wav, feats_dir, use_length_feat):
    if 'mp3' in complete_file_pth:
        aud_links = [complete_file_pth]
    else:
        with open(complete_file_pth, 'r') as f:
            aud_links = f.readlines()
        aud_links = [u.strip() for u in aud_links]

    # Download all the audios
    print("Downloading audios")
    for aud_link in tqdm(aud_links):
        download_audio(aud_link, aud_dest_pth_mp3, aud_dest_pth_wav)
    print("Downloaded all the audios")

    # Extract the features
    feats = get_features(aud_links, aud_dest_pth_wav, feats_dir)
    # Make the predictions
    pred_labels, true_labels = make_prediction(feats, model_ckpt_pth, scaler_model_ckpt_pth, use_length_feat)

    print("Predicted Labels : {}".format(pred_labels))
    if 'mp3' in complete_file_pth:
        if pred_labels[0] == 2:
            res_class = 'PUB'
        else:
            res_class = 'REJ'
        print("Predicted class : {}".format(res_class))

    # Get the results
    results = confusion_matrix(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    print("Confusion matrix: {}".format(results))
    print("Accuracy : {}".format(accuracy))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--complete_file_pth", type=str,
        help="Path for text file containing all the audio urls to be evaluated", required=True,
        default="/home/oniondev/Desktop/aman/gramvaani/ml_datasets/content-moderation/scripts/new_preprocessing/latest_short_acc_rej.xlsx")
    parser.add_argument("--model_ckpt", type=str, help="Path for the saved checkpoint", required=True,
        default="models/accept_reject_model_noisy_based_no_length_11_08_2020.pkl")
    parser.add_argument("--scaler_model_ckpt", type=str, help="Path for the saved scaler checkpoint",
        required=True,
        default="models/scaler_acc_rej_no_len_11_08_2020.pkl")
    parser.add_argument("--audio_dest_pth_mp3", type=str,
        help="Path of the folder where audio files need to downloaded", required=True,
        default="audio_files_mp3")
    parser.add_argument("--audio_dest_pth_wav", type=str,
        help="Path of the folder where wav files need to be saved", required=True,
        default="audio_files_wav")
    parser.add_argument("--feats_dir", type=str,
        help="Path of the folder where audio features have to be saved", required=True,
        default="feats_dir")
    parser.add_argument("--use_length_feat", type=str, help="Flag for using length as an audio feature",
        default="True", required=False)
    args = parser.parse_args()
    main(args.complete_file_pth, args.model_ckpt, args.scaler_model_ckpt, args.audio_dest_pth_mp3, args.audio_dest_pth_wav, args.feats_dir, args.use_length_feat)
