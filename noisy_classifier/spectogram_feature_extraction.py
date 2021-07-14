# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import requests
import subprocess
import os
import sys
import matplotlib.pyplot as plt

#for loading and visualizing audio files
import librosa
import librosa.display

from tqdm import tqdm

pData = pd.read_excel('noisy_classifier_in_data.xlsx')

noisy = pData[pData['Accept/Noisy label']==0][:10]
accepted = pData[pData['Accept/Noisy label']==1][:10]

def get_audios(aud_type, noisy_str):
    for url in tqdm(aud_type['Recording audio link']):
        try:
            url_id = url[(url.rindex('/')+1):-4]
            audio = requests.get(url)
            path = "./data/wave_files_{}/".format(noisy_str)
            if not os.path.exists(path):
                os.makedirs(path)

            filename = path+url_id + ".mp3"
            filename_wave = path+url_id+".wav"
            file1 = open(filename, 'wb')
            file1.write(audio.content)
            file1.close()
            subprocess.call(['ffmpeg', '-i',filename,filename_wave],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
            os.remove(filename)

            image_path_colorbar = "./data/test_{}/colorbar/".format(noisy_str)
            image_path_simple = "./data/test_{}/simple/".format(noisy_str)

            if not os.path.exists(image_path_colorbar):
                os.makedirs(image_path_colorbar)
            if not os.path.exists(image_path_simple):
                os.makedirs(image_path_simple)

            # Extract the audio spectrogram
            x,sr = librosa.load(filename_wave, sr=44100)
            S = librosa.feature.melspectrogram(x, sr=sr, n_mels=128,fmax=8000)
            os.remove(filename_wave)

            plt.figure(figsize=(14,5))
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=8000)
            plt.colorbar()

            plt.savefig(image_path_colorbar+url_id,pad_inches = 0,bbox_inches='tight')
            plt.figure(figsize=(8, 5),frameon=False)
            plt.axis("off")
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=sr,fmax=8000)
            plt.plot()
            plt.savefig(image_path_simple+url_id,pad_inches = 0,bbox_inches='tight')

            plt.close("all")
        except Exception as e:
            print("Error: {}".format(str(e)))
            sys.exit()


get_audios(noisy, 'noisy')
get_audios(accepted, 'accepted')
