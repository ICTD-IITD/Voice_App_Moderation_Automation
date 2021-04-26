import argparse
import os
import sys
import time

import pandas as pd

#from pydub import AudioSegment
from tqdm import tqdm

def main(in_file, dest_pth_mp3):
    filedf = pd.read_excel(in_file)
    aud_links = filedf['Recording audio link'].fillna('').tolist()
    present_audios = os.listdir(dest_pth_mp3)
    for audio_link in tqdm(aud_links):
        aud_name = audio_link.split('/')[-1].split('.mp3')[0]
        in_file = aud_name + '.mp3'
        if in_file not in present_audios:
            try:
                os.system('wget -P ' + dest_pth_mp3 + ' ' + audio_link + ' >/dev/null 2>&1') # Hides the console output, can also use subprocess
            except Exception as e:
                print("Download error for link : {} with error : {}".format(audio_link, str(e)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, help="File containing the data from the master server",
        default="data/gender_classifier.xlsx", required=True)
    parser.add_argument("--out_folder_mp3", type=str, help="Folder containing downloaded mp3 files",
        default="audio_files_mp3", required=True)
    args = parser.parse_args()
    in_file = args.in_file
    dest_pth_mp3 = args.out_folder_mp3
    if not os.path.exists(os.path.join(os.getcwd(), dest_pth_mp3)):
        os.mkdir(os.path.join(os.getcwd(), dest_pth_mp3))
    
    start_time = time.time()

    main(in_file, dest_pth_mp3)

    end_time = time.time()

    print("Time taken to Download the audios : {:.2f} seconds".format(end_time-start_time))

