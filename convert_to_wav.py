import argparse
import os
import sys

from tqdm import tqdm

def main(in_folder, out_folder):
    in_files = os.listdir(in_folder)
    out_files = os.listdir(out_folder)
    for fname in tqdm(in_files):
        if fname.replace('mp3','wav') not in out_files:
            os.system('ffmpeg -i '+ os.path.join(in_folder, fname) + ' -acodec pcm_s16le -ac 1 -ar 16000 ' +
                os.path.join(out_folder, fname.replace('mp3','wav'))+' >/dev/null 2>&1')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", type=str, help="Folder containing mp3 files",
        default="audio_files_mp3", required=True)
    parser.add_argument("--out_folder", type=str, help="Folder containing generated wav files",
        default="audio_files_wav", required=True)
    args = parser.parse_args()
    main(args.in_folder, args.out_folder)
