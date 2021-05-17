# Gender Classifier

## Data acquisition

The script to get data for the gender classifier is through `get_gender_classifier_data.py`

- This script runs in a python 2 environment present in the Master Server

Command:
```python
sudo python get_gender_classifier_data.py --thresh=500 --email_ids=aman.khullar@oniondev.com --start_date=2019-01-01 --end_date=2019-03-31 --output=/tmp/gender_classifier.xlsx
```

- Thresh is the number of items in the training data for each class. In the example above there will be 500 items for Female class, 500 items for the Male class

- Here the Male class is '1' and Female class is '2'

- The data generated from the above command is stored in the 'data' folder by the name `gender_classifier.xlsx`


### Download the audios and convert them into wav files

##### Activate the python 2 environment as in Master server
source activate <environment_name>
> NOTE: The requirements.txt has been created to install the dependencies

Download the audios of only the clean data as produced in the previous step and then convert them into wav files which are easier for feature extraction.

Download audio files:
```python
python download_audios.py --in_file="<in_file>.xlsx" --out_folder_mp3="<folder name>"
```
- Time taken to download 100 audios : 27.30 seconds
- Time taken to download 6000 audios : 1541.53 seconds

> Hardware specification: Processor: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz, 16GB RAM

Convert mp3 files to wav files:
```python
python convert_to_wav.py --in_folder="<folder created in previous step>" --out_folder="<folder containing wav audios>"
```

- Time taken to Convert the audios to Wave : 9.36 seconds
- Time taken to Convert 6000 (actually 5964) audios to Wave : 833.69 seconds

### Audio features extraction

The 136 features are extracted for all the audio files using this script. Corresponding feature files in the form of numpy arrays are saved. The audio features are extracted using [PyAudioAnalysis library](https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction). 4 values (min, max, mean, stddev) from the 34 features are extracted to get the 136 (34*4) features.

- We use a 50ms framesize with a 25ms stepsize
- Hence, for example if we have an audio of 100ms, there will be 3 frames and for a 1min audio there will be 2399 frames
- There will be 34 features for each frame
- Hence the feature matrix size will be 34*2399 for a 1 minute audio
- After that we take the min, max, mean, stddev for each feature across all the frames
- We finally get a 34*4=136 feature vector for any audio file

Extract and store audio features - in both npy files and csv format:
```python
python get_features.py --in_file="<cleaned data file>.xlsx" --auds_dir="<wav files>" --out_file="<features csv>.csv" --feats_dir="<dir containing npy files for features>"
```

- Time taken to extract features of 50 audio files : 371.02 seconds


### Train the gender classifier

We are using an SVM classifier to label an audio into the 'Male' vs. 'Female' class.
```python
python train_gender_classifier.py --feats_file=gender_classifier_features.csv --model_name=gender_classifier_26_04_2021.pkl
```


