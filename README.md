# Accept Reject Classifier

# GV’s guide to the Blank classifier

This guide is an example documentation of the broader set of ML pipeline principles that we can start adopting and have been specified [here](https://docs.google.com/document/d/1FXRsISYdjVHRf6YVyfGMbYuA3TenWfHfb0SebJ6Cw54/edit?usp=sharing).

This document deals with the following topics:
- Data Acquisition
- Data Processing
- Training
- Evaluation
- Continuous data extraction for ground truth

## Data Acquisition

The training and the test data was obtained from the <strong>dim_items</strong> table, present in the <strong>BI</strong> server.

> This step can be done using the existing scripts to fetch the data from the Alpha server as well.

The fields included for each item include - instance id, item id, state, audio url, title, transcription, rating, gender, source and tags. Date of creation should also be included in the data. These fields have been explained [here](https://docs.google.com/document/d/1c14cZRFRo5_2_oKxkCfqVFDgEIPyHFGU3qflhmO4oA8/edit?usp=sharing). 

SQL command for generating data from the dim items table:

```
select instance_id, id, current_state, audio_url, title, transcription, rating, gender, source, tags from dim_items where creation_time between <start_data> and <end_date>;
```


SQL command for alpha server (mnews table):

```
select ai_id, id, state, concat('http://voice.gramvaani.org/fsmedia/recordings/',ai_id,'/',detail_id,'.mp3') as audio_url, title, tags, transcript, rating, gender_id, source, time from mnews_news where time like "%2020-09-09%" FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';
```

> Store this in an excel file and copy the file to your local machine.

- The above command will give all the items in REJ (other than blank reason) as well as UNM items
- Hence, please extract those items which are accepted with a rating 4,5 or rejected because of blank reason as is explained below

## Data Processing

### Clean the data

The data used for training is a subset of all the collected data from the previous step. 

The training data comprises of acceptable items as well as rejectionable items. In order to increase the margin between acceptable items and rejectionable items, we take the items with following supporting fields :

#### Acceptable Item
- State of the item is ‘PUB’ <b>or</b> state of the item is ‘ARC’ with rating 4 or 5.

#### Rejectionable Item
- State of the item is ‘REJ’ <b>and</b> rating of the item is : 0,1 or 2 <b>and</b> tag is ‘blank’ or ‘noisy’.

<compile_latest_data.py> script is responsible for creating the required clean data with 50,000 items with 25,000 accepted items and 25,000 rejected items.

<code>compile_blank_classifier_dataset.py</code> script is responsible for taking the subset of only the acceptable and rejectionable items for training. This can be avoided if in the acquisition phase, we take only this subset and not create a wider set.

```python
python compile_latest_data.py --input_file_pth="<the file from previous step.xlsx>" --out_file_name="<Name of out file.xlsx>"
```

> <b>Note</b>: Ensure that the number of accepted and rejected items are the same if you require a distribution other than 25k items.

> <b>TODO</b>: Take the required items as a command line argument and automate the output acceptable and rejected items accordingly.

### Download the audios and convert them into wav files

Download the audios of only the clean data as produced in the previous step and then convert them into wav files which are easier for feature extraction.

Download audio files:
```python
python download_audios.py --in_file="<in_file>.xlsx" --out_folder_mp3="<folder name>"
```

Convert mp3 files to wav files:
```python
python convert_to_wav.py --in_folder="<folder created in previous step>" --out_folder="<folder containing wav audios>"
```

### Audio features extraction

The 136 + 1 (audio length) features are extracted for all the audio files using this script. Corresponding feature files in the form of numpy arrays are saved. The audio features are extracted using [PyAudioAnalysis library](https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction). 4 values (first quartile, median, third quartile and max) from the 34 features are extracted to get the 136 (34*4) features. The last feature (+1)corresponds to audio length and may or may not be used according to the use-case.

- We use a 50ms framesize with a 25ms stepsize
- Hence, for example if we have an audio of 100ms, there will be 3 frames and for a 1min audio there will be 2399 frames
- There will be 34 features for each frame
- Hence the feature matrix size will be 34*2399 for a 1 minute audio
- After that we take the first quartile, median, third quartile and max value for each feature across all the frames
- We finally get a 34*4=136 feature vector for any audio file (in the script we are taking 1 feature for length of audio but did not use it eventually)

Extract and store audio features - in both npy files and csv format:
```python
python get_features.py --in_file="<cleaned data file>.xlsx" --auds_dir="<wav files>" --out_file="<features csv>.csv" --feats_dir="<dir containing npy files for features>"
```

> Note : This step takes the maximum amount of time. It took around 1 second per audio file for feature extraction. Hence it is important to keep storing these features along the way.


### Divide the data into train and test splits

Randomize the data and divide 80% of the data into the training set and 20% of the data into the test set.
```python
python create_train_test_splits.py --feats_csv="<out fie created in previous step>" --train_csv="csv containing training set" --test_csv="<csv containing test set>"
```

## Training

##### New version for blank classifier
Upload the training data to Google Drive and run the created jupyter notebook in [Google Colab](https://colab.research.google.com/drive/1GidaFgeJ70MZIgamq2FSofL99xmaGDfj?usp=sharing) (Please use ODT account).

##### Older version for noisy+blank classifier
Upload the training data to Google Drive and run the created jupyter notebook in [Google Colab](https://colab.research.google.com/drive/1TuHW9jskxx9Snhe817MzxeuMgOi82uDl?usp=sharing) (Please use ODT account).

## Evaluation

##### New version for blank classifier
Upload the test data to Google drive and run the test cell in the [Google Colab](https://colab.research.google.com/drive/1GidaFgeJ70MZIgamq2FSofL99xmaGDfj?usp=sharing) (Please use ODT account).

##### Older version for noisy+blank classifier
Upload the test data to Google drive and run the test cell in the [Google Colab](https://colab.research.google.com/drive/1TuHW9jskxx9Snhe817MzxeuMgOi82uDl?usp=sharing) (Please use ODT account).

## Test Ad-hoc

Run this script to test any url or a list of urls in an ad-hoc manner. If just a single url needs to be tested, pass only that url in the `complete_file_pth` command line argument else if many urls have to be tested, write them line by line in a text file and pass that file as the argument `complete_file_pth`.

```python
python test_new_ad_hoc.py --complete_file_pth="<audio url> or <text file containing urls>" --model_ckpt="<trained pickle model>" --scaler_model_ckpt="trained scaler normalization model" --audio_dest_pth_mp3="<folder containing mp3 files>" --audio_dest_pth_wav="<folder containing wav files>" --feats_dir="<folder containing audio features>" --use_length_feat="<True> or <False>"
```
## New ground truth

In order to get new ground truth, we should aim at getting around 't' new items each month.
We should stop the ML predicitons from <a,b,c> instances for the first 'x' days each month.

