## Prediction Pipeline

- The assumption is that the complete data is in a single excel sheet

```python
python download_audios.py --in_file='data/all_banking_data.xlsx' --aud_col_name='Where are you currently talking from?' --out_folder_mp3=audio_files_mp3

python convert_to_wav.py --in_folder=audio_files_mp3/ --out_folder=audio_files_wav/

python get_features.py --in_file='data/all_banking_data.xlsx' --aud_col_name='Where are you currently talking from?' --auds_dir=audio_files_wav --out_file=banking_genders_features.csv --feats_dir=feats_dir

python make_predictions.py --in_file=data/all_banking_data.xlsx --aud_col_name='Where are you currently talking from?' --feats_file='banking_genders_features.csv' --model_name=../gender_classifier_temp_17_05_2021.pkl --scaler_name=../gender_scaler --out_file=data/banking_data_pred_genders.xlsx
```

