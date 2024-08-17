"""
Script to create a training and validation CSV file.
"""

import pandas as pd
import shutil
import os

def train_valid_split(all_images_folder=None, all_annots_csv=None, split=0.15):
    all_df = pd.read_csv(all_annots_csv)
    # Shuffle the CSV file rows.
    all_df.sample(frac=1)
    len_df = len(all_df)
    train_split = int((1-split)*len_df)

    train_df = all_df[:train_split]
    valid_df = all_df[train_split:]

    os.makedirs('..\\input\\train_images', exist_ok=True)
    os.makedirs('..\\input\\valid_images', exist_ok=True)

    # Copy training images.
    train_images = train_df['file_name'].tolist()
    for image in train_images:
        shutil.copy(
            f"..\\input\\TrainIJCNN2013\\{image}", 
            f"..\\input\\train_images\\{image}"
        )
    train_df.to_csv(r'..\input\train.csv', index=False)

    # Copy validation images.
    valid_images = valid_df['file_name'].tolist()
    for image in valid_images:
        shutil.copy(
            f"..\\input\\TrainIJCNN2013\\{image}", 
            f"..\\input\\valid_images\\{image}"
        )
    valid_df.to_csv(r'..\input\valid.csv', index=False)

train_valid_split(all_annots_csv=r'..\input\all_annots.csv')