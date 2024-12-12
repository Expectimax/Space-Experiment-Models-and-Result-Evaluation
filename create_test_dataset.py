# this is a simple script to take a specified amount of the initial dataset
# and store it at a different path for testing after the model is trained

import os
import random
import shutil
import pandas as pd


initial_path = "C:/Users/ferdi/Phenomenological_test"
new_path = "C:/Users/ferdi/Phenomenological_test_small"


def create_test_dataset(test_data_ratio, load_path, save_path):
    dirs = os.listdir(load_path)
    for directory in dirs:
        dir_path = os.path.join(load_path, directory)
        path_to_move = os.path.join(save_path, directory)
        filelist = os.listdir(dir_path)
        n_to_move = int(round(test_data_ratio * len(filelist), 0))
        files_to_move = random.sample(filelist, n_to_move)
        for file in files_to_move:
            old_path = os.path.join(dir_path, file)
            new = os.path.join(path_to_move, file)
            shutil.move(old_path, new)


def draw_subset_of_test_set(n, load_path, save_path):
    images = os.listdir(load_path)
    sample = random.sample(images, n)
    for image in sample:
        old_path = os.path.join(load_path, image)
        new_path = os.path.join(save_path, image)
        shutil.move(old_path, new_path)


def create_excel_for_small_test_set(excel_load_path, excel_save_path, folder):
    df = pd.read_excel(excel_load_path)
    small_test_set = os.listdir(folder)
    for index in df.index:
        image_name = df.loc[index, 'Filename']
        if image_name in small_test_set:
            pass
        else:
            df.drop(index, inplace=True)
    df.to_excel(excel_save_path, index=False)


excel_load_path = "C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Intuitive_predictions.xlsx"
excel_save_path = "C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Intuitive_predictions_small.xlsx"
folder = "C:/Users/ferdi/Intuitive_test_small"

create_excel_for_small_test_set(excel_load_path, excel_save_path, folder)
