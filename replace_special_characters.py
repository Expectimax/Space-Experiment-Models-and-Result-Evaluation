import os
import pandas as pd
# this file is used to exclude any special characters in the filenames, for both the filenames on the disk and
# the filenames in the Excel files
folder = 'C:/Users/ferdi/Formal_test_small'
excel_path = 'C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Phenomenological_predictions.xlsx'
excel_path_save = ('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions'
                   '/Phenomenological_predictions_replaced_characters.xlsx')


def replace_special_characters_in_filenames(path):
    filenames = os.listdir(path)
    for file in filenames:
        if 'ä' or 'Ä' or 'ü' or 'Ü' or 'ö' or 'Ö' or 'ß' or ' ' in file:
            new_name = file.replace('ä', 'ae').replace('Ä', 'Ae').replace('ö', 'oe').replace('Ö', 'Oe').replace('ü',
                                                                                                                'ue').replace(
                'Ü', 'Ue').replace('ß', 'ss').replace(' ', '_').replace(',', '').replace(':', '')

            os.rename(os.path.join(path, file), os.path.join(path, new_name))
    print('Successfully replaced the special characters in the filenames')


def replace_special_characters_in_excel_files(filepath):
    df = pd.read_excel(filepath)
    for index in df.index:
        image_name = df.loc[index, 'Filename']
        if 'Ä' or 'Ö' or 'Ü' or 'ä' or 'ö' or 'ü' or 'ß' or ' ' in image_name:
            new_name = (image_name.replace('ä', 'ae').replace('Ä', 'Ae').replace('ö', 'oe').replace('Ö', 'Oe')
                        .replace('ü', 'ue').replace('Ü', 'Ue').replace('ß', 'ss').replace(' ', '_').replace(',', '')
                        .replace(':', ''))
            df.loc[index, 'Filename'] = new_name
        else:
            pass
    df.to_excel(excel_path_save, index=False)

