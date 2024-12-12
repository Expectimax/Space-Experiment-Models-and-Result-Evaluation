from utils_results_evaluation import (perform_participant_regression_delegate, gender_regression,
                                      perform_participant_regression_not_delegate, perform_image_regression,
                                      age_regression, herd_regression, self_fulfilling_prophecy_regression,
                                      feedback_regression, outlier_detection_z_score, outlier_detection_iqr)
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# this file is used as the "main" to perform the regression analysis for the different cases
path_delegate = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/regression_files_participants/can_delegate'
path_not_delegate = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/regression_files_participants/cannot_delegate'
image_path = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/regression_results_images/seperated_diff_and_easy'
path_to_save = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/regression_results_participants'
image_path_to_save = ('C:/Users/ferdi/OneDrive/Masterarbeit/human '
                      'results/image_regression_results_final')
main_path = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results'
splitted_path = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/splitted_by_treatment_group'
age_path_to_save = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/age_results'
order_path_to_save = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/herd_behavior_results'
sfp_path_to_save = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/self-fulfilling_prophecy_results'
feedback_path_to_save = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/feedback_results'

# 1) perform multiclass regression for the participants. Set the boolean variables to True or False to only perform
# parts of the calculations
not_delegate = False
delegate = False
image = False
age = False
gender = False
order = False
sfp = True
feedback = False

# perform the multivariate regression for all participants who were allowed to delegate
if delegate:
    file_list = os.listdir(path_delegate)
    for file in file_list:
        file_path = os.path.join(path_delegate, file)
        setting = file.split('.')[0]
        df = pd.read_excel(file_path)
        if '_formal' in setting:
            df.drop(['negative_prophecy_formal', 'positive_prophecy_formal'], axis=1, inplace=True)
        elif '_social' in setting:
            df.drop(['negative_prophecy_social', 'positive_prophecy_social'], axis=1, inplace=True)
        elif '_pheno' in setting:
            df.drop(['negative_prophecy_pheno', 'positive_prophecy_pheno'], axis=1, inplace=True)
        elif '_intuitive' in setting:
            df.drop(['negative_prophecy_intuitive', 'positive_prophecy_intuitive'], axis=1, inplace=True)

        df.drop(['country_usa', 'country_ger', 'country_uk', 'country_india', 'country_china', 'country_spain',
                 'country_brazil', 'country_greece'], axis=1, inplace=True)
        X = df.iloc[:, :-2]
        y_acc = df.iloc[:, -2]
        y_del = df.iloc[:, -1]

        perform_participant_regression_delegate(X, y_acc, y_del, path_to_save, setting=setting)

# perform the multivariate regression for all participants who were not allowed to delegate
if not_delegate:
    file_list = os.listdir(path_not_delegate)
    for file in file_list:
        file_path = os.path.join(path_not_delegate, file)
        setting = file.split('.')[0]
        df = pd.read_excel(file_path)
        df.drop(['country_usa', 'country_ger', 'country_uk', 'country_india', 'country_china', 'country_spain',
                 'country_brazil', 'country_greece'], axis=1, inplace=True)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        perform_participant_regression_not_delegate(X, y, path_to_save, setting=setting)

# Perform single class regression for the images:
if image:
    image_file_list = os.listdir(image_path)
    for file in image_file_list:
        if 'overall' in file:
            setting = file.split('.')[0]
            image_regression_df = pd.read_excel(os.path.join(image_path, file))
            x = pd.DataFrame(image_regression_df.iloc[:, 1])
            y = image_regression_df.iloc[:, 2]
            outlier_index_z = outlier_detection_z_score(y)
            outlier_index_iqr = outlier_detection_iqr(y)
            to_drop = outlier_index_z + outlier_index_iqr
            to_drop_unique = list(set(to_drop))
            x = x.drop(index=to_drop_unique)
            y = y.drop(index=to_drop_unique)
            x_array = x.to_numpy()
            x_plot = [x for xs in x_array for x in xs]
            y_plot = y.to_numpy()
            sns.scatterplot(x=x_plot, y=y_plot, marker='o', facecolor='none', edgecolor='blue', )
            plt.title(setting)
            plt.ylabel('Delegation Rate')
            plt.xlabel('Accuracy')
            plt.yticks(np.arange(0, 1.0, 0.1))
            plt.savefig(image_path_to_save + '/' + setting + '.png')
            plt.clf()
            perform_image_regression(x, y, image_path_to_save, setting=setting)

# Perform single class regression
# a) age
if age:
    df_age = pd.read_excel(main_path + '/regression_delegate.xlsx')
    age_regression(df_age, age_path_to_save, mode='formal')
    age_regression(df_age, age_path_to_save, mode='social')
    age_regression(df_age, age_path_to_save, mode='pheno')
    age_regression(df_age, age_path_to_save, mode='intuitive')
# b) gender
if gender:
    gender_path_to_save = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/gender_results'
    df_gender = pd.read_excel(main_path + '/regression_delegate.xlsx')
    gender_regression(df_gender, gender_path_to_save, mode='formal')
    gender_regression(df_gender, gender_path_to_save, mode='social')
    gender_regression(df_gender, gender_path_to_save, mode='pheno')
    gender_regression(df_gender, gender_path_to_save, mode='intuitive')
# c) order
if order:
    df_order = pd.read_excel(main_path + '/regression_delegate.xlsx')
    herd_regression(df_order, order_path_to_save, mode='formal')
    herd_regression(df_order, order_path_to_save, mode='social')
    herd_regression(df_order, order_path_to_save, mode='pheno')
    herd_regression(df_order, order_path_to_save, mode='intuitive')
# d) self-fulfilling prophecy
if sfp:
    df_sfp = pd.read_excel(main_path + '/delete_outliers.xlsx')
    self_fulfilling_prophecy_regression(df_sfp, sfp_path_to_save, mode='first')
    self_fulfilling_prophecy_regression(df_sfp, sfp_path_to_save, mode='second')
    self_fulfilling_prophecy_regression(df_sfp, sfp_path_to_save, mode='third')
    self_fulfilling_prophecy_regression(df_sfp, sfp_path_to_save, mode='fourth')

# e) early negative feedback
if feedback:
    df_feedback = pd.read_excel(splitted_path + '/feedback.xlsx')
    feedback_regression(df_feedback, feedback_path_to_save, mode='first')
    feedback_regression(df_feedback, feedback_path_to_save, mode='second')
    feedback_regression(df_feedback, feedback_path_to_save, mode='third')
    feedback_regression(df_feedback, feedback_path_to_save, mode='fourth')




