import os
import pandas as pd
from utils_results_evaluation import (merge_result_dataframes_and_preprocess, compute_formal_accuracy_and_add_to_df,
                                      compute_social_accuracy_and_add_to_df, compute_pheno_accuracy_and_add_to_df,
                                      compute_intuitive_accuracy_and_add_to_df, delete_outliers_from_df,
                                      split_dataframe_by_treatment_group, split_dataframe_to_overall_image_results,
                                      split_dataframe_to_overall_image_results_by_treatment_group,
                                      add_overall_accuracy_and_delegation_rate_to_the_final_df,
                                      compute_average_accuracy_and_delegation_rate_and_add_to_new_df,
                                      add_ai_accuracy_and_delegation_rate_to_df, calculate_ai_delegation_group_accuracy,
                                      split_overall_dataframe_into_delegate_and_not_delegate_by_task)

# this file serves as the main to perform the necessary calculation in chronological order. It uses the two Excel files
# coming from the website and then creates new dataframe and Excel files as needed for the regression analysis
path_to_save = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results'
path_to_save_image = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/regression_results_images'
path_splitted_dfs = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/splitted_by_treatment_group'
path_regression = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/regression_files_participants'
# take the two results file, merge them and delete instances that either
# answered the "antibot"-question incorrectly, or didn't finish the experiment

path_read_participants = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/Participants.xlsx'
path_read_results = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/Results.xlsx'
df_merged, length = merge_result_dataframes_and_preprocess(path_read_participants, path_read_results)

# compute all accuracies and all delegation rates and add them to the dataframe
# formal:
image_formal_df = pd.read_excel('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Formal_predictions_small.xlsx')
df1 = compute_formal_accuracy_and_add_to_df(df_merged, image_formal_df)
# social:
image_social_df = pd.read_excel('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Social_predictions_small.xlsx')
df2 = compute_social_accuracy_and_add_to_df(df1, image_social_df)
# pheno:
image_pheno_df = pd.read_excel('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions'
                               '/Phenomenological_predictions_small.xlsx')
df3 = compute_pheno_accuracy_and_add_to_df(df2, image_pheno_df)
# intuitive:
image_intuitive_df = pd.read_excel('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions'
                                   '/Intuitive_predictions_small.xlsx')
df_acc = compute_intuitive_accuracy_and_add_to_df(df3, image_intuitive_df, path_to_save)
df_helper_acc = df_acc.copy()

# delete participants who did not solve the experiment adequately
delete_outliers_from_df(df_acc, path_to_save)
df_del = pd.read_excel('C:/Users/ferdi/OneDrive/Masterarbeit/human results/delete_outliers.xlsx')
df_helper_del = df_del.copy()

# preprocess data
# 1) drop unnecessary columns for the regression model:
df_del.drop(['session_id', 'formal_images', 'social_images', 'pheno_images', 'intuitive_images',
             'antibot'], axis=1, inplace=True)

# 2) convert gender variable to dummy variable:
gender_list = df_del['gender'].to_list()
gender_dummy = []
for gender in gender_list:
    if gender == 'male':
        dummy = 0
        gender_dummy.append(dummy)
    elif gender == 'female':
        dummy = 1
        gender_dummy.append(dummy)
    else:
        print('non_binary_gender: ', gender)
df_del['gender'] = gender_dummy

# preprocess country variable:
country_list = df_del['country'].to_list()
country_list_clean = []
for string in country_list:
    string_lower = string.lower()
    if string_lower == 'china' or string_lower == 'macao':
        string_final = 'China'
    elif string_lower == 'india':
        string_final = 'India'
    elif string_lower == 'brazil':
        string_final = 'Brazil'
    elif string_lower == 'germany' or string_lower == 'germany ':
        string_final = 'Germany'
    elif string_lower == 'spain':
        string_final = 'Spain'
    elif string_lower == 'greece':
        string_final = 'Greece'
    elif (string_lower == 'usa' or string_lower == 'united states' or string_lower == 'unitedstates of america'
          or string_lower == 'california' or string_lower == 'united states of america'
          or string_lower == 'united state' or string_lower == 'new york' or string_lower == 'united states america'
          or string_lower == 'us' or string_lower == 'texas' or string_lower == 'indiana' or string_lower == 'fl'
          or string_lower == 'louisville' or string_lower == 'georgia' or string_lower == 'nothing'
          or string_lower == 'usa ' or string_lower == 'unitedstates of america '):
        string_final = 'USA'
    elif string_lower == 'english':
        string_final = 'UK'
    else:
        print('country not assigned: ', string_lower)
        print('length: ', len(string_lower))
        string_final = string_lower
    country_list_clean.append(string_final)

# create dummy variables for the country data:
ger_dummy_list = []
usa_dummy_list = []
china_dummy_list = []
greece_dummy_list = []
india_dummy_list = []
brazil_dummy_list = []
spain_dummy_list = []
uk_dummy_list = []
for country in country_list_clean:
    if country == 'USA':
        usa_dummy = 1
    else:
        usa_dummy = 0
    usa_dummy_list.append(usa_dummy)

    if country == 'UK':
        uk_dummy = 1
    else:
        uk_dummy = 0
    uk_dummy_list.append(uk_dummy)

    if country == 'Germany':
        ger_dummy = 1
    else:
        ger_dummy = 0
    ger_dummy_list.append(ger_dummy)

    if country == 'Spain':
        spain_dummy = 1
    else:
        spain_dummy = 0
    spain_dummy_list.append(spain_dummy)

    if country == 'India':
        india_dummy = 1
    else:
        india_dummy = 0
    india_dummy_list.append(india_dummy)

    if country == 'Brazil':
        brazil_dummy = 1
    else:
        brazil_dummy = 0
    brazil_dummy_list.append(brazil_dummy)

    if country == 'Greece':
        greece_dummy = 1
    else:
        greece_dummy = 0
    greece_dummy_list.append(greece_dummy)

    if country == 'China':
        china_dummy = 1
    else:
        china_dummy = 0
    china_dummy_list.append(china_dummy)

df_del['country_usa'] = usa_dummy_list
df_del['country_ger'] = ger_dummy_list
df_del['country_spain'] = spain_dummy_list
df_del['country_brazil'] = brazil_dummy_list
df_del['country_india'] = india_dummy_list
df_del['country_china'] = brazil_dummy_list
df_del['country_greece'] = greece_dummy_list
df_del['country_uk'] = uk_dummy_list
df_del.drop('country', axis=1, inplace=True)

# create dummy variable for the competence data:
competency_list = df_del['competency'].to_list()
competency_high_list = []
competency_medium_list = []
competency_low_list = []

for comp in competency_list:
    if comp == 'High':
        competency_high = 1
    else:
        competency_high = 0
    competency_high_list.append(competency_high)

    if comp == 'Medium':
        competency_medium = 1
    else:
        competency_medium = 0
    competency_medium_list.append(competency_medium)

    if comp == 'Low':
        competency_low = 1
    else:
        competency_low = 0
    competency_low_list.append(competency_low)

df_del['competency_high'] = competency_high_list
df_del['competency_medium'] = competency_medium_list
df_del['competency_low'] = competency_low_list
df_del.drop('competency', axis=1, inplace=True)

# create dummy variable for the task order:
first_task_list = df_del['first_task'].to_list()
second_task_list = df_del['second_task'].to_list()
third_task_list = df_del['third_task'].to_list()
fourth_task_list = df_del['fourth_task'].to_list()
first_task_formal = []
first_task_social = []
first_task_pheno = []
first_task_intuitive = []
second_task_formal = []
second_task_social = []
second_task_pheno = []
second_task_intuitive = []
third_task_formal = []
third_task_social = []
third_task_pheno = []
third_task_intuitive = []
fourth_task_formal = []
fourth_task_social = []
fourth_task_pheno = []
fourth_task_intuitive = []

for task in first_task_list:
    if task == 'formal':
        first_task_formal_dummy = 1
    else:
        first_task_formal_dummy = 0
    first_task_formal.append(first_task_formal_dummy)

    if task == 'social':
        first_task_social_dummy = 1
    else:
        first_task_social_dummy = 0
    first_task_social.append(first_task_social_dummy)

    if task == 'pheno':
        first_task_pheno_dummy = 1
    else:
        first_task_pheno_dummy = 0
    first_task_pheno.append(first_task_pheno_dummy)

    if task == 'intuitive':
        first_task_intuitive_dummy = 1
    else:
        first_task_intuitive_dummy = 0
    first_task_intuitive.append(first_task_intuitive_dummy)

df_del['first_task_formal'] = first_task_formal
df_del['first_task_social'] = first_task_social
df_del['first_task_pheno'] = first_task_pheno
df_del['first_task_intuitive'] = first_task_intuitive

for task in second_task_list:
    if task == 'formal':
        second_task_formal_dummy = 1
    else:
        second_task_formal_dummy = 0
    second_task_formal.append(second_task_formal_dummy)

    if task == 'social':
        second_task_social_dummy = 1
    else:
        second_task_social_dummy = 0
    second_task_social.append(second_task_social_dummy)

    if task == 'pheno':
        second_task_pheno_dummy = 1
    else:
        second_task_pheno_dummy = 0
    second_task_pheno.append(second_task_pheno_dummy)

    if task == 'intuitive':
        second_task_intuitive_dummy = 1
    else:
        second_task_intuitive_dummy = 0
    second_task_intuitive.append(second_task_intuitive_dummy)

df_del['second_task_formal'] = second_task_formal
df_del['second_task_social'] = second_task_social
df_del['second_task_pheno'] = second_task_pheno
df_del['second_task_intuitive'] = second_task_intuitive

for task in third_task_list:
    if task == 'formal':
        third_task_formal_dummy = 1
    else:
        third_task_formal_dummy = 0
    third_task_formal.append(third_task_formal_dummy)

    if task == 'social':
        third_task_social_dummy = 1
    else:
        third_task_social_dummy = 0
    third_task_social.append(third_task_social_dummy)

    if task == 'pheno':
        third_task_pheno_dummy = 1
    else:
        third_task_pheno_dummy = 0
    third_task_pheno.append(third_task_pheno_dummy)

    if task == 'intuitive':
        third_task_intuitive_dummy = 1
    else:
        third_task_intuitive_dummy = 0
    third_task_intuitive.append(third_task_intuitive_dummy)

df_del['third_task_formal'] = third_task_formal
df_del['third_task_social'] = third_task_social
df_del['third_task_pheno'] = third_task_pheno
df_del['third_task_intuitive'] = third_task_intuitive

for task in fourth_task_list:
    if task == 'formal':
        fourth_task_formal_dummy = 1
    else:
        fourth_task_formal_dummy = 0
    fourth_task_formal.append(fourth_task_formal_dummy)

    if task == 'social':
        fourth_task_social_dummy = 1
    else:
        fourth_task_social_dummy = 0
    fourth_task_social.append(fourth_task_social_dummy)

    if task == 'pheno':
        fourth_task_pheno_dummy = 1
    else:
        fourth_task_pheno_dummy = 0
    fourth_task_pheno.append(fourth_task_pheno_dummy)

    if task == 'intuitive':
        fourth_task_intuitive_dummy = 1
    else:
        fourth_task_intuitive_dummy = 0
    fourth_task_intuitive.append(fourth_task_intuitive_dummy)

df_del['fourth_task_formal'] = fourth_task_formal
df_del['fourth_task_social'] = fourth_task_social
df_del['fourth_task_pheno'] = fourth_task_pheno
df_del['fourth_task_intuitive'] = fourth_task_intuitive

df_del.to_excel(path_to_save + '/preprocessed.xlsx')

# split the dataframe by the treatment groups and save them:
split_dataframe_by_treatment_group(df_del, path_splitted_dfs, length,
                                   image_formal_df, image_social_df, image_pheno_df, image_intuitive_df)

# compute average accuracies and delegation rates and save them in new dataframe:
file_list = os.listdir(path_splitted_dfs)

name_list = []
for file in file_list:
    name = file.split('.')[0]
    name_list.append(name)

df_list = []
for file in file_list:
    file_path = os.path.join(path_splitted_dfs, file)
    df = pd.read_excel(file_path)
    df_list.append(df)

df_final = compute_average_accuracy_and_delegation_rate_and_add_to_new_df(df_list, name_list)

# add overall acc and delegation rate to the final df for the complete merged df and for the merged df without outliers:
df_overall = add_overall_accuracy_and_delegation_rate_to_the_final_df(df_final, df_helper_acc, 'merged')
df_without_outliers = add_overall_accuracy_and_delegation_rate_to_the_final_df(df_overall, df_helper_del, 'outliers')

# add AI accuracy and delegation_rates to the final df, then finally save it:
add_ai_accuracy_and_delegation_rate_to_df(df_without_outliers, path_to_save)

# in a next step, now build a file for the accuracies and delegation rates for each single image and for each task:
create_image_files = True
if create_image_files:
    split_dataframe_to_overall_image_results_by_treatment_group(image_formal_df, image_social_df, image_pheno_df,
                                                                image_intuitive_df, path_splitted_dfs,
                                                                path_to_save_image)
    split_dataframe_to_overall_image_results(image_formal_df, image_social_df, image_pheno_df, image_intuitive_df,
                                             path_to_save, path_to_save_image)

# drop all columns that aren't needed for regression and reorder the columns
# 1) for the overall file that contains all participants
df_regression_overall = pd.read_excel(path_to_save + '/preprocessed.xlsx')
df_regression_overall.drop(['Unnamed: 0', 'answers_formal', 'answers_social', 'answers_pheno', 'answers_intuitive',
                            'first_task', 'second_task', 'third_task', 'fourth_task'],
                           axis=1, inplace=True)
new_columns_overall = ['experiment_group', 'gender', 'age', 'country_usa', 'country_ger', 'country_uk', 'country_india',
                       'country_china', 'country_spain', 'country_brazil', 'country_greece', 'first_task_formal',
                       'first_task_social', 'first_task_pheno', 'first_task_intuitive', 'second_task_formal',
                       'second_task_social', 'second_task_pheno', 'second_task_intuitive', 'third_task_formal',
                       'third_task_social', 'third_task_pheno', 'third_task_intuitive', 'fourth_task_formal',
                       'fourth_task_social', 'fourth_task_pheno', 'fourth_task_intuitive', 'competency_high',
                       'competency_medium', 'competency_low', 'expectation', 'perceived_help', 'complexity_formal',
                       'complexity_social', 'complexity_pheno', 'complexity_intuitive', 'negative_prophecy_formal',
                       'positive_prophecy_formal', 'negative_prophecy_social', 'positive_prophecy_social',
                       'negative_prophecy_pheno', 'positive_prophecy_pheno', 'negative_prophecy_intuitive',
                       'positive_prophecy_intuitive', 'accuracy_formal',
                       'accuracy_social', 'accuracy_pheno', 'accuracy_intuitive', 'delegation_rate_formal',
                       'delegation_rate_social', 'delegation_rate_pheno', 'delegation_rate_intuitive', ]
df_regression_overall = df_regression_overall[new_columns_overall]

df_regression_overall.to_excel(path_to_save + '/regression_overall.xlsx', index=False)
split_overall_dataframe_into_delegate_and_not_delegate_by_task(df_regression_overall, path_regression)

# 2) for the treatment group files which only contains participants of specific treatment groups:
file_list = os.listdir(path_splitted_dfs)

for file in file_list:
    treatment_group = file.split('.')[0]
    file_path = os.path.join(path_splitted_dfs, file)
    df_regression_treatment = pd.read_excel(file_path)
    df_regression_treatment.drop(['experiment_group', 'answers_formal', 'answers_social', 'answers_pheno',
                                  'answers_intuitive', ], axis=1, inplace=True)

    if treatment_group == 'ai_delegation' or treatment_group == 'baseline' or treatment_group == 'human_delegation':
        new_columns_del = ['gender', 'age', 'country_usa', 'country_ger', 'country_uk', 'country_india',
                           'country_china', 'country_spain', 'country_brazil', 'country_greece', 'first_task_formal',
                           'first_task_social', 'first_task_pheno', 'first_task_intuitive', 'second_task_formal',
                           'second_task_social', 'second_task_pheno', 'second_task_intuitive', 'third_task_formal',
                           'third_task_social', 'third_task_pheno', 'third_task_intuitive', 'fourth_task_formal',
                           'fourth_task_social', 'fourth_task_pheno', 'fourth_task_intuitive', 'competency_high',
                           'competency_medium', 'competency_low', 'expectation', 'perceived_help', 'complexity_formal',
                           'complexity_social', 'complexity_pheno', 'complexity_intuitive', 'negative_prophecy_formal',
                           'positive_prophecy_formal', 'negative_prophecy_social', 'positive_prophecy_social',
                           'negative_prophecy_pheno', 'positive_prophecy_pheno', 'negative_prophecy_intuitive',
                           'positive_prophecy_intuitive', 'accuracy_formal',
                           'accuracy_social', 'accuracy_pheno', 'accuracy_intuitive', 'delegation_rate_formal',
                           'delegation_rate_social', 'delegation_rate_pheno', 'delegation_rate_intuitive', ]
        df_regression_treatment = df_regression_treatment[new_columns_del]

        if treatment_group == 'baseline' or treatment_group == 'ai_delegation':
            df_regression_treatment.drop(['expectation', 'perceived_help', 'delegation_rate_formal',
                                          'delegation_rate_social', 'delegation_rate_pheno', 'delegation_rate_intuitive'
                                             , 'negative_prophecy_formal', 'positive_prophecy_formal',
                                          'negative_prophecy_social', 'positive_prophecy_social',
                                          'negative_prophecy_pheno', 'positive_prophecy_pheno',
                                          'negative_prophecy_intuitive', 'positive_prophecy_intuitive'],
                                         axis=1, inplace=True)

            new_columns_no_del = ['gender', 'age', 'country_usa', 'country_ger', 'country_uk', 'country_india',
                                  'country_china', 'country_spain', 'country_brazil', 'country_greece',
                                  'first_task_formal', 'first_task_social', 'first_task_pheno', 'first_task_intuitive',
                                  'second_task_formal', 'second_task_social', 'second_task_pheno',
                                  'second_task_intuitive', 'third_task_formal', 'third_task_social', 'third_task_pheno',
                                  'third_task_intuitive', 'fourth_task_formal', 'fourth_task_social',
                                  'fourth_task_pheno', 'fourth_task_intuitive', 'competency_high',
                                  'competency_medium', 'competency_low', 'complexity_formal', 'complexity_social',
                                  'complexity_pheno', 'complexity_intuitive', 'accuracy_formal', 'accuracy_social',
                                  'accuracy_pheno', 'accuracy_intuitive']
            df_regression_treatment = df_regression_treatment[new_columns_no_del]
    else:
        if treatment_group == 'delegation_counter':
            new_columns_del = ['gender', 'age', 'order', 'country_usa', 'country_ger', 'country_uk', 'country_india',
                               'country_china', 'country_spain', 'country_brazil', 'country_greece',
                               'first_task_formal', 'first_task_social', 'first_task_pheno', 'first_task_intuitive',
                               'second_task_formal', 'second_task_social', 'second_task_pheno', 'second_task_intuitive',
                               'third_task_formal', 'third_task_social', 'third_task_pheno', 'third_task_intuitive',
                               'fourth_task_formal', 'fourth_task_social', 'fourth_task_pheno', 'fourth_task_intuitive',
                               'competency_high', 'competency_medium', 'competency_low', 'expectation',
                               'perceived_help', 'complexity_formal', 'complexity_social', 'complexity_pheno',
                               'complexity_intuitive', 'negative_prophecy_formal',
                               'positive_prophecy_formal', 'negative_prophecy_social', 'positive_prophecy_social',
                               'negative_prophecy_pheno', 'positive_prophecy_pheno', 'negative_prophecy_intuitive',
                               'positive_prophecy_intuitive', 'accuracy_formal', 'accuracy_social', 'accuracy_pheno',
                               'accuracy_intuitive', 'delegation_rate_formal', 'delegation_rate_social',
                               'delegation_rate_pheno', 'delegation_rate_intuitive']
            df_regression_treatment = df_regression_treatment[new_columns_del]
        else:
            new_columns_del = ['gender', 'age', 'negative_feedback_first', 'negative_feedback_second',
                               'negative_feedback_third', 'negative_feedback_fourth', 'country_usa', 'country_ger',
                               'country_uk', 'country_india', 'country_china', 'country_spain', 'country_brazil',
                               'country_greece', 'first_task_formal', 'first_task_social', 'first_task_pheno',
                               'first_task_intuitive', 'second_task_formal', 'second_task_social', 'second_task_pheno',
                               'second_task_intuitive', 'third_task_formal', 'third_task_social', 'third_task_pheno',
                               'third_task_intuitive', 'fourth_task_formal', 'fourth_task_social', 'fourth_task_pheno',
                               'fourth_task_intuitive', 'competency_high', 'competency_medium', 'competency_low',
                               'expectation', 'perceived_help', 'complexity_formal', 'complexity_social',
                               'complexity_pheno', 'complexity_intuitive', 'negative_prophecy_formal',
                               'positive_prophecy_formal', 'negative_prophecy_social', 'positive_prophecy_social',
                               'negative_prophecy_pheno', 'positive_prophecy_pheno', 'negative_prophecy_intuitive',
                               'positive_prophecy_intuitive', 'accuracy_formal', 'accuracy_social',
                               'accuracy_pheno', 'accuracy_intuitive', 'delegation_rate_formal',
                               'delegation_rate_social', 'delegation_rate_pheno', 'delegation_rate_intuitive']
            df_regression_treatment = df_regression_treatment[new_columns_del]

    df_regression_treatment.to_excel(path_splitted_dfs + '/' + treatment_group + '.xlsx', index=False)

# lastly, build a file for regression that only has the data for each respective task
file_list = os.listdir(path_splitted_dfs)

for file in file_list:
    treatment_group = file.split('.')[0]
    file_path = os.path.join(path_splitted_dfs, file)
    df_regression = pd.read_excel(file_path)

    if treatment_group == 'baseline' or treatment_group == 'ai_delegation':
        df_regression_formal = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 18, 22, 26, 27, 28, 29, 33]]
        df_regression_social = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15, 19, 23, 26, 27, 28, 30, 34]]
        df_regression_pheno = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 26, 27, 28, 31, 35]]
        df_regression_intuitive = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17, 21, 25, 26, 27, 28, 32,
                                                         36]]
    elif treatment_group == 'delegation_counter':
        df_regression_formal = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 19, 23, 27, 28, 29, 30,
                                                      31, 32, 36, 37, 44, 48]]
        df_regression_social = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 27, 28, 29, 30,
                                                      31, 33, 38, 39, 45, 49]]
        df_regression_pheno = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 17, 21, 25, 27, 28, 29, 30,
                                                     31, 34, 40, 41, 46, 50]]
        df_regression_intuitive = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 18, 22, 26, 27, 28, 29,
                                                         30, 31, 35, 42, 43, 47, 51]]
    elif treatment_group == 'feedback':
        df_regression_formal = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 22, 26, 30,
                                                      31, 32, 33, 34, 35, 39, 40, 47, 51]]
        df_regression_social = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 19, 23, 27, 30,
                                                      31, 32, 33, 34, 36, 41, 42, 48, 52]]
        df_regression_pheno = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 20, 24, 28, 30,
                                                     31, 32, 33, 34, 37, 43, 44, 49, 53]]
        df_regression_intuitive = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 21, 25, 29,
                                                         30, 31, 32, 33, 34, 38, 45, 46, 50, 54]]
    else:
        df_regression_formal = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 18, 22, 26, 27, 28, 29, 30,
                                                      31, 35, 36, 43, 47]]
        df_regression_social = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15, 19, 23, 26, 27, 28, 29, 30,
                                                      32, 37, 38, 44, 48]]
        df_regression_pheno = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 26, 27, 28, 29, 30,
                                                     33, 39, 40, 45, 49]]
        df_regression_intuitive = df_regression.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17, 21, 25, 26, 27, 28, 29,
                                                         30, 34, 41, 42, 46, 50]]

    df_regression_formal.to_excel(path_regression + '/' + treatment_group + '_formal.xlsx', index=False)
    df_regression_social.to_excel(path_regression + '/' + treatment_group + '_social.xlsx', index=False)
    df_regression_pheno.to_excel(path_regression + '/' + treatment_group + '_pheno.xlsx', index=False)
    df_regression_intuitive.to_excel(path_regression + '/' + treatment_group + '_intuitive.xlsx', index=False)

# add AI -> human delegation accuracies to the result dataframe:
df_del_start = pd.read_excel('C:/Users/ferdi/OneDrive/Masterarbeit/human results/delete_outliers.xlsx')
df_image_results_formal = pd.read_excel(
    'C:/Users/ferdi/OneDrive/Masterarbeit/human results/regression_results_images/image_results_formal.xlsx')
df_image_results_social = pd.read_excel(
    'C:/Users/ferdi/OneDrive/Masterarbeit/human results/regression_results_images/image_results_social.xlsx')
df_image_results_pheno = pd.read_excel(
    'C:/Users/ferdi/OneDrive/Masterarbeit/human results/regression_results_images/image_results_pheno.xlsx')
df_image_results_intuitive = pd.read_excel(
    'C:/Users/ferdi/OneDrive/Masterarbeit/human results/regression_results_images/image_results_intuitive.xlsx')

image_formal_df = pd.read_excel('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Formal_predictions_small.xlsx')
image_social_df = pd.read_excel('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Social_predictions_small.xlsx')
image_pheno_df = pd.read_excel('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions'
                               '/Phenomenological_predictions_small.xlsx')
image_intuitive_df = pd.read_excel('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions'
                                   '/Intuitive_predictions_small.xlsx')
final_results_df = pd.read_excel('C:/Users/ferdi/OneDrive/Masterarbeit/human results/final_results.xlsx')
feedback_splitted_df = pd.read_excel(
    'C:/Users/ferdi/OneDrive/Masterarbeit/human results/splitted_by_treatment_group/feedback.xlsx')
human_delegation_splitted_df = pd.read_excel(
    'C:/Users/ferdi/OneDrive/Masterarbeit/human results/splitted_by_treatment_group/human_delegation.xlsx')
delegation_counter_splitted_df = pd.read_excel(
    'C:/Users/ferdi/OneDrive/Masterarbeit/human results/splitted_by_treatment_group/delegation_counter.xlsx')

formal_acc, social_acc, pheno_acc, intuitive_acc = (calculate_ai_delegation_group_accuracy
                                                    (df_del_start, image_formal_df, image_social_df, image_pheno_df,
                                                     image_intuitive_df, df_image_results_formal,
                                                     df_image_results_social, df_image_results_pheno,
                                                     df_image_results_intuitive, ))
df_difficulty = df_del_start.loc[:, ['complexity_formal', 'complexity_social', 'complexity_pheno', 'complexity_intuitive']]
avg_formal_diff = df_difficulty['complexity_formal'].mean()
avg_social_diff = df_difficulty['complexity_social'].mean()
avg_pheno_diff = df_difficulty['complexity_pheno'].mean()
avg_intuitive_diff = df_difficulty['complexity_intuitive'].mean()
diff_list = [avg_formal_diff, avg_social_diff, avg_pheno_diff, avg_intuitive_diff, 0, 0, 0, 0]
ai_del_rate = final_results_df.loc[:, ['AI']].tail(4).to_numpy().tolist()
ai_del_rate_list = [x for xs in ai_del_rate for x in xs]
acc_list = [formal_acc, social_acc, pheno_acc, intuitive_acc] + ai_del_rate_list
feedback_series = final_results_df['feedback'].to_list()
human_delegation_series = final_results_df['human_delegation'].to_list()
delegation_counter_series = final_results_df['delegation_counter'].to_list()
human_del_acc_list = []
for i in range(len(feedback_series)):
    human_del_acc = ((feedback_series[i] * len(feedback_splitted_df) +
                      human_delegation_series[i] * len(human_delegation_splitted_df) +
                      delegation_counter_series[i] * len(delegation_counter_splitted_df))
                     / (len(feedback_splitted_df) + len(human_delegation_splitted_df) + len(delegation_counter_splitted_df)))
    human_del_acc_list.append(human_del_acc)

final_results_df['AI -> human delegation'] = acc_list
final_results_df['Human -> AI delegation'] = human_del_acc_list
final_results_df['Perceived Difficulty'] = diff_list
final_results_df.to_excel('C:/Users/ferdi/OneDrive/Masterarbeit/human results/final_results.xlsx', index=False)

