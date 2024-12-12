import pandas as pd
import json
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os

# different functions to do the statistical evaluation


def merge_result_dataframes_and_preprocess(path_read_participants, path_read_results):

    # the data from the website comes in the form of two Excel files. One with the data of the participants and one
    # with each answer. They are linked via a primary key session_id which is a cookie that was saved in each
    # participant's browser. This function merges both files

    # read both files
    participant_data = pd.read_excel(path_read_participants)
    results_data = pd.read_excel(path_read_results)
    length = len(results_data)

    # combine dataframes:
    merged_df = participant_data.merge(results_data, left_on='session_id', right_on='user_session_id', how='inner')
    merged_df.drop(['id_x', 'id_y', 'user_session_id'], axis=1, inplace=True)
    merged_df.reindex()

    # delete participants who haven't finished the experiment or who have answered the bot question incorrectly:
    merged_df.drop(merged_df[merged_df['antibot'] != 4].index, inplace=True)
    merged_df.drop(merged_df[merged_df['perceived_help'] == 0].index, inplace=True)
    merged_df.drop(merged_df[merged_df['complexity_formal'] == 0].index, inplace=True)
    merged_df.drop(merged_df[merged_df['complexity_social'] == 0].index, inplace=True)
    merged_df.drop(merged_df[merged_df['complexity_pheno'] == 0].index, inplace=True)
    merged_df.drop(merged_df[merged_df['complexity_intuitive'] == 0].index, inplace=True)

    return merged_df, length


def compute_formal_accuracy_and_add_to_df(df, image_df):

    # this function computes the accuracy, the delegation rate, and the self-fulfilling prophecy variables for each
    # participant in the formal space

    formal_answers = df['answers_formal']
    accuracy_formal = []
    delegation_rate_formal = []
    negative_prophecy_formal_list = []
    positive_prophecy_formal_list = []

    # since django doesn't allow to save dictionaries, the answers are saved in a json string which is converted to a
    # dictionary here
    for answer in formal_answers:
        answer_string = json.loads(answer)
        answer_object = ast.literal_eval(answer_string)
        # for the feedback treatment group the json string represents a list of dictionaries which is converted to a
        # single dictionary here
        if type(answer_object) is list:
            answer_dict = {k: v for dictionary in answer_object for (k, v) in dictionary.items()}
        else:
            answer_dict = answer_object
        # define counter variables to compute accuracy and delegation rate
        correctly_classified = 0
        incorrectly_classified = 0
        delegated = 0
        not_delegated = 0
        # for each answer check whether the answer was correct and whether an iamge was delegated or not
        for single_answer in answer_dict.items():
            correct = single_answer[0].split('_')[0]
            guess = single_answer[1].split(' ')[0]
            if guess == 'low':
                not_delegated += 1
                if correct == 'Low':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            elif guess == 'medium':
                not_delegated += 1
                if correct == 'Middle':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            elif guess == 'high':
                not_delegated += 1
                if correct == 'High':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            # if the participant delegated an image, get the AI answer and act as if it was the participant's answer
            elif guess == 'delegate':
                delegated += 1
                image_name = single_answer[0]
                index = image_df.index[image_df['Filename'] == image_name]
                correct_ai = image_df.loc[index, 'Correctly_Classified'].item()
                if correct_ai:
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

        # for the self-fulfilling prophecy case, if the number of images that were delegated is below 3, set the
        # negative prophecy to 1. If it is above 5 set the positive prophecy to 1. Save the variables in a list for each
        # participant
        if delegated < 3:
            negative_prophecy_formal = 1
        else:
            negative_prophecy_formal = 0
        negative_prophecy_formal_list.append(negative_prophecy_formal)

        if delegated > 5:
            positive_prophecy_formal = 1
        else:
            positive_prophecy_formal = 0
        positive_prophecy_formal_list.append(positive_prophecy_formal)

        # after looping through all answers compute the accuracy and the delegation rate then save it in a list for
        # each participant
        participant_accuracy = correctly_classified / (correctly_classified + incorrectly_classified)
        delegation_rate = delegated / (delegated + not_delegated)
        delegation_rate_formal.append(delegation_rate)
        accuracy_formal.append(participant_accuracy)

    # add accuracy list, delegation rate list, and self-fulfilling prophecies to dataframe
    df['negative_prophecy_formal'] = negative_prophecy_formal_list
    df['positive_prophecy_formal'] = positive_prophecy_formal_list
    df['accuracy_formal'] = accuracy_formal
    df['delegation_rate_formal'] = delegation_rate_formal

    return df


def compute_social_accuracy_and_add_to_df(df, image_df):
    # this function is equivalent to the one before, but computes the results for the social space
    social_answers = df['answers_social']
    accuracy_social = []
    delegation_rate_social = []
    negative_prophecy_social_list = []
    positive_prophecy_social_list = []

    for answer in social_answers:
        answer_string = json.loads(answer)
        answer_object = ast.literal_eval(answer_string)
        if type(answer_object) is list:
            answer_dict = {k: v for dictionary in answer_object for (k, v) in dictionary.items()}
        else:
            answer_dict = answer_object
        correctly_classified = 0
        incorrectly_classified = 0
        delegated = 0
        not_delegated = 0
        for single_answer in answer_dict.items():
            correct = single_answer[0].split('_')[0]
            guess = single_answer[1].split(' ')[0]
            if guess == 'urban':
                not_delegated += 1
                if correct == 'Urban':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            elif guess == 'rural':
                not_delegated += 1
                if correct == 'Rural':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            elif guess == 'delegate':
                delegated += 1
                image_name = single_answer[0]
                index = image_df.index[image_df['Filename'] == image_name]
                correct_ai = image_df.loc[index, 'Correctly_Classified'].item()
                if correct_ai:
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1
        if delegated < 3:
            negative_prophecy_social = 1
        else:
            negative_prophecy_social = 0
        negative_prophecy_social_list.append(negative_prophecy_social)

        if delegated > 5:
            positive_prophecy_social = 1
        else:
            positive_prophecy_social = 0
        positive_prophecy_social_list.append(positive_prophecy_social)

        participant_accuracy = correctly_classified / (correctly_classified + incorrectly_classified)
        delegation_rate = delegated / (delegated + not_delegated)
        accuracy_social.append(participant_accuracy)
        delegation_rate_social.append(delegation_rate)

    df['negative_prophecy_social'] = negative_prophecy_social_list
    df['positive_prophecy_social'] = positive_prophecy_social_list
    df['accuracy_social'] = accuracy_social
    df['delegation_rate_social'] = delegation_rate_social

    return df


def compute_pheno_accuracy_and_add_to_df(df, image_df):
    # this function is equivalent to the formal space, but computes the results for the phenomenological space
    pheno_answers = df['answers_pheno']
    accuracy_pheno = []
    delegation_rate_pheno = []
    negative_prophecy_pheno_list = []
    positive_prophecy_pheno_list = []

    for answer in pheno_answers:
        answer_string = json.loads(answer)
        answer_object = ast.literal_eval(answer_string)
        if type(answer_object) is list:
            answer_dict = {k: v for dictionary in answer_object for (k, v) in dictionary.items()}
        else:
            answer_dict = answer_object
        correctly_classified = 0
        incorrectly_classified = 0
        delegated = 0
        not_delegated = 0
        for single_answer in answer_dict.items():
            correct = single_answer[0].split('_')[0]
            guess = single_answer[1].split(' ')[0]
            if guess == 'Paris':
                not_delegated += 1
                if correct == 'Paris':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            elif guess == 'New':
                not_delegated += 1
                if correct == 'New':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            elif guess == 'Los':
                not_delegated += 1
                if correct == 'Los':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            elif guess == 'Berlin':
                not_delegated += 1
                if correct == 'Berlin':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            elif guess == 'delegate':
                delegated += 1
                image_name = single_answer[0]
                index = image_df.index[image_df['Filename'] == image_name]
                correct_ai = image_df.loc[index, 'Correctly_Classified'].item()
                if correct_ai:
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1
        if delegated < 3:
            negative_prophecy_pheno = 1
        else:
            negative_prophecy_pheno = 0
        negative_prophecy_pheno_list.append(negative_prophecy_pheno)

        if delegated > 5:
            positive_prophecy_pheno = 1
        else:
            positive_prophecy_pheno = 0
        positive_prophecy_pheno_list.append(positive_prophecy_pheno)

        participant_accuracy = correctly_classified / (correctly_classified + incorrectly_classified)
        delegation_rate = delegated / (delegated + not_delegated)
        delegation_rate_pheno.append(delegation_rate)
        accuracy_pheno.append(participant_accuracy)

    df['negative_prophecy_pheno'] = negative_prophecy_pheno_list
    df['positive_prophecy_pheno'] = positive_prophecy_pheno_list
    df['accuracy_pheno'] = accuracy_pheno
    df['delegation_rate_pheno'] = delegation_rate_pheno

    return df


def compute_intuitive_accuracy_and_add_to_df(df, image_df, path_to_save):
    # this function is equivalent to the formal space, but computes the results for the intuitive space
    intuitive_answers = df['answers_intuitive']
    accuracy_intuitive = []
    delegation_rate_intuitive = []
    negative_prophecy_intuitive_list = []
    positive_prophecy_intuitive_list = []

    for answer in intuitive_answers:
        answer_string = json.loads(answer)
        answer_object = ast.literal_eval(answer_string)
        if type(answer_object) is list:
            answer_dict = {k: v for dictionary in answer_object for (k, v) in dictionary.items()}
        else:
            answer_dict = answer_object
        correctly_classified = 0
        incorrectly_classified = 0
        delegated = 0
        not_delegated = 0

        for single_answer in answer_dict.items():
            correct = single_answer[0].split('_')[0]
            guess = single_answer[1].split(' ')[0]
            if guess == 'low':
                not_delegated += 1
                if correct == 'Low':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            elif guess == 'medium':
                not_delegated += 1
                if correct == 'Middle':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            elif guess == 'high':
                not_delegated += 1
                if correct == 'High':
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1

            elif guess == 'delegate':
                delegated += 1
                image_name = single_answer[0]
                index = image_df.index[image_df['Filename'] == image_name]
                correct_ai = image_df.loc[index, 'Correctly_Classified'].item()
                if correct_ai:
                    correctly_classified += 1
                else:
                    incorrectly_classified += 1
        if delegated < 3:
            negative_prophecy_intuitive = 1
        else:
            negative_prophecy_intuitive = 0
        negative_prophecy_intuitive_list.append(negative_prophecy_intuitive)

        if delegated > 5:
            positive_prophecy_intuitive = 1
        else:
            positive_prophecy_intuitive = 0
        positive_prophecy_intuitive_list.append(positive_prophecy_intuitive)
        participant_accuracy = correctly_classified / (correctly_classified + incorrectly_classified)
        delegation_rate = delegated / (delegated + not_delegated)
        accuracy_intuitive.append(participant_accuracy)
        delegation_rate_intuitive.append(delegation_rate)

    df['negative_prophecy_intuitive'] = negative_prophecy_intuitive_list
    df['positive_prophecy_intuitive'] = positive_prophecy_intuitive_list
    df['accuracy_intuitive'] = accuracy_intuitive
    df['delegation_rate_intuitive'] = delegation_rate_intuitive
    df.to_excel(path_to_save + '/merged.xlsx')

    return df


def delete_outliers_from_df(df, path_to_save):
    # this function deletes participants who haven't produced meaningful results from the overall dataframe
    delete_counter = 0
    keep_counter = 0
    for index in df.index:
        counter = 0
        if df.loc[index, 'accuracy_formal'] < 0.34:
            counter += 1
        if df.loc[index, 'accuracy_social'] < 0.5:
            counter += 1
        if df.loc[index, 'accuracy_pheno'] < 0.25:
            counter += 1
        if df.loc[index, 'accuracy_intuitive'] < 0.34:
            counter += 1

        if counter >= 3:
            df.drop(index, inplace=True)
            delete_counter += 1
        else:
            keep_counter += 1

    delete_ratio = delete_counter / (delete_counter + keep_counter)
    print(delete_ratio)
    df.to_excel(path_to_save + '/delete_outliers.xlsx')

    return df


def compute_boolean_negative_feedback_values_and_add_to_df(df_feedback, image_formal_df, image_social_df,
                                                           image_pheno_df, image_intuitive_df, mode):
    # this function checks whether a participant received negative feedback in the first, second, third or fourth task.
    # Then it saves the results in the dataframe
    if mode == 'first':
        task_list = df_feedback['first_task'].to_list()
    elif mode == 'second':
        task_list = df_feedback['second_task'].to_list()
    elif mode == 'third':
        task_list = df_feedback['third_task'].to_list()
    else:
        task_list = df_feedback['fourth_task'].to_list()

    feedback_total = []
    df_feedback.reset_index(drop=True, inplace=True)
    for index, task in enumerate(task_list):
        if task == 'formal':
            image_df = image_formal_df
        elif task == 'social':
            image_df = image_social_df
        elif task == 'pheno':
            image_df = image_pheno_df
        else:
            image_df = image_intuitive_df

        answer = df_feedback.at[index, 'answers_' + task]
        answer_string = json.loads(answer)
        answer_object = ast.literal_eval(answer_string)
        if type(answer_object) is list:
            answer_dict = {k: v for dictionary in answer_object for (k, v) in dictionary.items()}
        else:
            answer_dict = answer_object

        feedback_per_answer = []
        for single_answer in answer_dict.items():
            image_name = single_answer[0]
            guess = single_answer[1].split(' ')[0]

            if guess == 'delegate':
                index = image_df.index[image_df['Filename'] == image_name]
                feedback = image_df.loc[index, 'Correctly_Classified'].item()
            else:
                feedback = 'not_delegated'
            feedback_per_answer.append(feedback)
        feedback_total.append(feedback_per_answer)

    negative_feedback_list = []
    for feedback in feedback_total:
        if False in feedback:
            negative_feedback = 1
        else:
            negative_feedback = 0
        negative_feedback_list.append(negative_feedback)

    df_feedback['negative_feedback_' + mode] = negative_feedback_list


def split_dataframe_by_treatment_group(df, path_to_save, length,
                                       image_formal_df, image_social_df, image_pheno_df, image_intuitive_df):
    # split merged dataframe based on the treatments groups
    mask = df.experiment_group.str.contains("base")
    df_base = df[mask]
    df_rest = df[~mask]

    mask = df_rest.experiment_group.str.contains("ai_delegate")
    df_ai_delegate = df_rest[mask]
    df_rest1 = df_rest[~mask]

    mask = df_rest1.experiment_group.str.contains("feedback")
    df_feedback = df_rest1[mask]
    df_rest2 = df_rest1[~mask]

    mask = df_rest2.experiment_group.str.contains("human_delegate")
    df_human_delegate = df_rest2[mask]
    df_delegation_counter = df_rest2[~mask]

    # include order variable for the herd behavior group:
    index_list = df_delegation_counter.index.to_list()
    order = [length - x for x in index_list]
    df_delegation_counter['order'] = order

    # include 'negative feedback' variable for the feedback group:
    compute_boolean_negative_feedback_values_and_add_to_df(df_feedback, image_formal_df, image_social_df,
                                                           image_pheno_df, image_intuitive_df, mode='first')
    compute_boolean_negative_feedback_values_and_add_to_df(df_feedback, image_formal_df, image_social_df,
                                                           image_pheno_df, image_intuitive_df, mode='second')
    compute_boolean_negative_feedback_values_and_add_to_df(df_feedback, image_formal_df, image_social_df,
                                                           image_pheno_df, image_intuitive_df, mode='third')
    compute_boolean_negative_feedback_values_and_add_to_df(df_feedback, image_formal_df, image_social_df,
                                                           image_pheno_df, image_intuitive_df, mode='fourth')
    # save the results in an Excel file
    df_base.to_excel(path_to_save + '/baseline.xlsx', index=False)
    df_feedback.to_excel(path_to_save + '/feedback.xlsx', index=False)
    df_delegation_counter.to_excel(path_to_save + '/delegation_counter.xlsx', index=False)
    df_human_delegate.to_excel(path_to_save + '/human_delegation.xlsx', index=False)
    df_ai_delegate.to_excel(path_to_save + '/ai_delegation.xlsx', index=False)


def compute_average_accuracy_and_delegation_rate_and_add_to_new_df(list_of_dfs, name_list):
    # this function computes the average accuracy and the average delegation rate for each participant
    index = ['Formal Accuracy', 'Social Accuracy', 'Pheno Accuracy', 'Intuitive Accuracy',
             'Formal Delegation Rate', 'Social Delegation Rate', 'Pheno Delegation Rate', 'Intuitive Delegation Rate']
    df_final_results = pd.DataFrame(index=index)

    for i, df in enumerate(list_of_dfs):
        name = name_list[i]
        results = []
        formal_acc_list = df['accuracy_formal'].tolist()
        avg_formal_acc = sum(formal_acc_list) / len(formal_acc_list)
        results.append(avg_formal_acc)

        social_acc_list = df['accuracy_social'].tolist()
        avg_social_acc = sum(social_acc_list) / len(social_acc_list)
        results.append(avg_social_acc)

        pheno_acc_list = df['accuracy_pheno'].tolist()
        avg_pheno_acc = sum(pheno_acc_list) / len(pheno_acc_list)
        results.append(avg_pheno_acc)

        intuitive_acc_list = df['accuracy_intuitive'].tolist()
        avg_intuitive_acc = sum(intuitive_acc_list) / len(intuitive_acc_list)
        results.append(avg_intuitive_acc)

        formal_delegation_rate_list = df['delegation_rate_formal'].tolist()
        avg_formal_delegation_rate = sum(formal_delegation_rate_list) / len(formal_delegation_rate_list)
        results.append(avg_formal_delegation_rate)

        social_delegation_rate_list = df['delegation_rate_social'].tolist()
        avg_social_delegation_rate = sum(social_delegation_rate_list) / len(social_delegation_rate_list)
        results.append(avg_social_delegation_rate)

        pheno_delegation_rate_list = df['delegation_rate_pheno'].tolist()
        avg_pheno_delegation_rate = sum(pheno_delegation_rate_list) / len(pheno_delegation_rate_list)
        results.append(avg_pheno_delegation_rate)

        intuitive_delegation_rate_list = df['delegation_rate_intuitive'].tolist()
        avg_intutive_delegation_rate = sum(intuitive_delegation_rate_list) / len(intuitive_delegation_rate_list)
        results.append(avg_intutive_delegation_rate)

        df_final_results[name] = results

    return df_final_results


def add_overall_accuracy_and_delegation_rate_to_the_final_df(df_final, df_merged, mode):
    results = []
    overall_acc_formal_list = df_merged['accuracy_formal'].sum() / len(df_merged['accuracy_formal'])
    results.append(overall_acc_formal_list)
    overall_acc_social = df_merged['accuracy_social'].sum() / len(df_merged['accuracy_social'])
    results.append(overall_acc_social)
    overall_acc_pheno = df_merged['accuracy_pheno'].sum() / len(df_merged['accuracy_pheno'])
    results.append(overall_acc_pheno)
    overall_acc_intuitive = df_merged['accuracy_intuitive'].sum() / len(df_merged['accuracy_intuitive'])
    results.append(overall_acc_intuitive)

    # split df into treatments group that can delegate and those that can't
    mask = df_merged.experiment_group.str.contains("base")
    df_rest = df_merged[~mask]
    mask = df_rest.experiment_group.str.contains("ai_delegate")
    df_delegate = df_rest[~mask]

    overall_delegation_rate_formal = df_delegate['delegation_rate_formal'].sum() / len(
        df_delegate['delegation_rate_formal'])
    results.append(overall_delegation_rate_formal)
    overall_delegation_rate_social = df_delegate['delegation_rate_social'].sum() / len(
        df_delegate['delegation_rate_social'])
    results.append(overall_delegation_rate_social)
    overall_delegation_rate_pheno = df_delegate['delegation_rate_pheno'].sum() / len(
        df_delegate['delegation_rate_pheno'])
    results.append(overall_delegation_rate_pheno)
    overall_delegation_rate_intuitive = df_delegate['delegation_rate_intuitive'].sum() / len(
        df_delegate['delegation_rate_intuitive'])
    results.append(overall_delegation_rate_intuitive)

    if mode == 'merged':
        df_final['overall'] = results
    elif mode == 'outliers':
        df_final['overall_without_outliers'] = results

    return df_final


def add_ai_accuracy_and_delegation_rate_to_df(df_final, path_to_save):
    # this function computes the accuracy and the delegation rate of the AI for all four spaces and saves the results
    path_to_read_formal = 'C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Formal_predictions_small.xlsx'
    path_to_read_social = 'C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Social_predictions_small.xlsx'
    path_to_read_pheno = ('C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Phenomenological_predictions_small'
                          '.xlsx')
    path_to_read_intuitive = 'C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Intuitive_predictions_small.xlsx'

    df_formal = pd.read_excel(path_to_read_formal)
    df_social = pd.read_excel(path_to_read_social)
    df_pheno = pd.read_excel(path_to_read_pheno)
    df_intuitive = pd.read_excel(path_to_read_intuitive)

    results = []
    ai_correctly_classified_formal = df_formal['Correctly_Classified'].tolist()
    correctly_classified = 0

    for value in ai_correctly_classified_formal:
        if value:
            correctly_classified += 1
    ai_accuracy_formal = correctly_classified / len(ai_correctly_classified_formal)
    results.append(ai_accuracy_formal)

    ai_correctly_classified_social = df_social['Correctly_Classified'].tolist()
    correctly_classified = 0
    for value in ai_correctly_classified_social:
        if value:
            correctly_classified += 1
    ai_accuracy_social = correctly_classified / len(ai_correctly_classified_social)
    results.append(ai_accuracy_social)

    ai_correctly_classified_pheno = df_pheno['Correctly_Classified'].tolist()
    correctly_classified = 0
    for value in ai_correctly_classified_pheno:
        if value:
            correctly_classified += 1
    ai_accuracy_pheno = correctly_classified / len(ai_correctly_classified_pheno)
    results.append(ai_accuracy_pheno)

    ai_correctly_classified_intuitive = df_intuitive['Correctly_Classified'].tolist()
    correctly_classified = 0
    for value in ai_correctly_classified_intuitive:
        if value:
            correctly_classified += 1
    ai_accuracy_intuitive = correctly_classified / len(ai_correctly_classified_intuitive)
    results.append(ai_accuracy_intuitive)

    # delegation rates:
    delegated = 0
    ai_delegate_formal = df_formal['Delegate'].tolist()
    for val in ai_delegate_formal:
        if val:
            delegated += 1
    ai_delegation_rate_formal = delegated / len(ai_delegate_formal)
    results.append(ai_delegation_rate_formal)

    delegated = 0
    ai_delegate_social = df_social['Delegate'].tolist()
    for val in ai_delegate_social:
        if val:
            delegated += 1
    ai_delegation_rate_social = delegated / len(ai_delegate_social)
    results.append(ai_delegation_rate_social)

    delegated = 0
    ai_delegate_pheno = df_pheno['Delegate'].tolist()
    for val in ai_delegate_pheno:
        if val:
            delegated += 1
    ai_delegation_rate_pheno = delegated / len(ai_delegate_pheno)
    results.append(ai_delegation_rate_pheno)

    delegated = 0
    ai_delegate_intuitive = df_intuitive['Delegate'].tolist()
    for val in ai_delegate_intuitive:
        if val:
            delegated += 1
    ai_delegation_rate_intuitive = delegated / len(ai_delegate_intuitive)
    results.append(ai_delegation_rate_intuitive)

    df_final['AI'] = results
    df_final.to_excel(path_to_save + '/final_results.xlsx')


def create_file_with_accuracies_for_each_image(image_name_list, answers_series, treatment_group, df, image_df):
    # this function computes the average accuracy and the average delegation rate over all participants for each image
    accuracy_list = []
    delegation_rate_list = []
    for image_name in image_name_list:
        count = count_image_occurrence_in_answer_series(answers_series, image_name)
        correct = count_how_often_an_image_was_answered_correctly(answers_series, image_name, image_df)
        delegated = count_how_often_an_image_was_delegated(answers_series, image_name)
        if count == 0:
            accuracy = 0
            delegation_rate = 0
        else:
            accuracy = correct / count
            delegation_rate = delegated / count
        accuracy_list.append(accuracy)
        delegation_rate_list.append(delegation_rate)

    df[treatment_group + '_accuracy'] = accuracy_list
    df[treatment_group + '_delegation_rate'] = delegation_rate_list
    return df


def count_image_occurrence_in_answer_series(answer_series, image_name):
    # this function counts how often an image occurred for all participants combined
    list_of_answers = []
    for answer in answer_series:
        answer_string = json.loads(answer)
        answer_object = ast.literal_eval(answer_string)
        if type(answer_object) is list:
            answer_dict = {k: v for dictionary in answer_object for (k, v) in dictionary.items()}
        else:
            answer_dict = answer_object

        for single_answer in answer_dict:
            list_of_answers.append(single_answer)

    image_count = list_of_answers.count(image_name)
    return image_count


def count_how_often_an_image_was_answered_correctly(answer_series, image_name):
    # this function counts how often an image was answered correctly for all participants combined
    correctly = 0
    for answer in answer_series:
        answer_string = json.loads(answer)
        answer_object = ast.literal_eval(answer_string)
        if type(answer_object) is list:
            answer_dict = {k: v for dictionary in answer_object for (k, v) in dictionary.items()}
        else:
            answer_dict = answer_object

        for single_answer in answer_dict.items():

            if single_answer[0] == image_name:
                correct = single_answer[0].split('_')[0]
                guess = single_answer[1].split(' ')[0]

                if guess == 'low':
                    if correct == 'Low':
                        correctly += 1
                elif guess == 'medium':
                    if correct == 'Middle':
                        correctly += 1
                elif guess == 'high':
                    if correct == 'High':
                        correctly += 1
                elif guess == 'rural':
                    if correct == 'Rural':
                        correctly += 1
                elif guess == 'urban':
                    if correct == 'Urban':
                        correctly += 1
                elif guess == correct:
                    correctly += 1
                elif guess == 'delegate':
                    pass

    return correctly


def count_how_often_an_image_was_delegated(answer_series, image_name):
    # this function counts how often an image was delegated for all participants combined
    delegated = 0
    for answer in answer_series:
        answer_string = json.loads(answer)
        answer_object = ast.literal_eval(answer_string)
        if type(answer_object) is list:
            answer_dict = {k: v for dictionary in answer_object for (k, v) in dictionary.items()}
        else:
            answer_dict = answer_object

        for single_answer in answer_dict.items():

            if single_answer[0] == image_name:
                guess = single_answer[1].split(' ')[0]
                if guess == 'delegate':
                    delegated += 1

    return delegated


def split_overall_dataframe_into_delegate_and_not_delegate_by_task(df, path_to_save):
    # this function splits the overall dataframe into one dataframe for all treatment groups that could delegate and one
    # for all treatment groups that couldn't. This is done for all four tasks. It also turns the "treatment group
    # variable into a dummy variable.
    mask = df.experiment_group.str.contains("base")
    df_rest_delegate = df[~mask]

    mask = df_rest_delegate.experiment_group.str.contains("ai_delegate")
    df_delegate = df_rest_delegate[~mask]

    mask = df.experiment_group.str.contains("human_delegate")
    df_rest_not_delegate = df[~mask]

    mask = df_rest_not_delegate.experiment_group.str.contains("feedback")
    df_rest_not_delegate1 = df_rest_not_delegate[~mask]

    mask = df_rest_not_delegate1.experiment_group.str.contains("delegation_counter")
    df_not_delegate = df_rest_not_delegate1[~mask]

    # make the experiment group variable a dummy variable:
    # 1) can delegate:
    experiment_group_list = df_delegate['experiment_group'].to_list()
    feedback_dummy_list = []
    human_delegate_dummy_list = []
    delegation_counter_dummy_list = []

    for group in experiment_group_list:
        if group == 'feedback':
            feedback_dummy = 1
        else:
            feedback_dummy = 0
        feedback_dummy_list.append(feedback_dummy)

        if group == 'human_delegate':
            human_delegate_dummy = 1
        else:
            human_delegate_dummy = 0
        human_delegate_dummy_list.append(human_delegate_dummy)

        if group == 'delegation_counter':
            delegation_counter_dummy = 1
        else:
            delegation_counter_dummy = 0
        delegation_counter_dummy_list.append(delegation_counter_dummy)

    df_delegate['experiment_group_feedback'] = feedback_dummy_list
    df_delegate['experiment_group_human_delegate'] = human_delegate_dummy_list
    df_delegate['experiment_group_delegation_counter'] = delegation_counter_dummy_list
    df_delegate.drop('experiment_group', axis=1, inplace=True)

    # 2) cannot delegate:
    experiment_group_list_not = df_not_delegate['experiment_group'].to_list()
    baseline_dummy_list = []
    ai_delegate_dummy_list = []

    for group in experiment_group_list_not:
        if group == 'base':
            baseline_dummy = 1
        else:
            baseline_dummy = 0
        baseline_dummy_list.append(baseline_dummy)

        if group == 'ai_delegate':
            ai_delegate_dummy = 1
        else:
            ai_delegate_dummy = 0
        ai_delegate_dummy_list.append(ai_delegate_dummy)

    df_not_delegate['experiment_group_ai_delegate'] = ai_delegate_dummy_list
    df_not_delegate['experiment_group_baseline'] = baseline_dummy_list
    df_not_delegate.drop('experiment_group', axis=1, inplace=True)

    new_columns_not = ['experiment_group_ai_delegate', 'experiment_group_baseline', 'gender', 'age', 'country_usa',
                       'country_ger', 'country_uk', 'country_india',
                       'country_china', 'country_spain', 'country_brazil', 'country_greece', 'first_task_formal',
                       'first_task_social', 'first_task_pheno', 'first_task_intuitive', 'second_task_formal',
                       'second_task_social', 'second_task_pheno', 'second_task_intuitive', 'third_task_formal',
                       'third_task_social', 'third_task_pheno', 'third_task_intuitive', 'fourth_task_formal',
                       'fourth_task_social', 'fourth_task_pheno', 'fourth_task_intuitive', 'competency_high',
                       'competency_medium', 'competency_low', 'complexity_formal',
                       'complexity_social', 'complexity_pheno', 'complexity_intuitive', 'accuracy_formal',
                       'accuracy_social', 'accuracy_pheno', 'accuracy_intuitive']

    new_columns = ['experiment_group_feedback', 'experiment_group_human_delegate',
                   'experiment_group_delegation_counter',
                   'gender', 'age', 'country_usa', 'country_ger', 'country_uk', 'country_india',
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

    df_not_delegate = df_not_delegate[new_columns_not]
    df_delegate = df_delegate[new_columns]

    df_delegate_formal = df_delegate.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 21, 25, 29, 30,
                                              31, 32, 33, 34, 38, 39, 46, 50]]
    df_delegate_social = df_delegate.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 18, 22, 26, 29, 30,
                                              31, 32, 33, 35, 40, 41, 47, 51]]
    df_delegate_pheno = df_delegate.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 19, 23, 27, 29, 30,
                                             31, 32, 33, 36, 42, 43, 48, 52]]
    df_delegate_intuitive = df_delegate.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 28, 29,
                                                 30, 31, 32, 33, 37, 44, 45, 49, 53]]

    df_not_delegate_formal = df_not_delegate.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 28, 29, 30,
                                                      31, 35, ]]
    df_not_delegate_social = df_not_delegate.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 21, 25, 28, 29, 30,
                                                      32, 36]]
    df_not_delegate_pheno = df_not_delegate.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 18, 22, 26, 28, 29, 30,
                                                     33, 37]]
    df_not_delegate_intuitive = df_not_delegate.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 19, 23, 27, 28, 29,
                                                         30, 34, 38]]

    df_not_delegate_formal.to_excel(path_to_save + '/regression_not_delegate_formal.xlsx', index=False)
    df_not_delegate_social.to_excel(path_to_save + '/regression_not_delegate_social.xlsx', index=False)
    df_not_delegate_pheno.to_excel(path_to_save + '/regression_not_delegate_pheno.xlsx', index=False)
    df_not_delegate_intuitive.to_excel(path_to_save + '/regression_not_delegate_intuitive.xlsx', index=False)

    df_delegate_formal.to_excel(path_to_save + '/regression_delegate_formal.xlsx', index=False)
    df_delegate_social.to_excel(path_to_save + '/regression_delegate_social.xlsx', index=False)
    df_delegate_pheno.to_excel(path_to_save + '/regression_delegate_pheno.xlsx', index=False)
    df_delegate_intuitive.to_excel(path_to_save + '/regression_delegate_intuitive.xlsx', index=False)


def split_dataframe_to_overall_image_results_by_treatment_group(image_formal_df, image_social_df, image_pheno_df,
                                                                image_intuitive_df, path_splitted_dfs, path_to_save):
    # this function spits the overall dataframe with the results for each image into one dataframe with the image
    # results for each treatment group. This is done for each space.

    # formal:
    image_list_formal = image_formal_df['Filename'].to_list()
    list_of_files = os.listdir(path_splitted_dfs)
    dataframe_formal = pd.DataFrame(index=image_list_formal)

    for file in list_of_files:
        file_path = os.path.join(path_splitted_dfs, file)
        treatment_group = file.split('.')[0]
        df_answers_formal = pd.read_excel(file_path)
        series_formal = df_answers_formal['answers_formal']
        dataframe_formal = create_file_with_accuracies_for_each_image(image_list_formal, series_formal, treatment_group,
                                                                      dataframe_formal, image_formal_df)

    split_image_results_dataframe_by_treatment_group(dataframe_formal, path_to_save, space='formal')
    dataframe_formal.to_excel(path_to_save + '/image_results_formal.xlsx')

    # social:
    image_list_social = image_social_df['Filename'].to_list()
    list_of_files = os.listdir(path_splitted_dfs)
    dataframe_social = pd.DataFrame(index=image_list_social)

    for file in list_of_files:
        file_path = os.path.join(path_splitted_dfs, file)
        treatment_group = file.split('.')[0]
        df_answers_social = pd.read_excel(file_path)
        series_social = df_answers_social['answers_social']
        dataframe_social = create_file_with_accuracies_for_each_image(image_list_social, series_social, treatment_group,
                                                                      dataframe_social, image_social_df)
    split_image_results_dataframe_by_treatment_group(dataframe_social, path_to_save, space='social')
    dataframe_social.to_excel(path_to_save + '/image_results_social.xlsx')

    # pheno:
    image_list_pheno = image_pheno_df['Filename'].to_list()
    list_of_files = os.listdir(path_splitted_dfs)
    dataframe_pheno = pd.DataFrame(index=image_list_pheno)

    for file in list_of_files:
        file_path = os.path.join(path_splitted_dfs, file)
        treatment_group = file.split('.')[0]
        df_answers_pheno = pd.read_excel(file_path)
        series_pheno = df_answers_pheno['answers_pheno']
        dataframe_pheno = create_file_with_accuracies_for_each_image(image_list_pheno, series_pheno, treatment_group,
                                                                     dataframe_pheno, image_pheno_df)
    split_image_results_dataframe_by_treatment_group(dataframe_pheno, path_to_save, space='pheno')
    dataframe_pheno.to_excel(path_to_save + '/image_results_pheno.xlsx')

    # intuitive:
    image_list_intuitive = image_intuitive_df['Filename'].to_list()
    list_of_files = os.listdir(path_splitted_dfs)
    dataframe_intuitive = pd.DataFrame(index=image_list_intuitive)

    for file in list_of_files:
        file_path = os.path.join(path_splitted_dfs, file)
        treatment_group = file.split('.')[0]
        df_answers_intuitive = pd.read_excel(file_path)
        series_intuitive = df_answers_intuitive['answers_intuitive']
        dataframe_intuitive = create_file_with_accuracies_for_each_image(image_list_intuitive, series_intuitive,
                                                                         treatment_group,
                                                                         dataframe_intuitive, image_intuitive_df)
    split_image_results_dataframe_by_treatment_group(dataframe_intuitive, path_to_save, space='intuitive')

    dataframe_intuitive.to_excel(path_to_save + '/image_results_intuitive.xlsx')


def perform_participant_regression_delegate(X, y_acc, y_del, path_to_save, setting):
    # this function performs a multivariate multiple regression analysis. The independent variable is the participant's
    # accuracy or the participants delegation rate. The dependent variables include age, gender, negative feedback,
    # expectations, prior knowledge and so on. This is done for the treatment groups that are allowed to delegate.
    X_acc_train, X_acc_test, y_acc_train, y_acc_test = train_test_split(X, y_acc, test_size=0.35, random_state=42)
    X_del_train, X_del_test, y_del_train, y_del_test = train_test_split(X, y_del, test_size=0.35, random_state=42)

    # train the model:
    lm_acc = LinearRegression()
    lm_del = LinearRegression()
    lm_acc.fit(X_acc_train, y_acc_train)
    lm_del.fit(X_del_train, y_del_train)

    acc_co = lm_acc.coef_
    del_co = lm_del.coef_

    cdf_acc = pd.DataFrame(acc_co, X.columns, columns=['Coefficients_Accuracy_' + setting])
    cdf_del = pd.DataFrame(del_co, X.columns, columns=['Coefficients_Delegation_Rate_' + setting])

    cdf_acc.to_excel(path_to_save + '/acc_' + setting + '.xlsx')
    cdf_del.to_excel(path_to_save + '/del_' + setting + '.xlsx')

    # model summary:
    X2 = sm.add_constant(X)
    model_acc = sm.OLS(y_acc, X2)
    model_del = sm.OLS(y_del, X2)
    model2_acc = model_acc.fit()
    model2_del = model_del.fit()

    with open(path_to_save + '/summary_acc_' + setting + '.txt', 'w') as fh:
        fh.write(model2_acc.summary().as_text())
    with open(path_to_save + '/summary_del_' + setting + '.txt', 'w') as fh:
        fh.write(model2_del.summary().as_text())
    with open(path_to_save + '/summary_acc_' + setting + '.csv', 'w') as fh:
        fh.write(model2_acc.summary().as_csv())
    with open(path_to_save + '/summary_del_' + setting + '.csv', 'w') as fh:
        fh.write(model2_del.summary().as_csv())
    # predictions:
    predictions_acc = lm_acc.predict(X_acc_test)
    predictions_del = lm_del.predict(X_del_test)

    # evaluation:
    MAE_acc = mean_absolute_error(y_acc_test, predictions_acc)
    MSE_acc = mean_squared_error(y_acc_test, predictions_acc)
    RMSE_acc = math.sqrt(mean_squared_error(y_acc_test, predictions_acc))
    MAE_del = mean_absolute_error(y_del_test, predictions_del)
    MSE_del = mean_squared_error(y_del_test, predictions_del)
    RMSE_del = math.sqrt(mean_squared_error(y_del_test, predictions_del))

    print("MAE_acc: ", MAE_acc)
    print("MSE_acc: ", MSE_acc)
    print("RMSE_acc: ", RMSE_acc)
    print("MAE_del: ", MAE_del)
    print("MSE_del: ", MSE_del)
    print("RMSE_del: ", RMSE_del)


def perform_participant_regression_not_delegate(X, y, path_to_save, setting):
    # this function performs a multivariate multiple regression analysis. The independent variable is the participant's
    # accuracy. The dependent variables include age, gender, negative feedback, expectations, prior knowledge and so on.
    # This is done for the treatment groups that are not allowed to delegate.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    # train the model:
    lm = LinearRegression(fit_intercept=True)
    lm.fit(X_train, y_train)

    co = lm.coef_

    cdf = pd.DataFrame(co, X.columns, columns=['Coefficients_Accuracy_' + setting])
    cdf.to_excel(path_to_save + '/acc_' + setting + '.xlsx')

    # model summary:
    X2 = sm.add_constant(X)
    model = sm.OLS(y, X2)
    model2 = model.fit()

    with open(path_to_save + '/summary_acc_' + setting + '.txt', 'w') as fh:
        fh.write(model2.summary().as_text())
    with open(path_to_save + '/summary_acc_' + setting + '.csv', 'w') as fh:
        fh.write(model2.summary().as_csv())
    # predictions:
    predictions = lm.predict(X_test)

    # evaluation:
    MAE = mean_absolute_error(y_test, predictions)
    MSE = mean_squared_error(y_test, predictions)
    RMSE = math.sqrt(mean_squared_error(y_test, predictions))

    print("MAE: ", MAE)
    print("MSE: ", MSE)
    print("RMSE: ", RMSE)


def split_image_results_dataframe_by_treatment_group(df, path_to_save, space):
    # this function splits the image result dataframe for each treatment group into one dataframe for the easy and one
    # for the difficult images
    df_ai_delegation = pd.DataFrame(df.iloc[:, [0, 1]])
    df_ai_delegation = df_ai_delegation.drop(df_ai_delegation[df_ai_delegation.ai_delegation_accuracy == 0].index)
    df_ai_delegation_easy = df_ai_delegation.loc[df['ai_delegation_accuracy'] >= 0.5]
    df_ai_delegation_diff = df_ai_delegation.loc[df['ai_delegation_accuracy'] < 0.5]
    df_ai_delegation.to_excel(path_to_save + '/ai_delegation_' + space + '.xlsx')
    df_ai_delegation_easy.to_excel(path_to_save + '/seperated_diff_and_easy/ai_delegation_' + space + '_easy.xlsx')
    df_ai_delegation_diff.to_excel(path_to_save + '/seperated_diff_and_easy/ai_delegation_' + space + '_diff.xlsx')
    df_baseline = pd.DataFrame(df.iloc[:, [2, 3]])
    df_baseline = df_baseline.drop(df_baseline[df_baseline.baseline_accuracy == 0].index)
    df_baseline_easy = df_baseline.loc[df['baseline_accuracy'] >= 0.5]
    df_baseline_diff = df_baseline.loc[df['baseline_accuracy'] < 0.5]
    df_baseline.to_excel(path_to_save + '/baseline_' + space + '.xlsx')
    df_baseline_easy.to_excel(path_to_save + '/seperated_diff_and_easy/baseline_' + space + '_easy.xlsx')
    df_baseline_diff.to_excel(path_to_save + '/seperated_diff_and_easy/baseline_' + space + '_diff.xlsx')
    df_delegation_counter = pd.DataFrame(df.iloc[:, [4, 5]])
    df_delegation_counter = df_delegation_counter.drop(
        df_delegation_counter[df_delegation_counter.delegation_counter_accuracy == 0].index)
    df_delegation_counter_easy = df_delegation_counter.loc[df['delegation_counter_accuracy'] >= 0.5]
    df_delegation_counter_diff = df_delegation_counter.loc[df['delegation_counter_accuracy'] < 0.5]
    df_delegation_counter.to_excel(path_to_save + '/delegation_counter_' + space + '.xlsx')
    df_delegation_counter_easy.to_excel(
        path_to_save + '/seperated_diff_and_easy/delegation_counter_' + space + '_easy.xlsx')
    df_delegation_counter_diff.to_excel(
        path_to_save + '/seperated_diff_and_easy/delegation_counter_' + space + '_diff.xlsx')
    df_feedback = pd.DataFrame(df.iloc[:, [6, 7]])
    df_feedback = df_feedback.drop(df_feedback[df_feedback.feedback_accuracy == 0].index)
    df_feedback_easy = df_feedback.loc[df['ai_delegation_accuracy'] >= 0.5]
    df_feedback_diff = df_feedback.loc[df['ai_delegation_accuracy'] < 0.5]
    df_feedback.to_excel(path_to_save + '/feedback_' + space + '.xlsx')
    df_feedback_easy.to_excel(path_to_save + '/seperated_diff_and_easy/feedback_' + space + '_easy.xlsx')
    df_feedback_diff.to_excel(path_to_save + '/seperated_diff_and_easy/feedback_' + space + '_diff.xlsx')
    df_human_delegation = pd.DataFrame(df.iloc[:, [8, 9]])
    df_human_delegation = df_human_delegation.drop(
        df_human_delegation[df_human_delegation.human_delegation_accuracy == 0].index)
    df_human_delegation_easy = df_human_delegation.loc[df['human_delegation_accuracy'] > 0.5]
    df_human_delegation_diff = df_human_delegation.loc[df['human_delegation_accuracy'] <= 0.5]
    df_human_delegation.to_excel(path_to_save + '/human_delegation_' + space + '.xlsx')
    df_human_delegation_easy.to_excel(
        path_to_save + '/seperated_diff_and_easy/human_delegation_' + space + '_easy.xlsx')
    df_human_delegation_diff.to_excel(
        path_to_save + '/seperated_diff_and_easy/human_delegation_' + space + '_diff.xlsx')


def perform_image_regression(x, y, path_to_save, setting):
    # this function performs a regression analysis for the image results. The dependent variable is the average accuracy
    # of an image over all participants. The independent variable is the average delegation rate of an image over all
    # participants
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

    # train the model:
    lm = LinearRegression()
    lm.fit(x_train, y_train)

    co = lm.coef_.tolist()
    intercept = lm.intercept_
    co.append(intercept)

    cdf = pd.DataFrame(co, [x.columns, 'intercept'], columns=['Coefficients_Accuracy_' + setting])
    cdf.to_excel(path_to_save + '/acc_' + setting + '.xlsx')

    # model summary:
    x2 = sm.add_constant(x)
    model = sm.OLS(y, x2)
    model2 = model.fit()

    with open(path_to_save + '/summary_image_delegation_rate' + setting + '.txt', 'w') as fh:
        fh.write(model2.summary().as_text())

    with open(path_to_save + '/summary_image_delegation_rate' + setting + '.csv', 'w') as fh:
        fh.write(model2.summary().as_csv())
    # predictions:
    predictions = lm.predict(x_test)

    # evaluation:
    MAE = mean_absolute_error(y_test, predictions)
    MSE = mean_squared_error(y_test, predictions)
    RMSE = math.sqrt(mean_squared_error(y_test, predictions))

    print("MAE: ", MAE)
    print("MSE: ", MSE)
    print("RMSE: ", RMSE)


def split_dataframe_to_overall_image_results(image_formal_df, image_social_df, image_pheno_df, image_intuitive_df,
                                             path_overall, path_to_save):
    # this function combines the image results of all treatment groups to one dataframe for each space. It excludes the
    # baseline and the ai_delegation treatment group

    image_list_formal = image_formal_df['Filename'].to_list()
    dataframe_formal_overall = pd.DataFrame(index=image_list_formal)
    filename_overall = path_overall + '/delete_outliers.xlsx'
    df_answers_formal_all = pd.read_excel(filename_overall)
    mask = df_answers_formal_all.experiment_group.str.contains("base")
    df_rest_delegate_formal = df_answers_formal_all[~mask]

    mask = df_rest_delegate_formal.experiment_group.str.contains("ai_delegate")
    df_answers_formal = df_rest_delegate_formal[~mask]
    series_formal = df_answers_formal['answers_formal']
    treatment_group = 'overall'
    dataframe_formal_overall = create_file_with_accuracies_for_each_image(image_list_formal, series_formal,
                                                                          treatment_group,
                                                                          dataframe_formal_overall, image_formal_df)

    dataframe_formal_overall = dataframe_formal_overall.drop(
        dataframe_formal_overall[dataframe_formal_overall.overall_accuracy == 0].index)
    df_formal_easy = dataframe_formal_overall.loc[dataframe_formal_overall['overall_accuracy'] >= 0.495]
    df_formal_diff = dataframe_formal_overall.loc[dataframe_formal_overall['overall_accuracy'] < 0.495]
    df_formal_easy.to_excel(
        path_to_save + '/seperated_diff_and_easy/overall_formal_easy.xlsx')
    df_formal_diff.to_excel(
        path_to_save + '/seperated_diff_and_easy/overall_formal_diff.xlsx')
    dataframe_formal_overall.to_excel(path_to_save + '/overall_formal.xlsx')

    image_list_social = image_social_df['Filename'].to_list()
    dataframe_social_overall = pd.DataFrame(index=image_list_social)
    filename_overall = path_overall + '/delete_outliers.xlsx'
    df_answers_social_all = pd.read_excel(filename_overall)
    mask = df_answers_social_all.experiment_group.str.contains("base")
    df_rest_delegate_social = df_answers_social_all[~mask]

    mask = df_rest_delegate_social.experiment_group.str.contains("ai_delegate")
    df_answers_social = df_rest_delegate_social[~mask]
    series_social = df_answers_social['answers_social']
    treatment_group = 'overall'
    dataframe_social_overall = create_file_with_accuracies_for_each_image(image_list_social, series_social,
                                                                          treatment_group,
                                                                          dataframe_social_overall, image_social_df)

    dataframe_social_overall = dataframe_social_overall.drop(
        dataframe_social_overall[dataframe_social_overall.overall_accuracy == 0].index)
    df_social_easy = dataframe_social_overall.loc[dataframe_social_overall['overall_accuracy'] >= 0.651]
    df_social_diff = dataframe_social_overall.loc[dataframe_social_overall['overall_accuracy'] < 0.651]
    df_social_easy.to_excel(
        path_to_save + '/seperated_diff_and_easy/overall_social_easy.xlsx')
    df_social_diff.to_excel(
        path_to_save + '/seperated_diff_and_easy/overall_social_diff.xlsx')
    dataframe_social_overall.to_excel(path_to_save + '/overall_social.xlsx')

    image_list_pheno = image_pheno_df['Filename'].to_list()
    dataframe_pheno_overall = pd.DataFrame(index=image_list_pheno)
    filename_overall = path_overall + '/delete_outliers.xlsx'
    df_answers_pheno_all = pd.read_excel(filename_overall)
    mask = df_answers_pheno_all.experiment_group.str.contains("base")
    df_rest_delegate_pheno = df_answers_pheno_all[~mask]

    mask = df_rest_delegate_pheno.experiment_group.str.contains("ai_delegate")
    df_answers_pheno = df_rest_delegate_pheno[~mask]
    series_pheno = df_answers_pheno['answers_pheno']
    treatment_group = 'overall'
    dataframe_pheno_overall = create_file_with_accuracies_for_each_image(image_list_pheno, series_pheno,
                                                                         treatment_group,
                                                                         dataframe_pheno_overall, image_pheno_df)

    dataframe_pheno_overall = dataframe_pheno_overall.drop(
        dataframe_pheno_overall[dataframe_pheno_overall.overall_accuracy == 0].index)
    df_pheno_easy = dataframe_pheno_overall.loc[dataframe_pheno_overall['overall_accuracy'] >= 0.392]
    df_pheno_diff = dataframe_pheno_overall.loc[dataframe_pheno_overall['overall_accuracy'] < 0.392]
    df_pheno_easy.to_excel(
        path_to_save + '/seperated_diff_and_easy/overall_pheno_easy.xlsx')
    df_pheno_diff.to_excel(
        path_to_save + '/seperated_diff_and_easy/overall_pheno_diff.xlsx')
    dataframe_pheno_overall.to_excel(path_to_save + '/overall_pheno.xlsx')

    image_list_intuitive = image_intuitive_df['Filename'].to_list()
    dataframe_intuitive_overall = pd.DataFrame(index=image_list_intuitive)
    filename_overall = path_overall + '/delete_outliers.xlsx'
    df_answers_intuitive_all = pd.read_excel(filename_overall)
    mask = df_answers_intuitive_all.experiment_group.str.contains("base")
    df_rest_delegate_intuitive = df_answers_intuitive_all[~mask]

    mask = df_rest_delegate_intuitive.experiment_group.str.contains("ai_delegate")
    df_answers_intuitive = df_rest_delegate_intuitive[~mask]
    series_intuitive = df_answers_intuitive['answers_intuitive']
    treatment_group = 'overall'
    dataframe_intuitive_overall = create_file_with_accuracies_for_each_image(image_list_intuitive, series_intuitive,
                                                                             treatment_group,
                                                                             dataframe_intuitive_overall,
                                                                             image_intuitive_df)

    dataframe_intuitive_overall = dataframe_intuitive_overall.drop(
        dataframe_intuitive_overall[dataframe_intuitive_overall.overall_accuracy == 0].index)
    df_intuitive_easy = dataframe_intuitive_overall.loc[dataframe_intuitive_overall['overall_accuracy'] >= 0.413]
    df_intuitive_diff = dataframe_intuitive_overall.loc[dataframe_intuitive_overall['overall_accuracy'] < 0.413]
    df_intuitive_easy.to_excel(path_to_save + '/seperated_diff_and_easy/overall_intuitive_easy.xlsx')
    df_intuitive_diff.to_excel(path_to_save + '/seperated_diff_and_easy/overall_intuitive_diff.xlsx')
    dataframe_intuitive_overall.to_excel(path_to_save + '/overall_intuitive.xlsx')


def perform_single_class_regression(x, y, path_to_save, setting):
    # this is a generic function to perform a single variable regression analysis
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

    # train the model:
    lm = LinearRegression()
    lm.fit(x_train, y_train)

    co = lm.coef_.tolist()
    intercept = lm.intercept_
    co.append(intercept)

    cdf = pd.DataFrame(co, [x.columns, 'intercept'], columns=['Coefficients_Accuracy_' + setting])
    cdf.to_excel(path_to_save + '/acc_' + setting + '.xlsx')

    # model summary:
    x2 = sm.add_constant(x)
    model = sm.OLS(y, x2)
    model2 = model.fit()

    with open(path_to_save + '/summary_single_class' + setting + '.txt', 'w') as fh:
        fh.write(model2.summary().as_text())
    with open(path_to_save + '/summary_single_class' + setting + '.csv', 'w') as fh:
        fh.write(model2.summary().as_csv())
    # predictions:
    predictions = lm.predict(x_test)

    # evaluation:
    MAE = mean_absolute_error(y_test, predictions)
    MSE = mean_squared_error(y_test, predictions)
    RMSE = math.sqrt(mean_squared_error(y_test, predictions))

    print("MAE: ", MAE)
    print("MSE: ", MSE)
    print("RMSE: ", RMSE)

    return co


def gender_regression(df, gender_path_to_save, mode):
    # this is a function to perform a single variable regression analysis. The dependent variable is gender.
    # The independent variable is either the accuracy or the delegation rate of a participant. It also plots and saves
    # the results.
    df_gender = df.iloc[:, [3, 38, 39, 40, 41, 42, 43, 44, 45]]
    if mode == 'formal':
        df_gender = df_gender.iloc[:, [0, 1, 5]]
    elif mode == 'social':
        df_gender = df_gender.iloc[:, [0, 2, 6]]
    elif mode == 'pheno':
        df_gender = df_gender.iloc[:, [0, 3, 7]]
    else:
        df_gender = df_gender.iloc[:, [0, 4, 8]]

    df_male = df_gender.loc[df_gender['gender'] == 0]
    df_female = df_gender.loc[df_gender['gender'] == 1]
    acc_male = pd.DataFrame(df_male.iloc[:, 1])
    del_rate_male = df_male.iloc[:, 2]
    acc_female = pd.DataFrame(df_female.iloc[:, 1])
    del_rate_female = df_female.iloc[:, 2]
    acc_male_array = acc_male.to_numpy()
    acc_female_array = acc_female.to_numpy()
    acc_male_plot = [x for xs in acc_male_array for x in xs]
    acc_female_plot = [x for xs in acc_female_array for x in xs]

    co_male = perform_single_class_regression(acc_male, del_rate_male, gender_path_to_save, setting=mode + '_male')
    co_female = perform_single_class_regression(acc_female, del_rate_female, gender_path_to_save,
                                                setting=mode + '_female')

    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y_male = [co_male[0] * x + co_male[1] for x in x]
    y_female = [co_female[0] * x + co_female[1] for x in x]
    sns.lineplot(x=x, y=y_male, color='blue', label='Male delegation behavior')
    sns.lineplot(x=x, y=y_female, color='orange', label='Female delegation behavior')
    sns.scatterplot(x=acc_male_plot, y=del_rate_male, marker='o', facecolor='none', edgecolor='blue', alpha=0.5)
    sns.scatterplot(x=acc_female_plot, y=del_rate_female, marker='o', facecolor='none', edgecolor='orange', alpha=0.5)
    plt.title('Gender Differences ' + mode)
    plt.xlabel('Accuracy ' + mode)
    plt.ylabel('Delegation Rate ' + mode)
    plt.legend(loc='upper left')
    plt.savefig(gender_path_to_save + '/' + mode + '.png')
    plt.clf()


def age_regression(df, age_path_to_save, mode):
    # this is a function to perform a single variable regression analysis. The dependent variable is age.
    # The independent variable is either the accuracy or the delegation rate of a participant. It also plots and saves
    # the results.
    df_age = df.loc[:, ['age', 'delegation_rate_' + mode]]
    age = pd.DataFrame(df_age.iloc[:, 0])
    del_rate = df_age.iloc[:, 1]
    co_age = perform_single_class_regression(age, del_rate, age_path_to_save, setting=mode)
    age_array = age.to_numpy()
    age_plot = [x for xs in age_array for x in xs]
    x = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    y = [co_age[0] * x + co_age[1] for x in x]
    sns.lineplot(x=x, y=y, color='blue', label='Delegation behavior based on age')
    sns.scatterplot(x=age_plot, y=del_rate, marker='o', facecolor='none', edgecolor='blue', alpha=0.7)
    plt.title('Age Differences ' + mode)
    plt.xlabel('Age')
    plt.ylabel('Delegation Rate')
    plt.legend(loc='upper left')
    plt.savefig(age_path_to_save + '/' + mode + '.png')
    plt.clf()


def herd_regression(df, order_path_to_save, mode):
    # this is a function to perform a single variable regression analysis. The dependent variable is the order
    # (i.e. the first or the 100th participant). The independent variable is either the accuracy or the delegation rate
    # of a participant. It also plots and saves the results.
    df_order = df.loc[:,
               ['experiment_group_delegation_counter', 'experiment_group_human_delegate', 'delegation_rate_' + mode]]
    df_herd = df_order.loc[df_order['experiment_group_delegation_counter'] == 1]
    df_no_herd = df_order.loc[df_order['experiment_group_human_delegate'] == 1]
    del_rate_herd = df_herd.iloc[:, 2].reset_index(drop=True)
    del_rate_no_herd = df_no_herd.iloc[:, 2].reset_index(drop=True)
    order_herd = pd.DataFrame(range(1, len(del_rate_herd) + 1))
    order_no_herd = pd.DataFrame(range(1, len(del_rate_no_herd) + 1))
    co_herd = perform_single_class_regression(order_herd, del_rate_herd, order_path_to_save, setting=mode + '_herd')
    co_no_herd = perform_single_class_regression(order_no_herd, del_rate_no_herd, order_path_to_save,
                                                 setting=mode + '_no_herd')
    order_herd_array = order_herd.to_numpy()
    order_no_herd_array = order_no_herd.to_numpy()
    order_herd_plot = [x for xs in order_herd_array for x in xs]
    order_no_herd_plot = [x for xs in order_no_herd_array for x in xs]
    x_herd = range(1, len(order_no_herd_plot) + 1)
    y_herd = [co_herd[0] * x + co_herd[1] for x in x_herd]
    x_no_herd = range(1, len(order_no_herd_plot) + 1)
    y_no_herd = [co_no_herd[0] * x + co_no_herd[1] for x in x_no_herd]
    sns.scatterplot(x=order_herd_plot, y=del_rate_herd, marker='o', facecolor='none', edgecolor='blue', alpha=0.7)
    sns.lineplot(x=x_herd, y=y_herd, color='blue',
                 label='Delegation behavior with information' + '\n' + 'about other participants')
    sns.scatterplot(x=order_no_herd_plot, y=del_rate_no_herd, marker='o', facecolor='none', edgecolor='orange',
                    alpha=0.7)
    sns.lineplot(x=x_no_herd, y=y_no_herd, color='orange',
                 label='Delegation behavior without information' + '\n' + 'about other participants')
    plt.title('Herd Behavior ' + mode)
    plt.ylabel('Delegation Rate ' + mode)
    plt.xlabel('Number of prior participants ' + mode)
    plt.legend(loc='upper left')
    plt.savefig(order_path_to_save + '/' + mode + '_herd.png')
    plt.clf()


def self_fulfilling_prophecy_regression(df, sfp_path_to_save, mode):
    # this is a function to perform a single variable regression analysis. The dependent variable is the number of
    # positive or negative feedback messages for the self-fulfilling prophecy experiment. The independent variable is
    # either the accuracy or the delegation rate of a participant. It also plots and saves the results.
    mask = df.experiment_group.str.contains("base")
    df_rest = df[~mask]

    mask = df_rest.experiment_group.str.contains("ai_delegate")
    df_sfp = df_rest[~mask]
    df_sfp.reset_index(drop=True, inplace=True)
    df_sfp = df_sfp.loc[:, ['expectation', 'first_task', 'second_task', 'third_task', 'fourth_task',
                            'negative_prophecy_formal', 'positive_prophecy_formal', 'negative_prophecy_social',
                            'positive_prophecy_social', 'negative_prophecy_pheno', 'positive_prophecy_pheno',
                            'negative_prophecy_intuitive', 'positive_prophecy_intuitive', 'delegation_rate_formal',
                            'delegation_rate_social', 'delegation_rate_pheno', 'delegation_rate_intuitive']]

    task_list = df_sfp[mode + '_task']
    expectation = pd.DataFrame(df_sfp.iloc[:, 0])
    del_rate = []
    for index, task in enumerate(task_list):
        del_rate_val = df_sfp['delegation_rate_' + task][index]
        del_rate.append(del_rate_val)
    if mode == 'first':
        co_first = perform_single_class_regression(expectation, del_rate, sfp_path_to_save, setting=mode)
        x = [1, 2, 3, 4, 5]
        y = [co_first[0] * x + co_first[1] for x in x]
        expectation_array = expectation.to_numpy()
        expectation_plot = [x for xs in expectation_array for x in xs]
        sns.scatterplot(x=expectation_plot, y=del_rate, marker='o', facecolor='none', edgecolor='blue', alpha=0.7)
        sns.lineplot(x=x, y=y, color='blue', label='Delegation behavior based on the prior expectations')
        plt.xlabel('Prior expectation')
        plt.xticks([1, 2, 3, 4, 5])
        plt.ylabel('Delegation Rate ' + mode)
        plt.title('Influence of the prior expectation on the delegation rate')
        plt.legend(loc='upper left')
        plt.savefig(sfp_path_to_save + '/' + mode + '.png')
        plt.clf()
    else:
        if mode == 'second':
            neg_list, pos_list = return_negative_and_positive_prophecy_list(df_sfp, 2)
            neg_df = pd.DataFrame(neg_list)
            pos_df = pd.DataFrame(pos_list)
            co_neg = perform_single_class_regression(neg_df, del_rate, sfp_path_to_save, setting=mode + '_neg')
            co_pos = perform_single_class_regression(pos_df, del_rate, sfp_path_to_save, setting=mode + '_pos')
            x_neg = [0, 1]
            x_pos = [0, 1]
        elif mode == 'third':
            neg_list, pos_list = return_negative_and_positive_prophecy_list(df_sfp, 3)
            neg_df = pd.DataFrame(neg_list)
            pos_df = pd.DataFrame(pos_list)
            co_neg = perform_single_class_regression(neg_df, del_rate, sfp_path_to_save, setting=mode + '_neg')
            co_pos = perform_single_class_regression(pos_df, del_rate, sfp_path_to_save, setting=mode + '_pos')
            x_neg = [0, 1, 2]
            x_pos = [0, 1, 2]
        else:
            neg_list, pos_list = return_negative_and_positive_prophecy_list(df_sfp, 4)
            neg_df = pd.DataFrame(neg_list)
            pos_df = pd.DataFrame(pos_list)
            co_neg = perform_single_class_regression(neg_df, del_rate, sfp_path_to_save, setting=mode + '_neg')
            co_pos = perform_single_class_regression(pos_df, del_rate, sfp_path_to_save, setting=mode + '_pos')
            x_neg = [0, 1, 2, 3]
            x_pos = [0, 1, 2, 3]

        y_neg = [co_neg[0] * x + co_neg[1] for x in x_neg]
        sns.lineplot(x=x_neg, y=y_neg, color='blue',
                     label='Delegation Behavior based on the number' + '\n' + 'of negative feedback loops')
        sns.scatterplot(x=neg_list, y=del_rate, marker='o', facecolor='none', edgecolor='blue', alpha=0.7)
        plt.title('Negative prior communication in the ' + mode + ' task')
        plt.ylabel('Delegation Rate')
        plt.xlabel('Number of negative feedback messages in prior tasks')
        plt.xticks(x_neg)
        plt.legend(loc='upper left')
        plt.savefig(sfp_path_to_save + '/' + mode + '_neg.png')
        plt.clf()
        y_pos = [co_pos[0] * x + co_pos[1] for x in x_pos]
        sns.lineplot(x=x_pos, y=y_pos, color='blue',
                     label='Delegation Behavior based on the number' + '\n' + 'of positive feedback loops')
        sns.scatterplot(x=pos_list, y=del_rate, marker='o', facecolor='none', edgecolor='blue', alpha=0.7)
        plt.title('Positive prior communication in the ' + mode + ' task')
        plt.ylabel('Delegation Rate')
        plt.xlabel('Number of positive feedback messages in prior tasks')
        plt.xticks(x_pos)
        plt.legend(loc='upper left')
        plt.savefig(sfp_path_to_save + '/' + mode + '_pos.png')
        plt.clf()


def feedback_regression(df, feedback_path_to_save, mode):
    # this is a function to perform a single variable regression analysis. The dependent variable is whether a
    # participant received negative feedback or not. The independent variable is either the accuracy or the delegation
    # rate of a participant. It also plots and saves the results.
    df_feedback = df.loc[:,
                  ['negative_feedback_first', 'negative_feedback_second', 'negative_feedback_third',
                   'negative_feedback_fourth', 'first_task_formal', 'first_task_social',
                   'first_task_pheno', 'first_task_intuitive', 'second_task_formal', 'second_task_social',
                   'second_task_pheno', 'second_task_intuitive',
                   'third_task_formal', 'third_task_social', 'third_task_pheno', 'third_task_intuitive',
                   'fourth_task_formal', 'fourth_task_social', 'fourth_task_pheno',
                   'fourth_task_intuitive', 'delegation_rate_formal', 'delegation_rate_social',
                   'delegation_rate_pheno', 'delegation_rate_intuitive']]

    feedback_first = df_feedback['negative_feedback_first']
    feedback_second = df_feedback['negative_feedback_second']
    feedback_third = df_feedback['negative_feedback_third']
    feedback_fourth = df_feedback['negative_feedback_fourth']

    first_task_formal = df_feedback['first_task_formal'].to_list()
    second_task_formal = df_feedback['second_task_formal'].to_list()
    third_task_formal = df_feedback['third_task_formal'].to_list()
    fourth_task_formal = df_feedback['fourth_task_formal'].to_list()
    first_task_social = df_feedback['first_task_social'].to_list()
    second_task_social = df_feedback['second_task_social'].to_list()
    third_task_social = df_feedback['third_task_social'].to_list()
    fourth_task_social = df_feedback['fourth_task_social'].to_list()
    first_task_pheno = df_feedback['first_task_pheno'].to_list()
    second_task_pheno = df_feedback['second_task_pheno'].to_list()
    third_task_pheno = df_feedback['third_task_pheno'].to_list()
    fourth_task_pheno = df_feedback['fourth_task_pheno'].to_list()

    first_task_list = []
    second_task_list = []
    third_task_list = []
    fourth_task_list = []
    for i in range(len(first_task_formal)):
        if first_task_formal[i] == 1:
            first_task = 'formal'
        elif first_task_social[i] == 1:
            first_task = 'social'
        elif first_task_pheno[i] == 1:
            first_task = 'pheno'
        else:
            first_task = 'intuitive'
        first_task_list.append(first_task)

        if second_task_formal[i] == 1:
            second_task = 'formal'
        elif second_task_social[i] == 1:
            second_task = 'social'
        elif second_task_pheno[i] == 1:
            second_task = 'pheno'
        else:
            second_task = 'intuitive'
        second_task_list.append(second_task)

        if third_task_formal[i] == 1:
            third_task = 'formal'
        elif third_task_social[i] == 1:
            third_task = 'social'
        elif third_task_pheno[i] == 1:
            third_task = 'pheno'
        else:
            third_task = 'intuitive'
        third_task_list.append(third_task)

        if fourth_task_formal[i] == 1:
            fourth_task = 'formal'
        elif fourth_task_social == 1:
            fourth_task = 'social'
        elif fourth_task_pheno == 1:
            fourth_task = 'pheno'
        else:
            fourth_task = 'intuitive'
        fourth_task_list.append(fourth_task)

    del_rate = []
    if mode == 'first':
        task_list = first_task_list
        for index, task in enumerate(task_list):
            del_rate_val = df_feedback['delegation_rate_' + task][index]
            del_rate.append(del_rate_val)
        nr_neg_feedback = feedback_first
        x = [0, 1]
    elif mode == 'second':
        task_list = second_task_list
        for index, task in enumerate(task_list):
            del_rate_val = df_feedback['delegation_rate_' + task][index]
            del_rate.append(del_rate_val)
        nr_neg_feedback = [x + y for x, y in zip(feedback_first, feedback_second)]
        x = [0, 1, 2]
    elif mode == 'third':
        task_list = third_task_list
        for index, task in enumerate(task_list):
            del_rate_val = df_feedback['delegation_rate_' + task][index]
            del_rate.append(del_rate_val)
        feedback_list = [feedback_first, feedback_second, feedback_third]
        nr_neg_feedback = [sum(x) for x in zip(*feedback_list)]
        x = [0, 1, 2, 3]
    else:
        task_list = fourth_task_list
        for index, task in enumerate(task_list):
            del_rate_val = df_feedback['delegation_rate_' + task][index]
            del_rate.append(del_rate_val)
        feedback_list = [feedback_first, feedback_second, feedback_third, feedback_fourth]
        nr_neg_feedback = [sum(x) for x in zip(*feedback_list)]
        x = [0, 1, 2, 3, 4]
    df_f = pd.DataFrame(nr_neg_feedback)
    co = perform_single_class_regression(df_f, del_rate, feedback_path_to_save, mode)
    y = [co[0] * x + co[1] for x in x]
    sns.lineplot(x=x, y=y, color='blue',
                 label='Delegation behavior based on the number' + '\n' + 'of negative feedbacks in prior tasks or the current one')
    sns.scatterplot(x=nr_neg_feedback, y=del_rate, marker='o', facecolor='none', edgecolor='blue', alpha=0.7)
    plt.title('Negative Feedback ' + mode + ' task')
    plt.xlabel('Number of Tasks where participants received negative feedback')
    plt.ylabel('Delegation Rate')
    plt.xticks(x)
    plt.legend(loc='upper left')
    plt.savefig(feedback_path_to_save + '/' + mode + '.png')
    plt.clf()


def return_negative_and_positive_prophecy_list(df, end):
    # this function computes how often a participant received a positive or a negative feedback message prior to the
    # second, third or fourth task.

    # x=2 because prior to the first task there cannot be any feedback message
    x = 2
    complete_negative_list = []
    complete_positive_list = []
    while x <= end:
        if x == 2:
            mode = 'second'
        elif x == 3:
            mode = 'third'
        else:
            mode = 'fourth'
        task_list = df[mode + '_task']
        negative_loop = []
        positive_loop = []
        for index, task in enumerate(task_list):
            negative_loop_val = df['negative_prophecy_' + task][index]
            positive_loop_val = df['positive_prophecy_' + task][index]
            negative_loop.append(negative_loop_val)
            positive_loop.append(positive_loop_val)
        complete_negative_list.append(negative_loop)
        complete_positive_list.append(positive_loop)
        x += 1

    if len(complete_negative_list) == 1:
        nr_of_negative_experiences = [x1 for x1 in complete_negative_list[0]]
        nr_of_positive_experiences = [x1 for x1 in complete_positive_list[0]]
    elif len(complete_negative_list) == 2:
        nr_of_negative_experiences = [x + y for x, y in zip(complete_negative_list[0], complete_negative_list[1])]
        nr_of_positive_experiences = [x + y for x, y in zip(complete_positive_list[0], complete_positive_list[1])]
    else:
        nr_of_negative_experiences = [sum(x) for x in zip(*complete_negative_list)]
        nr_of_positive_experiences = [sum(x) for x in zip(*complete_positive_list)]

    return nr_of_negative_experiences, nr_of_positive_experiences


def calculate_ai_delegation_group_accuracy(df, formal_image_df, social_image_df, pheno_image_df, intuitive_image_df,
                                           df_image_results_formal, df_image_results_social, df_image_results_pheno,
                                           df_image_results_intuitive):
    # this function computes the accuracy for the ai_delegation treatment group which only received images where the AI
    # decided to delegate the image. In order to get a comparable accuracy result it was therefore necessary to combine
    # the accuracy of the AI and the accuracy of the human participants. This was done in the following way:

    # 1. act as if this team setting has to solve the exact same images as a participant in the baseline group, meaning
    # for each participant in the baseline group, we checked which images each participant had to classify

    # 2. for each image check whether the AI wants to delegate or not. If it doesn't delegate use the AI answers to mark
    # it as correct or incorrect. If it delegates use the average accuracy of the human participant in the ai_delegation
    # treatment group over all participants for that image

    # 3. this way calculate an accuracy score in the ai_delegation treatment group for each participant in the baseline
    # group

    # 4. average these scores to get one final average accuracy score for the ai_delegation treatment group

    # get the list of images that were seen by participants in all spaces
    df_images_seen = df.loc[:, ['experiment_group', 'formal_images', 'social_images', 'pheno_images', 'intuitive_images']]
    df_image_results_formal = df_image_results_formal.loc[:, ['Unnamed: 0', 'ai_delegation_accuracy']]
    df_image_results_social = df_image_results_social.loc[:, ['Unnamed: 0', 'ai_delegation_accuracy']]
    df_image_results_pheno = df_image_results_pheno.loc[:, ['Unnamed: 0', 'ai_delegation_accuracy']]
    df_image_results_intuitive = df_image_results_intuitive.loc[:, ['Unnamed: 0', 'ai_delegation_accuracy']]

    # split these images to include only those seen by the baseline group
    mask = df_images_seen.experiment_group.str.contains("base")
    df_base = df_images_seen[mask]
    df_base.drop(['experiment_group'], axis=1, inplace=True)
    df_image_results_formal = (df_image_results_formal.drop
                               (df_image_results_formal[df_image_results_formal.ai_delegation_accuracy == 0].index))
    df_image_results_social = (df_image_results_social.drop
                               (df_image_results_social[df_image_results_social.ai_delegation_accuracy == 0].index))
    df_image_results_pheno = (df_image_results_pheno.drop
                              (df_image_results_pheno[df_image_results_pheno.ai_delegation_accuracy == 0].index))
    df_image_results_intuitive = (df_image_results_intuitive.drop(df_image_results_intuitive[
                                       df_image_results_intuitive.ai_delegation_accuracy == 0].index))

    base_formal = df_base['formal_images']
    base_social = df_base['social_images']
    base_pheno = df_base['pheno_images']
    base_intuitive = df_base['intuitive_images']
    # for each space, for all images seen by the baseline group, calculate the ai_delegation accuracy score
    for index in base_formal.index:
        # get the list of images seen:
        image_list = ast.literal_eval(json.loads(base_formal[index]))
        correctly = 0
        incorrectly = 0
        accuracy_p = []
        # for all images in this list: if it doesn't get delegated use the AI answers
        for image in image_list:
            i = formal_image_df.index[formal_image_df['Filename'] == image]
            delegate = formal_image_df.loc[i, 'Delegate'].item()
            if not delegate:
                correct_ai = formal_image_df.loc[i, 'Correctly_Classified'].item()
                if correct_ai:
                    correctly += 1
                else:
                    incorrectly += 1
            # otherwise use the average human accuracy for this image
            else:
                id_ = df_image_results_formal.index[df_image_results_formal['Unnamed: 0'] == image]
                if len(id_) < 1:
                    pass
                else:
                    avg = df_image_results_formal.loc[id_, 'ai_delegation_accuracy'].item()
                    correctly += avg
                    incorrectly += (1 - avg)

        participant_acc = correctly/(correctly + incorrectly)
        accuracy_p.append(participant_acc)
    avg_acc_formal = sum(accuracy_p)/len(accuracy_p)

    for index in base_social.index:
        image_list = ast.literal_eval(json.loads(base_social[index]))
        correctly = 0
        incorrectly = 0
        accuracy_p = []
        for image in image_list:
            i = social_image_df.index[social_image_df['Filename'] == image]
            delegate = social_image_df.loc[i, 'Delegate'].item()
            if not delegate:
                correct_ai = social_image_df.loc[i, 'Correctly_Classified'].item()
                if correct_ai:
                    correctly += 1
                else:
                    incorrectly += 1
            else:
                id_ = df_image_results_social.index[df_image_results_social['Unnamed: 0'] == image]
                if len(id_) < 1:
                    pass
                else:
                    avg = df_image_results_social.loc[id_, 'ai_delegation_accuracy'].item()
                    correctly += avg
                    incorrectly += (1 - avg)

        participant_acc = correctly/(correctly + incorrectly)
        accuracy_p.append(participant_acc)
    avg_acc_social = sum(accuracy_p)/len(accuracy_p)

    for index in base_pheno.index:
        image_list = ast.literal_eval(json.loads(base_pheno[index]))
        correctly = 0
        incorrectly = 0
        accuracy_p = []
        for image in image_list:
            i = pheno_image_df.index[pheno_image_df['Filename'] == image]
            delegate = pheno_image_df.loc[i, 'Delegate'].item()
            if not delegate:
                correct_ai = pheno_image_df.loc[i, 'Correctly_Classified'].item()
                if correct_ai:
                    correctly += 1
                else:
                    incorrectly += 1
            else:
                id_ = df_image_results_pheno.index[df_image_results_pheno['Unnamed: 0'] == image]
                if len(id_) < 1:
                    pass
                else:
                    avg = df_image_results_pheno.loc[id_, 'ai_delegation_accuracy'].item()
                    correctly += avg
                    incorrectly += (1 - avg)

        participant_acc = correctly/(correctly + incorrectly)
        accuracy_p.append(participant_acc)
    avg_acc_pheno = sum(accuracy_p)/len(accuracy_p)

    for index in base_intuitive.index:
        image_list = ast.literal_eval(json.loads(base_intuitive[index]))
        correctly = 0
        incorrectly = 0
        accuracy_p = []
        for image in image_list:
            i = intuitive_image_df.index[intuitive_image_df['Filename'] == image]
            delegate = intuitive_image_df.loc[i, 'Delegate'].item()
            if not delegate:
                correct_ai = intuitive_image_df.loc[i, 'Correctly_Classified'].item()
                if correct_ai:
                    correctly += 1
                else:
                    incorrectly += 1
            else:
                id_ = df_image_results_intuitive.index[df_image_results_intuitive['Unnamed: 0'] == image]
                if len(id_) < 1:
                    pass
                else:
                    avg = df_image_results_intuitive.loc[id_, 'ai_delegation_accuracy'].item()
                    correctly += avg
                    incorrectly += (1 - avg)

        participant_acc = correctly/(correctly + incorrectly)
        accuracy_p.append(participant_acc)
    avg_acc_intuitive = sum(accuracy_p)/len(accuracy_p)

    return avg_acc_formal, avg_acc_social, avg_acc_pheno, avg_acc_intuitive


def outlier_detection_z_score(data):
    # performs a simple outlier detection using the z-score method and returns the index of the outliers
    mean = np.mean(data)
    std = np.std(data)
    threshold = 3
    outliers = []
    outlier_index = []

    for i, x_z in enumerate(data):
        z = (x_z - mean) / std
        if np.abs(z) > threshold:
            outliers.append(x_z)
            outlier_index.append(i)

    return outlier_index


def outlier_detection_iqr(data):
    # performs a simple outlier detection using the inter quantile range method and returns the index of the outliers
    data_sorted = sorted(data)
    q1, q3 = np.quantile(data_sorted, [0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = []
    outlier_index = []

    for id_, x_iqr in enumerate(data):
        if x_iqr < lower_bound or x_iqr > upper_bound:
            outliers.append(x_iqr)
            outlier_index.append(id_)
    return outlier_index
