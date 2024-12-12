import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# this file is used to create the plots for the image regression analysis which is divided into difficult and
# easy images. The splitting criterion is the average human accuracy in each space
folder = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/regression_results_images'
margin_path = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/image_regression_results_final'
results_plots_folder = 'C:/Users/ferdi/OneDrive/Masterarbeit/human results/results_plots'
filelist = os.listdir(folder)
for file in filelist:
    if file.endswith('.xlsx'):
        if 'overall' in file:
            name = file.split('.')[0]
            file_name_margin_easy = 'acc_' + name + '_easy.xlsx'
            file_name_margin_diff = 'acc_' + name + '_diff.xlsx'
            file_path_margin_easy = os.path.join(margin_path, file_name_margin_easy)
            file_path_margin_diff = os.path.join(margin_path, file_name_margin_diff)
            m_easy_df = pd.read_excel(file_path_margin_easy)
            m_diff_df = pd.read_excel(file_path_margin_diff)
            m_easy = m_easy_df.iloc[:, 1].to_list()[0]
            m_diff = m_diff_df.iloc[:, 1].to_list()[0]
            i_easy = m_easy_df.iloc[:, 1].to_list()[1]
            i_diff = m_diff_df.iloc[:, 1].to_list()[1]
            if 'formal' in name:
                x_diff = [0, 0.1, 0.2, 0.3, 0.4, 0.4949]
                x_easy = [0.495, 0.6, 0.7, 0.8, 0.9, 1.0]
                avg_acc = str(49.5)
            elif 'social' in name:
                x_diff = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6509]
                x_easy = [0.651, 0.7, 0.8, 0.9, 1.0]
                avg_acc = str(65.1)
            elif 'pheno' in name:
                x_diff = [0, 0.1, 0.2, 0.3, 0.3919]
                x_easy = [0.392, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                avg_acc = str(39.2)
            else:
                x_diff = [0, 0.1, 0.2, 0.3, 0.4, 0.4129]
                x_easy = [0.413, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                avg_acc = str(41.3)
            header = name.split('_')[1]
            y_easy = [i_easy + (m_easy * x) for x in x_easy]
            y_diff = [i_diff + (m_diff * x) for x in x_diff]
            filepath = os.path.join(folder, file)
            df = pd.read_excel(filepath)
            acc = df.iloc[:, 1].to_list()
            del_rate = df.iloc[:, 2].to_list()
            sns.scatterplot(x=acc, y=del_rate, marker='o', facecolors='none', edgecolors='blue')
            sns.lineplot(x=x_easy, y=y_easy, color='blue', label='Delegation behavior depending on the' + '\n' +
                                                                 'image difficulty (difficult < ' + avg_acc +
                                                                 '%, easy >= ' + avg_acc + '%)')
            sns.lineplot(x=x_diff, y=y_diff, color='blue')
            plt.yticks(np.arange(0, 1, 0.1))
            plt.xticks(np.arange(0, 1, 0.1))
            plt.title(header)
            plt.xlabel('Accuracy')
            plt.ylabel('Delegation Rate')
            plt.legend(loc='upper left')
            if 'overall' in name:
                plt.savefig(results_plots_folder + '/' + name + '.png')
                plt.clf()
            else:
                plt.clf()











