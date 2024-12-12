import pandas as pd

# this is a simple file to check how many images get delegated in each space depending on different certainty scores
excel_file = 'C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Intuitive_predictions.xlsx'
save_path = 'C:/Users/ferdi/OneDrive/Masterarbeit/Models/Predictions/Intuitive_predictions_with_delegate.xlsx'
df = pd.read_excel(excel_file)
delegate_80_list = []
delegate_70_list = []
delegate_60_list = []
delegate_50_list = []
delegate_40_list = []
delegate_30_list = []
delegate_52_list = []

for index in df.index:
    probability = df.loc[index, 'Probabilities'][2:-2]
    probability_list = probability.split(' ')

    float_prob = [float(x) for x in probability_list if x != '']
    maximum = max(float_prob)

    if maximum > 0.8:
        delegate_80 = False
    else:
        delegate_80 = True
    delegate_80_list.append(delegate_80)

    if maximum > 0.7:
        delegate_70 = False
    else:
        delegate_70 = True
    delegate_70_list.append(delegate_70)

    if maximum > 0.6:
        delegate_60 = False
    else:
        delegate_60 = True
    delegate_60_list.append(delegate_60)

    if maximum > 0.52:
        delegate_52 = False
    else:
        delegate_52 = True
    delegate_52_list.append(delegate_52)

    if maximum > 0.5:
        delegate_50 = False
    else:
        delegate_50 = True
    delegate_50_list.append(delegate_50)

    if maximum > 0.4:
        delegate_40 = False
    else:
        delegate_40 = True
    delegate_40_list.append(delegate_40)

    if maximum > 0.3:
        delegate_30 = False
    else:
        delegate_30 = True
    delegate_30_list.append(delegate_30)

df['delegate_80'] = delegate_80_list
df['delegate_70'] = delegate_70_list
df['delegate_60'] = delegate_60_list
df['delegate_52'] = delegate_52_list
df['delegate_50'] = delegate_50_list
df['delegate_40'] = delegate_40_list
df['delegate_30'] = delegate_30_list

df.to_excel(save_path, index=False)
