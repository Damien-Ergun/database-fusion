import pandas as pd
from pprint import pprint
import numpy as np
import math
import random

import pandas.core.frame
from pyitlib import discrete_random_variable as drv

dataset = [['youth', 'no', 'no', 'just so-so', 'no'],
           ['youth', 'no', 'no', 'good', 'no'],
           ['youth', 'yes', 'no', 'good', 'yes'],
           ['youth', 'yes', 'yes', 'just so-so', 'yes'],
           ['youth', 'no', 'no', 'just so-so', 'no'],
           ['midlife', 'no', 'no', 'just so-so', 'no'],
           ['midlife', 'no', 'no', 'good', 'no'],
           ['midlife', 'yes', 'yes', 'good', 'yes'],
           ['midlife', 'no', 'yes', 'great', 'yes'],
           ['midlife', 'no', 'yes', 'great', 'yes'],
           ['geriatric', 'no', 'yes', 'great', 'yes'],
           ['geriatric', 'no', 'yes', 'good', 'yes'],
           ['geriatric', 'yes', 'no', 'good', 'yes'],
           ['geriatric', 'yes', 'no', 'great', 'yes'],
           ['geriatric', 'no', 'no', 'just so-so', 'no'],
           ['youth', 'yes', 'no', 'just so-so', 'no'],
           ['youth', 'no', 'yes', 'just so-so', 'yes'],
           ['youth', 'yes', 'no', 'good', 'yes'],
           ['midlife', 'no', 'yes', 'just so-so', 'no'],
           ['geriatric', 'yes', 'no', 'just so-so', 'yes'],
           ['youth', 'yes', 'no', 'just so-so', 'no'],
           ['midlife', 'no', 'yes', 'good', 'yes'],
           ['midlife', 'yes', 'yes', 'good', 'yes'],
           ['midlife', 'no', 'yes', 'great', 'yes'],
           ['midlife', 'no', 'yes', 'good', 'no'],
           ['midlife', 'yes', 'yes', 'great', 'no'],
           ['geriatric', 'no', 'yes', 'good', 'yes'],
           ['youth', 'yes', 'yes', 'good', 'yes'],
           ['midlife', 'no', 'yes', 'good', 'yes'],
           ['geriatric', 'no', 'no', 'just so-so', 'no']
           ]
# Construct dataset
features = ['age', 'work', 'house', 'feelings', 'credit']


# creation de data pour test

def create_dataset():
    return dataset, features


def create_dataframe():
    dataframe = pd.DataFrame(dataset, columns=features)
    return dataframe


# atomisation et fusion des bdd

def atomisation(dataframe: pandas.core.frame.DataFrame):
    all_rows = []
    for i in range(dataframe.shape[0]):
        row = dataframe.iloc[[i]]
        all_rows.append(row)

    return all_rows


def fusion_dataframe(dataframe1: pandas.core.frame.DataFrame, dataframe2: pandas.core.frame.DataFrame):
    dataframe_fusion = pd.concat([dataframe1, dataframe2], ignore_index=True)
    return dataframe_fusion


# calcul de l'entropie  et du delta impureté


def calc_shannon(variable: pandas.core.series.Series):
    return drv.entropy(variable, estimator='ML')


def spread(variable: pandas.core.series.Series):
    if variable.shape[0] == 1:
        return 0
    std = variable.std()
    return abs(std / variable.mean())


def calc_mean_shannon_dataset(dataframe: pandas.core.frame.DataFrame):
    entropy_variable = {}
    for column in data:
        if dataframe[column].dtype == np.int64 or data[column].dtype == np.float64:
            entropy_variable[column] = spread(dataframe[column])
        else:
            entropy_variable[column] = calc_shannon(dataframe[column])
    return np.array(list(entropy_variable.values())).mean()


def calc_delta_impurity(dataframe1: pandas.core.frame.DataFrame, dataframe2: pandas.core.frame.DataFrame):
    delta_impurity = {}
    data = pd.concat([dataframe1, dataframe2], ignore_index=True)
    p1 = dataframe1.shape[0] / data.shape[0]
    p2 = dataframe2.shape[0] / data.shape[0]
    for column in data:
        if data[column].dtype == np.int64 or data[column].dtype == np.float64:
            delta_impurity[column] = p1 * spread(dataframe1[column]) + p2 * spread(dataframe2[column])
        else:
            delta_impurity[column] = calc_shannon(data[column]) - p1 * calc_shannon(
                dataframe1[column]) - p2 * calc_shannon(dataframe2[column])
    return np.array(list(delta_impurity.values())).mean()


# fusion des 2 bases ayant le delta le plus bas d'entre tous

def choice_fusion(all_rows: list):
    n = len(all_rows)
    impurities = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i != j:
                impurities[i][j] = calc_delta_impurity(all_rows[i], all_rows[j])
                impurities[j][i] = calc_delta_impurity(all_rows[j], all_rows[i])

            else:
                impurities[i][j] = math.inf
    index_min = np.unravel_index(np.argmin(impurities), impurities.shape)
    all_rows[index_min[0]] = pd.concat([all_rows[index_min[0]], all_rows[index_min[1]]], ignore_index=True)
    all_rows.pop(index_min[1])

    return dict(rows=all_rows, impurity=impurities[index_min[0]][index_min[1]])


# fusion de la base en fonction de l'entropie

def fusion(dataframe1: pandas.core.frame.DataFrame, dataframe2: pandas.core.frame.DataFrame, k=10, threshold=math.inf):
    fused = fusion_dataframe(dataframe1, dataframe2)
    rows = atomisation(fused)
    impurity = []
    mean_entropy = []
    while len(rows) > 1:
        print('il reste ', len(rows), ' données à fusionner')
        dic = choice_fusion(rows)
        print(dic)
        rows = dic['rows']
        impurity.append(dic['impurity'])
        mean_entropy.append([calc_mean_shannon_dataset(dataframe) for dataframe in rows])
        print(mean_entropy[-1])
        if impurity[-1] != 0:
            answer = input('on continue ? [y/n]')
            if answer == 'n':
                break
    return rows, impurity


def two_by_two(rows, seed):
    random.seed(seed)
    rand_rows = []
    while len(rows) > 1:
        index = [i for i in range(len(rows))]
        sample = random.sample(index, k=2)
        rand_rows.append(fusion_dataframe(rows[sample[0]], rows[sample[1]]))
        rows.pop(sample[0])
        rows.pop(sample[1] - 1)
    if len(rows) == 1:
        rand_rows.append(rows[0])

    return rand_rows


#
# def min_mutual_information(dataframe):
#     mutual_information = {}
#     for variable1 in dataframe :
#         for variable2 in dataframe:
#             if not variable1.equals(variable2) :




def local_min(ys):
    return [y for i, y in enumerate(ys)
            if ((i == 0) or (ys[i - 1] >= y))
            and ((i == len(ys) - 1) or (y < ys[i + 1]))]


if __name__ == '__main__':
    dataframe = create_dataframe()
    dataframe['test'] = [i for i in range(dataframe.shape[0])]
    data = pd.DataFrame([['youth', 'yes', 'no', 'just so-so', 'no']], columns=features)
    data['test'] = [-1]
    fused = fusion_dataframe(data, dataframe)
    fused['test'] = [i for i in range(fused.shape[0])]
    rows = atomisation(fused)
    pprint(fusion(dataframe, data))
    # dataset, features = create_dataset()
    # pprint([dataset, features])
    # print(dataframe)
    # decision_tree = create_decision_tree(dataset, features)
    # # Print the generated decision tree
    # print(decision_tree)
    # # Classify the new samples
    # features = ['age', 'work', 'house', 'credit']
    # test_example = ['midlife', 'yes', 'no', 'great']
    # print(classify(decision_tree, features, test_example))
