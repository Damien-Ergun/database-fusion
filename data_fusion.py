import pandas as pd
from pprint import pprint
import numpy as np
import math
import random
import copy
import re
import inquirer

import pandas.core.frame
from pyitlib import discrete_random_variable as drv

dataset = [['youth', 'yes', 'no', 'just so-so', 'yes'],
           ['youth', 'no', 'yes', 'good', 'yes'],
           ['youth', 'yes', 'no', 'just so-so', 'yes'],
           ['youth', 'yes', 'yes', 'good', 'yes'],
           ['youth', 'no', 'no', 'just so-so', 'yes'],
           ['midlife', 'no', 'yes', 'just so-so', 'no'],
           ['midlife', 'yes', 'no', 'good', 'no'],
           ['midlife', 'yes', 'no', 'good', 'yes'],
           ['midlife', 'no', 'no', 'great', 'no'],
           ['midlife', 'no', 'yes', 'great', 'yes'],
           ['geriatric', 'no', 'yes', 'great', 'no'], 
        ['midlife', 'no', 'yes', 'just so-so', 'no'],
           ['midlife', 'no', 'yes', 'great', 'no'],
           ['youth', 'no', 'yes', 'just so-so', 'no'],
           ['youth', 'yes', 'no', 'good', 'yes'],
           ['youth', 'no', 'yes', 'just so-so', 'no'],
           ['midlife', 'no', 'yes', 'just so-so', 'no'],
           ['midlife', 'no', 'no', 'good', 'yes'],
           ['midlife', 'no', 'yes', 'just so-so', 'no'],
           ['midlife', 'no', 'no', 'great', 'yes'],
           ['midlife', 'no', 'yes', 'great', 'no'],
           ['geriatric', 'yes', 'no', 'good', 'yes']
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
    final = copy.deepcopy(rows)
    impurity = []
    index = 0
    all = []
    while len(rows) > 1:
        print('il reste ', len(rows), ' données à fusionner')
        dic = choice_fusion(rows)
        rows = dic['rows']
        impurity.append(dic['impurity'])
        all.append(dic)
    a = local_min(impurity)
    information_str = display_information(all, a)
    # possible_choices = []
    # questions = [
    #     inquirer.List('size',
    #                   message="quelle fusion voulez vous ?",
    #                   choices=possible_choices,
    #                   ),
    # ]
    # answers = inquirer.prompt(questions)
    print(information_str)
    # for count, value in enumerate(a[1]):
    #     final = all[value]
    #     print(all[value])
    #     if all[value]['impurity'] != 0:
    #         answer = input('voulez vous cette fusion ? [y/n]')
    #         if answer == 'y':
    #             break



def display_information(all, a, dict=False) :
    information = {}
    information_str = {}
    for count, value in enumerate(a[1]):
        information[f'stage {value}'] = {}
        information_str[f'stage {value}'] = {}
        information[f'stage {value}']['impurity'] = a[0][count]
        information_str[f'stage {value}']['impurity'] = f'impurity : {a[0][count]}'
        information_str[f'stage {value}']['tables'] = ''
        for i, data in enumerate(all[value]['rows']):
            print (i)
            information[f'stage {value}'][f'table {i+1}'] = {}
            information[f'stage {value}'][f'table {i+1}']['taille'] = data.shape[-1]
            information_str[f'stage {value}']['tables'] += f'info dataframe {i} \n size : {data.shape[-1]} \n average entropy : {calc_mean_shannon_dataset(data)} '
            information[f'stage {value}'][f'table {i+1}']['entropie moyenne'] = calc_mean_shannon_dataset(data)
    if dict:
        return information, information_str
    return information_str


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
    return [[y for i, y in enumerate(ys)
             if ((i == 0) or (ys[i - 1] >= y))
             and ((i == len(ys) - 1) or (y < ys[i + 1]))], [i for i, y in enumerate(ys)
                                                            if ((i == 0) or (ys[i - 1] >= y))
                                                            and ((i == len(ys) - 1) or (y < ys[i + 1]))]]


if __name__ == '__main__':
    dataframe = create_dataframe()
    dataframe['test'] = [i for i in range(dataframe.shape[0])]
    data = pd.DataFrame([['youth', 'yes', 'no', 'good', 'no']], columns=features)
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
