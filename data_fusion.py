import pandas as pd
from pprint import pprint
import numpy as np
import math

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

def atomisation(dataframe):
    all_rows = []
    for i in range(dataframe.shape[0]):
        row = dataframe.iloc[[i]]
        all_rows.append(row)
    return all_rows


def fusion_dataframe(dataframe1, dataframe2):
    dataframe_fusion = pd.concat([dataframe1, dataframe2], ignore_index=True)
    return dataframe_fusion


# calcul de l'entropie  et du delta impureté


def calc_shannon(variable):
    return drv.entropy(variable, estimator='ML')


def calc_min_shannon_dataset(dataset):
    entropy_variable = {}
    for line in dataset:
        print(line)
        entropy_variable[line] = calc_shannon(dataset[line])
    print(entropy_variable)
    return min(entropy_variable.items(), key=lambda x: x[1])


def calc_delta_impurity(dataframe1, dataframe2):
    delta_impurity = {}
    data = pd.concat([dataframe1, dataframe2], ignore_index=True)
    p1 = dataframe1.shape[0] / data.shape[0]
    p2 = dataframe2.shape[0] / data.shape[0]
    for line in data:
        delta_impurity[line] = (
                calc_shannon(data[line]) - p1 * calc_shannon(dataframe1[line]) - p2 * calc_shannon(dataframe2[line]))
    return delta_impurity[min(delta_impurity, key=delta_impurity.get)]


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

def fusion(dataframe1, dataframe2, threshold=math.inf):
    fused = fusion_dataframe(dataframe1, dataframe2)
    rows = atomisation(fused)
    impurity = []
    while len(rows) > 1:
        print(len(rows))
        dic = choice_fusion(rows)
        rows = dic['rows']
        impurity.append(dic['impurity'])
    return rows, impurity


#
# def min_mutual_information(dataframe):
#     mutual_information = {}
#     for variable1 in dataframe :
#         for variable2 in dataframe:
#             if not variable1.equals(variable2) :


def information_mutual(dataframe: pandas.core.frame.DataFrame, variable: str):
    column = dataframe[variable]
    dataframe = dataframe.drop(columns=variable)
    return drv.information_mutual(dataframe, column, cartesian_product=True)


if __name__ == '__main__':
    dataframe = create_dataframe()
    data = pd.DataFrame([['youth', 'yes', 'no', 'just so-so', 'no']], columns=features)
    fused = fusion_dataframe(data, dataframe)
    # print(calc_delta_impurity2(data,dataframe))
    # print(calc_delta_impurity(data,dataframe))
    # print(calc_shannon2(data['credit']))
    rows = atomisation(fused)
    print(information_mutual(dataframe, 'age'))
    # print(fusion(dataframe, data))
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
