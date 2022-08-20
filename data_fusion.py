from math import log

import scipy
import pandas as pd
from pprint import pprint
import numpy as np

# Construct dataset
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
           ['geriatric', 'no', 'no', 'just so-so', 'no']]
features = ['age', 'work', 'house', 'feelings', 'credit']


def create_dataset():
    return dataset, features


def create_dataframe():
    dataframe = pd.DataFrame(dataset, columns=features)
    return dataframe


# Calculates the shannon entropy coefficient of the current set
def atomisation(dataframe):
    all_rows = []
    for i in range(dataframe.shape[0]):
        row = dataframe.iloc[i, :]
        row
        all_rows.append(row)
    return all_rows


def calcShannon(variable):
    pd_series = pd.Series(variable)
    counts = pd_series.value_counts()
    entropy = scipy.stats.entropy(counts)
    return entropy


def calc_min_shannon_dataset(dataset):
    entropy_variable = {}
    for line in dataset:
        print(line)
        entropy_variable[line] = calcShannon(dataset[line])
    print(entropy_variable)
    return min(entropy_variable.items(), key=lambda x: x[1])


def fusion_dataframe(dataframe1, dataframe2):
    dataframe_fusion = pd.concat([dataframe1, dataframe2], ignore_index=True)
    return dataframe_fusion


def calc_delta_impurity(dataframe1, dataframe2):
    delta_impurity = {}
    data = pd.concat([dataframe1, dataframe2], ignore_index=True)
    p1 = dataframe1.shape[0] / data.shape[0]
    p2 = dataframe2.shape[0] / data.shape[0]
    for line in data:
        delta_impurity[line] = (
                calcShannon(data[line]) - p1 * calcShannon(dataframe1[line]) - p2 * calcShannon(dataframe2[line]))
    return min(delta_impurity.items(), key=lambda x: x[1])


def choice_fusion(all_rows: list):
    n = len(all_rows)
    impurities = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i != j:
                impurities[i][j] = calc_delta_impurity(all_rows[i], all_rows[j])
                impurities[i][j] = calc_delta_impurity(all_rows[j], all_rows[i])
            else:
                impurities[i][j] = None
    index_min = np.unravel_index(np.argmin(impurities), impurities.shape)
    print([all_rows[index_min[0]].shape, all_rows[index_min[1]].shape, impurities[index_min[0]][index_min[1]]])
    all_rows[index_min[0]] = pd.concat([all_rows[index_min[0]], all_rows[index_min[1]]], ignore_index=True)
    all_rows[index_min[1]].pop()
    return all_rows


# Extract subset
# Function: first find all samples with axis tag value = value from the dataSet
# Then delete the axis tag value of these samples, and then extract them all into a new sample set
def create_sub_dataset(dataset, index, value):
    sub_dataset = []
    for example in dataset:
        current_list = []
        if example[index] == value:
            current_list = example[:index]
            current_list.extend(example[index + 1:])
            sub_dataset.append(current_list)
    return sub_dataset


# Divide the current sample set into a part with value of feature i and a part with value other than value (two points)
def split_dataset(dataset, index, value):
    sub_dataset1 = []
    sub_dataset2 = []
    for example in dataset:
        current_list = []
        if example[index] == value:
            current_list = example[:index]
            current_list.extend(example[index + 1:])
            sub_dataset1.append(current_list)
        else:
            current_list = example[:index]
            current_list.extend(example[index + 1:])
            sub_dataset2.append(current_list)
    return sub_dataset1, sub_dataset2


def choose_best_feature(dataset):
    # Total number of features
    numFeatures = len(dataset[0]) - 1
    # When there is only one feature
    if numFeatures == 1:
        return 0
    # Initialize optimal Gini coefficient
    bestGini = 1
    # Initialize optimal features
    index_of_best_feature = -1
    # Traverse all features to find the optimal feature and the optimal tangent point under the feature
    for i in range(numFeatures):
        # De duplication, each attribute value is unique
        uniqueVals = set(example[i] for example in dataset)
        # Each value in the Gini dictionary represents the Gini coefficient after dividing the current set with the key corresponding to the value as the tangent point
        Gini = {}
        # For each value of the current feature
        for value in uniqueVals:
            # First find the two subsets divided by the value
            sub_dataset1, sub_dataset2 = split_dataset(dataset, i, value)
            # Find the proportion coefficient prob1 and prob2 of the two subsets in the original set
            prob1 = len(sub_dataset1) / float(len(dataset))
            prob2 = len(sub_dataset2) / float(len(dataset))
            # Calculate Gini coefficients for subset 1
            Gini_of_sub_dataset1 = calcShannon(sub_dataset1)
            # Calculate Gini coefficients for subset 2
            Gini_of_sub_dataset2 = calcShannon(sub_dataset2)
            # Calculate the final Gini coefficient divided by the current optimal tangent point
            Gini[value] = prob1 * Gini_of_sub_dataset1 + prob2 * Gini_of_sub_dataset2
            # Update the optimal features and optimal cut points
            if Gini[value] < bestGini:
                bestGini = Gini[value]
                index_of_best_feature = i
                best_split_point = value
    return index_of_best_feature, best_split_point


# Returns the value of the tag with the largest number of samples ('yes' or 'no')
def find_label(classList):
    # Initialize the dictionary for counting the times of each label
    # The key is each label, and the corresponding value is the number of times the label appears
    labelCnt = {}
    for key in classList:
        if key not in labelCnt.keys():
            labelCnt[key] = 0
        labelCnt[key] += 1
    # Sort classCount values in descending order
    # For example: sorted_labelCnt = {'yes': 9, 'no': 6}
    sorted_labelCnt = sorted(labelCnt.items(), key=lambda a: a[1], reverse=True)
    # There is a problem with the following way of writing
    # sortedClassCount = sorted(labelCnt.iteritems(), key=operator.itemgetter(1), reverse=True)
    # Sorted_ The first value of the first element in labelcnt is the desired value
    return sorted_labelCnt[0][0]


def create_decision_tree(dataset, features):
    # Find the labels of all samples in the training set
    # For the initial dataset, its label_list = ['no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    label_list = [example[-1] for example in dataset]
    # First write two cases where recursion ends:
    # If the labels of all samples in the current set are equal (that is, the samples have been classified as "pure")
    # The label value is directly returned as a leaf node
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    # If all features of the training set have been used, there are no available features, but the sample has not been classified as "pure"
    # The label with the most samples is returned as the result
    if len(dataset[0]) == 1:
        return find_label(label_list)
    # The following is the formal establishment process
    # Select the subscript and the best segmentation point of the best feature for branching
    index_of_best_feature, best_split_point = choose_best_feature(dataset)
    # Get the best feature
    best_feature = features[index_of_best_feature]
    # Initialize decision tree
    decision_tree = {best_feature: {}}
    # Delete the current best feature after it has been used
    del (features[index_of_best_feature])
    # Sub feature = current feature (because the used feature has been deleted just now)
    sub_labels = features[:]
    # Recursive call to create_decision_tree to generate new nodes
    # Generate a binary set divided by the optimal tangent point
    sub_dataset1, sub_dataset2 = split_dataset(dataset, index_of_best_feature, best_split_point)
    # Construct left subtree
    decision_tree[best_feature][best_split_point] = create_decision_tree(sub_dataset1, sub_labels)
    # Construct right subtree
    decision_tree[best_feature]['others'] = create_decision_tree(sub_dataset2, sub_labels)
    return decision_tree


# Use the decision tree trained above to classify the new samples
def classify(decision_tree, features, test_example):
    # The attribute represented by the root node
    first_feature = list(decision_tree.keys())[0]
    # second_dict is the value of the first classification attribute (also a dictionary)
    second_dict = decision_tree[first_feature]
    # The position of the attribute represented by the tree root in the attribute tag, that is, the first attribute
    index_of_first_feature = features.index(first_feature)
    # For second_ Every key in Dict
    for key in second_dict.keys():
        # key not equal to 'others'
        if key != 'others':
            if test_example[index_of_first_feature] == key:
                # If current second_ The value of the key of dict is a dictionary
                if type(second_dict[key]).__name__ == 'dict':
                    # Recursive query is required
                    classLabel = classify(second_dict[key], features, test_example)
                # If current second_ The value of the key of dict is a separate value
                else:
                    # Is the tag value to find
                    classLabel = second_dict[key]
            # If the value of the test sample in the current feature is not equal to key, it means that its value in the current feature belongs to 'others'
            else:
                # If second_ If the value of dict ['others'] is a string, it is output directly
                if isinstance(second_dict['others'], str):
                    classLabel = second_dict['others']
                # If second_ If the value of dict ['others'] is a dictionary, the query is recursive
                else:
                    classLabel = classify(second_dict['others'], features, test_example)
    return classLabel


if __name__ == '__main__':
    dataframe = create_dataframe()
    data = pd.DataFrame([['youth', 'yes', 'no', 'just so-so', 'no']], columns=features)
    print(type(dataframe))
    dataframe1 = pd.DataFrame(dataframe.iloc[5, :], columns=features)
    print(dataframe1)
    dataframe3 = pd.DataFrame(dataframe.iloc[1, :], columns=features)

    dataframe4 = pd.concat([dataframe1, dataframe3], ignore_index=True)
    print(dataframe4)
    print(dataframe4.transpose())

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
