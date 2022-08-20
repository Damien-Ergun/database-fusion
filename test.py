import pandas as pd

data = pd.read_csv("C:/Users/damie/Downloads/notes.csv")

# print(data.groupby('matiere').mean())

data_2 = data.groupby('nom').mean()

var = data_2.loc[data_2['note'] < 10, :]

print(var)