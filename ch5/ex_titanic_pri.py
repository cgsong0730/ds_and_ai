import pandas as pd

titanic = pd.read_csv('dataset/titanic.csv')
attributes = ['Name', 'Sex', 'Age', 'Ticket']
unique_profiles = titanic[attributes].drop_duplicates().shape[0]
print(unique_profiles, len(titanic))
print(titanic[attributes].head(3))
