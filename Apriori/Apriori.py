import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)]) #list of lists
    
#Training apriori method on the dataset
from apyori import apriori
#products bought 3x a day, 7 days, min support = 3*7/7500
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualize the results
results = list(rules)

AR = list(rules)
result = [list(AR[i][0]) for i in range(0, len(AR))]

